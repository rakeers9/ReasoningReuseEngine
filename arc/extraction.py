"""IS2 in number + unit form (config.EXTRACT_FORM == "number_unit").

DEVIATION from the paper, whose IS2 copies each value as written ("$150k", "ten moles").
Builder-written parsers then have to cope with every way a number can be written, and one
read "$150k" as 150. Here the small model returns, for each value:

  number    the full number in plain digits: no shorthand, no separators, no unit
            ("150000" for "$150k", "10" for "ten"); empty if the value is not a number
  unit      its unit or currency ("$", "%", "years", "°C"); empty if it has none
  text      the value, for values that are not numbers ("monthly", "methane")
  evidence  the exact words in the question that state it

Plain code then checks the model's work and rebuilds one value for the cached code, in the
style of that input's format examples ("$150,000", "25 years", "5.5%"):
  - evidence must be in the question (an implied value may leave it empty only when
    implied values are allowed),
  - if the evidence has digits, the number must equal a number in the evidence after
    expanding its shorthand in code (catches "150" for "$150k" and dropped zeros),
  - numbers written as words carry no digits to check and are trusted to the model.
Any failed check raises ExtractionError, so the question goes to the model, never to the
cache with a wrong input.

This runs after the gate: retrieval embeds the question's value-free structure and Clef
reads the raw question, so these fields never reach either.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from typing import Literal

from pydantic import ConfigDict, create_model

from . import config, llm
from .cache import ParamSpec
from .checks import numbers
from .normalize import expand_shorthand

CURRENCY = {"$", "£", "€", "¥", "₹", "US$", "USD", "EUR", "GBP"}

ANSWER_RULES = """\
Then, about the ANSWER the question asks for:
- output_format: unit = the unit the question asks the answer to be given in, copied. If it asks \
for none, but its own inputs include the SAME kind of quantity as the answer (temperatures when it \
asks for a final temperature, lengths when it asks for a length, money when it asks for an amount \
of money), unit = the unit of those inputs as written (the first one if they differ), and the \
evidence is that input as written; the reader expects the answer in the units they used. \
Otherwise "". rounding = how the question asks the answer to be rounded, written as "dp:N" (N \
decimal places), "sf:N" (N significant figures), "nearest:X" (to the nearest X, e.g. nearest:1, \
nearest:1000, nearest:0.01) or "" if it asks for no rounding; evidence = the exact words that ask \
for the unit or the rounding ("" if none).
- extras: every OTHER instruction about the answer itself, for example "explain your reasoning", \
"show each step", "give the answer as a fraction", "answer in words", "also compare it with...". \
Each item: kind ("explanation", "format" or "other"), text, and evidence (the exact words). The \
problem's own setup, method and assumptions are NOT extras ("assuming ideal behaviour", \
"compounded monthly", "using the perfect gas equation", "taking g = 9.81"): leave them out. Use \
an empty list if there are none.
"""

PROMPT = """\
Extract the following values from the input text. For each value fill four fields:
- evidence: the exact words in the input that state the value, copied character for character.{implied}
- number: if the value is a quantity, the full number in plain digits, with no thousands \
separators, no shorthand and no unit: 150000 for "$150k", 2000000 for "2.0 million", 10 for \
"ten", and 5.5 for "5.5%" (not 0.055). Keep the sign and every decimal digit as written. Leave \
empty if the value is not a number.
- unit: the unit or currency of that number, as written ("$", "%", "years", "°C", "dm3", \
"mol"). Leave empty if the number has no unit.
- text: only for a value that is not a number (a name, a method, a frequency such as \
"monthly"); otherwise empty.
Never compute or convert a value.

Values to extract:
{lines}

{answer_rules}
Input text:
{request}"""

IMPLIED = (" If a value is implied rather than stated, fill number, unit and text in the style "
           "of the format examples and leave evidence empty.")
NO_IMPLIED = " Never assume a value that is not stated."


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", s).strip()


def _decimal(text: str) -> Decimal | None:
    t = (text or "").strip().replace(",", "").replace(" ", "").replace("−", "-")
    if not t:
        return None
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


_SCALE_WORDS = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_SCALE_LETTERS = {"k": 3, "m": 6, "mm": 6, "mn": 6, "b": 9, "bn": 9, "t": 12, "tn": 12}
_CURRENCY_RE = r"(?:[$£€¥₹]|US\$|USD|EUR|GBP)"


def fold_scale(field: dict) -> dict:
    """Move a scale word the model wrote into the unit back into the number: number "0.42"
    with unit "$ million" becomes "420000" with unit "$". Words (thousand, million, ...) are
    folded wherever they stand; single letters (k, M, bn) only right after a currency sign,
    so real units (m, mm, km, K, kg) are never touched."""
    num, unit = (field.get("number") or "").strip(), (field.get("unit") or "").strip()
    d = _decimal(num)
    if d is None or not unit:
        return field
    m = re.fullmatch(rf"(?P<cur>{_CURRENCY_RE})?\s*(?P<s>{'|'.join(_SCALE_WORDS)})\b\s*(?P<rest>.*)", unit, re.I)
    if m:
        exp, rest = _SCALE_WORDS[m["s"].lower()], ((m["cur"] or "") + " " + m["rest"]).strip()
    else:
        m = re.fullmatch(rf"(?P<cur>{_CURRENCY_RE})\s*(?P<s>{'|'.join(sorted(_SCALE_LETTERS, key=len, reverse=True))})",
                         unit, re.I)
        if not m:
            return field
        exp, rest = _SCALE_LETTERS[m["s"].lower()], m["cur"]
    scaled = (d.scaleb(exp)).normalize()
    return {**field, "number": format(scaled, "f"), "unit": rest}


def check_value(field: dict, request: str, allow_implied: bool) -> str | None:
    """Why this extracted value can't be trusted, or None if it passes."""
    ev, num, text = field.get("evidence", ""), field.get("number", ""), field.get("text", "")
    if not ev.strip():
        if not allow_implied:
            return "no evidence for the value"
    elif _norm(ev) not in _norm(request):
        return f"evidence {ev!r} is not in the question"
    if not num.strip() and not text.strip():
        return "no value extracted"
    if num.strip():
        d = _decimal(num)
        if d is None:
            return f"number {num!r} is not plain digits"
        if re.search(r"\d", ev):
            stated = numbers(expand_shorthand(ev))
            if not any(abs(Decimal(repr(x)) - d) <= abs(d) * Decimal("1e-12") for x in stated):
                return f"number {num!r} does not match the evidence {ev!r}"
    return None


def _uses_separators(examples: list[str]) -> bool:
    return any(re.search(r"\d,\d{3}", e) for e in examples)


def render(field: dict, param: ParamSpec) -> str:
    """One value for the cached code, in the style of the parameter's format examples."""
    num, unit, text = field.get("number", "").strip(), field.get("unit", "").strip(), field.get("text", "").strip()
    if not num:
        return text
    d = _decimal(num)
    sign, d = ("-", -d) if d < 0 else ("", d)
    digits = format(d, "f")
    if _uses_separators(param.format_examples):
        whole, _, frac = digits.partition(".")
        digits = f"{int(whole):,}" + (f".{frac}" if frac else "")
    if unit in CURRENCY:
        return f"{sign}{unit}{'' if len(unit) == 1 or unit == 'US$' else ' '}{digits}"
    if unit == "%":
        return f"{sign}{digits}%"
    return f"{sign}{digits} {unit}" if unit else f"{sign}{digits}"


_ROUNDING = re.compile(r"^(dp:\d{1,2}|sf:\d{1,2}|nearest:\d+(?:\.\d+)?)?$")


def check_answer_form(output_format: dict | None, extras: list | None, request: str) -> list[str]:
    """Problems with what the question asks of the answer (empty list = fine to serve)."""
    problems = []
    fmt = output_format or {}
    if (fmt.get("unit") or fmt.get("rounding")) and fmt.get("evidence") and _norm(fmt["evidence"]) not in _norm(request):
        problems.append(f"output format evidence {fmt['evidence']!r} is not in the question")
    if not _ROUNDING.match((fmt.get("rounding") or "").strip()):
        problems.append(f"rounding {fmt.get('rounding')!r} is not one the formatter can apply")
    for x in extras or []:   # any instruction the cache can't carry out: let the model answer
        problems.append(f"the question also asks: {x.get('text') or x.get('evidence')!r} ({x.get('kind')})")
    return problems


def extract_number_unit(request: str, params: list[ParamSpec], tag: str | None = None,
                        allow_implied: bool = True, form_out: dict | None = None) -> dict[str, str]:
    from .serve import ExtractionError
    Value = create_model("ExtractedValue", __config__=ConfigDict(extra="forbid"),
                         evidence=(str, ...), number=(str, ...), unit=(str, ...), text=(str, ...))
    OutputFormat = create_model("OutputFormat", __config__=ConfigDict(extra="forbid"),
                                unit=(str, ...), rounding=(str, ...), evidence=(str, ...))
    Extra = create_model("Extra", __config__=ConfigDict(extra="forbid"),
                         kind=(Literal["explanation", "format", "other"], ...), text=(str, ...), evidence=(str, ...))
    Model = create_model("ExtractedValues", __config__=ConfigDict(extra="forbid"),
                         **{f"p{i}": (Value, ...) for i in range(len(params))},
                         output_format=(OutputFormat, ...), extras=(list[Extra], ...))
    lines = [f"p{i}: {p.description} (format examples: {', '.join(map(repr, p.format_examples)) or 'none'})"
             for i, p in enumerate(params)]
    prompt = PROMPT.format(implied=IMPLIED if allow_implied else NO_IMPLIED, lines="\n".join(lines),
                           request=request, answer_rules=ANSWER_RULES)
    data = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Model,
                    max_tokens=2048, tag=tag).parsed.model_dump()
    values, problems = {}, []
    output_format, extras = data.get("output_format") or {}, data.get("extras") or []
    problems += check_answer_form(output_format, extras, request)
    for i, p in enumerate(params):
        field = fold_scale(data[f"p{i}"])
        why = check_value(field, request, allow_implied)
        if why:
            problems.append(f"{p.description}: {why}")
        else:
            values[p.description] = render(field, p)
    if form_out is not None:
        form_out.update(fields=data, values=values, problems=problems, output_format=output_format, extras=extras)
    if problems:
        raise ExtractionError("extraction failed its checks: " + "; ".join(problems))
    return values



# ----------------------------------------------------------------------------- text form

TEXT_PROMPT = """\
Extract the following values from the input text. Copy each value exactly as it appears in \
the text, with its unit and any shorthand ("$150k", "2.0 million", "4.860 dm3", "27°C", \
"ten moles"). Do not convert, expand or compute anything.{implied}

Values to extract:
{lines}

{answer_rules}
Input text:
{request}"""

TEXT_IMPLIED = (" If a value is implied rather than stated, write it in the style of the format "
                "examples.")
TEXT_NO_IMPLIED = " Never assume a value that is not stated."


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or "").lower())


_DIGITS_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _digit_tokens(text: str) -> set[Decimal]:
    """Every number written in the text, as a value: "4.860" and "4.86" are the same number."""
    out = set()
    for m in _DIGITS_RE.finditer(unicodedata.normalize("NFKC", text or "")):
        d = _decimal(m.group())
        if d is not None:
            out.add(d.normalize())
    return out


def check_text_value(value: str, request: str) -> str | None:
    """Every number a value states must appear in the question (catches made-up numbers such
    as an assumed "300 K"). Wording may differ: "10 years" for "a 10-year loan" is fine.
    Implied values without digits ("monthly") pass."""
    if not (value or "").strip():
        return "no value extracted"
    missing = sorted(_digit_tokens(value) - _digit_tokens(request))
    if missing:
        return f"value {value!r} states {', '.join(format(m, 'f') for m in missing)}, which is not in the question"
    return None


def extract_text(request: str, params: list[ParamSpec], tag: str | None = None,
                 allow_implied: bool = True, form_out: dict | None = None) -> dict[str, str]:
    """config.EXTRACT_FORM == "text" (default, user decision 2026-11-15): each value copied
    exactly as written, unit and shorthand included, in one field; the Sonnet-built code reads
    it. Plus output_format and extras (what the question asks of the answer)."""
    from .serve import ExtractionError
    OutputFormat = create_model("OutputFormat", __config__=ConfigDict(extra="forbid"),
                                unit=(str, ...), rounding=(str, ...), evidence=(str, ...))
    Extra = create_model("Extra", __config__=ConfigDict(extra="forbid"),
                         kind=(Literal["explanation", "format", "other"], ...), text=(str, ...), evidence=(str, ...))
    Model = create_model("ExtractedText", __config__=ConfigDict(extra="forbid"),
                         **{f"p{i}": (str, ...) for i in range(len(params))},
                         output_format=(OutputFormat, ...), extras=(list[Extra], ...))
    lines = [f"p{i}: {p.description} (format examples: {', '.join(map(repr, p.format_examples)) or 'none'})"
             for i, p in enumerate(params)]
    prompt = TEXT_PROMPT.format(implied=TEXT_IMPLIED if allow_implied else TEXT_NO_IMPLIED,
                                lines="\n".join(lines), answer_rules=ANSWER_RULES, request=request)
    data = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Model,
                    max_tokens=2048, tag=tag).parsed.model_dump()
    output_format, extras = data.get("output_format") or {}, data.get("extras") or []
    problems = check_answer_form(output_format, extras, request)
    values = {}
    for i, p in enumerate(params):
        value = data.get(f"p{i}") or ""
        why = check_text_value(value, request)
        if why:
            problems.append(f"{p.description}: {why}")
        values[p.description] = value
    if form_out is not None:
        form_out.update(fields=data, values=values, problems=problems, output_format=output_format, extras=extras)
    if problems:
        raise ExtractionError("extraction failed its checks: " + "; ".join(problems))
    return values


# ----------------------------------------------------------------------------- evidence form

# config.EXTRACT_FORM == "evidence" (DEVIATION, opt-in until measured against "text"). The text
# form's only per-value check is "every number in the value is somewhere in the question", so a
# value in the wrong slot, a guessed "implied" value without digits ("monthly") and a number used
# twice all pass. Here the fixed rules go in the system prompt, the entry's build question is
# shown as a worked example (its original values are already stored), and every value comes with
# the words in the question that state it, which plain code checks. A value may be written in
# other words than the question uses ("carbon dioxide" for "CO2(g)"), but never without words in
# the question to back it, and never as a converted number.
EVIDENCE_SYSTEM = """\
A saved program will handle the user's question. You read the question and fill in the \
program's inputs. You only find and copy; the program does all the work (converting units, \
calculating, applying rules).

For each input:
1. evidence: copy the words in the question that state this input: the value plus the words \
that show what it is, e.g. "a $150k mortgage", "served on March 3, 2025", "an 11-year useful \
life".
2. value: the input as the question states it, with its unit and any shorthand ("$150k", \
"March 3, 2025", "4.860 dm3", "ten moles"). If the question states it in other words than the \
format examples use, you may write it in the style of the format examples ("carbon dioxide" for \
"CO2(g)", "monthly" for "monthly payments"). Never convert, expand, round or compute a number.

Every input must come from words in the question. If the question does not state an input, \
even in other words, leave its evidence and value empty: never guess, never assume a usual \
value, and never copy a value from the example question. A value in the question fills at most \
one input, unless the question says it applies to several ("both start at 25°C").

Some inputs are the same kind of value: two amounts of money, two dates, two temperatures, two \
percentages. Decide which value belongs to which input from the words next to it ("cost" vs \
"salvage", "sensitivity" vs "specificity", "filed" vs "served"). Go by order only when the \
question itself ties values to inputs by order ("$50k and $5k, respectively"). If the question \
does not make clear which value belongs to which input, set ambiguous to true.

""" + ANSWER_RULES


def evidence_message(request: str, params: list[ParamSpec], example: str | None = None) -> str:
    """The user message: the worked example (when the entry's build question and every original
    value are known), the inputs to fill, then the question."""
    parts = []
    if example and params and all((p.original_value or "").strip() for p in params):
        shown = "\n".join(f'p{i} ({p.description}): "{p.original_value}"' for i, p in enumerate(params))
        parts.append(f"Example. The program was built from this question:\n<<<{example}>>>\n"
                     f"Its inputs in that question were:\n{shown}\n")
    lines = "\n".join(f"p{i}: {p.description} (format examples: {', '.join(map(repr, p.format_examples)) or 'none'})"
                      for i, p in enumerate(params))
    parts.append(f"Inputs to fill:\n{lines}\n")
    parts.append(f"Question:\n<<<{request}>>>")
    return "\n".join(parts)


def _words(text: str) -> str:
    """Text as its words only, for matching evidence: case, spacing, punctuation and quote or
    dash styles are ignored ("Cost: $50k," and "cost $50k" match); the words and their order
    are not."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text or "").lower()))


def check_evidence_value(evidence: str, value: str, request: str) -> str | None:
    """Why one extracted input can't be trusted, or None. The value must be stated (no guesses),
    its evidence must be words in the question (punctuation aside), and every number in the
    value must be in its OWN evidence, not just somewhere in the question. A value without
    digits may reword its evidence ("carbon dioxide" for "CO2(g)")."""
    if not (value or "").strip():
        return "not stated in the question"
    ev = _words(evidence)
    if not ev:
        return f"no evidence for {value!r}"
    if f" {ev} " not in f" {_words(request)} ":
        return f"evidence {evidence!r} is not in the question"
    missing = sorted(_digit_tokens(value) - _digit_tokens(evidence))
    if missing:
        return f"value {value!r} states {', '.join(format(m, 'f') for m in missing)}, which is not in its evidence {evidence!r}"
    return None


def extract_evidence(request: str, params: list[ParamSpec], tag: str | None = None,
                     example: str | None = None, form_out: dict | None = None) -> dict[str, str]:
    """config.EXTRACT_FORM == "evidence": like the text form (values copied as written, so cached
    code is unchanged), plus evidence per value, a worked example, and an ambiguity flag."""
    from .serve import ExtractionError
    Slot = create_model("Slot", __config__=ConfigDict(extra="forbid"), evidence=(str, ...), value=(str, ...))
    OutputFormat = create_model("OutputFormat", __config__=ConfigDict(extra="forbid"),
                                unit=(str, ...), rounding=(str, ...), evidence=(str, ...))
    Extra = create_model("Extra", __config__=ConfigDict(extra="forbid"),
                         kind=(Literal["explanation", "format", "other"], ...), text=(str, ...), evidence=(str, ...))
    Model = create_model("ExtractedEvidence", __config__=ConfigDict(extra="forbid"),
                         **{f"p{i}": (Slot, ...) for i in range(len(params))},
                         ambiguous=(bool, ...), output_format=(OutputFormat, ...), extras=(list[Extra], ...))
    data = llm.call("extractor", evidence_message(request, params, example), model=config.TARGET_MODEL,
                    schema=Model, system=EVIDENCE_SYSTEM, max_tokens=2048, tag=tag).parsed.model_dump()
    output_format, extras = data.get("output_format") or {}, data.get("extras") or []
    problems = check_answer_form(output_format, extras, request)
    if data.get("ambiguous"):
        problems.append("the question doesn't make clear which value belongs to which input")
    values = {}
    for i, p in enumerate(params):
        slot = data.get(f"p{i}") or {}
        value = (slot.get("value") or "").strip()
        why = check_evidence_value(slot.get("evidence", ""), value, request)
        if why:
            problems.append(f"{p.description}: {why}")
        values[p.description] = value
    if form_out is not None:
        form_out.update(fields=data, values=values, problems=problems, output_format=output_format, extras=extras)
    if problems:
        raise ExtractionError("extraction failed its checks: " + "; ".join(problems))
    return values
