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

from pydantic import ConfigDict, create_model

from . import config, llm
from .cache import ParamSpec
from .checks import numbers
from .normalize import expand_shorthand

CURRENCY = {"$", "£", "€", "¥", "₹", "US$", "USD", "EUR", "GBP"}

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


def extract_number_unit(request: str, params: list[ParamSpec], tag: str | None = None,
                        allow_implied: bool = True, form_out: dict | None = None) -> dict[str, str]:
    from .serve import ExtractionError
    Value = create_model("ExtractedValue", __config__=ConfigDict(extra="forbid"),
                         evidence=(str, ...), number=(str, ...), unit=(str, ...), text=(str, ...))
    Model = create_model("ExtractedValues", __config__=ConfigDict(extra="forbid"),
                         **{f"p{i}": (Value, ...) for i in range(len(params))})
    lines = [f"p{i}: {p.description} (format examples: {', '.join(map(repr, p.format_examples)) or 'none'})"
             for i, p in enumerate(params)]
    prompt = PROMPT.format(implied=IMPLIED if allow_implied else NO_IMPLIED, lines="\n".join(lines),
                           request=request)
    data = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Model,
                    max_tokens=2048, tag=tag).parsed.model_dump()
    values, problems = {}, []
    for i, p in enumerate(params):
        field = data[f"p{i}"]
        why = check_value(field, request, allow_implied)
        if why:
            problems.append(f"{p.description}: {why}")
        else:
            values[p.description] = render(field, p)
    if form_out is not None:
        form_out.update(fields=data, values=values, problems=problems)
    if problems:
        raise ExtractionError("extraction failed its checks: " + "; ".join(problems))
    return values
