"""Output checks: cheap rules every correct answer must satisfy, run on every cache hit.

DEVIATION from the paper, which serves whatever the cached code returns. All of the build's
checking happens once, at build time; on a hit nothing looks at the answer itself. Here each
entry carries a small `check(inputs, outputs)` function, and a hit whose answer breaks a rule
goes to the model instead (serve.use, config.OUTPUT_CHECKS).

What a check is for: catching answers that are impossible, not answers that are slightly off.
Book value below salvage or above cost, a monthly payment larger than the loan, interest larger
than simple interest on the full principal, a negative pressure. These are the signatures of
unit slips, dropped "/100"s, swapped inputs and sign errors. A check never recomputes the exact
answer: that would be a second implementation (a different, later idea), and a rule tight
enough to pin the answer would also flag correct answers that use another valid convention.

Who writes it: a model (CHECKER_MODEL, Sonnet by default) that sees WHAT the entry does (its
original question, inputs, outputs, stated conventions) and never HOW (no code), the same line
the independent tester (arc/tester.py) never crosses. Written once per entry; validated on the
entry's own answers (a check that flags them is repaired once, then dropped).

Stored in the cache as cache.checks[entry_id] = {"source", "rules", "model", "status"}
(checks.json beside registry.json).
"""

from __future__ import annotations

import math
import re
import time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from . import config, llm, sandbox

ROLE = "checker"
CHECK_SECONDS = 2.0

_NUM = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
# Scale shorthand. A bare letter only scales when it is glued to the number ("150k", "$2.5m"):
# "294.65 K" is kelvin and "5 m" is metres. m/b letters also need a currency sign before the
# number ("5m" alone could be metres). Spelled-out words scale with or without a space.
_SCALE_WORDS = [(re.compile(r"^\s*thousand\b", re.I), 1e3), (re.compile(r"^\s*million\b", re.I), 1e6),
                (re.compile(r"^\s*billion\b", re.I), 1e9)]
_SCALE_K = re.compile(r"^[kK](?![a-zA-Z])")
_SCALE_MB = [(re.compile(r"^(?:m|mm|mn|M)(?![a-zA-Z])"), 1e6), (re.compile(r"^(?:b|bn|B)(?![a-zA-Z])"), 1e9)]
_CURRENCY = re.compile(r"[$€£¥₹]\s*$")
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}


def _word_number(s: str) -> float | None:
    """The first number written in words: "twenty-five years" -> 25, "one hundred and five" -> 105."""
    total, cur, seen = 0, 0, False
    for w in re.findall(r"[a-z]+", s.lower()):
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred" and seen:
            cur = (cur or 1) * 100
        elif w == "thousand" and seen:
            total, cur = total + (cur or 1) * 1000, 0
        elif w == "and" and seen:
            continue
        elif seen:
            break
        else:
            continue
        seen = True
    return float(total + cur) if seen else None


def number(text: Any) -> float | None:
    """The first number in `text` as a float: '$329,000' -> 329000, '$150k' -> 150000,
    '7.5%' -> 7.5, '10 years' -> 10, 'ten years' -> 10. None if there is no number."""
    if text is None:
        return None
    if isinstance(text, (int, float, Decimal)) and not isinstance(text, bool):
        return float(text)
    s = str(text)
    m = _NUM.search(s)
    if not m:
        return _word_number(s)
    try:
        v = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    rest = s[m.end():]
    for pat, f in _SCALE_WORDS:
        if pat.match(rest):
            return v * f
    if _SCALE_K.match(rest):
        return v * 1e3
    if _CURRENCY.search(s[:m.start()]):
        for pat, f in _SCALE_MB:
            if pat.match(rest):
                return v * f
    return v


def _plain(value: Any) -> Any:
    """Outputs as plain floats where they are numbers (Decimal, int, numeric text)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("$", "").strip())
        except ValueError:
            return value
    return value


def load_check(source: str):
    """Compile a check's source in the sandbox's restricted builtins; return `check`."""
    namespace = {"__builtins__": sandbox._safe_builtins(), "math": math, "number": number}
    exec(compile(source, "<output_check>", "exec"), namespace)
    fn = namespace.get("check")
    if not callable(fn):
        raise ValueError("check source defines no check(inputs, outputs)")
    return fn


_LOADED: dict[str, Any] = {}


def run(cache, entry_id: str, values: dict, output: dict) -> list[str] | None:
    """Rules this answer breaks ([] = plausible). None when the entry has no usable check.
    A check that crashes counts as broken: an answer we can't vouch for goes to the model."""
    from .guards import usable
    rec, _ = usable(cache, entry_id, "checks")       # missing, failed or stale -> no check
    if rec is None:
        return None
    src = rec["source"]
    try:
        fn = _LOADED.get(src) or _LOADED.setdefault(src, load_check(src))
        with sandbox.time_limit(CHECK_SECONDS):
            out = fn(dict(values), {k: _plain(v) for k, v in output.items()})
    except Exception as e:     # noqa: BLE001 - any failure of the check itself
        return [f"check crashed: {type(e).__name__}: {str(e)[:120]}"]
    if not isinstance(out, (list, tuple)):
        return [f"check returned {type(out).__name__}, not a list"]
    return [str(x) for x in out]


# ----------------------------------------------------------------------------- writing checks

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckSpec(_Strict):
    rules: list[str]      # each rule in plain words, and why every correct answer satisfies it
    source: str           # Python defining check(inputs, outputs)


SYSTEM = """You write sanity checks for a cached calculator. You never see its code.

Write Python defining exactly:

    def check(inputs, outputs):
        '''Return a list of short strings, one per rule the answer breaks; [] if none.'''

- inputs: dict, input description -> the raw text extracted from the question
  (e.g. "$329,000", "10 years", "7.5%", "monthly"). A value may be "" or unreadable.
- outputs: dict, output name -> float (or str when not numeric).
- number(text) is available: the first number in the text as a float, or None.
  "$329,000" -> 329000.0, "$150k" -> 150000.0, "$0.42 million" -> 420000.0, "7.5%" -> 7.5 (NOT 0.075),
  "twenty-five years" -> 25.0. It ALREADY applies k / thousand / million / billion: never scale its
  result again. It does not convert units (g vs kg, °C vs K, months vs years): you do that.
- `math` is available. No imports, no I/O.

Write only NECESSARY conditions: rules that hold for every correct answer, for every valid input,
under every reasonable convention for this kind of question:
- signs and hard bounds (a book value lies between salvage value and cost; a pressure is > 0)
- inequalities against the inputs (a level loan payment times the number of payments is at least
  the principal; total interest on an amortizing loan is below simple interest on the full
  principal for the whole term)
- identities between outputs that follow from what the outputs mean, allowing for rounding

Never:
- recompute the exact answer and compare to it, or pin a value to a narrow band
- rules that depend on a method choice the question does not fix
- ranges guessed from "typical" values

Robustness:
- If an input or output a rule needs is missing or number() returns None, skip that rule.
- Compare with a tolerance: abs 0.01 for money plus 1e-6 relative. When a rule combines a rounded
  output with a count (payment x number of payments), rounding errors add up: allow 0.01 per term.
- Edge cases are valid inputs: a 0% rate, zero salvage, a period past the useful life. A rule must
  hold for them too (with the tolerance above), or skip them.
- Never raise: the function must return a list for any inputs.

Each string you return names the rule broken and the numbers involved, for example
"book value 412.0 below salvage value 49100.0".

Also list each rule in plain words in `rules`, with one line on why every correct answer obeys it."""


def _brief(cache, entry_id: str) -> str:
    eps = [e for e in cache.entry_points if e.problem_id == entry_id]
    orig = next((e for e in eps if e.kind == "original"), eps[0])
    root = cache.functions[orig.root_id]
    ins = "\n".join(f"- {p.description} (format examples: {', '.join(map(repr, p.format_examples)) or 'none'})"
                    for p in orig.params)
    specs = (getattr(cache, "outputs", None) or {}).get(entry_id) or {}
    outs = "\n".join(f"- {o}" + (f" ({specs[o]})" if o in specs else "") for o in root.outputs)
    conv = (getattr(cache, "conventions", None) or {}).get(entry_id) or []
    conv_txt = ("\n\nConventions the entry states (to understand what the outputs mean; they may be wrong, "
                "so never build a rule that only holds under one of them):\n" + "\n".join(f"- {c}" for c in conv)
                ) if conv else ""
    return (f"What the calculator does: {root.description}\n\n"
            f"A question it answers:\n{orig.text}\n\n"
            f"inputs keys (exactly these strings):\n{ins}\n\n"
            f"outputs keys (exactly these strings):\n{outs}{conv_txt}")


FORMAT_COMBOS = 8   # input combinations built from the entry's format examples, per entry


def format_combos(cache, entry_id: str, n: int = FORMAT_COMBOS) -> list[dict]:
    """FREE. Inputs written the ways the entry says it accepts them (each input's format examples:
    "$150k", "$0.42 million", "five years", "180 months", "0.214 cal/(g·°C)"), combined across
    inputs, with the entry's answer. The build question writes each value one way; a guard
    validated only on it can still misread the others (the first reference double-scaled
    "$25.4k"). A combination the entry rejects (raises) is skipped."""
    from .serve import dispatch_root
    ep = next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None)
    if ep is None or not ep.params:
        return []
    import itertools
    import random
    options = [[x for x in p.format_examples if str(x).strip()] or [p.original_value or ""] for p in ep.params]
    combos = list(itertools.islice(itertools.product(*options), 5000))
    random.Random(0).shuffle(combos)                     # fixed seed: the same combos every time
    # edge cases: each input's number set to zero in turn ("0%", "$0", "0 years"). A guard must
    # hold there too (a 0% loan's interest is rounding dust, not negative). The entry skips any
    # zero it rejects, so this is safe for every entry.
    base = [o[0] for o in options]
    zeros = []
    for j, v in enumerate(base):
        z = re.sub(r"\d[\d,]*(?:\.\d+)?", "0", str(v), count=1)
        if z != v:
            zeros.append(tuple(base[:j] + [z] + base[j + 1:]))
    def run(combo, label):
        values = {p.description: v for p, v in zip(ep.params, combo)}
        try:
            output, _ = dispatch_root(cache, ep.root_id, values)
        except Exception:   # noqa: BLE001 - the entry rejects this input: not a case to check
            return None
        return {"question": label, "inputs": values, "outputs": output, "synthetic": True}

    edge = [x for x in (run(c, "(edge case: one input set to zero)") for c in zeros) if x]
    out = []
    for combo in combos:
        x = run(combo, "(inputs written in the entry's own format examples)")
        if x:
            out.append(x)
        if len(out) >= n:
            break
    return edge + out


def _examples(cache, entry_id: str, tag: str, k: int = 3) -> list[dict]:
    """Known-good (inputs, outputs) pairs: the entry's answers to its own build questions
    (PAID: one extraction each), then to its own format examples (free). A guard that disagrees
    with one of these is repaired."""
    from .serve import dispatch_root, extract_parameters
    out = []
    for ep in [e for e in cache.entry_points if e.problem_id == entry_id][:k]:
        try:
            values = extract_parameters(ep.text, ep.params, tag=f"{tag}:examples", example=ep.text)
            output, _ = dispatch_root(cache, ep.root_id, values)
        except Exception:   # noqa: BLE001 - an example we can't get is just skipped
            continue
        out.append({"question": ep.text, "inputs": values, "outputs": output})
    return out + format_combos(cache, entry_id)


def write(cache, entry_id: str, tag: str = "outchecks", model: str | None = None,
          examples: list[dict] | None = None) -> dict:
    """Write, validate and store cache.checks[entry_id]. PAID: one checker call (two if the first
    check flags a known-good answer), plus one Haiku extraction per example."""
    model = model or config.CHECKER_MODEL
    examples = examples if examples is not None else _examples(cache, entry_id, tag)
    prompt = _brief(cache, entry_id)
    rec: dict = {"model": model, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "examples": len(examples)}
    for attempt in range(3):
        r = llm.call(ROLE, prompt, model=model, system=SYSTEM, schema=CheckSpec, effort="medium",
                     max_tokens=8000, tag=f"{tag}:{entry_id}")
        spec: CheckSpec = r.parsed
        rec.update(source=spec.source, rules=spec.rules)
        flagged = []
        try:
            fn = load_check(spec.source)
            for ex in examples:
                bad = fn(dict(ex["inputs"]), {k: _plain(v) for k, v in ex["outputs"].items()})
                if bad:
                    flagged.append((ex, bad))
        except Exception as e:   # noqa: BLE001
            flagged.append(({"question": "(loading or running the check)", "outputs": {}}, [repr(e)[:200]]))
        if not flagged:
            rec["status"] = "ok"
            break
        ex, bad = flagged[0]
        rec.update(status="flagged_good_answer", flagged=[str(b) for b in bad][:5])
        prompt = (_brief(cache, entry_id) + "\n\nYour previous check:\n" + spec.source +
                  f"\n\nflagged a CORRECT answer. Question: {ex['question']}\n"
                  f"inputs: {ex.get('inputs')}\noutputs: { {k: str(v) for k, v in ex['outputs'].items()} }\n"
                  f"It returned: {bad}\nThat rule is not a necessary condition (or has a bug). "
                  "Rewrite the check without it.")
    cache.checks[entry_id] = rec
    return rec
