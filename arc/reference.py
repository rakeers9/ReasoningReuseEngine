"""Reference implementations: a second, independently written program per entry, run on every hit.

DEVIATION from the paper. Output checks (arc/outchecks.py) catch answers that are impossible;
they can't catch answers that are plausible but wrong (half the depreciation rate, a payment one
cent high). Here a model that never sees the entry's code writes its own `reference(inputs)`
from WHAT the entry does. On every hit both programs run on the same extracted values; the
answer is served only if they agree at the precision shown to the user. Two implementations
written separately rarely share a bug (N-version programming), and running both costs no model
call per hit.

What the writer sees: the entry's description, its original question, input and output names,
the entry's stated conventions, and the entry's answers to its own build questions (those were
verified against the production model at build time; the tester sees production answers too).
Never the code. Conventions are given as the spec to follow: a convention is a choice about what
the answer means (round the payment first), not a bug, and the build side owns those choices.

What it doesn't catch: anything both programs get from the same place. Both read the same
extracted values (a misread input fools both) and the same conventions (a wrong convention is
copied). Those are for extraction checks and the build side.

Stored as cache.references[entry_id] = {"source", "status", "model", ...} (references.json).
"""

from __future__ import annotations

import math
import time
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from . import config, llm, sandbox
from .outchecks import _brief, _examples, number

ROLE = "reference"
RUN_SECONDS = 2.0


def load_reference(source: str):
    namespace = {"__builtins__": sandbox._safe_builtins(), "math": math, "number": number, "Decimal": Decimal}
    exec(compile(source, "<reference>", "exec"), namespace)
    fn = namespace.get("reference")
    if not callable(fn):
        raise ValueError("source defines no reference(inputs)")
    return fn


def _shown(output: dict, specs: dict | None, output_format: dict | None) -> dict:
    """Each output as the user would see it (arc/present.py), so agreement is judged at the
    precision shown: a difference below that precision never reaches anyone."""
    from . import present
    try:
        return present.apply(output, output_format, specs)
    except Exception:   # noqa: BLE001 - fall back to the raw values
        return dict(output)


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def disagreements(entry_out: dict, ref_out: dict, specs: dict | None = None,
                  output_format: dict | None = None) -> list[str]:
    """Outputs on which the two programs differ, at the shown precision. A key the reference
    doesn't return is not compared (it may only compute the main answer); a numeric output with
    no rounding spec must agree to 1e-9 relative."""
    a, b = _shown(entry_out, specs, output_format), _shown(ref_out, specs, output_format)
    out = []
    for key, va in a.items():
        if key not in b:
            continue
        x, y = _num(va), _num(b[key])
        if x is None or y is None:
            if str(va).strip().lower() != str(b[key]).strip().lower():
                out.append(f"{key}: entry {va!r}, reference {b[key]!r}")
            continue
        if not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12):
            out.append(f"{key}: entry {va}, reference {b[key]}")
    return out


_LOADED: dict[str, Any] = {}


def run(cache, entry_id: str, values: dict, output: dict, output_format: dict | None = None) -> list[str] | None:
    """Disagreements between the entry's answer and the reference's ([] = agree). None when the
    entry has no usable reference. A reference that crashes or returns nothing comparable counts
    as a disagreement: an answer we can't confirm goes to the model."""
    from .guards import usable
    rec, _ = usable(cache, entry_id, "references")   # missing, failed or stale -> no reference
    if rec is None:
        return None
    src = rec["source"]
    try:
        fn = _LOADED.get(src) or _LOADED.setdefault(src, load_reference(src))
        with sandbox.time_limit(RUN_SECONDS):
            ref = fn(dict(values))
    except Exception as e:   # noqa: BLE001
        return [f"reference crashed: {type(e).__name__}: {str(e)[:120]}"]
    if not isinstance(ref, dict) or not set(ref) & set(output):
        return ["reference returned no comparable output"]
    specs = (getattr(cache, "outputs", None) or {}).get(entry_id)
    return disagreements(output, ref, specs, output_format)


# ----------------------------------------------------------------------------- writing

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceSpec(_Strict):
    approach: str     # the method, in a few sentences
    source: str       # Python defining reference(inputs) -> dict


SYSTEM = """You write a reference implementation of a calculator, independently. You never see the
calculator's code; your program is used to double-check its answers.

Write Python defining exactly:

    def reference(inputs):
        '''Return a dict with the outputs keys (exactly those strings) and their values.'''

- inputs: dict, input description -> the raw text extracted from a question
  (e.g. "$329,000", "10 years", "7.5%", "monthly").
- number(text) is available: the first number in the text as a float, or None.
  "$329,000" -> 329000.0, "$150k" -> 150000.0, "$0.42 million" -> 420000.0, "7.5%" -> 7.5 (NOT 0.075),
  "twenty-five years" -> 25.0. It ALREADY applies k / thousand / million / billion: never scale its
  result again. It does not convert units (g vs kg, °C vs K, months vs years): you do that.
  Read units and words yourself where they matter (months vs years, "monthly", °C vs K).
- Decimal and math are available. No imports, no I/O. Use Decimal (getcontext().prec is not
  available: work at default precision, or use floats where precision is ample).
- Follow the calculator's stated conventions exactly: they define the expected answer
  (what is rounded, when, and how).
- Return values UNROUNDED unless a convention says the calculator rounds that value itself;
  final rounding for display happens outside both programs.
- If the inputs can't be read, raise ValueError. Never guess a missing value."""


def write(cache, entry_id: str, tag: str = "reference", model: str | None = None,
          examples: list[dict] | None = None) -> dict:
    """Write, validate and store cache.references[entry_id]. PAID: one writer call (two if the
    first disagrees with a verified example), plus one Haiku extraction per example."""
    model = model or config.REFERENCE_MODEL
    examples = examples if examples is not None else _examples(cache, entry_id, tag)
    specs = (getattr(cache, "outputs", None) or {}).get(entry_id)
    shown = [{"question": ex["question"], "inputs": ex["inputs"],
              "outputs": {k: str(v) for k, v in _shown(ex["outputs"], specs, None).items()}}
             for ex in examples if not ex.get("synthetic")]
    prompt = _brief(cache, entry_id)
    if shown:
        prompt += ("\n\nVerified answers (checked against the production model) your program must reproduce "
                   "at the shown precision:\n" + "\n".join(
                       f"- question: {s['question']}\n  inputs: {s['inputs']}\n  outputs: {s['outputs']}" for s in shown))
    rec: dict = {"model": model, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "examples": len(examples)}
    for attempt in range(3):
        r = llm.call(ROLE, prompt, model=model, system=SYSTEM, schema=ReferenceSpec, effort="medium",
                     max_tokens=8000, tag=f"{tag}:{entry_id}")
        spec: ReferenceSpec = r.parsed
        rec.update(source=spec.source, approach=spec.approach)
        problems = []
        try:
            fn = load_reference(spec.source)
            for ex in examples:
                ref = fn(dict(ex["inputs"]))
                if not isinstance(ref, dict) or not set(ref) & set(ex["outputs"]):
                    problems.append((ex, ["returned no comparable output"]))
                    continue
                d = disagreements(ex["outputs"], ref, specs)
                if d:
                    problems.append((ex, d))
        except Exception as e:   # noqa: BLE001
            problems.append(({"question": "(loading or running the reference)"}, [repr(e)[:200]]))
        if not problems:
            rec["status"] = "ok"
            break
        ex, d = problems[0]
        rec.update(status="disagrees_with_verified", problems=[str(x) for x in d][:5])
        prompt += (f"\n\nYour previous program:\n{spec.source}\n\ngot this wrong on: {ex['question']}\n"
                   f"inputs: {ex.get('inputs')}\n{d}\nFind the mistake and rewrite it. Remember number() already "
                   "applies k / million / billion: never scale its result again.")
    cache.references[entry_id] = rec
    return rec
