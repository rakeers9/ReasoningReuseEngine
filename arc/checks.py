"""Checks that use no language model, shown next to LLM judge verdicts.

number_check: do the numbers in a reference's final answer appear in another answer,
EXACTLY to the precision they are written in? Owner rule (2026-11-15): "If an answer
requires an exact amount, then it requires an exact amount." Two numbers agree only if
they differ by at most half a unit in the last stated digit of the coarser one: $5,053.07
vs $5,053.27 disagree, 18.57 vs 18.6 atm agree, 1,008 vs 1,009 disagree. There is no
percentage floor or cap. SI-prefix rescaling (kPa vs Pa) is allowed only when an SI unit is
written and no money is involved, so "$150k" read as 150 can never match by rescaling.

In the paper path (builder, evaluate) the result is shown next to the judges' verdicts and
decides nothing. In the label-free path (certify._compare) a "mismatch" vetoes agreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SCI = re.compile(r"(\d(?:[\d.]*))\s*(?:×|x|\*|·)\s*10\s*\^?\s*\(?\s*([-−+]?\d+)\)?")
_SIGNED_CURRENCY = re.compile(r"([-+])\s*[£$€¥₹]\s*(?=\d)")
_NUM = re.compile(r"(?<![\w.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-−+]?\d+)?(?!\.?\d)")


_FLOAT_EPS = 1e-9            # relative slack for binary floating point only
_SCALES = (1, 1e3, 1e-3, 1e6, 1e-6)   # SI-prefix rescaling (kPa vs Pa, MPa vs Pa)
_NEAR = 0.05                 # a value this close but not equal is a conflicting answer
# Rescaling is allowed only when an SI unit is written and no money is involved: a cache
# that read "$150k" as 150 answers 0.92 for $921.13, and 921.13e-3 would "match" 0.92.
_SI_UNIT = re.compile(r"(?<![A-Za-z])(?:[kMGmµu]?(?:Pa|g|m|L|l|J|W|N|V|A|s|mol|Hz)|bar|mbar|kWh)(?![A-Za-z])")
# Money: a currency sign before a number, or a currency word. A bare "$" is not money, because
# LaTeX math ("$$p \approx 5.13\ \text{MPa}$$") uses it as a delimiter.
_MONEY = re.compile(r"[£€¥₹]|\\\$|(?<!\$)\$\s?[-+]?\d|\b(?:USD|EUR|GBP|JPY|INR|dollars?|euros?|pounds?|cents?|yen|rupees?)\b", re.I)


def _scales(*texts: str) -> tuple[float, ...]:
    joined = " ".join(texts)
    if _MONEY.search(joined) or not _SI_UNIT.search(joined):
        return (1,)
    return _SCALES


@dataclass(frozen=True)
class Written:
    """A number as written: its value, half a unit in its last stated digit, and the unit
    word written right after it ("atm", "mm", "people"; "" when none)."""
    value: float
    half_unit: float
    unit: str = ""


# Units the comparison understands. A word after a number that isn't here counts as no unit.
_UNITS = {"atm", "bar", "mbar", "psi", "mmhg", "torr", "pa", "hpa", "kpa", "mpa", "gpa",
          "m", "km", "cm", "mm", "µm", "um", "nm", "ft", "in", "mi", "l", "ml", "dl", "dm3", "cm3", "m3",
          "dm³", "cm³", "m³", "g", "kg", "mg", "µg", "ug", "lb", "lbs", "oz", "s", "ms", "min", "h", "hr",
          "k", "°c", "°f", "°", "c", "f", "j", "kj", "mj", "cal", "kcal", "w", "kw", "mw", "kwh", "n", "kn",
          "mol", "mmol", "µmol", "umol", "%", "mph", "km/h", "m/s", "hz", "khz", "mhz", "v", "mv", "kv",
          "a", "ma", "ka", "mol/l", "mmol/l", "µmol/l", "umol/l", "mg/dl", "g/l", "ml/min", "n/m", "kn/m"}
_SI_BASES = {"pa", "g", "m", "l", "j", "w", "n", "mol", "hz", "v", "a", "bar", "m3", "m³", "m/s", "mol/l", "n/m"}


def _base(unit: str) -> str:
    if unit and unit[0] in "kmgµuhd" and len(unit) > 1 and unit[1:] in _SI_BASES:
        return unit[1:]
    return unit


def _units_compatible(a: str, b: str) -> bool:
    """A missing unit fits anything; otherwise the same unit up to an SI prefix."""
    return not a or not b or _base(a) == _base(b)


# the unit written after a number, skipping spaces and LaTeX spacing ("50.7\ \text{atm}")
_UNIT_AFTER = re.compile(r"(?:\s|\\[ ,;!:]|~|\\text\{|\\mathrm\{|\\rm\{|\{)*([A-Za-zµ°%][A-Za-zµ°%/·²³0-9]*)")


def _tokens(text: str) -> list[tuple[str, str]]:
    t = (text or "").translate(_SUPERSCRIPT).replace("−", "-")
    t = _SIGNED_CURRENCY.sub(r"\1", t)   # "-£62,164.50" -> "-62,164.50" (keep the sign)
    t = _SCI.sub(lambda m: f"{m.group(1)}e{m.group(2)}", t)
    out = []
    for m in _NUM.finditer(t):
        s = m.group(0).replace(",", "")
        if re.search(r"\d", s):
            u = _UNIT_AFTER.match(t, m.end())
            unit = u.group(1).lower().rstrip("/·") if u else ""
            out.append((s, unit if unit in _UNITS else ""))
    return out


def _half_unit(token: str) -> float:
    """Half a unit in the last written digit: "5053.27" -> 0.005, "1008" -> 0.5,
    "1.88e6" -> 0.005e6 (the mantissa's decimals, times the exponent)."""
    mant, _, exp = token.lower().lstrip("+-").partition("e")
    decimals = len(mant.split(".", 1)[1]) if "." in mant else 0
    return 0.5 * 10.0 ** (int(exp or 0) - decimals)


def written_numbers(text: str) -> list[Written]:
    out = []
    for s, unit in _tokens(text):
        try:
            out.append(Written(float(s), _half_unit(s), unit))
        except ValueError:
            pass
    return out


_LONG_DECIMAL = re.compile(r"(?<![\w.])([-+]?)(\d+\.\d+)(?![\d.])")


def round_long_numbers(text: str, sig: int = 12, min_decimals: int = 6) -> str:
    """Shorten every decimal number in `text` to `sig` significant digits, but never to fewer
    than `min_decimals` decimals (so money keeps its cents however large). Shorter numbers,
    integers and words are left as written.

    ONLY for comparing two machine-computed full-precision values (the convention check: cached
    code vs a program, both in Decimal at 50-80 digits), where tail digits are numerical noise
    and an 80-digit string makes the judge balk. Never use it on a customer-facing answer: those
    are compared to the precision they are written in (number_check)."""
    from decimal import ROUND_HALF_EVEN, Decimal

    def short(m: re.Match) -> str:
        d = Decimal(m.group(2))
        keep = max(sig - 1 - d.adjusted(), min_decimals) if d else min_decimals
        if len(m.group(2).split(".")[1]) <= keep:
            return m.group(0)
        return m.group(1) + format(d.quantize(Decimal(1).scaleb(-keep), rounding=ROUND_HALF_EVEN), "f")

    return _LONG_DECIMAL.sub(short, text)


def numbers(text: str) -> list[float]:
    """Every number in the text, with 1.2×10^6, 1.2 × 10⁶ and 1.2e6 all read as 1.2e6."""
    return [w.value for w in written_numbers(text)]


# Unit -> (dimension, factor to that dimension's base). When both numbers state a unit, the
# rescale factor is the one these imply, never a guessed power of 1000 (5.13 MPa is 5130 kPa,
# not 5130000 kPa). Lowercased, as written_numbers stores units ("mpa" is MPa).
_FACTORS: dict[str, tuple[str, float]] = {
    "pa": ("pressure", 1.0), "hpa": ("pressure", 1e2), "kpa": ("pressure", 1e3), "mpa": ("pressure", 1e6),
    "gpa": ("pressure", 1e9), "bar": ("pressure_bar", 1.0), "mbar": ("pressure_bar", 1e-3),
    "g": ("mass", 1.0), "mg": ("mass", 1e-3), "kg": ("mass", 1e3), "µg": ("mass", 1e-6), "ug": ("mass", 1e-6),
    "m": ("length", 1.0), "mm": ("length", 1e-3), "cm": ("length", 1e-2), "km": ("length", 1e3),
    "µm": ("length", 1e-6), "um": ("length", 1e-6), "nm": ("length", 1e-9),
    "m3": ("volume", 1.0), "m³": ("volume", 1.0), "dm3": ("volume", 1e-3), "dm³": ("volume", 1e-3),
    "l": ("volume", 1e-3), "dl": ("volume", 1e-4), "ml": ("volume", 1e-6), "cm3": ("volume", 1e-6),
    "cm³": ("volume", 1e-6),
    "j": ("energy", 1.0), "kj": ("energy", 1e3), "mj": ("energy", 1e6),
    "w": ("power", 1.0), "kw": ("power", 1e3), "mw": ("power", 1e6),
    "n": ("force", 1.0), "kn": ("force", 1e3),
    "mol": ("amount", 1.0), "mmol": ("amount", 1e-3), "µmol": ("amount", 1e-6), "umol": ("amount", 1e-6),
    "mol/l": ("molarity", 1.0), "mmol/l": ("molarity", 1e-3), "µmol/l": ("molarity", 1e-6),
    "umol/l": ("molarity", 1e-6),
    "s": ("time", 1.0), "ms": ("time", 1e-3),
    "hz": ("frequency", 1.0), "khz": ("frequency", 1e3), "mhz": ("frequency", 1e6),
    "v": ("voltage", 1.0), "mv": ("voltage", 1e-3), "kv": ("voltage", 1e3),
    "n/m": ("stiffness", 1.0), "kn/m": ("stiffness", 1e3),
}


def _agree(a: Written, r: Written, scale: float = 1.0) -> bool:
    """Equal to the precision of the coarser of the two, in compatible units (51 bar never
    equals 50.6 atm). Both units known: the factor they imply is used and `scale` is ignored.
    Same unit written on both: no rescale. A unit on one side only: `scale` (a power of 1000,
    allowed by _scales) applies."""
    if a.unit and r.unit:
        if a.unit == r.unit:
            scale = 1.0
        elif a.unit in _FACTORS and r.unit in _FACTORS and _FACTORS[a.unit][0] == _FACTORS[r.unit][0]:
            scale = _FACTORS[r.unit][1] / _FACTORS[a.unit][1]
        else:
            return False
    target = r.value * scale
    tol = max(a.half_unit, r.half_unit * scale) + _FLOAT_EPS * max(abs(a.value), abs(target))
    return abs(a.value - target) <= tol


def values_agree(a: str, b: str) -> bool:
    """Shared helper: does the first number in `a` equal the first in `b` to the precision
    written (SI-prefix rescaling allowed)? e.g. values_agree("18.57", "18.6") is True."""
    xa, xb = written_numbers(a), written_numbers(b)
    if not xa or not xb:
        return False
    return any(_agree(xa[0], xb[0], f) for f in _scales(a, b))


_ANSWER_LABEL = re.compile(r"^[\s#>*_-]*(?:the\s+)?(?:final\s+answers?|answers?|result)\b[\s*_:.=-]*", re.I)


def final_line(reference: str) -> str:
    """The part of a reference that states the answer.

    A short label ("50.7 atm") is returned whole. A worked response (production answers,
    re-solves) is searched for its answer:
      1. the last line labelled "Final answer(s)" / "Answer(s)" / "Result", or holding a
         \\boxed{}; a label standing alone takes the block that follows it;
      2. no label: the first paragraph AND the last paragraph with a number, because some
         responses lead with the answer and others end with it. number_check only needs one
         close pair, so the givens in a first paragraph cannot hide the answer any more."""
    text = (reference or "").strip()
    lines = [ln.strip() for ln in text.splitlines()]
    labelled = [i for i, ln in enumerate(lines) if _ANSWER_LABEL.match(ln) or "\\boxed{" in ln]
    if labelled:
        i = labelled[-1]
        if re.search(r"[\w$£€¥₹]", _ANSWER_LABEL.sub("", lines[i], count=1)):
            return lines[i]
        block, k = [], i + 1                     # the label stands alone: take the next block
        while k < len(lines) and not lines[k]:
            k += 1
        while k < len(lines) and lines[k]:
            block.append(lines[k])
            k += 1
        return "\n".join(block) if block else lines[i]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) > 1:
        with_number = [p for p in paras if re.search(r"\d", p)]
        last = (with_number or paras)[-1]
        return paras[0] if last == paras[0] else f"{paras[0]}\n\n{last}"
    return text


def all_values_match(reference: str, answer: str) -> bool:
    """Shared helper for scoring: every number in `reference` (zero included) equals some
    number in `answer` to the written precision. For short texts such as a truth string vs a
    cache output; use number_check for worked answers."""
    ref, ans = written_numbers(reference), written_numbers(answer)
    if not ref:
        return True
    if not ans:
        return False
    scales = _scales(reference, answer)
    return all(any(_agree(a, r, f) for f in scales for a in ans) for r in ref)


def number_check(answer: str, reference: str, rel_tol: float | None = None) -> dict:
    """Compare the numbers in a reference's final answer with an answer's final answer.

    Per reference number: "exact" if some answer number equals it to the written precision
    (SI rescaling allowed). "Conflict" if not exact, an answer number is within 5% of it, the
    two are the same kind of quantity (money, or the same unit word written after both), and
    that answer number equals no other reference number. Example: $5,053.07 vs $5,053.27.
    Any conflict -> mismatch, even when another number matches (a right payment can't cover a
    wrong interest). Otherwise at least one exact -> match; none -> mismatch. Asides in other
    units ("51.3 bar" next to "50.7 atm") and rougher restatements ("about 318 mm" next to
    "317.5 mm") are not conflicts.
    rel_tol is ignored (kept so old callers don't break); there is no percentage tolerance."""
    ref_text, ans_text = final_line(reference), final_line(answer)
    ref = [w for w in written_numbers(ref_text) if w.value != 0]
    ans = written_numbers(ans_text)
    scales = _scales(ref_text, ans_text)
    base = {"rule": "exact to the written precision", "reference_numbers": [w.value for w in ref[:8]],
            "answer_numbers": [w.value for w in ans[:16]]}
    if not ref or not ans:
        return {**base, "status": "unclear", "detail": "no numbers to compare"}
    money = bool(_MONEY.search(ref_text + " " + ans_text))
    accounted = {i for i, a in enumerate(ans) if any(_agree(a, r, f) for f in scales for r in ref)}
    exact, conflicts = [], []
    for r in ref:
        hit = next(((a, f) for f in scales for a in ans if _agree(a, r, f)), None)
        if hit:
            exact.append((r, *hit))
            continue
        near = [a for i, a in enumerate(ans) if i not in accounted
                and abs(a.value - r.value) <= _NEAR * abs(r.value)
                and (money or (r.unit and (r.unit == a.unit or (r.unit in _FACTORS and a.unit in _FACTORS
                                                               and _FACTORS[r.unit] == _FACTORS[a.unit]))))]
        if near:
            conflicts.append((r, min(near, key=lambda a: abs(a.value - r.value))))
    if conflicts:
        r, a = conflicts[0]
        return {**base, "status": "mismatch", "closest_reference": r.value, "closest_answer": a.value,
                "detail": f"reference {r.value:.12g} vs answer {a.value:.12g}: differ beyond the written "
                          f"precision (+-{max(r.half_unit, a.half_unit):g})"}
    if exact:
        r, a, f = exact[0]
        scale = "" if f == 1 else f" after scaling by {f:g} (a unit prefix)"
        return {**base, "status": "match", "closest_reference": r.value, "closest_answer": a.value,
                "scale": f, "detail": f"reference {r.value:.12g} equals answer {a.value:.12g} to the written precision{scale}"}
    r, a = min(((r, a) for r in ref for a in ans), key=lambda p: abs(p[1].value - p[0].value) / abs(p[0].value))
    return {**base, "status": "mismatch", "closest_reference": r.value, "closest_answer": a.value,
            "detail": f"no answer number equals a reference number; closest {r.value:.12g} vs {a.value:.12g}"}


def prompt_sections(prompt: str) -> list[dict]:
    """Split a '## Heading' style prompt into labeled sections (used to show judge evidence)."""
    sections, title, buf = [], "", []
    for line in (prompt or "").splitlines():
        if line.startswith("## "):
            if title or any(x.strip() for x in buf):
                sections.append({"title": title, "text": "\n".join(buf).strip()})
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if title or any(x.strip() for x in buf):
        sections.append({"title": title, "text": "\n".join(buf).strip()})
    return sections
