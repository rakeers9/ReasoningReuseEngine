"""The final formatter: the one place a cached answer is rounded and put in its output unit.

DEVIATION from the paper, whose cached code formats its own outputs. Generated code rounded
and converted units on its own, and that is where the wrong answers came from. Now the code
computes at full precision and this module, plain tested code, does the presentation once:

  1. unit     if the question asks for an output unit (extraction's output_format.unit), it must
              be the unit the entry already gives (aliases like "$" / "dollars" count). There is
              no unit library: any other unit is a PresentError, and the model answers instead.
  2. rounding exact decimal arithmetic (ROUND_HALF_UP), only here:
                - the question's own request first: "dp:N", "sf:N", "nearest:X";
                - else the entry's output spec (cache.outputs): money to the cent, counts exact,
                  measurements to MEASURE_SIG_FIGS significant figures (ROUNDING_POLICY);
                - else (no spec) full precision, as computed.
Text and boolean outputs pass through unchanged.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from .runtime import CacheInvocationError


# The one rounding policy (user decision 2026-11-16). The formatter applies it to every cached
# answer, and the re-solve, tie-break and conventions prompts quote this exact text, so every
# part of the system rounds the same way.
MEASURE_SIG_FIGS = 4
ROUNDING_POLICY = (
    "Rounding policy: never round any intermediate value; round only the final answer, and only "
    "once. If the question says how to round, do exactly that. Otherwise: money to the cent "
    "(2 decimal places, halves rounded up), counts as exact whole numbers, and every other "
    f"quantity (measurements, rates, percentages) to {MEASURE_SIG_FIGS} significant figures, "
    "halves rounded up."
)


class PresentError(CacheInvocationError):
    """The answer can't be put in the form the question asks for: let the model answer."""


_NUMERIC_TEXT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _num(v: Any) -> Decimal | None:
    """A number, or text that is exactly a number (cached code often returns str(Decimal) to
    keep full precision; that must be rounded too). Text with anything else in it is left as is."""
    if isinstance(v, str) and _NUMERIC_TEXT.fullmatch(v.strip()):
        v = Decimal(v.strip())
    if isinstance(v, bool) or not isinstance(v, (int, float, Decimal)):
        return None
    try:
        d = Decimal(str(v))
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def round_to(value: Decimal, rule: str) -> Decimal:
    """'dp:2' -> 2 decimal places, 'sf:3' -> 3 significant figures, 'nearest:1000'."""
    kind, _, arg = rule.partition(":")
    if kind == "dp":
        return value.quantize(Decimal(1).scaleb(-int(arg)), rounding=ROUND_HALF_UP)
    if kind == "sf":
        n = int(arg)
        if value == 0 or n < 1:
            return value
        exp = value.adjusted() - n + 1
        return value.quantize(Decimal(1).scaleb(exp), rounding=ROUND_HALF_UP)
    if kind == "nearest":
        step = Decimal(arg)
        if step <= 0:
            raise PresentError(f"cannot round to the nearest {arg}")
        return (value / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step
    raise PresentError(f"unknown rounding {rule!r}")


def _text(d: Decimal) -> str:
    """Plain digits, keeping the rounded precision ("7.759", "5100", "0.0001234")."""
    if d == d.to_integral_value() and d.as_tuple().exponent >= 0:
        return f"{int(d)}"
    return format(d, "f")


_ALIASES = {"$": "usd", "us$": "usd", "usd": "usd", "dollar": "usd", "dollars": "usd",
            "£": "gbp", "gbp": "gbp", "pound": "gbp", "pounds": "gbp",
            "€": "eur", "eur": "eur", "euro": "eur", "euros": "eur",
            "%": "%", "percent": "%", "per cent": "%", "percentage": "%"}


def _same_unit(a: str, b: str) -> bool:
    na, nb = (a or "").strip().lower(), (b or "").strip().lower()
    return _ALIASES.get(na, na.replace(" ", "")) == _ALIASES.get(nb, nb.replace(" ", ""))


def apply(output: dict, output_format: dict | None = None, specs: dict | None = None) -> dict:
    """The cached output, converted and rounded for the reader. Raises PresentError."""
    fmt, specs = output_format or {}, specs or {}
    target = (fmt.get("unit") or "").strip()
    rule = (fmt.get("rounding") or "").strip()
    out, converted = {}, 0
    for key, value in output.items():
        d = _num(value)
        if d is None:
            out[key] = value
            continue
        spec = specs.get(key) or {}
        unit, kind = (spec.get("unit") or "").strip(), spec.get("kind")
        # No unit library (user decision 2026-11-15: the model's code does conversions). A
        # requested unit must already be the output's unit; anything else, let the model answer.
        if target and unit and not _same_unit(unit, target):
            raise PresentError(f"the question asks for {key!r} in {target}; the entry gives {unit}")
        if target and unit:
            converted += 1
        if rule:
            d = round_to(d, rule)
        elif kind == "money":
            d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        elif kind == "count" and d == d.to_integral_value():
            d = d.quantize(Decimal(1))
        elif kind == "measure":
            d = round_to(d, f"sf:{MEASURE_SIG_FIGS}")
        out[key] = _text(d)
    if target and not converted:
        raise PresentError(f"the question asks for the answer in {target}, and no output is known to be in a "
                           "compatible unit")
    return out
