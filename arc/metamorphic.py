"""Metamorphic checks: test cached code with no answer key and no model.

DEVIATION from the paper (used by arc/labelfree.py). A metamorphic relation says how
the output must behave when the input changes in a known way (Chen et al. 1998; Segura
et al., TSE 2016). Here: writing the same quantity a different way must not change the
answer. "25 m/s" and "90 km/h" are the same speed, and "€2.0 million" and "€2,000,000"
are the same amount. Cached code that keeps the number but drops the unit fails this.

These checks are decorrelated from every model in the pipeline: no LLM writes, runs or
grades them. That matters because label-free verification otherwise relies on models
agreeing with each other, and same-family models share mistakes.

Only rewrites that are unambiguous are made: physical units from a fixed table, and
scale words (thousand, million, billion). Percent <-> fraction is NOT used: "0.0784"
for a percentage slot is ambiguous to a reader, so it is not a safe equivalence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .checks import numbers

# spelling (lower case) -> (dimension, factor to the SI unit)
_UNITS: dict[str, tuple[str, float]] = {}
_SPELL = {
    "length": [("m", 1.0, ["m", "metre", "metres", "meter", "meters"]),
               ("cm", 0.01, ["cm"]), ("km", 1000.0, ["km"]), ("mm", 0.001, ["mm"]),
               ("ft", 0.3048, ["ft", "feet", "foot"])],
    "volume": [("L", 1e-3, ["l", "litre", "litres", "liter", "liters"]), ("mL", 1e-6, ["ml"]),
               ("m3", 1.0, ["m3", "m^3", "m³"]), ("dm3", 1e-3, ["dm3", "dm^3", "dm³"]),
               ("cm3", 1e-6, ["cm3", "cm^3", "cm³", "cc"])],
    "mass": [("kg", 1.0, ["kg"]), ("g", 1e-3, ["g", "grams", "gram"]), ("mg", 1e-6, ["mg"])],
    "time": [("s", 1.0, ["s", "sec", "secs", "second", "seconds"]), ("min", 60.0, ["min", "mins", "minutes"]),
             ("h", 3600.0, ["h", "hr", "hrs", "hour", "hours"])],
    "speed": [("m/s", 1.0, ["m/s", "m s^-1", "m s-1", "ms^-1"]), ("km/h", 1 / 3.6, ["km/h", "kph", "km/hr"]),
              ("mph", 0.44704, ["mph"])],
    "pressure": [("kPa", 1e3, ["kpa"]), ("atm", 101325.0, ["atm"]), ("Pa", 1.0, ["pa"]),
                 ("bar", 1e5, ["bar"]), ("MPa", 1e6, ["mpa"])],
    "amount": [("mol", 1.0, ["mol", "mole", "moles"]), ("mmol", 1e-3, ["mmol"])],
    "energy": [("J", 1.0, ["j"]), ("kJ", 1e3, ["kj"])],
    "accel": [("m/s^2", 1.0, ["m/s^2", "m/s²", "m s^-2"]), ("ft/s^2", 0.3048, ["ft/s^2", "ft/s²"])],
}
_CANON: dict[str, list[tuple[str, float]]] = {}
for _dim, _rows in _SPELL.items():
    _CANON[_dim] = [(name, f) for name, f, _ in _rows]
    for _name, _f, _spellings in _rows:
        for _s in _spellings:
            _UNITS[_s] = (_dim, _f)
_TEMPS = {"k": "K", "kelvin": "K", "°c": "C", "ºc": "C", "degc": "C", "deg c": "C", "° c": "C",
          "degrees c": "C", "degrees celsius": "C", "celsius": "C"}

_VALUE = re.compile(r"^\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(.*?)\s*\.?\s*$")
_SCALE = re.compile(r"^\s*([£$€¥]?)\s*(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion)\b\s*(.*)$", re.I)
_PLAIN_MONEY = re.compile(r"^\s*([£$€¥])\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.0+)?\s*$")
_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def rewrites(value: str) -> list[str]:
    """Other ways to write the same value. Empty when no safe rewrite exists."""
    if not isinstance(value, str) or not value.strip():
        return []
    out: list[str] = []
    m = _SCALE.match(value)
    if m:  # "€2.0 million" -> "€2,000,000"
        cur, num, word, rest = m.groups()
        x = float(num.replace(",", "")) * _WORDS[word.lower()]
        out.append(f"{cur}{x:,.0f}{(' ' + rest) if rest else ''}".strip())
        return out
    m = _PLAIN_MONEY.match(value)
    if m:  # "€2,000,000" -> "€2 million"
        cur, num = m.groups()
        x = float(num.replace(",", ""))
        if x >= 1e6:
            out.append(f"{cur}{_fmt(x / 1e6)} million")
        return out
    m = _VALUE.match(value)
    if not m:
        return []
    mag, unit = float(m.group(1).replace(",", "")), m.group(2).strip()
    key = unit.lower()
    if key in _TEMPS:
        if _TEMPS[key] == "K":
            out.append(f"{_fmt(mag - 273.15)} °C")
        else:
            out.append(f"{_fmt(mag + 273.15)} K")
        return out
    if key not in _UNITS:
        return []
    dim, f_src = _UNITS[key]
    for name, f_dst in _CANON[dim]:
        if abs(f_dst - f_src) > 1e-15 * max(f_dst, f_src):
            out.append(f"{_fmt(mag * f_src / f_dst)} {name}")
            break
    return out


# ----------------------------------------------------------------------------- relation check

_FACTORS = (1.0, 1e3, 1e-3, 1e6, 1e-6)


def same_numbers(a: str, b: str, rel_tol: float = 1e-4) -> bool:
    """Does output text b state the same values as output text a? Every number in a must
    have a close number in b (a power-of-1000 rescale allowed, e.g. kPa vs Pa).

    Both texts are raw outputs of the same cached code on the same quantity written two ways,
    so they should agree to float precision; 1e-4 (tightened from 1% on 2026-11-15) leaves room
    only for a conversion constant written to fewer digits. A unit-dropping bug is off by 1.5x
    or more."""
    xs, ys = numbers(a), numbers(b)
    if not xs:
        return True
    if not ys:
        return False
    for x in xs:
        if x == 0:
            if not any(abs(y) < 1e-9 for y in ys):
                return False
            continue
        if not any(abs(y - x * f) <= rel_tol * abs(x * f) for f in _FACTORS for y in ys):
            return False
    return True


@dataclass
class Relation:
    slot: str
    original: str
    rewritten: str
    status: str          # held | violated | crashed
    detail: str = ""


def check(run, values: dict[str, str], base_output: dict, max_relations: int = 4) -> list[Relation]:
    """run(values) -> output dict. Rewrites one slot at a time; each rewrite must leave
    every output value the same. A crash on a rewrite is recorded but is not a violation
    (the cache falls back on a crash, so it never serves a wrong answer from it)."""
    out: list[Relation] = []
    for slot, value in values.items():
        for new in rewrites(value)[:1]:
            if len(out) >= max_relations:
                return out
            try:
                got = run({**values, slot: new})
            except Exception as e:  # noqa: BLE001 - any failure inside cached code
                out.append(Relation(slot, value, new, "crashed", f"{type(e).__name__}: {e}"[:200]))
                continue
            bad = [k for k, v in base_output.items()
                   if k not in got or not same_numbers(str(v), str(got[k]))]
            out.append(Relation(slot, value, new, "violated" if bad else "held",
                                ("changed: " + "; ".join(f"{k}: {base_output[k]} -> {got.get(k)}" for k in bad))[:400]
                                if bad else ""))
    return out
