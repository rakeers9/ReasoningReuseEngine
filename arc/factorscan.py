"""Find unit-conversion factors typed by hand into generated code.

DEVIATION (label-free builds, strict units). Cached code must do every unit conversion
through the runtime's unit primitive (UNIT_FN, implemented in arc/units.py), never with a
typed factor: a typed factor is how "$150k" became 150 and how a litre/cm3 slip made an entry
1000x off. The builder gets this scan's findings back as feedback and rewrites the function.

Two kinds of literal are flagged:
  ALWAYS   factors that almost never appear for any other reason (101.325, 273.15, 0.0254, ...)
  CONTEXT  common numbers (1000, 1e-6, 60, 760, ...) only when the same statement names a
           unit (volume_ml / 1000, "kPa"), so `count * 1000` or `range(60)` stay clean.
A literal inside a call to the unit primitive (convert_unit(x, "K", "degC") or a factor
written inside its arguments) is never flagged.
"""

from __future__ import annotations

import ast
import re

UNIT_FN = "convert_unit"

# Factors and offsets with essentially no other use in this kind of code.
ALWAYS = (101.325, 101325.0, 101.3, 273.15, 0.0254, 25.4, 2.54, 0.3048, 3.28084, 1.609, 1.609344,
          0.621371, 0.62137, 3.6, 2.20462, 0.453592, 0.45359237, 28.3495, 4.184, 4184.0, 133.322,
          133.3224, 6894.76, 6.89476, 14.696, 14.7, 88.4, 113.12, 0.0113, 459.67, 0.264172, 3.78541,
          1.852, 0.514444)
# Common numbers that are a unit factor only when the statement is about units.
CONTEXT = (1000.0, 0.001, 1e6, 1e-6, 1e9, 1e-9, 1e5, 1e-5, 100000.0, 760.0, 1.8, 60.0, 3600.0, 86400.0,
           1e12, 1e-12)

_UNIT_WORDS = re.compile(
    r"^(?:[kmµuncdh]?(?:pa|g|m|l|j|w|n|mol|s)|kpa|mpa|gpa|hpa|bar|mbar|atm|torr|mmhg|psi|"
    r"litres?|liters?|ml|dl|dm3|cm3|m3|cc|kg|mg|ug|lbs?|oz|ounces?|ft|feet|foot|inch(?:es)?|in|"
    r"miles?|mi|yards?|yd|km|cm|mm|nm|um|kelvin|celsius|fahrenheit|degc|degf|hours?|hr|hrs|"
    r"min|mins|minutes?|sec|secs|seconds?|days?|kmh|kph|mph|knots?|kj|kwh|cal|kcal|joules?|"
    r"watts?|kw|pascals?|grams?|kilograms?|metres?|meters?|gallons?|gal|mmol|umol|µmol)$", re.I)


def _is(value: float, table: tuple[float, ...]) -> bool:
    return any(abs(value - f) <= 1e-9 * abs(f) for f in table)


def _unit_context(stmt: ast.AST) -> bool:
    """Does the statement name a unit in an identifier part or a string?"""
    words: list[str] = []
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name):
            words += re.split(r"[_\W]+", n.id)
        elif isinstance(n, ast.Attribute):
            words += re.split(r"[_\W]+", n.attr)
        elif isinstance(n, ast.arg):
            words += re.split(r"[_\W]+", n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            words += re.split(r"[\s_,;:()/\[\]{}'\"=.-]+", n.value)
    return any(w and _UNIT_WORDS.match(w) for w in words)


_SCALE_WORDS = re.compile(r"^(?:k|thousands?|millions?|mn|mil|bn|billions?|trillions?|tn|lakhs?|crores?)$", re.I)


def _scale_context(stmt: ast.AST) -> bool:
    """A statement about scale words ("k", "million") uses 1000 / 1e6 as a scale, not a unit
    factor ({"k": 1000, "million": 1e6}). Bare "m" stays ambiguous (metre or million)."""
    for n in ast.walk(stmt):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and _SCALE_WORDS.match(n.value.strip()):
            return True
        if isinstance(n, ast.Name) and any(_SCALE_WORDS.match(w) for w in re.split(r"[_\W]+", n.id) if w):
            return True
    return False


def _has(stmt: ast.AST, value: float) -> bool:
    """Fahrenheit: "x * 1.8 + 32" names no unit, but 1.8 with 32 is the conversion."""
    return any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and n.value == value
               for n in ast.walk(stmt))


def _inside_unit_call(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    p = parents.get(node)
    while p is not None:
        if isinstance(p, ast.Call):
            f = p.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name == UNIT_FN:
                return True
        p = parents.get(p)
    return False


def _statement(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    p = parents.get(node)
    while p is not None and not isinstance(p, ast.stmt):
        p = parents.get(p)
    return p or node


def typed_factors(code: str) -> list[str]:
    """One message per hand-typed unit factor found; [] if clean (or unparsable: the
    builder's own static check reports syntax errors)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            continue
        v = abs(float(node.value))
        if v == 0 or _inside_unit_call(node, parents):
            continue
        stmt = _statement(node, parents)
        context_hit = _is(v, CONTEXT) and not _scale_context(stmt) and (
            _unit_context(stmt) or (_is(v, (1.8,)) and _has(stmt, 32)))
        if _is(v, ALWAYS) or context_hit:
            found.append(f"line {node.lineno}: {node.value!r} looks like a hand-typed unit conversion "
                         f"factor; convert with {UNIT_FN}(value, from_unit, to_unit) instead")
    return found
