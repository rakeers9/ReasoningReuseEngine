"""Exact unit conversion for cached code. DEVIATION: the CPU-ARC paper has no such helper.

Why: every wrong or blocked cached answer so far came from model-written conversion code
(a litre/cm3 factor 1000x off, umol/L creatinine left unconverted, km/h left as is). Cached
code calls `convert_unit(...)` (this module's `convert`) instead of typing factors by hand.

    convert(value, from_unit, to_unit, substance=None) -> float | Decimal
    same_quantity(unit_a, unit_b, substance=None)      -> bool

Rules (production-strict; a refusal costs one model call, a wrong factor is served to every
matching question):
  - Every factor is an exact rational (fractions.Fraction) taken from a cited definition, and
    the arithmetic is exact. A float input is read by its shortest repr (0.1 means 1/10) and
    the result is the correctly rounded float. A Decimal input gives a Decimal result (exact
    when the decimal terminates, else 34 significant digits). Never rounded beyond that.
  - Unknown, ambiguous or mismatched units raise UnitError (a ValueError). Ambiguous
    spellings (bare "gal", "ton", "hp", "C", "F", "year", ...) are refused, never guessed.
  - Symbols are case-sensitive (mm != Mm, mL != ML); spelled-out names are not.
  - Temperature: a lone °C/°F/K/°R is an absolute temperature (offsets applied, nothing below
    absolute zero). Inside a compound unit (J/(g*°C), °C/min) a degree is a temperature
    difference. For a lone difference use "delta_degC" / "delta_degF" (or Δ°C / Δ°F).
  - Amount <-> mass (umol/L <-> mg/dL) needs `substance=`, looked up in SUBSTANCES (molar
    masses with sources). It is never inferred.

Sources cited below:
  [SI]   BIPM, The International System of Units (SI Brochure), 9th ed., 2019.
  [811]  NIST Special Publication 811 (2008), Appendix B: conversion factors.
  [YP]   International yard and pound agreement, 1959 (yd = 0.9144 m, lb = 0.45359237 kg).
  [WMA]  UK Weights and Measures Act 1985, Schedule 1 (imperial gallon = 4.54609 L).
  [IUPAC] CIAAW / IUPAC abridged standard atomic weights, 2021.
"""

from __future__ import annotations

import math
import numbers
import re
from decimal import Decimal, Inexact, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

__all__ = ["UnitError", "convert", "same_quantity", "SUBSTANCES"]


class UnitError(ValueError):
    """Unknown, ambiguous or incompatible unit, or a value that cannot be converted."""


F = Fraction

# ----------------------------------------------------------------------------- dimensions
# Exponents over the SI base units used here.
BASES = ("m", "kg", "s", "K", "mol", "A")
Dim = tuple  # tuple[int, ...] of len(BASES)


def _dim(**exps: int) -> Dim:
    return tuple(exps.get(b, 0) for b in BASES)


NONE = _dim()
LENGTH, MASS, TIME, TEMP, AMOUNT, CURRENT = (_dim(**{b: 1}) for b in BASES)


def _add(a: Dim, b: Dim, k: int = 1) -> Dim:
    return tuple(x + k * y for x, y in zip(a, b))


def _show(d: Dim) -> str:
    parts = [b if e == 1 else f"{b}^{e}" for b, e in zip(BASES, d) if e]
    return "*".join(parts) or "dimensionless"


class _Unit(NamedTuple):
    factor: Fraction            # SI value = value * factor (+ offset, absolute temperatures only)
    dim: Dim
    offset: Fraction | None = None   # not None: an absolute temperature scale (K and °R: 0)


# ----------------------------------------------------------------------------- unit table
_SYMBOLS: dict[str, _Unit] = {}   # case-sensitive symbols and multi-word spellings
_NAMES: dict[str, _Unit] = {}     # spelled-out names, matched lower-case
_AMBIGUOUS: dict[str, str] = {}   # spelling -> why it is refused (symbols and lower-case names)


def _put(table: dict, key: str, unit: _Unit) -> None:
    if key in table and table[key] != unit:
        raise RuntimeError(f"unit table conflict on {key!r}")   # caught by the test suite at import
    table[key] = unit


def _def(symbols: str | tuple = (), names: str | tuple = (), factor=1, dim: Dim = NONE,
         offset=None) -> _Unit:
    unit = _Unit(F(factor), dim, None if offset is None else F(offset))
    for s in (symbols,) if isinstance(symbols, str) else symbols:
        _put(_SYMBOLS, s, unit)
    for n in (names,) if isinstance(names, str) else names:
        _put(_NAMES, n.lower(), unit)
    return unit


def _plural(*names: str) -> tuple:
    return tuple(x for n in names for x in (n, n + "s"))


# SI prefixes [SI Table 7]. Only the ones listed per base unit are generated, so that rare
# combinations can't collide with common symbols (Mm vs mm, ML vs mL, Mg vs mg).
_PREFIX = {"G": (F(10) ** 9, "giga"), "M": (F(10) ** 6, "mega"), "k": (F(1000), "kilo"),
           "h": (F(100), "hecto"), "da": (F(10), "deca"), "d": (F(1, 10), "deci"),
           "c": (F(1, 100), "centi"), "m": (F(1, 1000), "milli"), "u": (F(1, 10**6), "micro"),
           "n": (F(1, 10**9), "nano"), "p": (F(1, 10**12), "pico")}


def _prefixed(symbol: str, names: tuple, factor, dim: Dim, prefixes: str) -> None:
    """Define the base unit and each listed prefix, with names like 'kilometre(s)'."""
    _def(symbol, _plural(*names), factor, dim)
    for p in prefixes.split():
        scale, word = _PREFIX[p]
        _def(p + symbol, _plural(*(word + n for n in names)), F(factor) * scale, dim)


SMALL = "k h da d c m u n p"

# --- length [SI]; inch/foot/yard/mile [YP]; nautical mile [811]; angstrom [811]
_prefixed("m", ("metre", "meter"), 1, LENGTH, SMALL)
_def(("in",), ("inch", "inches"), F("0.0254"), LENGTH)
_def(("ft",), ("foot", "feet"), F("0.3048"), LENGTH)
_def(("yd",), _plural("yard"), F("0.9144"), LENGTH)
_def(("mi",), _plural("mile"), F("1609.344"), LENGTH)            # 5280 ft
_def(("nmi",), _plural("nautical mile"), 1852, LENGTH)
_def(("Å",), _plural("angstrom"), F(1, 10**10), LENGTH)

# --- mass [SI]; pound, ounce (1/16 lb), stone (14 lb) [YP]; troy ounce [811]; tonne [SI]
_prefixed("g", ("gram", "gramme"), F(1, 1000), MASS, "k h da d c m u n p")
_def("mcg", _plural("microgram"), F(1, 10**9), MASS)
_def(("lb", "lbs"), _plural("pound"), F("0.45359237"), MASS)
_def("oz", _plural("ounce"), F("0.45359237") / 16, MASS)
_def(("ozt", "oz t"), _plural("troy ounce"), F("0.0311034768"), MASS)
_def((), ("stone",), F("0.45359237") * 14, MASS)
_def("t", _plural("tonne", "metric ton"), 1000, MASS)

# --- time [SI]
_prefixed("s", ("second",), 1, TIME, "m u n p")
_def(("sec", "secs"), (), 1, TIME)
_def(("min", "mins"), _plural("minute"), 60, TIME)
_def(("h", "hr", "hrs"), _plural("hour"), 3600, TIME)
_def("d", _plural("day"), 86400, TIME)
_def("wk", _plural("week"), 604800, TIME)

# --- temperature [SI 2.3.1]: T(K) = t(°C) + 273.15; T(K) = (t(°F) + 459.67) * 5/9 [811 B.9]
_def(("K", "°K"), _plural("kelvin"), 1, TEMP, offset=0)
_def(("°C", "degC", "deg C", "deg_C"), ("celsius", "degree celsius", "degrees celsius"),
     1, TEMP, offset=F("273.15"))
_def(("°F", "degF", "deg F", "deg_F"), ("fahrenheit", "degree fahrenheit", "degrees fahrenheit"),
     F(5, 9), TEMP, offset=F("459.67") * F(5, 9))
_def(("°R", "degR"), ("rankine", "degree rankine", "degrees rankine"), F(5, 9), TEMP, offset=0)
# temperature differences (no offset)
_def(("delta_degC", "Δ°C", "ΔdegC"), (), 1, TEMP)
_def(("delta_degF", "Δ°F", "ΔdegF"), (), F(5, 9), TEMP)
_def(("delta_K", "ΔK"), (), 1, TEMP)

# --- amount of substance [SI]; molar concentration M = mol/L [811]
_prefixed("mol", ("mole",), 1, AMOUNT, "k m u n p")
for _p in "m u n p".split():                       # clinical shorthand: "micromol/L"
    _def((), (_PREFIX[_p][1] + "mol",), _PREFIX[_p][0], AMOUNT)
_MOLAR = _add(AMOUNT, LENGTH, -3)
_def("M", ("molar",), 1000, _MOLAR)
for _p in "m u n p".split():
    _def(_p + "M", (), 1000 * _PREFIX[_p][0], _MOLAR)

# --- volume: litre = 1 dm^3 exactly [SI Table 8]
_VOL = _dim(m=3)
_prefixed("L", ("litre", "liter"), F(1, 1000), _VOL, SMALL)
for _p in SMALL.split():
    _def(_p + "l", (), F(1, 1000) * _PREFIX[_p][0], _VOL)
_def("l", (), F(1, 1000), _VOL)
_def("cc", (), F(1, 10**6), _VOL)
# US gallon = 231 in^3 exactly [811]; imperial gallon [WMA]; fluid ounces = gal/128, gal/160
_US_GAL = 231 * F("0.0254") ** 3
_IMP_GAL = F("4.54609") / 1000
_def(("US gal", "gal_us", "gal (US)"), _plural("US gallon"), _US_GAL, _VOL)
_def(("imp gal", "gal_imp", "gal (imp)"), _plural("imperial gallon"), _IMP_GAL, _VOL)
_def(("US qt",), _plural("US quart"), _US_GAL / 4, _VOL)
_def(("US pt",), _plural("US pint"), _US_GAL / 8, _VOL)
_def(("US fl oz", "fl oz (US)"), (), _US_GAL / 128, _VOL)
_def(("imp fl oz", "fl oz (imp)"), (), _IMP_GAL / 160, _VOL)

# --- area: hectare [SI Table 8]; acre = 43560 ft^2 (international foot) [811]
_def("ha", _plural("hectare"), 10**4, _dim(m=2))
_def((), _plural("acre"), 43560 * F("0.3048") ** 2, _dim(m=2))

# --- speed: knot = 1852 m/h [811]
_SPEED = _dim(m=1, s=-1)
_def("mph", (), F("1609.344") / 3600, _SPEED)
_def("kph", (), F(1000, 3600), _SPEED)
_def(("kn", "kt"), _plural("knot"), F(1852, 3600), _SPEED)

# --- force: newton [SI]; dyne [811]; lbf = lb * standard gravity 9.80665 m/s^2 [811, CGPM 1901]
_FORCE = _dim(kg=1, m=1, s=-2)
_prefixed("N", ("newton",), 1, _FORCE, "k m")
_def("dyn", _plural("dyne"), F(1, 10**5), _FORCE)
_LBF = F("0.45359237") * F("9.80665")
_def("lbf", ("pound-force", "pounds-force"), _LBF, _FORCE)

# --- pressure: pascal [SI]; bar, atm [SI Table 8 note, 811]; Torr = atm/760 [811];
#     mmHg conventional 133.322387415 Pa, inHg conventional 3386.389 Pa,
#     cmH2O conventional 98.0665 Pa [811 B.8]; psi = lbf/in^2 [811]
_PRES = _dim(kg=1, m=-1, s=-2)
_prefixed("Pa", ("pascal",), 1, _PRES, "G M k h")
_def("bar", _plural("bar"), 10**5, _PRES)
_def("mbar", _plural("millibar"), 100, _PRES)
_def("atm", _plural("atmosphere"), 101325, _PRES)
_def(("Torr", "torr"), (), F(101325, 760), _PRES)
_def(("mmHg", "mm Hg"), (), F("133.322387415"), _PRES)
_def(("inHg", "in Hg"), (), F("3386.389"), _PRES)
_def(("cmH2O", "cm H2O"), (), F("98.0665"), _PRES)
_def("psi", (), _LBF / F("0.0254") ** 2, _PRES)

# --- energy: joule [SI]; thermochemical calorie 4.184 J, BTU (IT) 1055.05585262 J [811];
#     electronvolt 1.602176634e-19 J exactly [SI 2019]; Wh = 3600 J; erg [811]
_ENERGY = _dim(kg=1, m=2, s=-2)
_prefixed("J", ("joule",), 1, _ENERGY, "G M k m")
_def("cal", _plural("calorie"), F("4.184"), _ENERGY)
_def("kcal", _plural("kilocalorie"), F("4184"), _ENERGY)
_def(("BTU", "Btu"), (), F("1055.05585262"), _ENERGY)
_prefixed("eV", ("electronvolt",), F("1.602176634e-19"), _ENERGY, "G M k")
_prefixed("Wh", ("watt-hour",), 3600, _ENERGY, "G M k")
_def("erg", _plural("erg"), F(1, 10**7), _ENERGY)

# --- power, current, voltage, resistance, charge, frequency [SI]
_POWER = _dim(kg=1, m=2, s=-3)
_prefixed("W", ("watt",), 1, _POWER, "G M k m")
_prefixed("A", ("ampere", "amp"), 1, CURRENT, "k m u")
_VOLT = _add(_POWER, CURRENT, -1)
_prefixed("V", ("volt",), 1, _VOLT, "k m")
_def(("Ω", "ohm"), _plural("ohm"), 1, _add(_VOLT, CURRENT, -1))
_def("kΩ", _plural("kiloohm"), 1000, _add(_VOLT, CURRENT, -1))
_def("Ah", (), 3600, _dim(A=1, s=1))
_def("mAh", (), F("3.6"), _dim(A=1, s=1))
_prefixed("Hz", ("hertz",), 1, _dim(s=-1), "G M k")

# --- dimensionless: percent, per mille, ppm, ppb [811 7.10.3]
_def("1", ("fraction",), 1, NONE)
_def("%", ("percent", "per cent"), F(1, 100), NONE)
_def("‰", ("permille", "per mille"), F(1, 1000), NONE)
_def("ppm", (), F(1, 10**6), NONE)
_def("ppb", (), F(1, 10**9), NONE)

# --- refused spellings: meaning depends on context, so the caller must say which
for _keys, _why in [
    (("gal", "gallon", "gallons"), "US or imperial? use 'US gal' or 'imp gal'"),
    (("qt", "quart", "quarts", "pt", "pint", "pints"), "US or imperial? use 'US qt' / 'US pt'"),
    (("fl oz", "floz", "fluid ounce", "fluid ounces"), "US or imperial? use 'US fl oz' or 'imp fl oz'"),
    (("cup", "cups", "tbsp", "tsp", "tablespoon", "teaspoon"), "kitchen measures vary by country"),
    (("ton", "tons"), "short, long or metric ton? use 't' for the metric tonne"),
    (("hp", "horsepower"), "mechanical or metric horsepower? convert from W"),
    (("C",), "'C' is the coulomb symbol; for Celsius use '°C' or 'degC'"),
    (("F",), "'F' is the farad symbol; for Fahrenheit use '°F' or 'degF'"),
    (("Cal", "Calorie", "Calories"), "food Calorie (kcal) or calorie? use 'kcal' or 'cal'"),
    (("yr", "year", "years", "y", "a", "mo", "month", "months"),
     "calendar lengths vary; use 'd' with an explicit day count"),
    (("ppt",), "parts per thousand or per trillion? use '‰' or a fraction"),
    (("Eq", "mEq", "eq", "meq"), "equivalents need the ion's valence; not supported"),
    (("IU", "U", "iu"), "international units are substance-specific; not supported"),
    (("°", "deg", "degree", "degrees"), "temperature scale not stated; use '°C', '°F' or 'K'"),
]:
    for _k in _keys:
        _AMBIGUOUS[_k] = _why

# ----------------------------------------------------------------------------- substances
class Substance(NamedTuple):
    molar_mass: Fraction   # g/mol
    source: str


# Molar masses in g/mol, each to the precision its source states. Formulas use [IUPAC] atomic
# weights (H 1.008, C 12.011, N 14.007, O 15.999, Ca 40.078); the tests recompute them.
SUBSTANCES: dict[str, Substance] = {
    "creatinine": Substance(F("113.12"), "C4H7N3O, PubChem CID 588: 113.12 g/mol"),
    "glucose": Substance(F("180.16"), "C6H12O6, PubChem CID 5793: 180.16 g/mol"),
    "urea": Substance(F("60.06"), "CH4N2O, PubChem CID 1176: 60.06 g/mol"),
    # BUN is reported as mass of urea nitrogen: 2 N per urea, 2 x 14.007 [IUPAC]
    "urea nitrogen": Substance(F("28.014"), "N2 in urea, 2 x 14.007 g/mol [IUPAC 2021]"),
    "calcium": Substance(F("40.078"), "Ca, IUPAC 2021 standard atomic weight 40.078"),
}
_SUBSTANCE_ALIASES = {"bun": "urea nitrogen", "blood urea nitrogen": "urea nitrogen",
                      "ca": "calcium", "creat": "creatinine", "cr": "creatinine"}


def _substance(name: str) -> Substance:
    key = " ".join(str(name).lower().split())
    key = _SUBSTANCE_ALIASES.get(key, key)
    if key not in SUBSTANCES:
        raise UnitError(f"no molar mass registered for {name!r}; known: {sorted(SUBSTANCES)}")
    return SUBSTANCES[key]


# ----------------------------------------------------------------------------- parsing
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUPER_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")
_TOKEN = re.compile(r"\s*(\(|\)|\*|/|\^\s*-?\d+|[^()*/^]+)")
_TRAILING_EXP = re.compile(r"^(.*?[^\d\s-])(-?\d+)$")


def _normalize(unit: str) -> str:
    s = unit.strip()
    s = s.replace("µ", "u").replace("μ", "u")          # micro sign and Greek mu
    s = s.replace("℃", "°C").replace("℉", "°F").replace("º", "°")
    s = re.sub(r"°\s+([CFKR])\b", r"°\1", s)
    s = _SUPER_RUN.sub(lambda m: "^" + m.group().translate(_SUPER), s)
    s = re.sub(r"[·⋅•]", "*", s)
    s = re.sub(r"\s+per\s+", "/", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s)


def _lookup(token: str) -> _Unit:
    t = token.strip()
    if t in _SYMBOLS:
        return _SYMBOLS[t]
    if t in _AMBIGUOUS:                         # exact case first: "Calorie" is refused, "calorie" is not
        raise UnitError(f"ambiguous unit {token!r}: {_AMBIGUOUS[t]}")
    if t.lower() in _NAMES:
        return _NAMES[t.lower()]
    if t.lower() in _AMBIGUOUS:
        raise UnitError(f"ambiguous unit {token!r}: {_AMBIGUOUS[t.lower()]}")
    m = _TRAILING_EXP.match(t)                  # dm3, cm2, s-1
    if m:
        base = _lookup(m.group(1))
        return _power(base, int(m.group(2)))
    raise UnitError(f"unknown unit {token!r}")


def _power(u: _Unit, n: int) -> _Unit:
    if n == 0:
        raise UnitError("a zero exponent is not a unit")
    return _Unit(u.factor ** n, tuple(e * n for e in u.dim))


def _parse(unit: str) -> _Unit:
    """Parse a unit string into (factor, dimension, offset). Raises UnitError."""
    if not isinstance(unit, str) or not unit.strip():
        raise UnitError(f"unit must be a non-empty string, got {unit!r}")
    return _parse_str(unit)


@lru_cache(maxsize=1024)
def _parse_str(unit: str) -> _Unit:
    s = _normalize(unit)
    if s in _SYMBOLS or s.lower() in _NAMES:       # whole-string spelling, incl. multi-word ones
        return _lookup(s)
    tokens = [t.strip() for t in _TOKEN.findall(s) if t.strip()]
    if "".join(t.replace(" ", "") for t in tokens) != s.replace(" ", ""):
        raise UnitError(f"cannot parse unit {unit!r}")
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def atom() -> tuple[_Unit, bool]:
        t = take() if peek() is not None else None
        if t is None or t in ("*", "/", ")") or t.startswith("^"):
            raise UnitError(f"cannot parse unit {unit!r}")
        if t == "(":
            u, single = expr()
            if peek() != ")":
                raise UnitError(f"unbalanced parentheses in {unit!r}")
            take()
            return u, single
        return _lookup(t), True

    def factor() -> tuple[_Unit, bool]:
        u, single = atom()
        if peek() is not None and peek().startswith("^"):
            u, single = _power(u, int(take()[1:].strip())), False
        return u, single

    def expr() -> tuple[_Unit, bool]:
        u, single = factor()
        while peek() in ("*", "/"):
            op = take()
            v, _ = factor()
            k = 1 if op == "*" else -1
            u, single = _Unit(u.factor * v.factor ** k, _add(u.dim, v.dim, k)), False
        return u, single

    u, single = expr()
    if pos != len(tokens):
        raise UnitError(f"cannot parse unit {unit!r}")
    if not single:                     # compound: any degree inside is a temperature difference
        return _Unit(u.factor, u.dim)
    return u


# ----------------------------------------------------------------------------- public API
def _exact(value) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, Decimal)):
        raise UnitError(f"value must be a number, got {type(value).__name__}")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise UnitError(f"value must be finite, got {value}")
        return F(value)
    if isinstance(value, numbers.Integral):
        return F(int(value))
    if isinstance(value, numbers.Rational):
        return F(value.numerator, value.denominator)
    v = float(value)
    if not math.isfinite(v):
        raise UnitError(f"value must be finite, got {v}")
    return F(repr(v))                  # shortest repr: 0.1 means one tenth, as written


def _bridge(a: _Unit, b: _Unit, substance) -> int | None:
    """Exponent k such that b.dim == a.dim + k*(mass - amount), via a molar mass; else None."""
    if substance is None:
        return None
    step = _add(MASS, AMOUNT, -1)
    for k in (1, -1):
        if _add(a.dim, step, k) == b.dim:
            return k
    return None


def _check_compatible(a: _Unit, b: _Unit, substance, fa: str, fb: str) -> int:
    """0 for same dimension, ±1 for a molar-mass bridge; raises UnitError otherwise."""
    if a.dim == b.dim:
        if (a.offset or b.offset) and (a.offset is None or b.offset is None):
            raise UnitError(f"cannot convert between an absolute temperature and a temperature "
                            f"difference ({fa!r} -> {fb!r}); use delta_degC/delta_degF for differences")
        return 0
    k = _bridge(a, b, substance)
    if k is None:
        hint = (" (amount <-> mass needs substance=...)"
                if _bridge(a, b, "any") is not None else "")
        raise UnitError(f"incompatible units {fa!r} ({_show(a.dim)}) and {fb!r} "
                        f"({_show(b.dim)}){hint}")
    return k


def convert(value, from_unit: str, to_unit: str, substance: str | None = None):
    """Convert `value` from `from_unit` to `to_unit` exactly.

    Returns a Decimal when given a Decimal, else a float. `substance` names a registered
    molar mass for amount <-> mass conversions (e.g. umol/L <-> mg/dL creatinine).
    Raises UnitError for unknown, ambiguous or incompatible units and invalid values.
    """
    x = _exact(value)
    a, b = _parse(from_unit), _parse(to_unit)
    k = _check_compatible(a, b, substance, from_unit, to_unit)
    si = x * a.factor + (a.offset or 0)
    if (a.offset is not None or b.offset is not None) and si < 0:
        raise UnitError(f"{value} {from_unit} is below absolute zero")
    if k:
        si *= (_substance(substance).molar_mass / 1000) ** k      # g/mol -> kg/mol
    out = (si - (b.offset or 0)) / b.factor
    if isinstance(value, Decimal):
        with localcontext() as c:
            c.prec = 34
            c.traps[Inexact] = False
            return Decimal(out.numerator) / Decimal(out.denominator)
    try:
        return float(out)
    except OverflowError as e:
        raise UnitError(f"result out of float range: {value} {from_unit} -> {to_unit}") from e


def same_quantity(unit_a: str, unit_b: str, substance: str | None = None) -> bool:
    """True if a value in `unit_a` can be converted to `unit_b` (with `substance`, if given).
    Unknown or ambiguous units give False, so callers can refuse values that don't fit."""
    try:
        a, b = _parse(unit_a), _parse(unit_b)
        k = _check_compatible(a, b, substance, unit_a, unit_b)
        if k:
            _substance(substance)
        return True
    except UnitError:
        return False
