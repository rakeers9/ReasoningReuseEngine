"""arc/units.py: exact unit conversion for cached code (DEVIATION), and its runtime primitive."""

from __future__ import annotations

import re
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from arc import embed, runtime, sandbox, serve, units
from arc.cache import Cache
from arc.units import UnitError, convert, same_quantity

D = Decimal


def dec_div(a, b) -> Decimal:
    with localcontext() as c:
        c.prec = 34
        return D(a) / D(b)


# ----------------------------------------------------------------------------- exact values

@pytest.mark.parametrize("value, src, dst, expected", [
    (1, "L", "cm3", 1000.0),
    (1, "L", "mL", 1000.0),
    (1, "dm3", "L", 1.0),
    (1, "m³", "L", 1000.0),
    (2.5, "L", "m^3", 0.0025),
    (100, "km/h", "m/s", float(Fraction(250, 9))),
    (36, "km/h", "m/s", 10.0),
    (60, "mph", "km/h", 96.56064),
    (1, "knot", "km/h", 1.852),
    (1, "atm", "kPa", 101.325),
    (1, "atm", "Pa", 101325.0),
    (1, "bar", "kPa", 100.0),
    (760, "Torr", "atm", 1.0),
    (1, "mmHg", "Pa", 133.322387415),
    (1, "kg", "lb", float(1 / Fraction("0.45359237"))),
    (1, "lb", "g", 453.59237),
    (16, "oz", "lb", 1.0),
    (1, "in", "cm", 2.54),
    (1, "mi", "km", 1.609344),
    (1, "US gal", "L", 3.785411784),
    (1, "imp gal", "L", 4.54609),
    (1, "kcal", "kJ", 4.184),
    (1, "kWh", "MJ", 3.6),
    (1, "h", "s", 3600.0),
    (90, "min", "h", 1.5),
    (1, "acre", "m^2", 4046.8564224),
    (1, "ha", "km2", 0.01),
    (35, "%", "1", 0.35),
    (0.35, "1", "%", 35.0),
    (250, "ppm", "%", 0.025),
    (1, "mg/dL", "g/L", 0.01),
    (5.5, "mmol/L", "umol/L", 5500.0),
    (2, "mM", "mmol/L", 2.0),
    (1, "J/(g*°C)", "J/(kg*K)", 1000.0),      # degree inside a compound is a difference
    (1, "W/(m*K)", "W/(m*°C)", 1.0),
    (1, "kg*m/s^2", "N", 1.0),
    (1, "J/kg/K", "J/(kg*K)", 1.0),
    (1, "s-1", "Hz", 1.0),
])
def test_exact_values(value, src, dst, expected):
    assert convert(value, src, dst) == expected


def test_float_input_read_as_written():
    # 0.1 is read as one tenth, so 0.1 L is exactly 100 mL (naive float maths gives 100.00000000000001)
    assert convert(0.1, "L", "mL") == 100.0
    assert convert(0.07, "L", "mL") == 70.0


def test_psi_from_exact_definitions():
    # psi = lbf / in^2, lbf = 0.45359237 kg * 9.80665 m/s^2 (NIST SP 811)
    assert convert(D(1), "psi", "Pa") == dec_div(D("0.45359237") * D("9.80665"), D("0.0254") ** 2)


def test_mmhg_is_not_torr():
    # conventional mmHg (133.322387415 Pa) differs from Torr (101325/760 Pa) in the 7th digit
    assert convert(D(760), "Torr", "atm") == D(1)
    assert convert(D(760), "mmHg", "atm") != D(1)


# ----------------------------------------------------------------------------- spellings

@pytest.mark.parametrize("spelling", ["mL", "ml", "cm3", "cm^3", "cm³", "cc", "millilitre", "Milliliters",
                                      "(cm)^3"])
def test_volume_spellings_are_one_millilitre(spelling):
    assert convert(1, spelling, "mL") == 1.0


@pytest.mark.parametrize("spelling", ["dm3", "dm^3", "dm³", "L", "l", "litre", "liters", "Liter"])
def test_litre_spellings(spelling):
    assert convert(1, spelling, "mL") == 1000.0


@pytest.mark.parametrize("spelling", ["µmol/L", "μmol/L", "umol/L", "micromol/L", "µM", "uM",
                                      "µmol·L⁻¹", "umol*L^-1", "micromoles per litre"])
def test_micro_spellings(spelling):
    assert convert(1, spelling, "nmol/mL") == 1.0


def test_micro_mass_spellings():
    for s in ("µg", "μg", "ug", "mcg", "microgram"):
        assert convert(1000, s, "mg") == 1.0


def test_speed_spellings():
    for s in ("km/h", "km/hr", "kph", "kilometres per hour", "km·h⁻¹"):
        assert convert(36, s, "m/s") == 10.0


def test_case_sensitive_symbols():
    assert convert(1, "mm", "m") == 0.001
    for bad in ("Mm", "ML", "Kg", "KM"):       # not defined, never guessed
        with pytest.raises(UnitError):
            convert(1, bad, "m")


def test_every_table_entry_converts_to_itself():
    for key in list(units._SYMBOLS) + list(units._NAMES):
        assert convert(D(1), key, key) == D(1), key


# ----------------------------------------------------------------------------- temperature

@pytest.mark.parametrize("value, src, dst, expected", [
    (D(25), "°C", "K", D("298.15")),
    (D(0), "K", "°C", D("-273.15")),
    (D(212), "°F", "°C", D(100)),
    (D(32), "degF", "K", D("273.15")),
    (D(-40), "°C", "°F", D(-40)),
    (D(100), "celsius", "fahrenheit", D(212)),
    (D("491.67"), "°R", "°F", D(32)),
    (D(37), "℃", "℉", D("98.6")),
    (D(20), "° C", "K", D("293.15")),
    (D(10), "delta_degC", "delta_degF", D(18)),
    (D(9), "Δ°F", "ΔK", D(5)),
    (D(5), "K", "delta_degC", D(5)),            # no offset on either side: plain scaling
])
def test_temperatures(value, src, dst, expected):
    assert convert(value, src, dst) == expected


def test_below_absolute_zero_raises():
    for v, u in ((-300, "°C"), (-1, "K"), (-500, "°F"), (-1, "°R")):
        with pytest.raises(UnitError, match="absolute zero"):
            convert(v, u, "K")


def test_absolute_and_difference_do_not_mix():
    # 10 °C (a temperature) is not a 10-degree difference; refuse rather than guess
    with pytest.raises(UnitError, match="difference"):
        convert(10, "°C", "delta_degF")
    with pytest.raises(UnitError, match="difference"):
        convert(10, "delta_degC", "°F")
    assert not same_quantity("°C", "delta_degC")


# ----------------------------------------------------------------------------- substances

def test_creatinine_mg_dl_to_umol_l():
    # 1 mg/dL = 10 mg/L; / 113.12 g/mol = 10000/113.12 umol/L
    assert convert(D(1), "mg/dL", "umol/L", substance="creatinine") == dec_div(10000, D("113.12"))
    assert convert(1.2, "mg/dL", "µmol/L", substance="creatinine") == float(Fraction(12000) / Fraction("113.12"))


def test_creatinine_umol_l_to_mg_dl():
    assert convert(D("88.4"), "umol/L", "mg/dL", substance="creatinine") == D("0.9999808")
    assert convert(88.4, "umol/L", "mg/dL", substance="Creatinine") == 0.9999808


def test_glucose_and_bun():
    assert convert(D(18016), "mg/dL", "mmol/L", substance="glucose") == D(1000)
    assert convert(D("28.014"), "mg/dL", "mmol/L", substance="BUN") == D(10)
    assert convert(D(1), "mmol/L", "mg/dL", substance="calcium") == D("4.0078")


def test_amount_to_mass_needs_substance():
    with pytest.raises(UnitError, match="substance"):
        convert(1, "mg/dL", "umol/L")
    assert not same_quantity("mg/dL", "umol/L")
    assert same_quantity("mg/dL", "umol/L", substance="creatinine")


def test_unknown_substance_raises():
    with pytest.raises(UnitError, match="molar mass"):
        convert(1, "mg/dL", "umol/L", substance="unobtainium")
    assert not same_quantity("mg/dL", "umol/L", substance="unobtainium")


def test_substance_ignored_when_not_needed():
    assert convert(1, "mg/dL", "g/L", substance="creatinine") == 0.01


def test_substance_cannot_bridge_unrelated_dimensions():
    with pytest.raises(UnitError):
        convert(1, "mg/dL", "m/s", substance="creatinine")


_ATOMIC = {"H": D("1.008"), "C": D("12.011"), "N": D("14.007"), "O": D("15.999"), "Ca": D("40.078")}
_FORMULA = {"creatinine": "C4H7N3O", "glucose": "C6H12O6", "urea": "CH4N2O",
            "urea nitrogen": "N2", "calcium": "Ca"}


@pytest.mark.parametrize("name", sorted(units.SUBSTANCES))
def test_molar_masses_match_their_formula(name):
    mass = sum(_ATOMIC[el] * int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", _FORMULA[name]))
    stated = D(str(units.SUBSTANCES[name].molar_mass.numerator)) / units.SUBSTANCES[name].molar_mass.denominator
    places = -stated.normalize().as_tuple().exponent
    assert mass.quantize(D(1).scaleb(-places)) == stated.quantize(D(1).scaleb(-places))
    assert units.SUBSTANCES[name].source


# ----------------------------------------------------------------------------- round trips

@pytest.mark.parametrize("src, dst, kw", [
    ("L", "cm3", {}), ("km/h", "m/s", {}), ("atm", "mmHg", {}), ("°C", "°F", {}),
    ("K", "°F", {}), ("psi", "kPa", {}), ("%", "ppm", {}), ("US gal", "imp gal", {}),
    ("mg/dL", "umol/L", {"substance": "creatinine"}), ("mmol/L", "mg/dL", {"substance": "glucose"}),
])
@pytest.mark.parametrize("value", [D("0.5"), D("1"), D("37.25"), D("1234.5678")])
def test_decimal_round_trip(src, dst, kw, value):
    there = convert(value, src, dst, **kw)
    back = convert(there, dst, src, **kw)
    # 34 significant digits each way; equal to the input at far beyond any stated precision
    assert back.quantize(D("1e-20")) == value.quantize(D("1e-20"))


# ----------------------------------------------------------------------------- types

def test_decimal_in_decimal_out():
    out = convert(D("1.5"), "L", "mL")
    assert isinstance(out, Decimal) and out == D(1500)


def test_float_and_int_in_float_out():
    assert isinstance(convert(1.5, "L", "mL"), float)
    assert isinstance(convert(2, "L", "mL"), float)
    assert isinstance(convert(Fraction(1, 2), "L", "mL"), float)


@pytest.mark.parametrize("value", [True, "5", None, float("nan"), float("inf"), D("NaN"), D("Infinity"), [1]])
def test_bad_values_raise(value):
    with pytest.raises(UnitError):
        convert(value, "L", "mL")


def test_unit_error_is_value_error():
    assert issubclass(UnitError, ValueError)


# ----------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("src, dst", [
    ("kg", "m"), ("km/h", "km"), ("°C", "kg"), ("%", "m"), ("L", "kg"), ("Pa", "J"), ("1", "s"),
])
def test_incompatible_units_raise(src, dst):
    with pytest.raises(UnitError, match="incompatible"):
        convert(1, src, dst)
    assert not same_quantity(src, dst)


@pytest.mark.parametrize("unit", sorted(units._AMBIGUOUS))
def test_ambiguous_units_refused(unit):
    with pytest.raises(UnitError, match="ambiguous"):
        convert(1, unit, "m")
    assert not same_quantity(unit, unit)


@pytest.mark.parametrize("unit", ["furlong", "", "   ", "m/", "(m", "m)", "m^", "m^0", "kg m", "**m", "1000*uL", None, 3, ["m"]])
def test_unknown_or_malformed_units_raise(unit):
    with pytest.raises(UnitError):
        convert(1, unit, "m")
    assert not same_quantity(unit, "m")


@pytest.mark.parametrize("a, b", [
    ("L", "cm3"), ("km/h", "m/s"), ("kPa", "mmHg"), ("°C", "K"), ("%", "1"), ("mg/dL", "g/L"),
    ("J/(g*°C)", "J/(kg*K)"), ("N", "lbf"),
])
def test_same_quantity_true(a, b):
    assert same_quantity(a, b)


# ----------------------------------------------------------------------------- runtime primitive

def test_paper_primitives_unchanged():
    # paper mode: the original eight primitives are still there, under the same names
    assert set(runtime.PAPER_PRIMITIVES) == {"load", "do", "narrate", "lm", "extract", "convert",
                                             "smart_convert", "invoke_tool"}
    for name, fn in runtime.PAPER_PRIMITIVES.items():
        assert runtime.PRIMITIVES[name] is fn
    assert runtime.PRIMITIVES["convert"] is runtime.convert       # type coercion, not units
    assert runtime.PRIMITIVES["convert"](None, "3", int) == 3


def test_unit_primitives_exposed():
    assert runtime.PRIMITIVES["convert_unit"] is units.convert
    assert runtime.PRIMITIVES["same_quantity"] is units.same_quantity
    assert "UnitError" not in runtime.PRIMITIVES     # cached code can't swallow a unit failure


SOURCE = '''
def semantic_function(ctx):
    scr = load(ctx, "serum creatinine in umol/L", float)
    if not same_quantity("umol/L", "mg/dL", substance="creatinine"):
        raise ValueError("units do not fit")
    mg = convert_unit(scr, "umol/L", "mg/dL", substance="creatinine")
    narrate(ctx, f"creatinine {mg} mg/dL")
    return {"creatinine in mg/dL": mg}
'''


@pytest.fixture
def one_key_embed(monkeypatch):
    monkeypatch.setattr(embed, "best_match", lambda q, keys: (0, 1.0) if keys else (-1, -1.0))


def test_sandbox_accepts_convert_unit():
    sandbox.check_source(SOURCE)


def test_cached_code_calls_convert_unit(one_key_embed):
    c = Cache("t")
    rec = c.add_function("convert creatinine", SOURCE)
    out, ctx = serve.dispatch_root(c, rec.id, {"serum creatinine": 88.4})
    assert out == {"creatinine in mg/dL": 0.9999808}
    assert "0.9999808 mg/dL" in ctx.narration()


def test_cached_code_survives_save_and_load(tmp_path, one_key_embed):
    c = Cache("t")
    rec = c.add_function("convert creatinine", SOURCE)
    c.save(tmp_path)
    loaded = Cache.load(tmp_path)
    out, _ = serve.dispatch_root(loaded, rec.id, {"serum creatinine": 88.4})
    assert out == {"creatinine in mg/dL": 0.9999808}


def test_unit_error_in_cached_code_propagates(one_key_embed):
    c = Cache("t")
    bad = "def semantic_function(ctx):\n    return {'v': convert_unit(1, 'gal', 'L')}\n"
    rec = c.add_function("bad gallons", bad)
    with pytest.raises(UnitError, match="ambiguous"):
        serve.dispatch_root(c, rec.id, {})


def test_sandbox_still_blocks_importing_arc():
    with pytest.raises(sandbox.UnsafeCodeError):
        sandbox.check_source("from arc.units import convert\ndef semantic_function(ctx):\n    return {}\n")
