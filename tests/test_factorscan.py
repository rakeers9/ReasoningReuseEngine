"""arc/factorscan.py: hand-typed unit factors in generated code are flagged; ordinary
numbers and factors inside the unit primitive's call are not."""

import pytest

from arc.factorscan import UNIT_FN, typed_factors


def fn(body: str) -> str:
    return "def semantic_function(ctx):\n" + "".join(f"    {line}\n" for line in body.splitlines())


@pytest.mark.parametrize("body", [
    "t_k = t_c + 273.15",
    "p_kpa = p_atm * 101.325",
    "p_pa = p_atm * 101325",
    "v_ms = v_kmh / 3.6",
    "x_m = x_in * 0.0254",
    "scr_mgdl = scr_umol / 88.4",
    "m_kg = m_lb * 0.45359237",
    "volume_l = volume_ml / 1000",
    "volume_l = volume_cm3 * 1e-3",
    "p = load(ctx, 'pressure in kPa') * 1000",
    "t_s = t_min * 60",
    "p_mmhg = p_atm * 760",
    "t_f = t_c * 1.8 + 32",
    "t_r = t_f + 459.67",
    "scr_mgdl = scr_umol * 0.0113",
])
def test_typed_factors_are_flagged(body):
    found = typed_factors(fn(body))
    assert found and "line 2" in found[0] and UNIT_FN in found[0]


@pytest.mark.parametrize("body", [
    f"t_k = {UNIT_FN}(t_c, 'degC', 'K')",
    f"t_k = {UNIT_FN}(t_c + 0, 'degC', 'K') + 0",
    "total = count * 1000",                   # no unit named: an ordinary number
    "for i in range(60):\n    pass",
    "months = years * 12",
    "rate = annual / 100",
    "R = 0.082057  # L atm / (mol K), a physical constant, not a unit factor",
    "payment = principal * r / (1 - (1 + r) ** -n)",
    "narrate(ctx, 'converted 273.15 K')",     # inside a string, not arithmetic
    "price = Decimal('1000.00')",
    "flag = True",
    "scales = {'k': 1000, 'thousand': 1000, 'million': 1e6, 'bn': 1e9}",    # scale words, not units
    "amount_usd = number * 1000 if suffix == 'k' else number",
    "principal = number * 10**3",
])
def test_ordinary_code_is_clean(body):
    assert typed_factors(fn(body)) == []


def test_factor_inside_the_unit_call_is_fine_but_outside_is_not():
    code = fn(f"a = {UNIT_FN}(x * 273.15, 'K', 'degC')\nb = y + 273.15")
    found = typed_factors(code)
    assert len(found) == 1 and found[0].startswith("line 3")


def test_unparsable_code_is_left_to_the_syntax_check():
    assert typed_factors("def semantic_function(ctx):\n    return {") == []
