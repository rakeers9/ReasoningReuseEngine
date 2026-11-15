"""eval/labelfree/families.py: the hand-written reference solvers must reproduce every
seed's known answer, and the generators must be deterministic and vary input forms."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("lf_families", ROOT / "eval" / "labelfree" / "families.py")
fam = importlib.util.module_from_spec(_s)
sys.modules["lf_families"] = fam   # dataclasses need the module registered
_s.loader.exec_module(fam)

from arc.checks import numbers  # noqa: E402


def close(a, b, rel=0.0005):
    return abs(a - b) <= rel * abs(b)


def test_seed_answers_are_reproduced():
    assert close(fam.gas_pressure_pa(10.0, 4.860e-3, 300.15) / 101325, 50.68)
    assert close(fam.projectile_range(20.0, 25.0, 35), 80.98)
    assert close(5 * 12000 * 6.0 ** 4 / (384 * 200e9 * 8.0e-5) * 1000, 12.66)
    assert fam.count_div(1000, 3, 5, 7) == 400
    assert fam.math.factorial(7) - 2 * fam.math.factorial(6) == 3600
    se, sp, p = 0.92, 0.96, 0.015
    assert close(se * p / (se * p + (1 - sp) * (1 - p)), 0.2594, rel=0.001)
    m, i = fam.loan_payment(350_000, 6.25, 30)
    assert close(m, 2155.01, 1e-6) and close(i, 425_803.67, 1e-6)
    assert close(fam.npv_value(120_000, 35_000, 5, 10_000, 0.08), 26_550.68, 1e-6)
    assert close(fam.bond_price(1000, 5, 10, 6.2), 911.55, 1e-5)
    assert close(fam.ddb_book(80_000, 8_000, 6, 3), 23_703.70, 1e-6)
    q = (196 / 2500) ** 0.5
    assert close(2 * (1 - q) * q * 2500, 1008)
    assert close((140 - 68) * 62 / (72 * 1.4) * 0.85, 37.64)
    assert close(-fam.math.log10(1.8e-5) + fam.math.log10(0.08 / 0.17), 4.417)


def test_ddb_never_goes_below_salvage():
    assert fam.ddb_book(10_000, 4_000, 4, 3) == 4_000


@pytest.mark.parametrize("family", sorted(fam.GENERATORS))
def test_generators_are_deterministic_and_well_formed(family):
    gen = fam.GENERATORS[family]
    a = [gen(random.Random(7), k) for k in ("positive", "near_miss")]
    b = [gen(random.Random(7), k) for k in ("positive", "near_miss")]
    assert [x.question for x in a] == [x.question for x in b]
    for item in a:
        assert item.family == family and item.question.strip()
        assert item.truth and all(numbers(t) for t in item.truth)
    assert a[0].kind == "positive" and a[1].kind == "near_miss"


@pytest.mark.parametrize("family", sorted(set(fam.GENERATORS) - {"bond_price_semiannual",
                                                                 "circular_seating_not_adjacent"}))
def test_positive_input_forms_vary(family):
    rng = random.Random(1)
    forms = {fam.GENERATORS[family](rng, "positive").form for _ in range(40)}
    assert len(forms) >= 2, forms


def test_unit_forms_carry_the_same_quantity():
    # a km/h question must have the same truth as the m/s question for the same speed
    assert close(fam.projectile_range(20.0, 90 / 3.6, 35), fam.projectile_range(20.0, 25.0, 35), 1e-9)
