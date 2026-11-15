"""arc/metamorphic.py: unit and scale rewrites, and the no-oracle relation check."""

import math

import pytest

from arc import metamorphic as mm


@pytest.mark.parametrize("value, expected", [
    ("25 m/s", "90 km/h"),
    ("90 km/h", "25 m/s"),
    ("4.860 dm3", "4860 mL"),   # same-factor spellings (dm3 = L) are skipped: magnitude must change
    ("500 mL", "0.5 L"),
    ("27°C", "300.15 K"),
    ("300 K", "26.85 °C"),
    ("10.0 mol", "10000 mmol"),
    ("20.0 m", "2000 cm"),
    ("101.325 kPa", "1 atm"),
    ("€2.0 million", "€2,000,000"),
    ("€2,000,000", "€2 million"),
])
def test_rewrites_are_equivalent(value, expected):
    assert mm.rewrites(value) == [expected]


@pytest.mark.parametrize("value", ["7.84%", "0.25", "C2H6", "", "35°", "Rule 12(a)", "$450", None])
def test_no_rewrite_when_not_safe(value):
    assert mm.rewrites(value) == []


def test_same_numbers():
    assert mm.same_numbers("pressure: 5130 kPa", "pressure: 5.13e6 Pa")      # prefix rescale
    # changed 2026-11-15: the tolerance is 1e-4, not 1% (owner's exact-amount rule)
    assert not mm.same_numbers("x: 42.0", "x: 42.2")
    assert mm.same_numbers("x: 42.0", "x: 42.000001")
    assert not mm.same_numbers("x: 42.0", "x: 11.7")
    assert not mm.same_numbers("x: 42.0", "x: none")
    assert mm.same_numbers("decision: reject", "decision: reject")            # no numbers


def _range_unit_aware(v):
    speed = float(v["speed"].split()[0]) * (1 / 3.6 if "km/h" in v["speed"] else 1.0)
    return {"range": speed ** 2 / 9.81}


def _range_drops_unit(v):  # the projectile bug: keeps the number, ignores the unit
    return {"range": float(v["speed"].split()[0]) ** 2 / 9.81}


def test_unit_aware_code_holds():
    values = {"speed": "25 m/s"}
    [r] = mm.check(_range_unit_aware, values, _range_unit_aware(values))
    assert r.status == "held" and r.rewritten == "90 km/h"


def test_code_that_drops_units_is_caught():
    values = {"speed": "25 m/s"}
    [r] = mm.check(_range_drops_unit, values, _range_drops_unit(values))
    assert r.status == "violated" and "range" in r.detail


def test_crash_on_rewrite_is_not_a_violation():
    def strict(v):
        if "km/h" in v["speed"]:
            raise ValueError("unit not supported")
        return {"range": 1.0}
    [r] = mm.check(strict, {"speed": "25 m/s"}, {"range": 1.0})
    assert r.status == "crashed"


def test_slots_without_rewrites_are_skipped_and_capped():
    values = {"name": "ethane", "a": "1 m", "b": "2 m", "c": "3 m"}
    rels = mm.check(lambda v: {"y": 1}, values, {"y": 1}, max_relations=2)
    assert [r.slot for r in rels] == ["a", "b"]
    assert math.isclose(float(rels[0].rewritten.split()[0]), 100)


def test_same_numbers_tolerance_is_tight():
    from arc.metamorphic import same_numbers
    assert same_numbers("p: 101325.0", "p: 101325.000001")
    assert same_numbers("p: 760.0 mmHg", "p: 760.02 mmHg")       # constant written to fewer digits
    assert not same_numbers("p: 760.0", "p: 761.0")               # 0.13% used to pass under 1%
