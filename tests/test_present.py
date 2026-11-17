"""The final formatter (arc/present.py): exact rounding once, at the end; output units."""

import sys
import types
from decimal import Decimal

import pytest

from arc import present
from arc.runtime import CacheInvocationError

MORTGAGE = {"monthly payment": 5053.067812, "total interest": 95368.1374, "note": "fixed rate"}
SPECS = {"monthly payment": {"kind": "money", "unit": "$"}, "total interest": {"kind": "money", "unit": "$"}}


def test_money_is_rounded_to_the_cent_once_at_the_end():
    out = present.apply(MORTGAGE, None, SPECS)
    assert out == {"monthly payment": "5053.07", "total interest": "95368.14", "note": "fixed rate"}


def test_half_up_not_bankers_rounding():
    assert present.apply({"x": 2.675}, None, {"x": {"kind": "money"}}) == {"x": "2.68"}   # float 2.675 is 2.67499..
    assert present.round_to(Decimal("0.125"), "dp:2") == Decimal("0.13")


def test_without_a_spec_full_precision_is_kept():
    assert present.apply({"pressure": 18.571302}) == {"pressure": "18.571302"}


def test_counts_stay_exact():
    assert present.apply({"ways": 3600.0}, None, {"ways": {"kind": "count"}}) == {"ways": "3600"}


@pytest.mark.parametrize("rule, want", [("dp:0", "5053"), ("dp:1", "5053.1"), ("sf:3", "5.05E+3"),
                                        ("nearest:100", "5100"), ("nearest:0.05", "5053.05")])
def test_the_questions_rounding_request_wins(rule, want):
    got = present.apply({"monthly payment": 5053.067812}, {"rounding": rule}, SPECS)["monthly payment"]
    assert Decimal(got) == Decimal(want)


def test_unknown_rounding_declines():
    with pytest.raises(present.PresentError):
        present.round_to(Decimal("1.5"), "banker:2")


def test_a_present_error_falls_back_to_the_model():
    assert issubclass(present.PresentError, CacheInvocationError)


def test_a_different_requested_unit_declines_without_a_unit_library():
    # user decision 2026-11-15: no unit table; the model's code converts, the formatter never does
    with pytest.raises(present.PresentError, match="entry gives atm"):
        present.apply({"pressure": 50.678}, {"unit": "kPa", "rounding": "dp:1"},
                      {"pressure": {"kind": "measure", "unit": "atm"}})


def test_requested_unit_that_cannot_be_confirmed_declines():
    with pytest.raises(present.PresentError, match="no output is known"):
        present.apply({"pressure": 50.678}, {"unit": "kPa"}, None)            # unit of the output unknown


def test_requested_unit_equal_to_the_outputs_unit_needs_no_conversion():
    out = present.apply({"payment": 921.1349}, {"unit": "dollars"}, {"payment": {"kind": "money", "unit": "$"}})
    assert out == {"payment": "921.13"}


def test_text_and_booleans_pass_through():
    assert present.apply({"ok": True, "label": "x"}) == {"ok": True, "label": "x"}



def test_the_formatter_never_imports_a_unit_library():
    src = (present.__file__ and open(present.__file__).read())
    assert "import units" not in src and "units." not in src.split('"""', 2)[2]


# ---- the one rounding policy ----

@pytest.mark.parametrize("value, want", [(7.758933821320725274725274725, "7.759"), (0.000123456, "0.0001235"),
                                        (1234567.0, "1235000"), (51.5249, "51.52"), (2.0, "2.000"),
                                        (-154974.8412973295, "-155000")])
def test_measurements_get_four_significant_figures(value, want):
    assert present.apply({"x": value}, None, {"x": {"kind": "measure", "unit": "bar"}}) == {"x": want}


def test_the_questions_rounding_beats_the_policy():
    out = present.apply({"x": 7.758933}, {"rounding": "dp:1"}, {"x": {"kind": "measure", "unit": "bar"}})
    assert out == {"x": "7.8"}


def test_the_policy_text_states_every_rule():
    p = present.ROUNDING_POLICY
    for must in ("never round any intermediate value", "If the question says how to round", "money to the cent",
                 "counts as exact whole numbers", "4 significant figures", "halves rounded up"):
        assert must in p, must


def test_a_number_returned_as_text_is_rounded_too():
    # regression (hidden run 2026-11-16): the code returned str(Decimal) and 50 digits were served
    spec = {"equilibrium temperature in K": {"kind": "measure", "unit": "K", "decimals": None}}
    out = present.apply({"equilibrium temperature in K": "301.01079281511303809228863425209043047383090740167"},
                        None, spec)
    assert out == {"equilibrium temperature in K": "301.0"}


@pytest.mark.parametrize("text", ["301.01 K", "about 5", "n/a", "1,234.5", ""])
def test_text_that_is_not_exactly_a_number_is_left_alone(text):
    assert present.apply({"x": text}, None, {"x": {"kind": "measure"}}) == {"x": text}


def test_money_returned_as_text_goes_to_the_cent():
    assert present.apply({"m": "2176.8612"}, None, {"m": {"kind": "money", "unit": "USD"}}) == {"m": "2176.86"}
