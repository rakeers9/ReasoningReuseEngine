"""eval/chat_loop.py scoring: a right answer counts in any equivalent unit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import chat_loop as cl  # noqa: E402

GAS = {"kind": "variant", "answer_value": 5.191, "answer_unit": "atm", "answer": "5.191 atm", "should_hit": True}


def test_pressure_in_kpa_pa_or_bar_counts_as_right():
    for text in ["P ≈ 5.19 atm", "P ≈ 5.26 × 10⁵ Pa (about 526 kPa)", "about 5.26 bar", "526 kPa"]:
        assert cl.score_message(GAS, text)[0] is True, text


def test_a_wrong_pressure_is_still_wrong_in_every_unit():
    assert cl.score_message(GAS, "P ≈ 7.9 atm (800 kPa)")[0] is False


def test_chitchat_is_not_scored():
    assert cl.score_message({"kind": "chitchat"}, "hi") == (None, "chitchat")


def test_money_answers_are_unchanged():
    m = {"kind": "variant", "answer_value": 1199.10, "answer_unit": "$", "answer": "$1,199.10", "should_hit": True}
    assert cl.score_message(m, "Payment $1,199.10/month")[0] is True
    assert cl.score_message(m, "Payment $1,250.00/month")[0] is False


# ---- exact to the stated precision (CLAUDE.md production rule) ----

LOAN = {"kind": "variant", "answer_value": 5053.067800, "answer_unit": "$", "answer": "$5,053.07", "should_hit": True}


def test_money_must_be_right_to_the_cent():
    assert cl.score_message(LOAN, "Monthly payment ≈ $5,053.07")[0] is True
    assert cl.score_message(LOAN, "Monthly payment ≈ $5,053.27")[0] is False      # 20 cents off: wrong
    assert cl.score_message(LOAN, "about $5,053 per month")[0] is False           # not given to the cent


def test_counts_must_be_exact():
    m = {"kind": "variant", "answer_value": 1008.0, "answer_unit": "people", "answer": "1,008 people", "should_hit": True}
    assert cl.score_message(m, "1,008 people")[0] is True
    assert cl.score_message(m, "1,009 people")[0] is False


def test_measurements_match_to_their_own_last_digit():
    gas = {"kind": "variant", "answer_value": 18.571302, "answer_unit": "atm", "answer": "18.57 atm", "should_hit": True}
    assert cl.score_message(gas, "P ≈ 18.6 atm")[0] is True          # 18.571 rounds to 18.6
    assert cl.score_message(gas, "P ≈ 18.57 atm")[0] is True
    assert cl.score_message(gas, "P ≈ 18.8 atm")[0] is False         # was "right" under the old 1%... 1.2% off
    assert cl.score_message(gas, "P ≈ 18.7 atm")[0] is False         # 0.7% off: the old 1% rule accepted this


def test_a_value_1000x_off_is_wrong_even_if_the_digits_match():
    m = {"kind": "variant", "answer_value": 921.13, "answer_unit": "$", "answer": "$921.13", "should_hit": True}
    assert cl.score_message(m, "the fixed monthly payment in dollars: 0.92") == (False, "scaled_without_unit")


def test_a_unit_label_before_the_number_counts():
    # regression (chat run 2026-11-15): "pressure of the gas in bar: 7.7589..." is 7.6575 atm, scored wrong
    m = {"kind": "variant", "answer_value": 7.657472305757299, "answer_unit": "atm", "answer": "7.657 atm",
         "should_hit": True}
    assert cl.score_message(m, "pressure of the gas in bar: 7.758933821320725274725274725")[0] is True
    assert cl.score_message(m, "pressure of the gas in atm: 7.758933821320725")[0] is False   # labelled atm: wrong
