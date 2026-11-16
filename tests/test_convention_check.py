"""arc/labelfree._convention_check: a program written from the build question plus the
entry's stated conventions must give the code's answer. Two false failures from the chat-loop
run (2026-11-15), each of which cost an unneeded Sonnet rebuild (owner approved the fix):

  gas (fam_f0004)       the program printed "692677.39 pascals"; the code returns bar and atm.
                        Fix: the program is told the code's output quantities and units.
  mortgage (fam_f0003)  both printed the same payment to 60-80 digits and differed only in the
                        tail digits. Fix: in this check only (code vs program, both computed at
                        full precision by machine), long numbers are compared at 12 significant
                        digits (never fewer than 6 decimals). Customer-facing answer checks are
                        unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc import checks, judges, labelfree
from arc.cache import Cache, EntryPoint
from arc import config
from test_labelfree_simple import FAM, family, pre, sim  # noqa: F401

# ----------------------------------------------------------------------------- the two real cases

GAS_Q = ("A 8.4 L flask holds 2.69 mol of nitrogen at 260.15 K. Assuming perfect-gas behaviour, "
         "what pressure does the gas exert?")
GAS_BAR = "6.92677385797090833333333333333333333333333333333333333333333"
GAS_ATM = "6.83619428371172793815280861912986265317871535488115798996628"
GAS_RULES = ["Pressure is computed as p = n x R x T / V with no intermediate rounding.",
             "Pressure in bar is the pressure in pascals divided by exactly 100000.",
             "Pressure in atm is the pressure in pascals divided by exactly 101325."]
GAS_OUTPUTS = {"pressure in bar": {"kind": "measure", "unit": "bar", "decimals": None},
               "pressure in atm": {"kind": "measure", "unit": "atm", "decimals": None}}

MORT_Q = ("A $366,000 mortgage has a 5.0% annual interest rate, compounded monthly, over 15 years. "
          "What is the monthly payment, and how much total interest is paid over the life of the loan?")
MORT_CODE_PAY = "2894.3046738740527894610937400956764335798156029526415526996822626632405821366840"
MORT_CODE_INT = "154974.841297329502102996873217221758044366808531475479485943"
MORT_PROG = ("monthly payment: 2894.3046738740527894610937400956764335798156029526415526996815000353048360838698 dollars\n"
             "total interest paid over the life of the loan: 154974.841297329502102996873217221758044366808531475479485943 dollars")
MORT_OUTPUTS = {"monthly payment in dollars": {"kind": "money", "unit": "USD", "decimals": 2},
                "total interest paid over the life of the loan in dollars": {"kind": "money", "unit": "USD",
                                                                             "decimals": 2}}


def _cache(entry_id, question, outputs: dict[str, str], specs, rules):
    c = Cache(entry_id)
    body = ", ".join(f"{k!r}: Decimal({v!r})" for k, v in outputs.items())
    c.add_function("root", "def semantic_function(ctx):\n    from decimal import Decimal\n"
                           f"    return {{{body}}}\n")
    c.entry_points = [EntryPoint(question, "original", list(c.functions)[0], [], entry_id)]
    c.outputs[entry_id] = specs
    c.conventions[entry_id] = rules
    return c


@pytest.fixture
def judge_log(monkeypatch):
    seen = []

    def judge(prompt, **k):
        seen.append(prompt)
        return SimpleNamespace(passed=True, critique="")
    monkeypatch.setattr(judges, "judge", judge)
    return seen


def test_gas_program_is_told_the_codes_output_units_and_then_agrees(fake_llm, judge_log):
    c = _cache("fam_f0004", GAS_Q, {"pressure in bar": GAS_BAR, "pressure in atm": GAS_ATM}, GAS_OUTPUTS, GAS_RULES)
    program = ('from decimal import Decimal, getcontext\ngetcontext().prec = 60\n'
               'p = Decimal("2.69") * Decimal("8.314462618") * Decimal("260.15") / (Decimal("8.4") / 1000)\n'
               'print(f"pressure in bar: {p / 100000} bar")\nprint(f"pressure in atm: {p / 101325} atm")\n')
    fake_llm.queue(labelfree.Program(program=program))
    out = labelfree._convention_check(c, "fam_f0004", {"question": GAS_Q})
    prompt = fake_llm.calls[0]["prompt"]
    assert "pressure in bar (bar)" in prompt and "pressure in atm (atm)" in prompt     # the code's units
    assert "6.92" not in prompt and "6.83" not in prompt                             # never the values
    assert out["passed"], out["detail"]


def test_gas_program_in_pascals_still_fails(fake_llm, judge_log):
    # a program that ignores the instruction can't pass by accident: it says nothing in bar or atm
    c = _cache("fam_f0004", GAS_Q, {"pressure in bar": GAS_BAR, "pressure in atm": GAS_ATM}, GAS_OUTPUTS, GAS_RULES)
    fake_llm.queue(labelfree.Program(program='print("pressure of the gas: 692677.39 pascals")'))
    assert not labelfree._convention_check(c, "fam_f0004", {"question": GAS_Q})["passed"]


def test_mortgage_tails_beyond_12_significant_digits_do_not_fail_the_check(fake_llm, judge_log):
    c = _cache("fam_f0003", MORT_Q, {"monthly payment in dollars": MORT_CODE_PAY,
                                     "total interest paid over the life of the loan in dollars": MORT_CODE_INT},
               MORT_OUTPUTS, ["Total interest = unrounded monthly payment x number of payments minus principal."])
    fake_llm.queue(labelfree.Program(program="print('''" + MORT_PROG + "''')"))
    out = labelfree._convention_check(c, "fam_f0003", {"question": MORT_Q})
    assert out["passed"], out["detail"]
    [prompt] = judge_log
    assert "2894.30467387" in prompt and MORT_CODE_PAY not in prompt and "29526415526996" not in prompt
    assert out["cache_answer"].startswith(f"monthly payment in dollars: {MORT_CODE_PAY}")   # raw kept for the record


def test_a_real_convention_difference_still_fails(fake_llm, judge_log):
    # the old rounded-payment convention: interest off by 4 cents, far above 12 significant digits
    c = _cache("fam_f0003", MORT_Q, {"monthly payment in dollars": MORT_CODE_PAY,
                                     "total interest paid over the life of the loan in dollars": MORT_CODE_INT},
               MORT_OUTPUTS, ["Total interest = monthly payment rounded to the cent x payments - principal."])
    fake_llm.queue(labelfree.Program(program="print('monthly payment: 2894.30 dollars')\n"
                                             "print('total interest paid over the life of the loan: 154974.00 dollars')"))
    out = labelfree._convention_check(c, "fam_f0003", {"question": MORT_Q})
    assert not out["passed"]


# ----------------------------------------------------------------------------- the rounding helper

@pytest.mark.parametrize("text,want", [
    (f"monthly payment in dollars: {MORT_CODE_PAY}", "monthly payment in dollars: 2894.30467387"),
    (f"pressure in bar: {GAS_BAR}", "pressure in bar: 6.92677385797"),
    ("total: 12345678901.23456789 dollars", "total: 12345678901.234568 dollars"),   # never < 6 decimals
    ("x: 0.000000123456789012345678", "x: 0.000000123456789012"),
    ("payment 2894.30 dollars, 360 payments, 6.83 atm", "payment 2894.30 dollars, 360 payments, 6.83 atm"),
    ("1.23456789012345678e-05 m3", "1.23456789012e-05 m3"),
    ("-154974.841297329502102996873217", "-154974.841297"),
])
def test_significant_digits_helper(text, want):
    assert checks.round_long_numbers(text) == want


def test_helper_leaves_short_text_and_words_alone():
    for s in ("pressure in m3: 4.86", "no numbers here", "v2 of 3.14"):
        assert checks.round_long_numbers(s) == s


def test_customer_facing_compare_is_unchanged(judge_log):
    # only the convention check shortens numbers; production-vs-cache comparisons see the raw text
    from arc import certify
    certify._compare("production vs cache", MORT_Q, "$2,894.30", f"monthly payment: {MORT_CODE_PAY}", "t")
    assert MORT_CODE_PAY in judge_log[-1]
    assert checks.number_check("2894.31 dollars", "2894.30 dollars")["status"] == "mismatch"


def test_output_specs_are_written_before_the_convention_check(pre, monkeypatch):
    order = []
    monkeypatch.setattr(config, "LABELFREE_CONVENTION_CHECK", True)
    monkeypatch.setattr(labelfree, "write_outputs", lambda cache, eid, t, model=None: order.append("outputs") or
                        cache.outputs.setdefault(eid, {"result": {"kind": "measure", "unit": "m", "decimals": None}}))
    monkeypatch.setattr(labelfree, "_convention_check", lambda cache, eid, rec: order.append(
        ("check", dict(cache.outputs.get(eid, {})))) or {"passed": True, "detail": ""})
    pre.answers = ["16"]
    assert labelfree.build_family(FAM, family(8)).status == "accepted"
    assert order == ["outputs", ("check", {"result": {"kind": "measure", "unit": "m", "decimals": None}})]


def test_a_failed_conventions_write_still_skips_the_check(pre, monkeypatch):
    called = []
    monkeypatch.setattr(config, "LABELFREE_CONVENTION_CHECK", True)

    def boom(cache, eid, t, model=None):
        raise ValueError("truncated JSON")
    monkeypatch.setattr(labelfree, "write_conventions", boom)
    monkeypatch.setattr(labelfree, "_convention_check", lambda *a: called.append(1) or {"passed": True})
    pre.answers = ["16"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.CONVENTION_REASON) and called == []


def test_text_outputs_are_not_asked_of_the_program():
    c = Cache("e")
    c.outputs["e"] = {"pressure in bar": {"kind": "measure", "unit": "bar", "decimals": 3},
                      "formatted pressure report": {"kind": "text", "unit": "", "decimals": None}}
    out = {"pressure in bar": 1, "formatted pressure report": "x", "pressure in atm": 2}
    assert labelfree._output_quantities(c, "e", out) == ["pressure in bar (bar)", "pressure in atm"]
