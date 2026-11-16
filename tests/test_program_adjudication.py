"""Tie-breaks run a program (owner decision 2026-11-15): when production and the cache
disagree, each blind solve is a short Python program Haiku writes from the QUESTION ONLY,
run in a separate process, so the arithmetic is exact. Regression: the stopped chat run's
mortgage family was right to the cent ($5,053.07 / $95,368.14) but every check came back
"unresolved", because hand-arithmetic blind solves couldn't agree to the cent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc import config, judges, labelfree, sandbox

MORTGAGE_PROGRAM = '''
from decimal import Decimal, getcontext
getcontext().prec = 50
principal = Decimal({P})          # dollars
annual_rate = Decimal("{rate}") / 100
months = {years} * 12
r = annual_rate / 12
payment = principal * r / (1 - (1 + r) ** -months)
interest = payment * months - principal
print(f"monthly payment: {{payment.quantize(Decimal('0.01'))}} dollars")
print(f"total interest: {{interest.quantize(Decimal('0.01'))}} dollars")
'''

CASES = [  # (principal, rate %, years, cache answer, a wrong production answer)
    (892000, "5.875", 20, "monthly_payment: 6326.41\ntotal_interest: 626337.72",
     "Monthly payment ≈ $6,326.89; total interest ≈ $626,450"),
    (511000, "3.5", 10, "monthly_payment: 5053.07\ntotal_interest: 95368.14",
     "Final answers: monthly payment ≈ **$5,053.27**; total interest ≈ **$95,392**"),
]


def q(P, rate, years):
    return f"A ${P:,} mortgage at {rate}% annual interest, compounded monthly, over {years} years. " \
           f"What is the monthly payment and the total interest paid?"


@pytest.fixture
def judge_passes(monkeypatch):
    """The judge always says yes, so the strict number check alone decides."""
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    monkeypatch.setattr(config, "LABELFREE_ADJUDICATOR", "program")


# ----------------------------------------------------------------------------- sandbox.run_program

def test_run_program_captures_printed_output():
    out = sandbox.run_program("from fractions import Fraction\nprint(Fraction(1, 3) * 3)\nprint('x: 2.50 m')")
    assert out.ok and out.stdout.strip().splitlines() == ["1", "x: 2.50 m"]


def test_run_program_rejects_unsafe_code_before_running():
    for src in ["import os\nprint(os.getcwd())", "print(open('/etc/passwd').read())",
                "print(().__class__)", "print(getattr(1, 'real'))", "exec('print(1)')"]:
        out = sandbox.run_program(src)
        assert not out.ok and "not allowed" in out.error and out.stdout == ""


def test_run_program_times_out():
    out = sandbox.run_program("while True:\n    pass", seconds=1)
    assert not out.ok and "timed out" in out.error


def test_run_program_reports_a_runtime_error():
    out = sandbox.run_program("print(1 / 0)")
    assert not out.ok and "ZeroDivisionError" in out.error


def test_run_program_blocks_imports_at_run_time_too():
    out = sandbox.run_program("m = 'o' + 's'\nprint(__import__(m))")   # __import__ is caught statically
    assert not out.ok
    out = sandbox.run_program("import importlib")
    assert not out.ok and "not allowed" in out.error


# ----------------------------------------------------------------------------- the real cases

@pytest.mark.parametrize("P,rate,years,cache,production", CASES)
def test_a_correct_program_clears_the_right_cache(fake_llm, judge_passes, P, rate, years, cache, production):
    prog = labelfree.Program(program=MORTGAGE_PROGRAM.format(P=P, rate=rate, years=years))
    fake_llm.queue(prog)
    fake_llm.queue(prog)
    outcome, answers, detail = labelfree.adjudicate(q(P, rate, years), production, cache, "t")
    assert outcome == "cache_right", (answers, detail)
    assert len(answers) == 2 and all("monthly payment" in a for a in answers)


def test_programs_see_only_the_question(fake_llm, judge_passes):
    P, rate, years, cache, production = CASES[1]
    prog = labelfree.Program(program=MORTGAGE_PROGRAM.format(P=P, rate=rate, years=years))
    fake_llm.queue(prog)
    fake_llm.queue(prog)
    labelfree.adjudicate(q(P, rate, years), production, cache, "t")
    prompts = [c["prompt"] for c in fake_llm.calls]
    assert len(prompts) == 2 and all(c["role"] == "adjudicator" for c in fake_llm.calls)
    for p in prompts:
        assert q(P, rate, years) in p
        assert "5053.07" not in p and "5,053.27" not in p and "95368.14" not in p and "semantic_function" not in p


def test_a_program_that_agrees_with_production_blames_the_cache(fake_llm, judge_passes):
    P, rate, years, _, _ = CASES[1]
    prog = labelfree.Program(program=MORTGAGE_PROGRAM.format(P=P, rate=rate, years=years))
    fake_llm.queue(prog)
    outcome, _, _ = labelfree.adjudicate(q(P, rate, years), "payment $5,053.07; interest $95,368.14",
                                         "monthly_payment: 5153.07\ntotal_interest: 107368.14", "t")
    assert outcome == "cache_wrong"


@pytest.mark.parametrize("bad_program", ["print(1 / 0)", "x = 1", "import os\nprint(1)", "while True:\n    pass"])
def test_a_failed_program_never_clears_the_cache(fake_llm, judge_passes, monkeypatch, bad_program):
    monkeypatch.setattr(config, "LABELFREE_PROGRAM_SECONDS", 1)
    P, rate, years, cache, production = CASES[1]
    fake_llm.queue(labelfree.Program(program=bad_program))
    fake_llm.queue(labelfree.Program(program=MORTGAGE_PROGRAM.format(P=P, rate=rate, years=years)))
    outcome, answers, _ = labelfree.adjudicate(q(P, rate, years), production, cache, "t")
    assert outcome == "unresolved" and "no answer" in answers[0]


def test_both_programs_must_side_with_the_cache(fake_llm, judge_passes):
    P, rate, years, cache, production = CASES[1]
    fake_llm.queue(labelfree.Program(program=MORTGAGE_PROGRAM.format(P=P, rate=rate, years=years)))
    fake_llm.queue(labelfree.Program(program="print('monthly payment: 5000.00 dollars')"))
    outcome, _, _ = labelfree.adjudicate(q(P, rate, years), production, cache, "t")
    assert outcome == "unresolved"


def test_the_old_hand_solve_path_is_kept_behind_the_flag(monkeypatch, judge_passes):
    from arc import explain
    monkeypatch.setattr(config, "LABELFREE_ADJUDICATOR", "blind")
    seen = []
    monkeypatch.setattr(explain, "_blind_solve", lambda question, tag: seen.append(tag) or
                        explain.Worked(working="", final_answer="monthly payment $5,053.07; interest $95,368.14"))
    P, rate, years, cache, production = CASES[1]
    assert labelfree.adjudicate(q(P, rate, years), production, cache, "t")[0] == "cache_right" and len(seen) == 2


def test_audit_uses_the_program_adjudicator(fake_llm, monkeypatch):
    # audit_hit compares first (judge says no here), then adjudicates with programs
    monkeypatch.setattr(config, "LABELFREE_ADJUDICATOR", "program")
    verdicts = iter([False, True, True, True, True])
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=next(verdicts), critique=""))
    P, rate, years, cache, production = CASES[1]
    prog = labelfree.Program(program=MORTGAGE_PROGRAM.format(P=P, rate=rate, years=years))
    fake_llm.queue(prog)
    fake_llm.queue(prog)
    assert labelfree.audit_hit(q(P, rate, years), cache, production, "t")["outcome"] == "cache_right"


# ----------------------------------------------------------------------------- conventions
# Owner decision 2026-11-15 (option A): the entry states its conventions; tie-break programs
# get the question plus those lines (never code or answers); the question wins over the entry.

ROUNDED = "total interest = monthly payment rounded to the cent x number of payments - principal"
UNROUNDED = "total interest = unrounded monthly payment x number of payments - principal"

ROUNDED_PROGRAM = '''
from decimal import Decimal, getcontext
getcontext().prec = 50
principal = Decimal(511000)       # dollars
r = Decimal("3.5") / 100 / 12
n = 120
payment = (principal * r / (1 - (1 + r) ** -n)).quantize(Decimal("0.01"))
print(f"monthly payment: {payment} dollars")
print(f"total interest: {(payment * n - principal).quantize(Decimal('0.01'))} dollars")
'''
UNROUNDED_PROGRAM = MORTGAGE_PROGRAM.format(P=511000, rate="3.5", years=10)
Q = q(511000, "3.5", 10)
PRODUCTION = "Final answers: monthly payment ≈ **$5,053.27**; total interest ≈ **$95,392**"


def _queue(fake_llm, *programs):
    for p in programs:
        fake_llm.queue(labelfree.Program(program=p))


def test_rounded_convention_clears_an_entry_that_states_it(fake_llm, judge_passes):
    _queue(fake_llm, ROUNDED_PROGRAM, ROUNDED_PROGRAM)
    outcome, answers, _ = labelfree.adjudicate(Q, PRODUCTION, "monthly_payment: 5053.07\ntotal_interest: 95368.40",
                                               "t", conventions=[ROUNDED])
    assert outcome == "cache_right" and "95368.40" in answers[0]
    for c in fake_llm.calls:
        assert ROUNDED in c["prompt"] and Q in c["prompt"]
        assert "95368.40" not in c["prompt"] and "5,053.27" not in c["prompt"]


def test_entry_that_silently_uses_another_convention_is_not_cleared(fake_llm, judge_passes):
    _queue(fake_llm, ROUNDED_PROGRAM, ROUNDED_PROGRAM)
    outcome, _, _ = labelfree.adjudicate(Q, PRODUCTION, "monthly_payment: 5053.07\ntotal_interest: 95368.14",
                                         "t", conventions=[ROUNDED])
    assert outcome != "cache_right"


def test_question_overrides_the_entry_convention(fake_llm, judge_passes):
    q_unrounded = Q + " Compute the total interest from the unrounded payment."
    _queue(fake_llm, UNROUNDED_PROGRAM, UNROUNDED_PROGRAM)       # a program that follows the question
    outcome, _, _ = labelfree.adjudicate(q_unrounded, PRODUCTION, "monthly_payment: 5053.07\ntotal_interest: 95368.14",
                                         "t", conventions=[ROUNDED])
    assert outcome == "cache_right"
    p = fake_llm.calls[0]["prompt"]
    assert "(1) anything the question itself says" in p and "(2) the rules" in p and q_unrounded in p


def test_default_convention_applies_when_the_entry_states_none(fake_llm, judge_passes):
    _queue(fake_llm, ROUNDED_PROGRAM)
    labelfree.adjudicate(Q, PRODUCTION, "total_interest: 95368.40", "t", max_solves=1)
    assert labelfree.DEFAULT_CONVENTION in fake_llm.calls[0]["prompt"]
    # owner decisions 2026-11-15/16: "never round until the end", now present.ROUNDING_POLICY
    from arc import present
    assert labelfree.DEFAULT_CONVENTION == present.ROUNDING_POLICY
    assert "never round any intermediate value" in labelfree.DEFAULT_CONVENTION
    assert "round only the final answer" in labelfree.DEFAULT_CONVENTION


def test_one_rounding_rule_everywhere():
    """Builder rules, re-solve and tie-break programs all carry "never round until the end";
    none of them still carries the old payment-rounding exception."""
    from arc import builder
    old = ("rounded to the cent before", "before any total", "money that is actually paid")
    prompts = {"builder strict rules": builder.STRICT_UNITS_RULES,
               "re-solve": labelfree.RESOLVE_RULES.format(default=labelfree.DEFAULT_CONVENTION),
               "tie-break program": labelfree.PROGRAM_PROMPT.format(
                   question="q", rounding=labelfree.DEFAULT_CONVENTION, and_rules="", rules="", print_as=""),
               "conventions writer": labelfree.CONVENTIONS_PROMPT,
               "default": labelfree.DEFAULT_CONVENTION}
    for name, text in prompts.items():
        assert not any(o in text for o in old), name
    assert "Never round until the end" in builder.STRICT_UNITS_RULES
    assert "unless the question itself" in builder.STRICT_UNITS_RULES          # the question still wins
    for name in ("re-solve", "tie-break program"):
        assert labelfree.DEFAULT_CONVENTION in prompts[name], name
    assert "(1) anything the question itself says" in prompts["tie-break program"]
    assert "follow anything the question says; otherwise" in prompts["re-solve"]


def _mortgage_cache(interest: str):
    from arc.cache import Cache, EntryPoint
    c = Cache("fam_m")
    c.add_function("mortgage", "def semantic_function(ctx):\n"
                               f"    return {{'monthly_payment': 5053.07, 'total_interest': {interest}}}\n")
    c.entry_points = [EntryPoint(Q, "original", list(c.functions)[0], [], "fam_m")]
    return c


def test_build_time_check_catches_a_line_that_disagrees_with_the_code(fake_llm, judge_passes):
    c = _mortgage_cache("95368.14")                               # code: unrounded
    c.conventions["fam_m"] = [ROUNDED]                            # line: rounded
    _queue(fake_llm, ROUNDED_PROGRAM)
    out = labelfree._convention_check(c, "fam_m", {"question": Q})
    assert not out["passed"] and "95368.40" in out["detail"] and "95368.14" in out["detail"]


def test_build_time_check_passes_when_line_and_code_agree(fake_llm, judge_passes):
    c = _mortgage_cache("95368.14")
    c.conventions["fam_m"] = [UNROUNDED]
    _queue(fake_llm, UNROUNDED_PROGRAM)
    assert labelfree._convention_check(c, "fam_m", {"question": Q})["passed"]
    c = _mortgage_cache("95368.40")
    c.conventions["fam_m"] = [ROUNDED]
    _queue(fake_llm, ROUNDED_PROGRAM)
    assert labelfree._convention_check(c, "fam_m", {"question": Q})["passed"]


def test_build_time_check_fails_safe_when_the_program_fails(fake_llm, judge_passes):
    c = _mortgage_cache("95368.14")
    _queue(fake_llm, "print(1/0)")
    out = labelfree._convention_check(c, "fam_m", {"question": Q})
    assert not out["passed"] and "program failed" in out["detail"]


def test_write_conventions_stores_rules_on_the_cache(fake_llm):
    from arc.source import Transcript
    c = _mortgage_cache("95368.14")
    fake_llm.queue(labelfree.Conventions(rules=[UNROUNDED, "  ", "final amounts are rounded to the cent"]))
    t = Transcript("fam_m", Q, "working", "", "m", 0.0)
    rules = labelfree.write_conventions(c, "fam_m", t)
    assert rules == [UNROUNDED, "final amounts are rounded to the cent"] and c.conventions["fam_m"] == rules
    call = fake_llm.calls[0]
    assert call["role"] == "builder" and "semantic_function" in call["prompt"] and "unrounded payment" in call["prompt"]


def test_conventions_survive_save_load_and_merge(tmp_path):
    from arc.cache import Cache
    c = _mortgage_cache("95368.14")
    c.conventions["fam_m"] = [UNROUNDED]
    c.save(tmp_path / "e")
    loaded = Cache.load(tmp_path / "e")
    assert loaded.conventions == {"fam_m": [UNROUNDED]}
    master = Cache("master")
    master.merge(loaded)
    assert master.conventions == {"fam_m": [UNROUNDED]}


def test_served_answer_states_its_conventions():
    from arc import serve
    c = _mortgage_cache("95368.14")
    c.conventions["fam_m"] = [UNROUNDED]
    res = serve.CacheResult({"total_interest": 95368.14}, "", c.entry_points[0], 0.9, {}, 0.0)
    assert serve.answer_text(c, res) == f"total_interest: 95368.14\nConventions: {UNROUNDED}"
    c.conventions.clear()
    assert serve.answer_text(c, res) == "total_interest: 95368.14"
