"""One rounding policy for every agent (owner request 2026-11-16): arc/present.py ROUNDING_POLICY,
quoted verbatim (imported, never copied) in the re-solve program prompt (FINAL ANSWER lines
only), the tie-break program prompt, the conventions writer and the builder's strict rules (as
context: the code returns full precision and must not round). The convention check's program
is the exception: it is compared with the code's exact output, so it prints full precision."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc import builder, certify, checks, judges, labelfree, present
from arc.cache import Cache, EntryPoint, ParamSpec
from arc.source import Transcript

POLICY = present.ROUNDING_POLICY


def test_policy_text_is_the_one_in_present():
    assert labelfree.DEFAULT_CONVENTION is POLICY
    assert "4 significant figures" in POLICY and present.MEASURE_SIG_FIGS == 4


def test_resolve_prompt_quotes_the_policy_for_the_final_answer_lines_only(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('x: 1 m')", final_answer="1 m"))
    labelfree._resolve("q", "m", "t")
    prompt = fake_llm.calls[0]["prompt"]
    assert POLICY in prompt
    assert "applies to these FINAL ANSWER lines only" in prompt
    assert "intermediate lines stay at full precision" in prompt


def test_tie_break_program_prompt_quotes_the_policy(fake_llm):
    fake_llm.queue(labelfree.Program(program="print('x: 1 m')"))
    labelfree._program_solve("q", "t")
    prompt = fake_llm.calls[0]["prompt"]
    assert POLICY in prompt and labelfree.FULL_PRECISION not in prompt


def test_convention_check_program_prints_full_precision_instead(fake_llm):
    # compared with the code's exact output: a 4-significant-figure answer would hide differences
    fake_llm.queue(labelfree.Program(program="print('x: 1 m')"))
    labelfree._program_solve("q", "t", ["rule"], print_as=["x (m)"])
    prompt = fake_llm.calls[0]["prompt"]
    assert labelfree.FULL_PRECISION in prompt and POLICY not in prompt


def test_conventions_writer_quotes_the_policy(fake_llm, monkeypatch):
    import arc.gatecard
    monkeypatch.setattr(arc.gatecard, "_entry_code", lambda cache, root: "def f(): pass")
    c = Cache("fam_m")
    c.entry_points = [EntryPoint("q", "original", "root", [ParamSpec("x")], "fam_m")]
    fake_llm.queue(labelfree.Conventions(rules=["r"]))
    labelfree.write_conventions(c, "fam_m", Transcript("fam_m", "q", "s", "", "m", 0.0))
    prompt = fake_llm.calls[-1]["prompt"]
    assert POLICY in prompt and "only if the question itself sets one" in prompt


def test_builder_strict_rules_quote_the_policy_as_context_only():
    rules = builder.STRICT_UNITS_RULES
    assert POLICY in rules and "must NOT apply it" in rules and "Return full precision" in rules
    assert POLICY not in builder.BUILDER_SYSTEM                       # paper mode unchanged


# ----------------------------------------------------------------------------- 4 significant figures

GAS_PROGRAM = (
    "from decimal import Decimal, getcontext, ROUND_HALF_UP\ngetcontext().prec = 50\n"
    "p = Decimal('7.758933416666666666666666666')  # bar, as computed\n"
    "print(f'pressure, unrounded: {p} bar')\nprint('FINAL ANSWER')\n"
    "print(f'pressure: {p.quantize(Decimal(\"0.001\"), rounding=ROUND_HALF_UP)} bar')\n")


def test_resolve_at_4_significant_figures_matches_production(fake_llm, monkeypatch):
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    fake_llm.queue(labelfree.ResolveOut(working="p = nRT/V", program=GAS_PROGRAM, final_answer="7.759 bar"))
    w = labelfree._resolve("What pressure does the gas exert?", "m", "t")
    assert w.final_answer == "pressure: 7.759 bar" and "7.758933416" in w.working   # exact value kept
    assert checks.number_check(w.final_answer, "7.759 bar")["status"] == "match"
    assert certify._compare("t", "q", "The pressure is 7.759 bar.", w.final_answer, "t").agree


@pytest.mark.parametrize("production", ["7.758 bar", "7.76 bar"])
def test_a_different_rounding_is_not_a_match(production):
    # strict: 7.759 vs 7.758 differs at the stated precision; 7.76 is coarser, so 7.759 rounds to it
    status = checks.number_check("pressure: 7.759 bar", production)["status"]
    assert status == ("mismatch" if production == "7.758 bar" else "match")
