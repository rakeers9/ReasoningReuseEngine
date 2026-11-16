"""The builder's reference values come from an executed program, never from hand arithmetic.

Chat-loop run, 2026-11-15, family fam_f0003 ($0.49 lost, owner asked for the fix). The question
was "$174k mortgage at 5.875%, 15 years". The re-solve (fam_f0003:resolve2, Haiku) did its
working by hand and slipped at "P*r = 174000 x 0.0048958333 = 851.9375" (the true value is
851.875), which gave M = 1456.693046. Its program printed the right 1456.59, which matched
production, so the build went ahead. But the builder was handed the hand working, and every
build judge rejected correct code ("does not match the expected 1456.693046"), 8 times over.
The fix has three parts:
  1. The transcript handed to the builder is the program source plus its executed output.
     The program prints labelled intermediate values, then FINAL ANSWER and the answer lines.
  2. The model's own working is dropped when its final numbers disagree with the program
     under the strict number check.
  3. Builder(reference="executed"): the build judges are told the values come from an executed
     program and must not be recomputed by hand. reference="transcript" (the default, the
     paper) is byte-for-byte unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc import builder, judges, labelfree
from arc.source import Transcript
from test_labelfree_simple import FAM, ENTRY, FakeBuilder, family, rec, sim, state  # noqa: F401

# ----------------------------------------------------------------------------- the real case (verbatim)

Q174 = ("Calculate the monthly installment and lifetime interest for a $174k mortgage at 5.875% annual "
        "interest compounded monthly, repaid over 15 years.")
WORKING174 = (
    "Given: principal P = 174000 USD; annual rate 5.875% compounded monthly; term 15 years, so n = 15 x 12 "
    "= 180 monthly payments. Monthly rate r = 0.05875 / 12 = 0.0048958333... = 47/9600. Formula (standard "
    "amortization): M = P*r*(1+r)^n / ((1+r)^n - 1). Lifetime interest = n*M - P. Step 1: ln(1+r) by series "
    "r - r^2/2 + r^3/3 - r^4/4 + ... with r^2 = 2.39691840e-5, r^3 = 1.1734913e-7, r^4 = 5.74507e-10: "
    "ln(1+r) = 0.00489583333 - 0.00001198459 + 0.00000003912 - 0.00000000014 = 0.00488388771464. Step 2: "
    "n*ln(1+r) = 180 x 0.00488388771464 = 0.87909978854. Step 3: (1+r)^180 = e^0.87909978854 = e^0.88 x "
    "e^-0.00090021146 = 2.41089970619 x 0.99910019360 = 2.40873036321 (call A). Step 4: A - 1 = "
    "1.40873036321; A/(A-1) = 1 + 1/1.40873036321 = 1 + 0.70985905189 = 1.70985905189. Step 5: P*r = "
    "174000 x 0.0048958333 = 851.9375. Step 6: M = 851.9375 x 1.70985905189 = 1456.693046 USD, which "
    "rounds to 1456.69 USD. Step 7: Total paid = 180 x 1456.693046 = 262204.748 USD. Step 8: Lifetime "
    "interest = 262204.748 - 174000 = 88204.748 USD, which rounds to 88204.75 USD.")
PROGRAM174 = (
    "from decimal import Decimal, getcontext, ROUND_HALF_UP\ngetcontext().prec = 60\nP = Decimal(174000)\n"
    "annual = Decimal('0.05875')\nr = annual / 12\nn = 180\nA = (1 + r) ** n\nM = P * r * A / (A - 1)\n"
    "interest = n * M - P\ncent = Decimal('0.01')\n"
    "print('Monthly installment: ' + str(M.quantize(cent, rounding=ROUND_HALF_UP)) + ' USD')\n"
    "print('Lifetime interest: ' + str(interest.quantize(cent, rounding=ROUND_HALF_UP)) + ' USD')")
FINAL174 = ("Monthly installment: $1,456.69 per month; Lifetime interest: $88,204.75 total over the 180 "
            "payments")
HAND_SLIPS = ("851.9375", "1456.693046", "1456.69", "88204.75", "262204.748")

# the same program in the new style: labelled intermediates, then FINAL ANSWER and the answer lines
STEPS174 = (
    "from decimal import Decimal, getcontext, ROUND_HALF_UP\ngetcontext().prec = 60\n"
    "P = Decimal(174000)  # dollars\nr = Decimal('0.05875') / 12  # per month\nn = 180  # payments\n"
    "A = (1 + r) ** n\nM = P * r * A / (A - 1)\ninterest = n * M - P\ncent = Decimal('0.01')\n"
    "print(f'monthly rate: {r} per month')\nprint(f'P x r: {P * r} dollars')\n"
    "print(f'growth factor (1+r)^n: {A}')\nprint(f'monthly installment, unrounded: {M} dollars')\n"
    "print(f'lifetime interest, unrounded: {interest} dollars')\nprint('FINAL ANSWER')\n"
    "print(f'Monthly installment: {M.quantize(cent, rounding=ROUND_HALF_UP)} USD')\n"
    "print(f'Lifetime interest: {interest.quantize(cent, rounding=ROUND_HALF_UP)} USD')\n")


def test_the_174k_hand_slip_never_reaches_the_builder(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(working=WORKING174, program=PROGRAM174, final_answer=FINAL174))
    w = labelfree._resolve(Q174, "claude-haiku-5-5", "fam_f0003:resolve2")
    assert w.final_answer == "Monthly installment: 1456.59 USD\nLifetime interest: 88185.51 USD"
    assert w.executed and not w.working_kept                       # hand numbers disagree -> dropped
    for slip in HAND_SLIPS:
        assert slip not in w.working, slip
    assert PROGRAM174 in w.working and "1456.59" in w.working and "88185.51" in w.working
    assert w.working_check["status"] == "mismatch"


def test_new_style_program_gives_the_builder_exact_intermediates(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(working="Monthly rate r = annual rate / 12; M = P r A / (A - 1).",
                                        program=STEPS174, final_answer="$1,456.59 and $88,185.51"))
    w = labelfree._resolve(Q174, "claude-haiku-5-5", "t")
    # only the lines after FINAL ANSWER are compared with production
    assert w.final_answer == "Monthly installment: 1456.59 USD\nLifetime interest: 88185.51 USD"
    assert w.executed and w.working_kept                            # agrees with the program -> kept
    assert "P x r: 851.8749999999999" in w.working                 # the true 851.875 (r repeats), executed
    assert "monthly installment, unrounded: 1456.586179630" in w.working
    assert "Monthly rate r = annual rate / 12" in w.working


def test_resolve_prompt_asks_for_executed_intermediates_and_no_hand_arithmetic(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('x: 1 m')", final_answer="1 m"))
    labelfree._resolve(Q174, "claude-haiku-5-5", "t")
    prompt = fake_llm.calls[0]["prompt"]
    assert "FINAL ANSWER" in prompt and "intermediate value" in prompt
    assert "Do not do any arithmetic by hand" in prompt


@pytest.mark.parametrize("stdout,steps,final", [
    ("a: 1\nb: 2\nFINAL ANSWER\nx: 3 m\n", "a: 1\nb: 2", "x: 3 m"),
    ("a: 1\nfinal answer:\nx: 3 m\ny: 4 s", "a: 1", "x: 3 m\ny: 4 s"),
    ("x: 3 m\n", "", "x: 3 m"),                                     # old style: the whole output
    ("FINAL ANSWER\nx: 1\nFINAL ANSWER\nx: 2\n", "FINAL ANSWER\nx: 1", "x: 2"),
])
def test_split_program_output(stdout, steps, final):
    assert labelfree._split_program_output(stdout) == (steps, final)


def test_marker_with_nothing_after_it_counts_as_printing_nothing(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('a: 1')\nprint('FINAL ANSWER')",
                                        final_answer="2 m"))
    w = labelfree._resolve(Q174, "m", "t")
    assert not w.executed and w.final_answer == "" and "no answer after FINAL ANSWER" in w.error


@pytest.mark.parametrize("program,error", [("print(1/0)", "ZeroDivisionError"), ("", "printed nothing"),
                                           ("while True:\n    pass", "timed out")])
def test_a_failed_program_gives_no_answer_and_no_hand_working(fake_llm, monkeypatch, program, error):
    # owner decision 2026-11-16: never fall back to the hand-written final_answer or working
    from arc import config
    monkeypatch.setattr(config, "LABELFREE_PROGRAM_SECONDS", 0.5)
    fake_llm.queue(labelfree.ResolveOut(working=WORKING174, program=program, final_answer=FINAL174))
    w = labelfree._resolve(Q174, "m", "t")
    assert not w.executed and w.final_answer == "" and w.working == "" and error in w.error


def test_hand_answer_with_no_numbers_is_dropped():
    assert not labelfree._hand_agrees("see the program", "x: 3 m")["agree"]
    assert labelfree._hand_agrees("x = 3 m", "x: 3 m")["agree"]
    assert not labelfree._hand_agrees("x = 3.1 m", "x: 3.0 m")["agree"]
    # an extra wrong hand number is caught in the other direction
    assert not labelfree._hand_agrees("payment $1,456.59, interest $88,204.75",
                                      "payment: 1456.59 USD\ninterest: 88185.51 USD")["agree"]


# ----------------------------------------------------------------------------- the build flow

def test_build_flow_builds_with_the_executed_reference(sim, monkeypatch):
    monkeypatch.setattr(labelfree, "_resolve", lambda q, m, tag, unit_hint=None: labelfree.Resolved(
        working="exec output", final_answer="6 m", executed=True, working_kept=False,
        working_check={"status": "mismatch"}))
    res = labelfree.build_family(FAM, [rec(2)])
    assert res.status == "accepted"
    assert FakeBuilder.references == ["executed"]
    s = state(sim)
    assert s["reference"] == "executed" and s["attempts"][-1]["working_kept"] is False


# ----------------------------------------------------------------------------- a crashed re-solve program

def test_a_crashed_program_falls_through_to_a_working_second_try(sim):
    sim.answers = [None, "6 m"]
    res = labelfree.build_family(FAM, [rec(2)])
    assert res.status == "accepted" and len(sim.calls) == 2
    [t] = FakeBuilder.built
    assert "ZeroDivisionError" not in t.solution and FakeBuilder.references == ["executed"]
    first, second = state(sim)["attempts"]
    assert first["agree"] is False and first["executed"] is False and "ZeroDivisionError" in first["program_error"]
    assert second["agree"] is True


def test_two_crashes_give_mismatch_with_nothing_built(sim):
    sim.answers = [None, None]
    res = labelfree.build_family(FAM, [rec(2)])
    assert res.status == "mismatch" and res.failure is None and not res.accepted
    assert res.reason.startswith(labelfree.PROGRAM_FAILED_REASON) and "ZeroDivisionError" in res.reason
    assert FakeBuilder.given == []                                   # no builder, no hand numbers


def test_a_crash_and_a_real_disagreement_is_the_usual_mismatch(sim):
    sim.answers = [None, "7 m"]
    res = labelfree.build_family(FAM, [rec(2)])
    assert res.status == "mismatch" and res.reason == labelfree.MISMATCH_REASON and FakeBuilder.given == []


def test_a_crashed_resolve_is_never_compared_with_production(sim, monkeypatch):
    seen = []
    monkeypatch.setattr(labelfree, "_compare", lambda name, q, ref, cand, tag: seen.append(cand) or
                        SimpleNamespace(agree=True))
    sim.answers = [None, "6 m"]
    labelfree.build_family(FAM, [rec(2)])
    assert seen == ["6 m"]


# ----------------------------------------------------------------------------- the build judges

PROBLEM = {"id": "p", "question": Q174, "golden_label": ""}
T = Transcript("p", Q174, "Program ... Executed output ...\nmonthly installment, unrounded: 1456.5861796",
               "", "m", 0.0)
SPEC = builder.PlannedFunction(id="f1", description="Compute the level monthly payment",
                               inputs=["principal", "monthly rate", "payments"],
                               outputs=["monthly payment in dollars"], calls=[],
                               transcript_fragment="M = P r A / (A - 1)")
PARAMS = builder.ParameterRecognition(problem_class="c", template="t", parameters=[])
CODE = ("def semantic_function(ctx):\n    p = load(ctx, 'principal', d_type=float)\n"
        "    narrate(ctx, 'payment')\n    return {'monthly payment in dollars': p}\n")
PAPER_RULE = ("Pass only if the output matches what the transcript computes for this step "
              "and the code uses its inputs rather than hard-coding the original values.")


@pytest.fixture
def prompts(monkeypatch):
    seen = []
    monkeypatch.setattr(judges, "judge", lambda prompt, **k: seen.append(prompt) or
                        SimpleNamespace(passed=True, critique="", model_dump=lambda: {}))
    return seen


def _implement(fake_llm, monkeypatch, **kw):
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict", **kw)
    monkeypatch.setattr(b, "_run_candidate", lambda code, payloads: ({"monthly payment in dollars": 1456.586}, "ok",
                                                                       None))
    fake_llm.queue(builder.Candidate(reuse_existing_function_id=None, description="d", code=CODE,
                                     test_payloads=[], expected_result="1456.693046"))
    b.implement(SPEC, PARAMS, is_root=False)
    return b


def test_executed_reference_judge_does_not_recompute_by_hand(fake_llm, monkeypatch, prompts):
    b = _implement(fake_llm, monkeypatch, reference="executed")
    b._judge_final({"monthly payment in dollars": 1456.586}, "", "end_to_end")
    step, final = prompts
    for p in (step, final):
        assert "executed program" in p and "Do not recompute" in p
    assert PAPER_RULE not in step
    assert "executed values win" in step               # the builder's own claim (1456.693046) loses


def test_default_judges_are_the_paper_prompts_byte_for_byte(fake_llm, monkeypatch, prompts):
    b = _implement(fake_llm, monkeypatch)
    assert b.reference == "transcript"
    b._judge_final({"monthly payment in dollars": 1456.586}, "", "end_to_end")
    step, final = prompts
    assert step.endswith("## Code\n" + CODE + "\n\n" + PAPER_RULE)
    assert final.endswith("Does the cache output give the correct final answer to the problem (agreeing "
                          "with the transcript)?")
    assert "executed program" not in step and "executed program" not in final


def test_unknown_reference_fails_loudly():
    with pytest.raises(ValueError):
        builder.Builder(PROBLEM, T, use_golden=False, reference="hand")
