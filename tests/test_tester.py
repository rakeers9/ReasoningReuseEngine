"""arc/tester.py: independent acceptance tests written by a different model from the builder
(owner decisions 2026-11-17). The builder, re-solve, gate and serving are faked by the
fixtures in test_labelfree_simple.py (the cached code doubles x; pre.wrong adds 1); the
test-maker's suite and its programs are faked here."""

from __future__ import annotations

import pytest

from arc import backlog, config, families, labelfree, sandbox, tester
from arc.cache import Cache, EntryPoint, ParamSpec
from test_labelfree_simple import ENTRY, FAM, FakeBuilder, family, pre, rec, sim, state  # noqa: F401

HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
FEEDBACK = ["it mishandles very large distances"]
HALF1 = [("double 20 m", "40"), ("double 0 m", "0"), ("double 1000000 m", "2000000"), ("double 0.5 m", "1")]
HALF2 = [("double 30 m", "60"), ("double 40 m", "80"), ("double 50 m", "100"), ("double 60 m", "120")]
LOOKALIKES = [("double 6 m, then halve the result", "6"), ("double 7 m, then halve the result", "7")]


def suite(cases=(), declines=()):
    """cases / declines: (question, answer its program prints, or None = program fails / no answer)."""
    return tester.TestSuite(
        cases=[tester.TestCase(question=q, purpose="p", program="FAIL" if a is None else f"print({a!r})")
               for q, a in cases],
        declines=[tester.DeclineCase(question=q, why_different="w", program="" if a is None else f"print({a!r})")
                  for q, a in declines])


@pytest.fixture
def tests_on(pre, monkeypatch):
    monkeypatch.setattr(config, "TESTER_CASES", 4)
    monkeypatch.setattr(config, "TESTER_DECLINES", 1)
    monkeypatch.setattr(config, "TESTER_MIN_CASES", 3)
    monkeypatch.setattr(config, "TESTER_MAX_SUITES", 3)
    monkeypatch.setattr(config, "TESTER_MODEL", SONNET)
    monkeypatch.setattr(config, "TESTER_FALLBACK_MODEL", OPUS)
    monkeypatch.setattr(config, "TESTER_REPAIRS", 0)          # repair tests turn it on
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", [OPUS])
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 1.0)

    def run_program(source, seconds=10, max_output=4000):
        if source == "FAIL":
            return sandbox.ProgramRun(False, "", "NameError: x")
        return sandbox.ProgramRun(True, eval(source[len("print("):-1]), "")
    monkeypatch.setattr(tester.sandbox, "run_program", run_program)
    pre.answers = ["16"]
    pre.suite = suite(HALF1 + HALF2, LOOKALIKES)
    for q, _ in LOOKALIKES:
        pre.p[q] = 0.1                             # the gate turns look-alikes away
    pre.suite_calls = []

    def write_suite(cache, entry_id, ctx, tag, model=None):
        pre.suite_calls.append({"entry": entry_id, "ctx": ctx, "model": model})
        return pre.suite
    monkeypatch.setattr(tester, "write_suite", write_suite)
    pre.feedback_calls = []

    def write_feedback(rows, questions, model, tag):
        pre.feedback_calls.append(([r["question"] for r in rows if r["outcome"] == "fail"], model))
        return list(FEEDBACK)
    monkeypatch.setattr(tester, "write_feedback", write_feedback)
    return pre


def published(s):
    return (s.path / "caches" / ENTRY / "registry.json").exists()


def _tests_of(s):
    return state(s)["tests"]


# ----------------------------------------------------------------------------- verdicts

def test_passing_half_publishes_and_records_the_tests(tests_on):
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and published(tests_on) and res.built_by == OPUS
    t = _tests_of(tests_on)
    assert t["passed"] and t["model"] == SONNET and t["confirmed"] == 4 and t["suite"] == "written"
    assert [r["question"] for r in t["rows"]] == [q for q, _ in HALF1] + [LOOKALIKES[0][0]]   # one half only
    assert [r["outcome"] for r in t["rows"]] == ["pass"] * 5
    [call] = tests_on.suite_calls
    assert call["model"] == SONNET and tests_on.feedback_calls == [] and "tests" in res.parts


def test_wrong_answer_on_an_independent_case_blocks_publishing(tests_on):
    tests_on.wrong = {"double 1000000 m"}          # cache adds 1; program tie-breaks side with the test
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and not published(tests_on) and res.failure == "code"
    assert res.reason.startswith(tester.FAILED_REASON) and "2000000" in res.reason
    assert state(tests_on)["prepublish"]["passed"]  # the builder-era checks alone would have published it


def test_wrong_answer_to_a_look_alike_is_a_gate_failure_with_no_repair(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_REPAIRS", 1)
    tests_on.p[LOOKALIKES[0][0]] = 0.95            # the gate lets it through; cache says 12, it should be 6
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.failure == "gate" and not published(tests_on)
    assert res.reason.startswith(tester.GATE_REASON)
    assert FakeBuilder.models == [OPUS] and tests_on.feedback_calls == []    # a rebuild can't fix the gate


def test_answering_a_look_alike_correctly_is_harmless(tests_on):
    tests_on.suite = suite(HALF1 + HALF2, [("double 6 m, then report it", "12")] * 2)
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and _tests_of(tests_on)["rows"][-1]["outcome"] == "harmless"


def test_look_alike_missing_a_value_must_be_refused(tests_on):
    tests_on.suite = suite(HALF1 + HALF2, [("double 9 m of the rope, length unknown", None)] * 2)
    res = labelfree.build_family(FAM, family(8))     # the gate lets it through (default P 0.95)
    assert res.status == "mismatch" and res.failure == "gate"


def test_a_wrong_test_does_not_count_against_a_right_entry(tests_on):
    tests_on.suite.cases[0].program = "print('41')"   # the tester's own program is wrong
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted"
    row = _tests_of(tests_on)["rows"][0]
    assert row["outcome"] == "pass" and row["detail"].startswith("cache_right")


def test_unresolved_disagreement_is_unchecked_not_a_failure(tests_on):
    tests_on.suite.cases[0].program = "print('41')"
    tests_on.blind = {"double 20 m": "13"}            # the tie-break sides with no one
    res = labelfree.build_family(FAM, family(8))
    t = _tests_of(tests_on)
    assert res.status == "accepted" and t["rows"][0]["outcome"] == "unchecked" and t["confirmed"] == 3


def test_unchecked_cases_do_not_count_as_confirmed_passes(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_MIN_CASES", 4)
    tests_on.suite.cases[0].program = "print('41')"
    tests_on.blind = {"double 20 m": "13"}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(tester.TOO_FEW_REASON) and res.failure == "tests"


def test_too_few_checkable_cases_blocks_publishing(tests_on):
    tests_on.suite = suite([(q, None) for q, _ in HALF1[:3]] + [HALF1[3]] + HALF2, LOOKALIKES)
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(tester.TOO_FEW_REASON) and not published(tests_on)


def test_entry_that_declines_most_cases_is_not_published(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_MIN_CASES", 1)
    tests_on.p.update({"double 20 m": 0.1, "double 0 m": 0.1, "double 0.5 m": 0.1})
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(tester.DECLINED_REASON)


def test_tester_cases_zero_turns_it_off(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_CASES", 0)
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and tests_on.suite_calls == [] and "tests" not in state(tests_on)


# ----------------------------------------------------------------------------- which model tests

def test_tester_switches_when_it_built_the_entry(tests_on, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", [HAIKU, SONNET])
    FakeBuilder.fail = {HAIKU}                     # Haiku's build fails, Sonnet builds the entry
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and res.built_by == SONNET
    assert [c["model"] for c in tests_on.suite_calls] == [OPUS]   # not Sonnet testing itself
    assert _tests_of(tests_on)["model"] == OPUS and "warning" not in _tests_of(tests_on)


def test_pick_model():
    assert tester.pick_model(HAIKU) == config.TESTER_MODEL
    assert tester.pick_model(config.TESTER_MODEL) == config.TESTER_FALLBACK_MODEL


def test_same_model_everywhere_is_flagged(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_MODEL", OPUS)
    monkeypatch.setattr(config, "TESTER_FALLBACK_MODEL", OPUS)
    labelfree.build_family(FAM, family(8))
    assert "not independent" in _tests_of(tests_on)["warning"]


# ----------------------------------------------------------------------------- repair and halves

def test_repair_is_judged_on_the_unseen_half_without_a_new_suite(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_REPAIRS", 1)
    tests_on.wrong, tests_on.bad_by = {"double 1000000 m"}, {OPUS}   # only the unrepaired build is wrong
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and published(tests_on)
    assert FakeBuilder.models == [OPUS, OPUS] and FakeBuilder.feedbacks == [[], FEEDBACK]
    assert len(tests_on.suite_calls) == 1          # one suite, two halves
    t = _tests_of(tests_on)
    assert t["suite"] == "reused" and [r["question"] for r in t["rows"][:4]] == [q for q, _ in HALF2]
    [(failed, model)] = tests_on.feedback_calls
    assert failed == ["double 1000000 m"] and model == SONNET
    first, second = res.cascade
    assert first["next"].startswith("independent tests failed") and second["repair"] == 1
    assert "repaired 1x" in state(tests_on)["reason"]


def test_repair_that_still_fails_is_not_published(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_REPAIRS", 1)
    tests_on.wrong = {"double 1000000 m", "double 30 m"}      # wrong on both halves
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and not published(tests_on) and FakeBuilder.models == [OPUS, OPUS]


def test_feedback_carries_to_the_next_builder_in_the_cascade(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_REPAIRS", 1)
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", [HAIKU, SONNET])
    tests_on.wrong, tests_on.bad_by = {"double 1000000 m", "double 30 m"}, {HAIKU, HAIKU + "+repaired"}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and res.built_by == SONNET
    assert FakeBuilder.models == [HAIKU, HAIKU, SONNET] and FakeBuilder.feedbacks[2] == FEEDBACK
    assert [c["model"] for c in tests_on.suite_calls] == [SONNET, OPUS]   # Sonnet's build needs Opus tests


def test_no_repair_without_feedback(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_REPAIRS", 1)
    monkeypatch.setattr(tester, "write_feedback", lambda *a: [])
    tests_on.wrong = {"double 1000000 m"}
    labelfree.build_family(FAM, family(8))
    assert FakeBuilder.models == [OPUS]


def test_extra_questions_from_the_tester_are_never_run(tests_on):
    many = [(f"double {n} m", str(2 * n)) for n in range(20, 40)]
    tests_on.suite = suite(many, [(f"double {n} m, then halve the result", None) for n in range(10)])
    for q in tests_on.suite.declines:
        tests_on.p[q.question] = 0.1
    labelfree.build_family(FAM, family(8))
    t = _tests_of(tests_on)
    assert len(t["rows"]) == config.TESTER_CASES + config.TESTER_DECLINES == 5
    assert t["dropped"] == (20 - 8) + (10 - 2)


def test_suite_limit_stops_the_build_before_paying_for_another(tests_on, monkeypatch):
    monkeypatch.setattr(config, "TESTER_MAX_SUITES", 1)
    monkeypatch.setattr(config, "TESTER_REPAIRS", 1)
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", [HAIKU, SONNET])
    tests_on.wrong = {"double 1000000 m", "double 30 m"}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and tester.LIMIT_REASON in res.reason and res.failure == "code"
    assert FakeBuilder.models == [HAIKU, HAIKU]     # Sonnet would need a second (Opus) suite: not built
    assert res.cascade[-1]["status"] == "skipped" and len(tests_on.suite_calls) == 1


# ----------------------------------------------------------------------------- context

def test_tester_sees_answered_examples_and_near_misses(tests_on):
    res = labelfree.build_family(FAM, family(8), near_misses=["triple 4 m"])
    assert res.status == "accepted"
    ctx = tests_on.suite_calls[0]["ctx"]
    assert {"question": "double 8 m", "answer": "16"} in ctx.examples     # newest, with production's answer
    assert ctx.near_misses == ["triple 4 m"] and ctx.history == []       # first build: no history
    assert _tests_of(tests_on)["context"] == {"examples": 8, "near_misses": 1, "history": 0}


def test_cold_start_family_still_gets_tested(tests_on, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_FIT_CHECKS", 0)    # too few records for the backlog check
    tests_on.answers = ["6"]
    res = labelfree.build_family(FAM, family(3))              # the minimum a family is built from
    ctx = tests_on.suite_calls[0]["ctx"]
    assert res.status == "accepted" and len(ctx.examples) == 3 and ctx.near_misses == []
    assert _tests_of(tests_on)["confirmed"] == 4               # the synthetic tests did the checking


def test_a_retry_shows_the_tester_what_failed_last_time(tests_on):
    tests_on.wrong = {"double 1000000 m"}
    labelfree.build_family(FAM, family(8))                    # fails
    tests_on.wrong, tests_on.answers = set(), ["18"]
    labelfree.build_family(FAM, family(9))                    # retried on a newer question
    history = tests_on.suite_calls[-1]["ctx"].history
    assert any(h.startswith(tester.FAILED_REASON) for h in history)


def test_pick_examples_spreads_across_the_family_and_keeps_the_newest():
    recs = [rec(i) for i in range(20)]
    got = tester.pick_examples(recs, k=5)
    assert len(got) == 5 and got[0]["question"] == "double 1 m" and got[-1]["question"] == "double 20 m"
    assert tester.pick_examples(recs[:2], k=5) == [{"question": "double 1 m", "answer": "2"},
                                                    {"question": "double 2 m", "answer": "4"}]


def test_prompt_shows_the_contract_but_not_how_it_was_built(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "TESTER_MODEL", SONNET)
    c = Cache(ENTRY)
    c.entry_points = [EntryPoint("double 3 m", "original", "root",
                                 [ParamSpec("length in metres", ["3 m", "300 cm"], "3 m", "0 to 1e6 m")], ENTRY)]
    c.conventions[ENTRY] = ["round only the final answer"]
    c.outputs[ENTRY] = {"result": {"kind": "measure", "unit": "m", "decimals": None}}
    ctx = tester.Context([{"question": "double 4 m", "answer": "8 m"}], ["triple 4 m"], ["mishandled cm"])
    fake_llm.queue(suite())
    tester.write_suite(c, ENTRY, ctx, "t")
    call = fake_llm.calls[-1]
    p = call["prompt"]
    assert call["role"] == tester.ROLE and call["model"] == SONNET and call["schema"] is tester.TestSuite
    assert "Q: double 4 m\nA: 8 m" in p and "round only the final answer" in p and "result: m" in p
    assert "length in metres; formats: 3 m, 300 cm; valid range: 0 to 1e6 m" in p
    assert "triple 4 m" in p and "mishandled cm" in p and "def " not in p


def test_cold_start_prompt_leaves_out_empty_sections(fake_llm):
    c = Cache(ENTRY)
    c.entry_points = [EntryPoint("double 3 m", "original", "root", [ParamSpec("x")], ENTRY)]
    fake_llm.queue(suite())
    tester.write_suite(c, ENTRY, tester.Context([{"question": "double 4 m", "answer": "8"}]), "t")
    p = fake_llm.calls[-1]["prompt"]
    assert "DIFFERENT task" not in p and "earlier attempts" not in p


def test_near_misses_are_real_questions_placed_in_another_family(tmp_path):
    path = tmp_path / "backlog.jsonl"
    for rid, q in [("a", "double 3 m"), ("b", "triple 3 m"), ("c", "halve 3 m"), ("d", "weather?")]:
        backlog.log_interaction(q, "x", HAIKU, path=path, id=rid)
    tr = families.FamilyTracker(path=tmp_path / "families.json", embed_fn=lambda t: None)
    tr.assignments = {"a": {"candidates": {"F": 0.9}, "family": "F"},          # a member, not a near miss
                      "b": {"candidates": {"F": 0.95}, "family": "G"},
                      "c": {"candidates": {"F": 0.91, "G": 0.99}, "family": "H"},
                      "d": {"candidates": {}, "family": "Z"}}
    assert families.near_misses(tr, "F", path) == ["triple 3 m", "halve 3 m"]
    assert families.near_misses(tr, "F", path, k=0) == []


def test_build_ready_passes_near_misses_only_to_build_fns_that_take_them():
    assert families._takes_near_misses(labelfree.build_family)
    assert not families._takes_near_misses(lambda fid, recs: None)


# ----------------------------------------------------------------------------- feedback

def test_feedback_redacts_numbers_from_the_failing_tests(fake_llm):
    rows = [{"outcome": "fail", "question": "double 1000000 m", "expected": "2,000,000 m",
             "cache_answer": "2000001 m", "purpose": "large values", "detail": ""}]
    fake_llm.queue(tester.Feedback(problems=["off by one above 2,000,000; 12 months a year is fine"]))
    out = tester.write_feedback(rows, ["double 3 m"], SONNET, "t")
    assert out == ["off by one above <value>; 12 months a year is fine"]
    assert fake_llm.calls[-1]["role"] == tester.ROLE and fake_llm.calls[-1]["model"] == SONNET


def test_feedback_lines_that_quote_a_test_question_are_dropped(fake_llm):
    rows = [{"outcome": "fail", "question": "a train travels 90 km in 1.5 hours at constant speed",
             "expected": "60 km/h", "cache_answer": "16.67 m/s", "purpose": "units", "detail": ""}]
    fake_llm.queue(tester.Feedback(problems=[
        "for a train travels 90 km in 1.5 hours at constant speed it is wrong",   # quotes the test
        "reports speed in m/s when the question gives km and hours; answer in km/h"]))
    out = tester.write_feedback(rows, ["q"], SONNET, "t")
    assert out == ["reports speed in m/s when the question gives km and hours; answer in km/h"]


def test_builder_is_told_to_fix_behaviour_not_cases():
    from arc import builder
    from arc.source import Transcript
    b = builder.Builder({"id": "p", "question": "q", "golden_label": ""}, Transcript("p", "q", "s", "", "m", 0.0),
                        use_golden=False, units="strict", feedback=["rates per month treated as yearly"])
    text = b._context_block["text"]
    assert "rates per month treated as yearly" in text and "do not special-case" in text
