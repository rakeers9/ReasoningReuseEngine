"""arc/labelfree.py builder cascade (DEVIATION, user decision 2026-11-15): label-free builds
use LABELFREE_BUILDERS in order (Haiku, then Sonnet 5.5, then Opus 5.5: user decision
2026-11-16). A build that fails for a code reason is rebuilt with the next model from the same re-solve, within the cost cap; failures a
better builder can't fix are not retried. Re-solve, builder, gate and judges are faked."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from arc import builder, config, labelfree, llm
from arc import metamorphic as mm
from test_labelfree_simple import ENTRY, FAM, FakeBuilder, family, pre, rec, sim, state  # noqa: F401

HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"


@pytest.fixture
def cascade(pre, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", [HAIKU, SONNET])
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 1.0)
    FakeBuilder.cost = {HAIKU: 0.002, SONNET: 0.03}
    pre.answers = ["16"]
    return pre


def published(s):
    return (s.path / "caches" / ENTRY / "registry.json").exists()


# ----------------------------------------------------------------------------- config

def test_default_cascade_is_haiku_sonnet_opus_and_builder_model_is_haiku(monkeypatch):
    for name in ("ARC_PROFILE", "ARC_BUILDER_MODEL", "ARC_LABELFREE_BUILDERS", "ARC_TESTER_MODEL",
                 "ARC_TESTER_FALLBACK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    try:
        c = importlib.reload(config)
        assert c.LABELFREE_BUILDERS == [HAIKU, SONNET, OPUS]
        assert c.TESTER_MODEL == SONNET and c.TESTER_FALLBACK_MODEL == OPUS
        assert c.BUILDER_MODEL == HAIKU                                   # every other build path
        assert c.PROFILE_BUILDER_MODEL["paper"] == "claude-opus-5-5"      # paper profile unchanged
        monkeypatch.setenv("ARC_LABELFREE_BUILDERS", f" {SONNET} ")
        assert importlib.reload(config).LABELFREE_BUILDERS == [SONNET]
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_builder_uses_the_model_it_is_given_else_config(fake_llm, monkeypatch):
    from arc.source import Transcript
    t = Transcript("p", "q", "s", "", "m", 0.0)
    problem = {"id": "p", "question": "q", "golden_label": ""}
    monkeypatch.setattr(config, "BUILDER_MODEL", "builder-default")
    assert builder.Builder(problem, t, use_golden=False).model == "builder-default"
    b = builder.Builder(problem, t, use_golden=False, units="strict", model=SONNET)
    fake_llm.queue(SimpleNamespace())
    b._ask("go", object, "tag")
    assert fake_llm.calls[-1]["model"] == SONNET and fake_llm.calls[-1]["role"] == "builder"


def test_conventions_and_output_specs_are_written_by_the_given_model(fake_llm):
    from arc.cache import Cache, EntryPoint, ParamSpec
    from arc.source import Transcript
    c = Cache("fam_m")
    c.entry_points = [EntryPoint("q", "original", "root", [ParamSpec("x")], "fam_m")]
    import arc.gatecard
    orig = arc.gatecard._entry_code
    arc.gatecard._entry_code = lambda cache, root: "def f(): pass"
    try:
        fake_llm.queue(labelfree.Conventions(rules=["r"]))
        labelfree.write_conventions(c, "fam_m", Transcript("fam_m", "q", "s", "", "m", 0.0), model=SONNET)
    finally:
        arc.gatecard._entry_code = orig
    assert fake_llm.calls[-1]["model"] == SONNET


# ----------------------------------------------------------------------------- not retried

def test_haiku_passes_so_sonnet_is_never_called(cascade):
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and res.built_by == HAIKU
    assert FakeBuilder.models == [HAIKU]
    assert [r["model"] for r in res.cascade] == [HAIKU] and res.cascade[0]["status"] == "accepted"
    st = state(cascade)
    assert st["built_by"] == HAIKU and "built by claude-haiku-5-5 and published" in st["reason"]


def test_resolve_mismatch_never_reaches_a_builder(cascade):
    cascade.answers = ["7", "9"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason == labelfree.MISMATCH_REASON
    assert FakeBuilder.models == [] and res.cascade == [] and res.built_by is None


def test_fit_only_failure_is_not_retried(cascade):
    cascade.p = {f"double {i} m": 0.2 for i in range(1, 8)}       # Clef: doesn't fit, no code errors
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.MISFIT_REASON)
    assert FakeBuilder.models == [HAIKU]
    [row] = res.cascade
    assert row["retry"] is False and "next" not in row and not published(cascade)


def test_gate_outage_is_not_retried(cascade):
    cascade.gate_error = {f"double {i} m" for i in range(1, 8)}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and FakeBuilder.models == [HAIKU]


def test_api_error_during_the_build_is_not_retried(cascade):
    FakeBuilder.fail = llm.LLMError("builder call failed (400): credit balance is too low")
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "rejected" and "credit balance" in res.reason
    assert FakeBuilder.models == [HAIKU] and res.cascade[0]["retry"] is False


def test_cost_cap_during_the_checks_is_not_retried(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.0122)    # re-solve 0.01 + build 0.002: ~2 checks
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and "cost cap" in res.reason and FakeBuilder.models == [HAIKU]


# ----------------------------------------------------------------------------- retried with Sonnet

def assert_sonnet_rebuilt(res, s, first_status="mismatch"):
    assert res.status == "accepted" and res.accepted and res.built_by == SONNET
    assert FakeBuilder.models == [HAIKU, SONNET]
    first, second = res.cascade
    assert first["model"] == HAIKU and first["status"] == first_status and first["retry"] is True
    assert first["next"] == f"code failure, rebuilding with {SONNET}"
    assert second["model"] == SONNET and second["status"] == "accepted"
    st = state(s)
    assert st["built_by"] == SONNET and f"after {HAIKU} failed" in st["reason"]
    assert len(st["attempts"]) == 1 and len(s.calls) == 1          # the same re-solve, not a new one
    assert FakeBuilder.given[0] is FakeBuilder.given[1]            # the same transcript
    assert published(s)


def test_haiku_code_wrong_so_sonnet_rebuilds_and_publishes(cascade):
    cascade.wrong, cascade.bad_by = {"double 5 m"}, {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert_sonnet_rebuilt(res, cascade)
    assert res.cascade[0]["reason"].startswith(labelfree.CODE_WRONG_REASON)
    assert res.cost == pytest.approx(0.01 + 0.002 + 0.03 + 10 * 0.0001)   # both attempts counted
    assert sum(r["cost"] for r in res.cascade) == pytest.approx(res.cost - 0.01)
    assert {r["outcome"] for r in state(cascade)["prepublish"]["rows"]} == {"agree"}   # Sonnet's check


def test_haiku_build_error_so_sonnet_rebuilds(cascade):
    FakeBuilder.fail = {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert_sonnet_rebuilt(res, cascade, first_status="rejected")
    assert "build failed" in res.cascade[0]["reason"]


def test_unusable_haiku_output_is_retried(cascade):
    err = llm.OutputError("builder call hit max_tokens=16000")
    err.models = {HAIKU}
    FakeBuilder.fail = err
    assert_sonnet_rebuilt(labelfree.build_family(FAM, family(8)), cascade, first_status="rejected")


def test_crashed_rows_that_cost_the_fit_are_retried(cascade):
    cascade.crash, cascade.bad_by = {"double 6 m", "double 7 m"}, {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert_sonnet_rebuilt(res, cascade)
    assert res.cascade[0]["reason"].startswith(labelfree.MISFIT_REASON)


def test_unresolved_row_is_retried(cascade):
    cascade.wrong, cascade.bad_by = {"double 5 m"}, {HAIKU}
    cascade.blind = {"double 5 m": "13"}                 # blind solve sides with neither
    res = labelfree.build_family(FAM, family(8))
    assert res.cascade[0]["reason"].startswith(labelfree.CODE_WRONG_REASON)
    assert "unresolved" in res.cascade[0]["reason"] and FakeBuilder.models == [HAIKU, SONNET]


def test_violated_relation_is_retried(cascade, monkeypatch):
    bad = [mm.Relation("principal", "$150k", "$150,000", "violated", "0.92 vs 921.13")]
    monkeypatch.setattr(mm, "check", lambda run, values, out, max_relations=6:
                        list(bad) if FakeBuilder.models[-1] == HAIKU else [])
    res = labelfree.build_family(FAM, family(8))
    assert_sonnet_rebuilt(res, cascade)
    assert res.cascade[0]["reason"].startswith(labelfree.RELATION_REASON)


def test_convention_mismatch_is_retried(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_CONVENTION_CHECK", True)
    monkeypatch.setattr(labelfree, "_convention_check", lambda cache, eid, r: {
        "passed": cache.built_by == SONNET, "detail": "program said 95368.40; code said 95368.14"})
    res = labelfree.build_family(FAM, family(8))
    assert_sonnet_rebuilt(res, cascade)
    assert res.cascade[0]["reason"].startswith(labelfree.CONVENTION_REASON)
    assert res.cascade[0]["convention_check"]["passed"] is False
    assert state(cascade)["convention_check"]["passed"] is True     # top level: the published attempt


def test_conventions_and_outputs_steps_are_retried_with_the_next_model(cascade, monkeypatch):
    seen = []

    def outputs(cache, eid, t, model=None):
        seen.append(model)
        if model == HAIKU:
            raise ValueError("truncated JSON")
        cache.outputs[eid] = {"result": {"kind": "measure", "unit": "m", "decimals": None}}
    monkeypatch.setattr(labelfree, "write_outputs", outputs)
    res = labelfree.build_family(FAM, family(8))
    assert_sonnet_rebuilt(res, cascade)
    assert seen == [HAIKU, SONNET] and res.cascade[0]["reason"].startswith(labelfree.OUTPUTS_REASON)


def test_sonnet_failing_too_reports_the_last_failure_and_publishes_nothing(cascade):
    cascade.wrong = {"double 5 m"}                      # both builders' code is wrong
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.CODE_WRONG_REASON)
    assert FakeBuilder.models == [HAIKU, SONNET] and res.built_by is None
    assert [r["retry"] for r in res.cascade] == [True, True] and "next" not in res.cascade[1]
    assert not published(cascade) and "built_by" not in state(cascade)


# ----------------------------------------------------------------------------- cost cap

def test_cap_reached_after_haiku_skips_sonnet(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.05)
    FakeBuilder.cost = {HAIKU: 0.05, SONNET: 0.03}       # Haiku's build alone uses up the cap
    FakeBuilder.fail = {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "rejected" and FakeBuilder.models == [HAIKU] and not published(cascade)
    assert "cost cap $0.05 reached" in res.reason and f"{SONNET} not tried" in res.reason
    assert "build failed" in res.reason                 # the reason Haiku failed is kept
    assert res.cascade[1] == {"model": SONNET, "status": "skipped", "reason": res.cascade[1]["reason"],
                              "retry": False, "cost": 0.0}


def test_sonnet_checks_get_only_what_is_left_of_the_cap(cascade, monkeypatch):
    budgets = []
    real = labelfree._prepublish_check

    def spy(cache, eid, others, budget):
        budgets.append(round(budget, 6))
        return real(cache, eid, others, budget)
    monkeypatch.setattr(labelfree, "_prepublish_check", spy)
    cascade.wrong, cascade.bad_by = {"double 5 m"}, {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert res.built_by == SONNET
    assert budgets == [round(1.0 - 0.012, 6), round(1.0 - 0.012 - 5 * 0.0001 - 0.03, 6)]


# ----------------------------------------------------------------------------- llm error kinds

def _response(stop_reason, parsed=None):
    usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)
    return SimpleNamespace(stop_reason=stop_reason, content=[], usage=usage, model=HAIKU,
                           parsed_output=parsed, stop_details=None)


@pytest.mark.parametrize("stop,schema,match", [("max_tokens", None, "max_tokens"),
                                                ("end_turn", dict, "no parseable output"),
                                                ("refusal", None, "refused")])
def test_unusable_output_raises_output_error(monkeypatch, stop, schema, match):
    api = SimpleNamespace(create=lambda **k: _response(stop), parse=lambda **k: _response(stop))
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(messages=api, beta=SimpleNamespace(messages=api)))
    monkeypatch.setattr(llm, "_log", lambda entry: None)
    with pytest.raises(llm.OutputError, match=match) as e:
        llm.call("builder", "hi", model=HAIKU, schema=schema)
    assert isinstance(e.value, llm.LLMError)            # existing `except LLMError` still catches it


def test_opus_guard_is_a_plain_llm_error_not_an_output_error():
    with pytest.raises(llm.LLMError) as e:
        llm.call("judge", "hi", model=OPUS)            # the builder may run Opus; other roles may not
    assert not isinstance(e.value, llm.OutputError)


def test_builder_role_reaches_opus_without_the_opt_in(monkeypatch):
    from conftest import RealAPICallBlocked
    monkeypatch.delenv("ARC_ALLOW_OPUS", raising=False)
    with pytest.raises(RealAPICallBlocked):           # past the guard, the call reaches the blocked client
        llm.call("builder", "hi", model=OPUS)
    assert OPUS in llm.OPUS_LIKE                      # Opus gets the server-side refusal fallback


# ----------------------------------------------------------------------------- three steps: Opus last

@pytest.fixture
def cascade3(pre, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", [HAIKU, SONNET, OPUS])
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 1.20)
    FakeBuilder.cost = {HAIKU: 0.002, SONNET: 0.03, OPUS: 0.06}
    pre.answers = ["16"]
    return pre


def test_three_steps_haiku_passes_so_nothing_else_runs(cascade3):
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and res.built_by == HAIKU and FakeBuilder.models == [HAIKU]


def test_three_steps_sonnet_fixes_haiku_so_opus_never_runs(cascade3):
    FakeBuilder.fail = {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert res.built_by == SONNET and FakeBuilder.models == [HAIKU, SONNET]


def test_three_steps_both_fail_code_so_opus_builds_and_publishes(cascade3):
    cascade3.wrong, cascade3.bad_by = {"double 5 m"}, {HAIKU, SONNET}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and res.built_by == OPUS and res.failure is None
    assert FakeBuilder.models == [HAIKU, SONNET, OPUS] and published(cascade3)
    h, s, o = res.cascade
    assert h["next"] == f"code failure, rebuilding with {SONNET}"
    assert s["next"] == f"code failure, rebuilding with {OPUS}" and o["status"] == "accepted"
    assert f"after {HAIKU}, {SONNET} failed" in state(cascade3)["reason"]
    assert FakeBuilder.given[0] is FakeBuilder.given[2] and len(cascade3.calls) == 1   # one re-solve


def test_three_steps_opus_failing_too_is_code(cascade3):
    FakeBuilder.fail = True
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "rejected" and res.failure == "code" and not published(cascade3)
    assert FakeBuilder.models == [HAIKU, SONNET, OPUS] and "next" not in res.cascade[2]


def test_three_steps_a_non_code_failure_stops_before_opus(cascade3):
    FakeBuilder.fail = {HAIKU}
    cascade3.p = {f"double {i} m": 0.2 for i in range(1, 8)}       # Sonnet's entry doesn't fit: not code
    res = labelfree.build_family(FAM, family(8))
    assert FakeBuilder.models == [HAIKU, SONNET] and res.failure is None


def test_three_steps_cap_stops_before_opus(cascade3, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.10)
    FakeBuilder.cost = {HAIKU: 0.002, SONNET: 0.09, OPUS: 0.06}   # Sonnet's build uses up the cap
    FakeBuilder.fail = {HAIKU, SONNET}
    res = labelfree.build_family(FAM, family(8))
    assert FakeBuilder.models == [HAIKU, SONNET] and res.cascade[2]["status"] == "skipped"
    assert f"{OPUS} not tried" in res.reason and res.failure == "code" and not published(cascade3)


# ----------------------------------------------------------------------------- extraction refusals

# Regression (hidden-set run 2026-11-16, runs/chat_loop_hidden/labelfree/fam_f0003/state.json):
# Clef said all 5 family questions fit, but extraction refused 4 of them (word numbers, an implied
# zero balance, months vs years), so 1 of 5 was served. Each refusal counted as a code crash, the
# cascade rebuilt Haiku -> Sonnet -> Opus ($1.08) and then the family still didn't fit. The
# messages below are the ones Haiku's entry got in that run.
F0003_DECLINES = [
    "extraction failed its checks: the nominal annual interest rate of the account with its unit, "
    "as a percentage per annum: value '8% APR' states 8, which is not in the question",
    "extraction failed its checks: the nominal annual interest rate of the account with its unit, "
    "as a percentage per annum: value '4.25% APR' states 4.25, which is not in the question",
    "extraction failed its checks: the starting balance of the account at time zero in dollars: "
    "value '0 dollars' states 0, which is not in the question",
    "extraction failed its checks: the investment horizon, meaning the length of time from the start "
    "to the evaluation date, as a duration in years: value '8 years' states 8, which is not in the question",
]


def test_f0003_extraction_refusals_end_the_cascade_with_no_failure(cascade3):
    cascade3.decline = {f"double {i} m": msg for i, msg in zip(range(3, 7), F0003_DECLINES)}
    res = labelfree.build_family(FAM, family(8))
    assert FakeBuilder.models == [HAIKU]                         # no Sonnet, no Opus
    assert res.status == "mismatch" and res.failure is None and not published(cascade3)
    assert res.reason.startswith(labelfree.MISFIT_REASON) and "1 of 5" in res.reason
    [row] = res.cascade
    assert row["retry"] is False and row["failure"] is None and "next" not in row
    st = state(cascade3)
    assert st["failure"] is None and st["status"] == "mismatch"
    pre_rows = st["prepublish"]["rows"]
    assert [r["outcome"] for r in pre_rows] == ["declined"] * 4 + ["agree"]
    assert not any(r["fits"] for r in pre_rows[:4]) and st["prepublish"]["code_error"] is False


def test_a_code_crash_among_refusals_still_rebuilds(cascade3):
    # the cached code itself raising (ValueError on 'eight percent') is a code fault
    cascade3.decline = {f"double {i} m": msg for i, msg in zip(range(3, 6), F0003_DECLINES)}
    cascade3.crash, cascade3.bad_by = {"double 6 m"}, {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert FakeBuilder.models == [HAIKU, SONNET] and res.cascade[0]["retry"] is True
    assert res.cascade[0]["failure"] == "code"


def test_a_wrong_answer_among_refusals_still_rebuilds(cascade3):
    cascade3.decline = {f"double {i} m": msg for i, msg in zip(range(3, 6), F0003_DECLINES)}
    cascade3.wrong, cascade3.bad_by = {"double 6 m"}, {HAIKU}
    res = labelfree.build_family(FAM, family(8))
    assert FakeBuilder.models[:2] == [HAIKU, SONNET] and res.cascade[0]["failure"] == "code"


def test_one_refusal_with_enough_fits_still_publishes(cascade):
    cascade.decline = {"double 6 m": F0003_DECLINES[0]}            # 4 of 5 served: need 4
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and FakeBuilder.models == [HAIKU]
