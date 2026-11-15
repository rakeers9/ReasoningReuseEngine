"""arc/labelfree.py simple flow (LABELFREE_CERTIFY=off, the default): newest record,
re-solved by the customer's own production model, build on a match, one retry, else
"mismatch". The re-solve, the judge and the builder are faked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arc import config, explain, labelfree, llm
from arc.cache import Cache, EntryPoint, ParamSpec
from arc.checks import numbers

FAM = "f0001"
ENTRY = f"fam_{FAM}"


def rec(i, model="claude-haiku-5-5", response=None):
    return {"id": f"r{i}", "question": f"double {i + 1} m", "timestamp": i, "model": model,
            "response": f"{2 * (i + 1)}" if response is None else response}


class FakeBuilder:
    built = []
    fail = False

    def __init__(self, problem, transcript, use_golden=True):
        assert use_golden is False and problem["golden_label"] == ""
        self.problem, self.transcript = problem, transcript

    def build(self):
        from arc.builder import BuildError
        if FakeBuilder.fail:
            raise BuildError("code did not run")
        FakeBuilder.built.append(self.transcript)
        c = Cache(self.problem["id"])
        c.entry_points = [EntryPoint(self.problem["question"], "original", "root", [ParamSpec("x")],
                                     self.problem["id"])]
        return c


@pytest.fixture
def sim(monkeypatch, tmp_path):
    import arc.builder
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "caches")
    monkeypatch.setattr(config, "cache_dir", lambda name: tmp_path / "caches" / name)
    monkeypatch.setattr(config, "GATE", "off")
    monkeypatch.setattr(config, "LABELFREE_CERTIFY", False)
    monkeypatch.setattr(config, "LABELFREE_RESOLVE_TRIES", 2)
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.15)
    monkeypatch.setattr(config, "LABELFREE_FIT_CHECKS", 0)    # the `pre` fixture turns the pre-publish check on
    monkeypatch.setattr(config, "LABELFREE_CONVENTION_CHECK", False)   # convention tests turn it on
    monkeypatch.setattr(config, "CLEF_SEEDS", 1)                  # seed tests raise it
    monkeypatch.setattr(labelfree, "write_conventions",
                        lambda cache, eid, t: cache.conventions.setdefault(eid, list(s.rules)))
    monkeypatch.setattr(arc.builder, "Builder", FakeBuilder)
    FakeBuilder.built, FakeBuilder.fail = [], False
    s = SimpleNamespace(path=tmp_path, answers=[], calls=[], error=None, rules=[])

    def resolve(question, model, tag):
        s.calls.append((question, model))
        llm.LEDGER.add(llm.Record(labelfree.RESOLVE_ROLE, model, 0, 0, 0, 0, 0.01))
        if s.error:
            raise s.error
        return explain.Worked(working="x times two", final_answer=s.answers.pop(0))

    monkeypatch.setattr(labelfree, "_resolve", resolve)
    monkeypatch.setattr(labelfree, "_compare",
                        lambda name, q, ref, cand, tag: SimpleNamespace(agree=numbers(ref)[:1] == numbers(cand)[:1]))
    return s


def state(s):
    return json.loads((s.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())


def test_default_flow_is_simple(monkeypatch):
    seen = []
    monkeypatch.setattr(labelfree, "_build_simple", lambda f, r: seen.append("simple") or
                        labelfree.BuildResult(False, None, "", 0.0))
    monkeypatch.setattr(labelfree, "_build_certified", lambda f, r: seen.append("certified") or
                        labelfree.BuildResult(False, None, "", 0.0))
    monkeypatch.setattr(config, "LABELFREE_CERTIFY", False)
    labelfree.build_family(FAM, [rec(0)])
    monkeypatch.setattr(config, "LABELFREE_CERTIFY", True)
    labelfree.build_family(FAM, [rec(0)])
    assert seen == ["simple", "certified"]


def test_match_builds_from_the_newest_record_with_its_own_model(sim):
    records = [rec(0, model="claude-haiku-5-5"), rec(2, model="cust-model-b"), rec(1, model="cust-model-a")]
    sim.answers = ["6 m"]
    res = labelfree.build_family(FAM, records)
    assert sim.calls == [("double 3 m", "cust-model-b")]          # newest by timestamp, its model
    assert res.status == "accepted" and res.accepted and res.entry_id == ENTRY and res.record_id == "r2"
    assert res.cost == pytest.approx(0.01) and res.parts["resolve"] == pytest.approx(0.01)
    [t] = FakeBuilder.built
    assert "x times two" in t.solution and "6 m" in t.solution and t.model == "cust-model-b"
    assert (sim.path / "caches" / ENTRY / "registry.json").exists()
    assert {e.problem_id for e in Cache.load(sim.path / "caches" / "master").entry_points} == {ENTRY}
    st = state(sim)
    assert st["status"] == "accepted" and st["record_id"] == "r2" and st["model"] == "cust-model-b"


def test_first_mismatch_is_retried_once(sim):
    sim.answers = ["7", "2"]
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "accepted" and len(sim.calls) == 2
    assert [a["agree"] for a in state(sim)["attempts"]] == [False, True]


def test_two_mismatches_return_mismatch_and_publish_nothing(sim):
    sim.answers = ["7", "9"]
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "mismatch" and not res.accepted and res.entry_id is None
    assert res.reason == labelfree.MISMATCH_REASON == "caching tried but answer didn't match"
    assert res.cost == pytest.approx(0.02) and len(sim.calls) == 2
    assert FakeBuilder.built == [] and not (sim.path / "caches" / ENTRY).exists()
    assert state(sim)["status"] == "mismatch" and state(sim)["record_id"] == "r0"
    assert "mismatch" in labelfree.STATUSES


def test_blocked_or_failed_model_call_is_rejected_not_retried(sim):
    sim.error = llm.LLMError("resolve: refusing to call some-opus-model")
    res = labelfree.build_family(FAM, [rec(0, model="some-opus-model")])
    assert res.status == "rejected" and "refusing" in res.reason and len(sim.calls) == 1
    assert not (sim.path / "caches" / ENTRY).exists()


def test_build_failure_is_rejected(sim):
    FakeBuilder.fail = True
    sim.answers = ["2"]
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "rejected" and "build failed" in res.reason
    assert not (sim.path / "caches" / ENTRY).exists()


def test_no_usable_records_is_free(sim):
    res = labelfree.build_family(FAM, [{"id": "r0", "question": "", "response": "", "timestamp": 0}])
    assert res.status == "need_more_data" and res.cost == 0.0 and sim.calls == []


def test_missing_model_falls_back_to_the_source_model(sim):
    r = rec(0)
    r.pop("model")
    sim.answers = ["2"]
    labelfree.build_family(FAM, [r])
    assert sim.calls[0][1] == config.SOURCE_MODEL


def test_cost_cap_stops_before_the_retry(sim, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.005)
    sim.answers = ["7", "2"]
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "rejected" and "cost cap" in res.reason and len(sim.calls) == 1


def test_resolve_uses_the_resolve_role_and_the_given_model(fake_llm):
    fake_llm.queue(explain.Worked(working="w", final_answer="2"))
    w = labelfree._resolve("double 1 m", "cust-model", "t")
    [c] = fake_llm.calls
    assert c["role"] == "resolve" and c["model"] == "cust-model" and c["schema"] is explain.Worked
    assert "double 1 m" in c["prompt"] and w.final_answer == "2"


@pytest.mark.parametrize("gate,writes_spec", [("spec", True), ("clef", True), ("v2", False), ("off", False)])
def test_gate_data_for_clef_includes_specs(monkeypatch, gate, writes_spec):
    from arc import gatecard, spec
    calls = []
    monkeypatch.setattr(config, "GATE", gate)
    monkeypatch.setattr(spec, "index_cache", lambda c: calls.append("spec"))
    monkeypatch.setattr(gatecard, "make_card", lambda *a, **k: calls.append("card") or {})
    labelfree._add_gate_data(Cache(ENTRY), {"id": ENTRY}, None)
    assert ("spec" in calls) == writes_spec


# ----------------------------------------------------------------------------- pre-publish check
# After a build, the newest other family questions go through the real serving path (faked
# here), forced onto the new entry: the gate must fit 4 of 5, the code's answers must equal
# production's (strict comparison; disagreements adjudicated by blind solves), and no
# metamorphic relation may break.

import re as _re

from arc import metamorphic as mm
from arc import serve


def _x(q):
    return float(_re.search(r"double (\S+)", q).group(1))


@pytest.fixture
def pre(sim, monkeypatch):
    """Fake serving: gate P per question from pre.p (default 0.95), served if P >= 0.8.
    The cached code doubles x, except questions in pre.wrong (adds 1) or pre.crash.
    Blind solves give the true answer unless pre.blind overrides it."""
    monkeypatch.setattr(config, "GATE", "clef")
    monkeypatch.setattr(config, "CLEF_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "LABELFREE_FIT_CHECKS", 5)
    monkeypatch.setattr(config, "LABELFREE_FIT_MIN", 0.8)
    monkeypatch.setattr(labelfree, "_add_gate_data", lambda *a: None)
    sim.p, sim.asked, sim.wrong, sim.crash, sim.gate_error, sim.blind = {}, [], set(), set(), set(), {}
    sim.relations = []

    def use(cache, q, tag=None, trace=None, shadow=False, entry_point=None):
        assert entry_point is not None and entry_point.problem_id == ENTRY and entry_point.kind == "original"
        sim.asked.append(q)
        llm.LEDGER.add(llm.Record("decider", "clef", 0, 0, 0, 0, 0.0001))
        if q in sim.gate_error:
            raise llm.LLMError("cloudflare 503")
        p = sim.p.get(q, 0.95)
        trace["clef_p"] = p
        if p < config.CLEF_THRESHOLD:
            trace.update(stage="gate_rejected", rejected_by=["clef"])
            return None
        if q in sim.crash:
            raise serve.CacheInvocationError("ValueError: could not parse '$150k'")
        y = 2 * _x(q) + (1 if q in sim.wrong else 0)
        return serve.CacheResult({"result": y}, "", entry_point, 0.9, {"x": f"{_x(q):g} m"}, 0.0)

    def blind(question, tag):
        ans = sim.blind.get(question, f"{2 * _x(question):g}")
        return explain.Worked(working="", final_answer=ans)

    monkeypatch.setattr(serve, "use", use)
    monkeypatch.setattr(explain, "_blind_solve", blind)
    monkeypatch.setattr(labelfree, "_program_solve",
                        lambda q, tag, conventions=None: (blind(q, tag).final_answer, ""))
    monkeypatch.setattr(mm, "check", lambda run, values, out, max_relations=6: list(sim.relations))
    return sim


def family(n):
    return [rec(i) for i in range(n)]       # r{n-1} is the newest, so it is the build record


def test_passing_check_publishes_and_records_every_question(pre):
    pre.answers = ["16"]
    pre.p = {"double 3 m": 0.30}                            # one of five turned away
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and (pre.path / "caches" / ENTRY).exists()
    assert pre.asked == [f"double {i} m" for i in range(3, 8)]   # newest 5 others, never the build record
    chk = state(pre)["prepublish"]
    assert chk["checked"] == 5 and chk["fits"] == 4 and chk["verified"] == 4 and chk["passed"]
    assert [r["outcome"] for r in chk["rows"]] == ["turned_away", "agree", "agree", "agree", "agree"]
    assert res.parts["prepublish"] > 0


def test_entry_built_from_a_near_miss_is_not_published(pre):
    pre.answers = ["16"]
    pre.p = {f"double {i} m": 0.2 for i in range(1, 8)}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and not res.accepted and res.entry_id is None and res.record_id == "r7"
    assert res.reason.startswith(labelfree.MISFIT_REASON) and "0 of 5" in res.reason and "need 4" in res.reason
    assert not (pre.path / "caches" / ENTRY).exists()


def test_four_of_five_must_fit_three_is_not_enough(pre):
    pre.answers = ["16"]
    pre.p = {"double 4 m": 0.1}
    assert labelfree.build_family(FAM, family(8)).status == "accepted"
    pre.answers = ["16"]
    pre.p = {"double 4 m": 0.1, "double 5 m": 0.1}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and "3 of 5" in res.reason


@pytest.mark.parametrize("others,need", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4)])
def test_needed_count_rounds_up(pre, others, need):
    c = Cache(ENTRY)
    c.entry_points = [EntryPoint("double 1 m", "original", "root", [ParamSpec("x")], ENTRY)]
    chk = labelfree._prepublish_check(c, ENTRY, [rec(i) for i in range(1, others + 1)], budget=1.0)
    assert chk["checked"] == others and chk["needed"] == need


def test_threshold_is_inclusive(pre):
    pre.answers = ["6"]
    pre.p = {"double 1 m": 0.8, "double 2 m": 0.8}
    assert labelfree.build_family(FAM, family(3)).status == "accepted"


def test_gate_error_counts_as_no_fit(pre):
    pre.answers = ["6"]
    pre.gate_error = {"double 1 m", "double 2 m"}
    res = labelfree.build_family(FAM, family(3))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.MISFIT_REASON)
    assert all("cloudflare 503" in r["detail"] for r in state(pre)["prepublish"]["rows"])


def test_code_that_misreads_an_input_is_not_published(pre):
    # the chat run's mortgage entry read "$150k" as 150; here the code's answer is off on one
    # question and the blind solves side with production
    pre.answers = ["16"]
    pre.wrong = {"double 5 m"}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.CODE_WRONG_REASON)
    assert "cache_wrong" in res.reason and "r4" in res.reason
    assert not (pre.path / "caches" / ENTRY).exists()


def test_production_mistake_does_not_block_a_right_entry(pre):
    # production answered one family question wrong; the blind solves side with the cache
    pre.answers = ["16"]
    recs = family(8)
    recs[4]["response"] = "11"                              # double 5 m is 10, production said 11
    res = labelfree.build_family(FAM, recs)
    assert res.status == "accepted"
    outcomes = {r["record_id"]: r["outcome"] for r in state(pre)["prepublish"]["rows"]}
    assert outcomes["r4"] == "cache_right"


def test_unresolved_disagreement_blocks(pre):
    pre.answers = ["16"]
    recs = family(8)
    recs[4]["response"] = "11"
    pre.blind = {"double 5 m": "13"}                        # the blind solve sides with neither
    res = labelfree.build_family(FAM, recs)
    assert res.status == "mismatch" and "unresolved" in res.reason


def test_a_crash_is_a_safe_decline_that_counts_as_not_served(pre):
    # changed 2026-11-15: a crash or an extraction refusal falls back to the model in
    # production, so it costs a fit, it doesn't block on its own
    pre.answers = ["16"]
    pre.crash = {"double 6 m"}
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted"
    rows = {r["record_id"]: r for r in state(pre)["prepublish"]["rows"]}
    assert rows["r5"]["outcome"] == "crashed" and not rows["r5"]["fits"]
    pre.answers = ["16"]
    pre.crash = {"double 6 m", "double 7 m"}               # two declines: 3 of 5 served
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.MISFIT_REASON) and "3 of 5" in res.reason


def test_violated_metamorphic_relation_blocks(pre):
    pre.answers = ["16"]
    pre.relations = [mm.Relation("principal", "$150k", "$150,000", "violated", "0.92 vs 921.13")]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.RELATION_REASON)
    assert "$150k" in res.reason


def test_cost_cap_during_checks_blocks(pre, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.0102)   # re-solve 0.01 leaves room for 2 checks
    pre.answers = ["16"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and "cost cap" in res.reason


def test_check_skipped_without_other_questions_or_when_off(pre, monkeypatch):
    pre.answers = ["2"]
    assert labelfree.build_family(FAM, family(1)).status == "accepted" and pre.asked == []
    monkeypatch.setattr(config, "LABELFREE_FIT_CHECKS", 0)
    pre.answers = ["6"]
    assert labelfree.build_family(FAM, family(3)).status == "accepted" and pre.asked == []
    assert "prepublish" not in state(pre)


def test_check_also_runs_with_the_gate_off(pre, monkeypatch):
    # changed 2026-11-15: code correctness doesn't depend on the gate, so the check runs anyway
    monkeypatch.setattr(config, "GATE", "off")
    pre.answers = ["16"]
    pre.wrong = {"double 5 m"}
    assert labelfree.build_family(FAM, family(8)).status == "mismatch"


# ----------------------------------------------------------------------------- conventions + seeds

def test_convention_mismatch_blocks_publishing(pre, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_CONVENTION_CHECK", True)
    monkeypatch.setattr(labelfree, "_convention_check",
                        lambda cache, eid, rec: {"passed": False, "detail": "program said 95368.40; code said 95368.14"})
    pre.answers = ["16"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.CONVENTION_REASON)
    assert "95368.40" in res.reason and pre.asked == []              # stopped before the pre-publish check
    assert not (pre.path / "caches" / ENTRY).exists()


def test_conventions_are_written_and_published_with_the_entry(pre):
    pre.rules = ["final amounts are rounded to the cent"]
    pre.answers = ["16"]
    assert labelfree.build_family(FAM, family(8)).status == "accepted"
    assert Cache.load(pre.path / "caches" / ENTRY).conventions == {ENTRY: pre.rules}
    assert Cache.load(pre.path / "caches" / "master").conventions == {ENTRY: pre.rules}
    assert state(pre)["conventions"] == pre.rules


def test_fit_check_sees_family_seeds_and_only_verified_questions_are_published(pre, monkeypatch):
    monkeypatch.setattr(config, "CLEF_SEEDS", 3)
    seen = {}
    real_use = serve.use

    def spy(cache, q, **kw):
        seen[q] = [e.text for e in cache.entry_points if e.kind == "family"]
        return real_use(cache, q, **kw)
    monkeypatch.setattr(serve, "use", spy)
    pre.answers = ["20"]
    pre.p = {"double 5 m": 0.1}                                  # turned away: never published
    res = labelfree.build_family(FAM, family(10))                # build r9; check r4..r8; r0..r3 unchecked
    assert res.status == "accepted"
    # seeds: the 2 newest questions NOT being checked (r3, r2), added oldest first
    assert all(v == ["double 3 m", "double 4 m"] for v in seen.values())
    fam = [e.text for e in Cache.load(pre.path / "caches" / ENTRY).entry_points if e.kind == "family"]
    assert set(fam) == {"double 6 m", "double 7 m", "double 8 m", "double 9 m"}   # verified only
    assert "double 3 m" not in fam and "double 5 m" not in fam
    assert state(pre)["seed_questions"] == ["double 4 m", "double 3 m"] and state(pre)["indexed"] == 4


def test_seeds_fill_from_checked_questions_when_the_family_is_small(pre, monkeypatch):
    monkeypatch.setattr(config, "CLEF_SEEDS", 3)
    c = Cache(ENTRY)
    c.entry_points = [EntryPoint("double 9 m", "original", "root", [ParamSpec("x")], ENTRY)]
    others = [rec(i) for i in range(3)]                          # all 3 are being checked
    picks = labelfree._add_seed_questions(c, ENTRY, others, 3)
    assert picks == ["double 3 m", "double 2 m"]
    assert [e.text for e in c.entry_points if e.kind == "family"] == ["double 2 m", "double 3 m"]
    labelfree._remove_seed_questions(c, ENTRY, picks)
    assert [e.kind for e in c.entry_points] == ["original"]
