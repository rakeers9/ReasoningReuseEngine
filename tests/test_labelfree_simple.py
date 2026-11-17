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
    """fail: True = every build raises BuildError; or a set of builder models that fail; or an
    exception instance to raise. Each build costs cost[model] (default 0) on the ledger."""
    built = []
    models = []
    given = []                  # every transcript a build was started from, failed builds included
    references = []             # Builder(reference=...) of every build
    fail = False
    cost = {}

    feedbacks = []              # Builder(feedback=...) of every build (repair rounds)

    def __init__(self, problem, transcript, use_golden=True, units="paper", model=None, reference="transcript",
                 feedback=None):
        FakeBuilder.feedbacks.append(list(feedback or []))
        self.feedback = list(feedback or [])
        assert use_golden is False and problem["golden_label"] == ""
        FakeBuilder.references.append(reference)
        assert units == "strict"            # label-free builds always use the strict unit rules
        self.problem, self.transcript, self.model = problem, transcript, model

    def build(self):
        from arc.builder import BuildError
        FakeBuilder.models.append(self.model)
        FakeBuilder.given.append(self.transcript)
        if FakeBuilder.cost.get(self.model):
            llm.LEDGER.add(llm.Record("builder", self.model, 0, 0, 0, 0, FakeBuilder.cost[self.model]))
        fail = FakeBuilder.fail
        if isinstance(fail, Exception) and (not hasattr(fail, "models") or self.model in fail.models):
            raise fail
        if fail is True or (isinstance(fail, set) and self.model in fail):
            raise BuildError("code did not run")
        FakeBuilder.built.append(self.transcript)
        c = Cache(self.problem["id"])
        c.built_by = self.model + ("+repaired" if self.feedback else "")
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
    monkeypatch.setattr(config, "LABELFREE_BUILDERS", ["claude-haiku-5-5"])   # cascade tests use two
    monkeypatch.setattr(config, "TESTER_CASES", 0)               # tests/test_tester.py turns it on
    monkeypatch.setattr(labelfree, "write_conventions",
                        lambda cache, eid, t, model=None: cache.conventions.setdefault(eid, list(s.rules)))
    monkeypatch.setattr(labelfree, "write_outputs",
                        lambda cache, eid, t, model=None: cache.outputs.setdefault(eid, {"result": {"kind": "measure", "unit": "m",
                                                                                     "decimals": None}}))
    monkeypatch.setattr(arc.builder, "Builder", FakeBuilder)
    FakeBuilder.built, FakeBuilder.models, FakeBuilder.given, FakeBuilder.fail, FakeBuilder.cost = [], [], [], False, {}
    FakeBuilder.references, FakeBuilder.feedbacks = [], []
    s = SimpleNamespace(path=tmp_path, answers=[], calls=[], error=None, rules=[])

    def resolve(question, model, tag, unit_hint=None):
        s.calls.append((question, model))
        llm.LEDGER.add(llm.Record(labelfree.RESOLVE_ROLE, model, 0, 0, 0, 0, 0.01))
        if s.error:
            raise s.error
        a = s.answers.pop(0)
        if a is None:                       # None: the re-solve program crashed (no answer, no working)
            return labelfree.Resolved(working="", final_answer="", executed=False, working_kept=False,
                                      error="ZeroDivisionError: division by zero")
        return labelfree.Resolved(working="x times two", final_answer=a, executed=True)

    monkeypatch.setattr(labelfree, "_resolve", resolve)
    monkeypatch.setattr(labelfree, "_compare",
                        lambda name, q, ref, cand, tag: SimpleNamespace(agree=numbers(ref)[:1] == numbers(cand)[:1]))
    return s


def state(s):
    return json.loads((s.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())


def test_default_flow_is_simple(monkeypatch):
    seen = []
    monkeypatch.setattr(labelfree, "_build_simple", lambda f, r, n=None: seen.append("simple") or
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
    # schema changed 2026-11-15: the re-solve also writes a program that computes the answer
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('result: 2 m')", final_answer="2"))
    w = labelfree._resolve("double 1 m", "cust-model", "t")
    [c] = fake_llm.calls
    assert c["role"] == "resolve" and c["model"] == "cust-model" and c["schema"] is labelfree.ResolveOut
    assert "double 1 m" in c["prompt"] and w.final_answer == "result: 2 m"


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
    The cached code doubles x, except questions in pre.wrong (adds 1), pre.crash (the code
    raises) or pre.decline (extraction refuses: serve.ExtractionError). Blind solves give the true answer unless pre.blind overrides it."""
    monkeypatch.setattr(config, "GATE", "clef")
    monkeypatch.setattr(config, "CLEF_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "LABELFREE_FIT_CHECKS", 5)
    monkeypatch.setattr(config, "LABELFREE_FIT_MIN", 0.8)
    monkeypatch.setattr(labelfree, "_add_gate_data", lambda *a: None)
    sim.p, sim.asked, sim.wrong, sim.crash, sim.gate_error, sim.blind = {}, [], set(), set(), set(), {}
    sim.decline = {}           # question -> ExtractionError message
    sim.relations = []
    sim.bad_by = None          # None: pre.wrong / pre.crash apply to every builder; else only these

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
        bad = sim.bad_by is None or getattr(cache, "built_by", None) in sim.bad_by
        if q in sim.decline and bad:
            raise serve.ExtractionError(sim.decline[q])
        if q in sim.crash and bad:
            raise serve.CacheInvocationError("ValueError: could not parse '$150k'")
        y = 2 * _x(q) + (1 if q in sim.wrong and bad else 0)
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


# ----------------------------------------------------------------------------- 2026-11-15 regressions
# (1) write_conventions hit max_tokens=2000; on another family the truncated JSON raised a
#     pydantic ValidationError that escaped build_family. (2) The re-solve said "approximately
#     $415,669 (about $415,669.60 using the rounded payment)" and gave a false mismatch against
#     production's "$415,670" ($368,000, 5.875%, 30y: 415,669.23 unrounded, 415,669.60 rounded).

def _validation_error():
    from pydantic import ValidationError
    try:
        labelfree.Conventions.model_validate_json('{"rules": ["total interest = pay')
    except ValidationError as e:
        return e


def test_conventions_parse_failure_is_a_mismatch_not_an_exception(pre, monkeypatch):
    def boom(cache, eid, t, model=None):
        raise _validation_error()
    monkeypatch.setattr(labelfree, "write_conventions", boom)
    pre.answers = ["16"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.CONVENTION_REASON)
    assert "ValidationError" in res.reason and not (pre.path / "caches" / ENTRY).exists()


def test_write_conventions_gets_room_to_think(fake_llm):
    from arc.source import Transcript
    c = Cache(ENTRY)
    c.add_function("f", "def semantic_function(ctx):\n    return {'y': 1}\n")
    c.entry_points = [EntryPoint("q", "original", list(c.functions)[0], [], ENTRY)]
    fake_llm.queue(labelfree.Conventions(rules=[]))
    labelfree.write_conventions(c, ENTRY, Transcript(ENTRY, "q", "w", "", "m", 0.0))
    assert fake_llm.calls[0]["max_tokens"] >= 16000


Q368 = ("A $368,000 mortgage at 5.875% annual interest, compounded monthly, over 30 years. "
        "What is the monthly payment and the total interest paid?")
PROD368 = "- Monthly payment: **$2,176.86**\n- Total interest over 30 years: **$415,670**"
PROGRAM368 = '''
from decimal import Decimal, getcontext
getcontext().prec = 50
principal = Decimal(368000)        # dollars
r = Decimal("5.875") / 100 / 12
n = 360
payment = (principal * r / (1 - (1 + r) ** -n)).quantize(Decimal("0.01"))   # this program rounds
print(f"monthly payment: {payment} dollars")
print(f"total interest: {(payment * n - principal).quantize(Decimal('0.01'))} dollars")
'''
# PROGRAM368 rounds the payment (the default before 2026-11-15's revision). The tests using it
# check only that the program's printed output is the compared answer, not the convention.


def test_resolve_states_one_answer_computed_by_its_program(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(
        working="P = 368000, r = 0.05875/12, n = 360 ...", program=PROGRAM368,
        final_answer="approximately $415,669 (about $415,669.60 using the rounded payment)"))
    w = labelfree._resolve(Q368, "cust-model", "t")
    assert w.final_answer == "monthly payment: 2176.86 dollars\ntotal interest: 415669.60 dollars"
    assert PROGRAM368.strip() in w.working and "P = 368000" in w.working        # the builder sees both
    call = fake_llm.calls[0]
    assert call["role"] == "resolve" and call["model"] == "cust-model" and call["schema"] is labelfree.ResolveOut
    assert labelfree.DEFAULT_CONVENTION in call["prompt"] and "ONE answer" in call["prompt"]


def test_resolved_answer_matches_production_at_its_stated_precision(fake_llm, monkeypatch):
    from arc import certify, judges
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    fake_llm.queue(labelfree.ResolveOut(working="...", program=PROGRAM368, final_answer="$415,669.60"))
    w = labelfree._resolve(Q368, "cust-model", "t")
    assert certify._compare("t", Q368, PROD368, w.final_answer, "t").agree
    # the old free-text answer gave the false mismatch
    old = "Monthly payment ≈ $2,176.86; total interest ≈ approximately $415,669 (about $415,669.60 using the rounded payment)"
    assert w.final_answer != old


# Regression (builder comparison, 2026-11-15): owner decision "never round until the end".
# Production said 1048.17 and 81560.86; with the old default (payment rounded to the cent before
# the total) the re-solve gave 81560.80, a false mismatch. Unrounded payment x 240 - P = 81560.86.
Q017 = "You borrow $0.17 million for 20 years at 4.2% APR (monthly compounding). Find the monthly payment and the total interest paid."
PROD017 = "1048.17 and 81560.86"
PROGRAM017 = '''
from decimal import Decimal, getcontext
getcontext().prec = 50
principal = Decimal("0.17") * 10**6   # dollars
r = Decimal("4.2") / 100 / 12
n = 20 * 12
payment = principal * r / (1 - (1 + r) ** -n)      # no intermediate rounding
interest = payment * n - principal
print(f"monthly payment: {payment.quantize(Decimal('0.01'))} dollars")
print(f"total interest: {interest.quantize(Decimal('0.01'))} dollars")
'''
OLD_ROUNDED017 = PROGRAM017.replace("payment = principal * r / (1 - (1 + r) ** -n)",
                                    "payment = (principal * r / (1 - (1 + r) ** -n)).quantize(Decimal('0.01'))")


def test_unrounded_resolve_matches_production_on_the_017_million_loan(fake_llm, monkeypatch):
    from arc import certify, judges
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    fake_llm.queue(labelfree.ResolveOut(working="...", program=PROGRAM017, final_answer="1048.17 and 81560.86"))
    w = labelfree._resolve(Q017, "cust-model", "t")
    assert "81560.86" in w.final_answer and "1048.17" in w.final_answer
    assert "never round any intermediate value" in fake_llm.calls[0]["prompt"]
    assert certify._compare("t", Q017, PROD017, w.final_answer, "t").agree
    # the old payment-rounding convention gave 81560.80, which the strict compare rejects
    fake_llm.queue(labelfree.ResolveOut(working="...", program=OLD_ROUNDED017, final_answer="x"))
    old = labelfree._resolve(Q017, "cust-model", "t2")
    assert "81560.80" in old.final_answer
    assert not certify._compare("t", Q017, PROD017, old.final_answer, "t").agree


def test_resolve_falls_back_to_the_stated_answer_if_its_program_fails(fake_llm):
    # owner decision 2026-11-16: no fallback to the hand-written answer (was: final_answer used)
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print(1/0)", final_answer="$415,669.60"))
    w = labelfree._resolve(Q368, "cust-model", "t")
    assert not w.executed and w.final_answer == "" and w.working == "" and "ZeroDivisionError" in w.error


# ----------------------------------------------------------------------------- output specs

def _two_output_cache():
    c = Cache(ENTRY)
    c.add_function("mortgage", "def semantic_function(ctx):\n"
                               "    return {'monthly payment in dollars': 5053.0678, 'number of payments': 120,\n"
                               "            'total interest in dollars': 95368.14, 'decision': 'ok'}\n")
    c.entry_points = [EntryPoint("q", "original", list(c.functions)[0], [], ENTRY)]
    return c


def test_write_outputs_records_a_spec_for_every_real_output(fake_llm):
    from arc.source import Transcript
    c = _two_output_cache()
    fake_llm.queue(labelfree.OutputSpecs(outputs=[
        labelfree.OutputItem(name="monthly payment in dollars", kind="money", unit="USD", decimals=None),
        labelfree.OutputItem(name="number of payments", kind="count", unit="payments", decimals=3),
        labelfree.OutputItem(name="decision", kind="text", unit="", decimals=None),
        labelfree.OutputItem(name="not a real output", kind="money", unit="USD", decimals=2),
    ]))
    specs = labelfree.write_outputs(c, ENTRY, Transcript(ENTRY, "q", "w", "", "m", 0.0))
    assert specs == {
        "monthly payment in dollars": {"kind": "money", "unit": "USD", "decimals": 2},     # money -> cents
        "number of payments": {"kind": "count", "unit": "payments", "decimals": 0},       # counts exact
        "total interest in dollars": {"kind": "measure", "unit": "", "decimals": None, "inferred": True},
        "decision": {"kind": "text", "unit": "", "decimals": None},
    }
    assert c.outputs[ENTRY] == specs
    prompt = fake_llm.calls[0]["prompt"]
    assert "- monthly payment in dollars: 5053.0678" in prompt and fake_llm.calls[0]["max_tokens"] >= 16000


def test_output_specs_survive_save_load_and_merge(tmp_path):
    c = _two_output_cache()
    c.outputs[ENTRY] = {"monthly payment in dollars": {"kind": "money", "unit": "USD", "decimals": 2}}
    c.save(tmp_path / "e")
    loaded = Cache.load(tmp_path / "e")
    assert loaded.outputs == c.outputs
    master = Cache("master")
    master.merge(loaded)
    assert master.outputs == c.outputs


def test_output_spec_failure_is_a_mismatch(pre, monkeypatch):
    def boom(cache, eid, t, model=None):
        raise _validation_error()
    monkeypatch.setattr(labelfree, "write_outputs", boom)
    pre.answers = ["16"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "mismatch" and res.reason.startswith(labelfree.OUTPUTS_REASON)
    assert not (pre.path / "caches" / ENTRY).exists()


def test_output_specs_are_published_with_the_entry(pre):
    pre.answers = ["16"]
    assert labelfree.build_family(FAM, family(8)).status == "accepted"
    want = {ENTRY: {"result": {"kind": "measure", "unit": "m", "decimals": None}}}
    assert Cache.load(pre.path / "caches" / ENTRY).outputs == want
    assert Cache.load(pre.path / "caches" / "master").outputs == want


def test_cost_cap_default_fits_the_three_step_cascade():
    # owner decisions: $0.60 (2026-11-15, a Sonnet build alone is $0.21-0.26), then $1.20
    # (2026-11-16, Haiku + Sonnet + Opus is ~$1.02 worst case, plus re-solve and checks)
    import os
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "arc" / "config.py").read_text()
    assert 'LABELFREE_MAX_COST = float(os.environ.get("ARC_LABELFREE_MAX_COST", 1.20))' in src
    if "ARC_LABELFREE_MAX_COST" not in os.environ:
        assert config.LABELFREE_MAX_COST == 1.20


def test_a_sonnet_priced_build_is_not_stopped_by_the_cap(pre, monkeypatch):
    import arc.builder
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.60)

    class PricedBuilder(FakeBuilder):
        def build(self):
            llm.LEDGER.add(llm.Record("builder", "claude-sonnet-5-5", 0, 0, 0, 0, 0.25))
            return super().build()
    monkeypatch.setattr(arc.builder, "Builder", PricedBuilder)
    pre.answers = ["16"]
    res = labelfree.build_family(FAM, family(8))
    assert res.status == "accepted" and res.cost > 0.25


# ----------------------------------------------------------------------------- unit hint for the re-solve
# Owner decision 2026-11-15: the re-solve is told the UNIT(S) production answered in, never the
# values, so both answers are in the same unit and the exact check works. Regression: production
# "9.445 atm" vs a re-solve "956988 Pa" was a false mismatch (the strict check never converts atm/Pa).

GAS_Q = "6.25 mol of argon gas is held in a 20.38 dm3 vessel at 97°C. Assuming ideal behaviour, calculate the pressure."
GAS_PROD = "**Given:** n = 6.25 mol, V = 20.38 L, T = 370.15 K\n\nP = nRT/V\n\n**Final answer:** P ≈ 9.315 atm"


@pytest.mark.parametrize("production,hint", [
    (GAS_PROD, "atm"),
    ("**Final answer:** P ≈ 18.6 atm (≈ 1.88 MPa)", "atm, MPa"),
    ("Final answers: monthly payment ≈ **$5,053.27**; total interest ≈ **$95,392**", "$ (dollars)"),
    ("NPV = -£62,164.50 (reject)", "£ (pounds)"),
    ("V ≈ 4.86 L", "L"),
    ("V = 4.860 dm3", "dm3"),
    ("**Answer:** 400 integers", None),            # a count with no unit: no hint
    ("t = 2.00 (df = 35); fail to reject H0", None),
    ("", None),
])
def test_unit_hint_reads_units_from_the_production_answer(production, hint):
    assert labelfree.unit_hint(production) == hint


def test_unit_hint_never_carries_production_numbers():
    for prod in (GAS_PROD, "**$5,053.27** and **$95,392**", "p = 4.73 × 10^5 Pa (473 kPa)", "4.860 dm3"):
        h = labelfree.unit_hint(prod)
        assert h and numbers(h) == []


def test_resolve_is_told_the_unit_and_its_atm_answer_matches(fake_llm, monkeypatch):
    from arc import certify, judges
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique=""))
    prog = ("from decimal import Decimal\nn = Decimal('6.25')\nV = Decimal('20.38')\nT = Decimal('370.15')\n"
            "R = Decimal('0.082057366')\nprint(f'pressure: {n * R * T / V} atm')")
    fake_llm.queue(labelfree.ResolveOut(working="...", program=prog, final_answer="9.445 atm"))
    w = labelfree._resolve(GAS_Q, "cust-model", "t", unit_hint=labelfree.unit_hint(GAS_PROD))
    prompt = fake_llm.calls[0]["prompt"]
    assert "Give the final answer in atm" in prompt and "9.315" not in prompt
    assert certify._compare("t", GAS_Q, GAS_PROD, w.final_answer, "t").agree
    # the same quantity in Pa (943,817 Pa) is what the strict check refused before
    assert not certify._compare("t", GAS_Q, GAS_PROD, "pressure: 943817 Pa", "t").agree


def test_no_hint_means_no_unit_line_in_the_prompt(fake_llm):
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('count: 400')", final_answer="400"))
    labelfree._resolve("How many integers ...?", "cust-model", "t", unit_hint=None)
    assert "Give the final answer in" not in fake_llm.calls[0]["prompt"]


def test_simple_flow_passes_the_hint_from_the_newest_record(sim, monkeypatch):
    seen = []
    monkeypatch.setattr(labelfree, "_resolve", lambda q, model, tag, unit_hint=None: seen.append(unit_hint) or
                        labelfree.Resolved(working="w", final_answer="2 m", executed=True))
    labelfree.build_family(FAM, [rec(0, response="x = 2 m")])
    assert seen == ["m"]
