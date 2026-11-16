"""arc/labelfree.py: certify and build entries from the backlog, no golden label.

Everything model-shaped is faked: serve.use (the gate + extraction), the cached code
(dispatch_root), the comparison judge, blind solves, the explainer and the builder.
The family: "double <x> m". Production answers "<2x>". The cache returns {"result": 2x}.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from arc import config, explain, labelfree, llm, serve
from arc.cache import Cache, EntryPoint, ParamSpec
from arc.checks import numbers

ENTRY = "fam_x"


def rec(i, x=None, response=None, t=None):
    x = i + 1 if x is None else x
    return {"id": f"r{i}", "question": f"double {x} m", "timestamp": i if t is None else t,
            "response": f"{2 * x}" if response is None else response}


def recs(n, start=0):
    return [rec(i) for i in range(start, start + n)]


def make_cache():
    c = Cache(ENTRY)
    c.entry_points = [EntryPoint("double 1 m", "original", "root", [ParamSpec("x")], ENTRY)]
    return c


def x_in_metres(value: str) -> float:
    v = float(value.split()[0])
    return v / 100 if "cm" in value else v


class World:
    """Configurable fakes. Each knob changes one behaviour."""

    def __init__(self, monkeypatch):
        self.turn_away = set()      # questions the gate rejects
        self.crash = set()          # questions the cached code crashes on
        self.cache_wrong_on = set()  # questions where the cache's answer is wrong
        self.blind = {}             # question -> list of blind answers (else the true answer)
        self.unit_bug = False       # cached code ignores units (metamorphic violation)
        self.solves = 0
        self.forced = []            # entry_point passed to each use() call
        self.sim = {}               # question -> IS1 cosine (default 0.95)
        monkeypatch.setattr(serve, "use", self.use)
        monkeypatch.setattr(serve, "dispatch_root", self.dispatch)
        monkeypatch.setattr(labelfree, "_compare", self.compare)
        monkeypatch.setattr(explain, "_blind_solve", self.blind_solve)
        monkeypatch.setattr(labelfree, "_program_solve",
                            lambda q, tag, conventions=None: (self.blind_solve(q, tag).final_answer, ""))

    def x_of(self, q):
        return float(re.search(r"double (\S+)", q).group(1))

    def dispatch(self, cache, root_id, values):
        x = float(values["x"].split()[0]) if self.unit_bug else x_in_metres(values["x"])
        return {"result": 2 * x}, None

    def use(self, cache, question, tag=None, trace=None, shadow=False, entry_point=None):
        trace = trace if trace is not None else {}
        self.forced.append(entry_point)
        trace["similarity"] = self.sim.get(question, 0.95)
        if entry_point is None and trace["similarity"] < config.ENTRY_THRESHOLD:
            trace["stage"] = "below_threshold"
            return None
        if question in self.turn_away:
            trace.update(stage="gate_rejected", rejected_by=["asked_for"])
            return None
        if question in self.crash:
            raise serve.CacheInvocationError("boom")
        values = {"x": f"{self.x_of(question):g} m"}
        out, _ = self.dispatch(cache, "root", values)
        if question in self.cache_wrong_on:
            out = {"result": out["result"] + 1}
        trace["stage"] = "dispatch"
        return serve.CacheResult(out, "", cache.entry_points[0], 0.95, values, 0.0)

    def compare(self, name, question, reference, candidate, tag):
        agree = numbers(reference)[:1] == numbers(candidate)[:1]
        return SimpleNamespace(agree=agree)

    def blind_solve(self, question, tag):
        self.solves += 1
        queue = self.blind.get(question)
        ans = queue.pop(0) if queue else f"{2 * self.x_of(question):g}"
        return explain.Worked(working="...", final_answer=ans)


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MIN_CHECKED", 10)
    monkeypatch.setattr(config, "LABELFREE_MAX_CHECKED", 40)
    monkeypatch.setattr(config, "LABELFREE_RELATIONS", 6)
    monkeypatch.setattr(config, "LABELFREE_FAIL_FAST", True)
    return World(monkeypatch)


# ----------------------------------------------------------------------------- helpers

def test_error_upper_bound_rule_of_three():
    assert labelfree.error_upper_bound(0) == 1.0
    assert abs(labelfree.error_upper_bound(10) - 0.259) < 0.001
    assert labelfree.error_upper_bound(29) < 0.10 < labelfree.error_upper_bound(28)


def test_usable_dedupes_drops_empty_and_sorts_by_time():
    rs = [rec(2, x=5, t=20), rec(1, x=5, t=10), {**rec(3), "response": "  "}, rec(4, x=9, t=5)]
    out = labelfree.usable(rs)
    assert [r["id"] for r in out] == ["r4", "r1"]   # oldest copy of "double 5 m" kept


# ----------------------------------------------------------------------------- adjudication

def test_adjudicate_blind_sides_with_production(world):
    world.blind["double 3 m"] = ["6"]
    outcome, answers, _ = labelfree.adjudicate("double 3 m", "6", "result: 7", "t")
    assert outcome == "cache_wrong" and answers == ["6"] and world.solves == 1


def test_adjudicate_clears_cache_only_if_every_solve_agrees(world):
    world.blind["double 3 m"] = ["6", "6"]
    outcome, _, detail = labelfree.adjudicate("double 3 m", "9", "result: 6", "t")
    assert outcome == "cache_right" and world.solves == 2 and "production looks wrong" in detail


def test_adjudicate_split_vote_is_unresolved(world):
    world.blind["double 3 m"] = ["6", "9"]
    outcome, _, _ = labelfree.adjudicate("double 3 m", "9", "result: 6", "t")
    assert outcome == "unresolved"


def test_adjudicate_solve_agreeing_with_neither_is_unresolved(world):
    world.blind["double 3 m"] = ["42"]
    outcome, _, _ = labelfree.adjudicate("double 3 m", "9", "result: 6", "t")
    assert outcome == "unresolved" and world.solves == 1


# ----------------------------------------------------------------------------- certify

def test_all_agree_is_accepted_with_a_bound(world):
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(12))
    s = cert.summary()
    assert cert.status == "accepted" and s["verified"] == 12 and s["cache_wrong"] == 0
    assert s["relations_held"] == 6 and s["error_upper_bound"] == pytest.approx(labelfree.error_upper_bound(12))
    assert s["serve_rate"] == 1.0


def test_too_few_verified_is_need_more_data(world):
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(6))
    assert cert.status == "need_more_data" and "only 6" in cert.reason


def test_production_mistake_does_not_count_against_cache(world):
    rs = recs(12)
    rs[3] = rec(3, response="999")          # production got this one wrong
    world.blind[rs[3]["question"]] = ["8", "8"]
    cert = labelfree.certify(make_cache(), ENTRY, "x", rs)
    s = cert.summary()
    assert cert.status == "accepted" and s["production_errors_found"] == 1 and s["verified"] == 12


def test_confirmed_wrong_answer_rejects_and_stops(world):
    rs = recs(12)
    world.cache_wrong_on.add(rs[2]["question"])
    cert = labelfree.certify(make_cache(), ENTRY, "x", rs)
    assert cert.status == "rejected" and "1 wrong" in cert.reason and "r2" in cert.reason
    assert len(cert.checks) == 3            # fail fast


def test_without_fail_fast_every_question_is_checked(world):
    rs = recs(12)
    world.cache_wrong_on.add(rs[2]["question"])
    cert = labelfree.certify(make_cache(), ENTRY, "x", rs, fail_fast=False)
    assert cert.status == "rejected" and len(cert.checks) == 12


def test_unresolved_disagreement_rejects(world):
    rs = recs(12)
    rs[1] = rec(1, response="999")
    world.blind[rs[1]["question"]] = ["123"]   # blind agrees with neither
    cert = labelfree.certify(make_cache(), ENTRY, "x", rs)
    assert cert.status == "rejected" and "1 unresolved" in cert.reason


def test_unit_dropping_code_is_rejected_by_a_relation(world):
    world.unit_bug = True
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(12))
    assert cert.status == "rejected" and "metamorphic relation violated" in cert.reason
    assert "cm" in cert.reason


def test_gate_turned_away_and_crashes_are_safe_but_lower_serve_rate(world):
    rs = recs(14)
    world.turn_away.update(r["question"] for r in rs[:2])
    world.crash.add(rs[2]["question"])
    cert = labelfree.certify(make_cache(), ENTRY, "x", rs)
    s = cert.summary()
    assert cert.status == "accepted" and s["not_served"] == 2 and s["crashed"] == 1
    assert s["verified"] == 11 and s["serve_rate"] == pytest.approx(11 / 14)


def test_max_checked_caps_cost(world, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_CHECKED", 11)
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(30))
    assert len(cert.checks) == 11 and cert.status == "accepted"


def test_prior_checks_accumulate(world):
    first = labelfree.certify(make_cache(), ENTRY, "x", recs(6))
    assert first.status == "need_more_data"
    second = labelfree.certify(make_cache(), ENTRY, "x", recs(5, start=6), prior=first.checks)
    assert second.status == "accepted" and second.summary()["verified"] == 11


# ----------------------------------------------------------------------------- build_family

class FakeBuilder:
    built = 0

    def __init__(self, problem, transcript, use_golden=True, units="paper"):
        assert use_golden is False and problem["golden_label"] == ""
        assert units == "strict"            # label-free builds always use the strict unit rules
        self.problem = problem

    def build(self):
        FakeBuilder.built += 1
        c = Cache(self.problem["id"])
        c.entry_points = [EntryPoint(self.problem["question"], "original", "root", [ParamSpec("x")],
                                     self.problem["id"])]
        return c


@pytest.fixture
def lab(world, monkeypatch, tmp_path):
    import arc.builder
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "caches")
    monkeypatch.setattr(config, "cache_dir", lambda name: tmp_path / "caches" / name)
    monkeypatch.setattr(config, "GATE", "off")
    monkeypatch.setattr(config, "LABELFREE_CERTIFY", True)      # these tests cover the certified flow
    monkeypatch.setattr(arc.builder, "Builder", FakeBuilder)
    monkeypatch.setattr(labelfree, "typical_first", lambda rs: list(rs))
    FakeBuilder.built = 0
    lab = SimpleNamespace(world=world, path=tmp_path, explained=[], refuse=set())

    def fake_explain(record, blind_tries=3):
        lab.explained.append(record["id"])
        llm.LEDGER.add(llm.Record("explainer", "claude-haiku-5-5", 0, 0, 0, 0, 0.001))
        if record["id"] in lab.refuse:
            return explain.Explanation(record["id"], record["question"], record["response"],
                                       decision="not_explained", reason="blind solves disagree")
        t = {"problem_id": record["id"], "question": record["question"], "solution": "2x",
             "thinking_summary": "", "model": "claude-haiku-5-5", "cost": 0.0}
        return explain.Explanation(record["id"], record["question"], record["response"],
                                   decision="explained", transcript=t)

    monkeypatch.setattr(explain, "explain", fake_explain)
    return lab


def test_too_few_records_returns_need_more_data_for_free(lab):
    res = labelfree.build_family("x", recs(10))
    assert res.status == "need_more_data" and res.cost == 0.0 and lab.explained == []
    assert res.accepted is False and res.entry_id is None


def test_accepted_build_is_published_to_entry_and_master(lab):
    res = labelfree.build_family("x", recs(15))
    assert res.status == "accepted" and res.accepted and res.entry_id == ENTRY
    assert res.certificate["verified"] == 14 and res.serve_rate == 1.0
    assert res.cost == pytest.approx(0.001) and res.parts["explain"] == pytest.approx(0.001)
    assert (lab.path / "caches" / ENTRY / "registry.json").exists()
    master = Cache.load(lab.path / "caches" / "master")
    assert {e.problem_id for e in master.entry_points} == {ENTRY}
    state = json.loads((lab.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())
    assert state["status"] == "accepted" and state["build_ids"] == ["r0"]


def test_build_question_is_never_used_to_certify(lab):
    res = labelfree.build_family("x", recs(15))
    state = json.loads((lab.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())
    assert "r0" not in {c["record_id"] for c in state["checks"]} and res.accepted


def test_explainer_refusals_try_next_records_then_reject(lab):
    lab.refuse.update({"r0", "r1", "r2"})
    res = labelfree.build_family("x", recs(15))
    assert res.status == "rejected" and lab.explained == ["r0", "r1", "r2"]
    assert "explainer" in res.reason and FakeBuilder.built == 0 and res.cost == pytest.approx(0.003)


def test_second_record_is_used_when_first_cannot_be_explained(lab):
    lab.refuse.add("r0")
    res = labelfree.build_family("x", recs(15))
    assert res.accepted and lab.explained == ["r0", "r1"]


def test_pending_entry_is_recertified_without_rebuilding(lab, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_CHECKED", 6)   # first pass can't reach 10
    first = labelfree.build_family("x", recs(15))
    assert first.status == "need_more_data" and FakeBuilder.built == 1
    assert (lab.path / "caches" / "pending" / ENTRY / "registry.json").exists()
    assert not (lab.path / "caches" / "master").exists()
    monkeypatch.setattr(config, "LABELFREE_MAX_CHECKED", 40)
    second = labelfree.build_family("x", recs(20))
    assert second.status == "accepted" and FakeBuilder.built == 1 and len(lab.explained) == 1
    state = json.loads((lab.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())
    ids = [c["record_id"] for c in state["checks"]]
    assert len(ids) == len(set(ids)) == 19          # no question checked twice, build question excluded


def test_rejected_build_publishes_nothing(lab):
    lab.world.cache_wrong_on.add("double 4 m")
    res = labelfree.build_family("x", recs(15))
    assert res.status == "rejected" and not res.accepted and res.entry_id is None
    assert not (lab.path / "caches" / "master").exists()


def test_result_matches_build_ready_contract(lab):
    res = labelfree.build_family("x", recs(15))
    for attr in ("accepted", "entry_id", "reason", "cost", "parts", "certificate", "serve_rate", "status"):
        assert hasattr(res, attr)
    assert res.status in labelfree.STATUSES


# ----------------------------------------------------------------------------- audit after launch

def test_audit_agree(world):
    assert labelfree.audit_hit("double 3 m", "result: 6", "6", "t")["outcome"] == "agree"


def test_audit_catches_wrong_cache_answer(world):
    out = labelfree.audit_hit("double 3 m", "result: 7", "6", "t")
    assert out["outcome"] == "cache_wrong" and labelfree.audit_should_disable(out["outcome"])


def test_budget_stops_certification_and_decides_on_what_was_checked(world, monkeypatch):
    spend = {"n": 0}
    real_compare = world.compare

    def costly_compare(*a, **k):
        spend["n"] += 1
        llm.LEDGER.add(llm.Record("cert_judge", "claude-haiku-5-5", 0, 0, 0, 0, 0.01))
        return real_compare(*a, **k)
    monkeypatch.setattr(labelfree, "_compare", costly_compare)
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(30), budget=0.05)
    assert len(cert.checks) == 5 and cert.status == "need_more_data"


def test_build_family_respects_the_total_cap(lab, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.0005)   # explain alone costs 0.001
    res = labelfree.build_family("x", recs(15))
    assert res.status == "need_more_data"
    state = json.loads((lab.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())
    assert state["checks"] == []


# ----------------------------------------------------------------------------- family index

def test_certify_forces_the_entry_and_records_the_cosine(world):
    world.sim = {f"double {x} m": 0.70 for x in range(1, 6)}       # IS1 would miss these
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(12))
    assert all(ep is not None and ep.problem_id == ENTRY for ep in world.forced)
    assert cert.status == "accepted" and cert.summary()["verified"] == 12
    assert cert.summary()["verified_below_threshold"] == 5
    assert cert.checks[0]["similarity"] == 0.70


def test_flag_off_keeps_the_old_search_path(world, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_FAMILY_INDEX", False)
    world.sim = {f"double {x} m": 0.70 for x in range(1, 6)}
    cert = labelfree.certify(make_cache(), ENTRY, "x", recs(12))
    assert world.forced == [None] * 12
    assert cert.status == "need_more_data" and cert.summary()["not_served"] == 5


def _checks(*rows):
    return [{"question": q, "outcome": o, "similarity": s, "relations": rel or []} for q, o, s, rel in rows]


def test_index_family_adds_only_verified_questions_lowest_cosine_first(monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_INDEX_MAX", 2)
    c = make_cache()
    c.cards[ENTRY] = {"index_texts": ["double 9 m"], "intent": "double x"}
    added = labelfree.index_family(c, ENTRY, _checks(
        ("double 2 m", "agree", 0.90, None),
        ("double 3 m", "cache_right", 0.70, None),
        ("triple 4 m", "not_served", 0.60, None),                   # near-miss: never indexed
        ("double 5 m", "cache_wrong", 0.50, None),
        ("double 6 m", "agree", 0.40, [{"status": "violated"}]),   # broke a relation
        ("double 7 m", "agree", 0.80, None),
        ("double 1 m", "agree", 0.10, None),                       # already the original
    ))
    assert added == ["double 3 m", "double 7 m"]
    fam = [e for e in c.entry_points if e.kind == "family"]
    assert [e.text for e in fam] == added
    assert all(e.problem_id == ENTRY and e.root_id == "root" and e.params == c.entry_points[0].params for e in fam)
    assert c.cards[ENTRY]["index_texts"] == ["double 9 m", "double 3 m", "double 7 m"]


def test_index_family_without_an_original_does_nothing():
    c = Cache(ENTRY)
    assert labelfree.index_family(c, ENTRY, _checks(("double 2 m", "agree", 0.5, None))) == []
    assert c.entry_points == []


def test_accepted_build_publishes_verified_questions_as_entry_points(lab, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_INDEX_MAX", 20)
    res = labelfree.build_family("x", recs(15))
    assert res.status == "accepted"
    for folder in (lab.path / "caches" / ENTRY, lab.path / "caches" / "master"):
        fam = [e for e in Cache.load(folder).entry_points if e.kind == "family"]
        assert len(fam) == 14 and {e.problem_id for e in fam} == {ENTRY}
        assert "double 1 m" not in {e.text for e in fam}           # the build question stays "original"
    state = json.loads((lab.path / "runs" / "labelfree" / ENTRY / "state.json").read_text())
    assert state["indexed"] == 14


def test_rejected_build_indexes_nothing(lab):
    lab.world.cache_wrong_on = {"double 30 m"}
    res = labelfree.build_family("y", [rec(i, x=i + 20) for i in range(15)])
    assert res.status == "rejected"
    assert not (lab.path / "caches" / "fam_y").exists()


def test_flag_off_publishes_no_family_entry_points(lab, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_FAMILY_INDEX", False)
    labelfree.build_family("x", recs(15))
    assert [e.kind for e in Cache.load(lab.path / "caches" / ENTRY).entry_points] == ["original"]
