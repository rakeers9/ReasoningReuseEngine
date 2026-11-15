"""Cache-worthiness meter: break-even per family, any production model (arc/families.py)."""

import argparse
import json

import numpy as np
import pytest

from arc import backlog, config, families

DAY = 86400.0
from pathlib import Path as _P
ROOT_DIR = _P(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Keep families.json / build_costs.jsonl out of runs/, and pin the settings under test."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.setattr(config, "FAMILY_SIMILARITY", 0.75)
    monkeypatch.setattr(config, "FAMILY_CHECK", "embed")   # Clef-mode tests set "clef" themselves
    monkeypatch.setattr(config, "FAMILY_HALF_LIFE_DAYS", 30.0)
    monkeypatch.setattr(config, "FAMILY_MIN_SIGHTINGS", 3.0)
    monkeypatch.setattr(config, "CACHE_ANSWER_COST", 0.0001)
    monkeypatch.setattr(config, "SERVE_RATE_DEFAULT", 1.0)


_WORDS: dict[str, int] = {}


def fake_embed(texts):
    """The first word picks the direction: 'mortgage ...' questions are one family."""
    out = np.zeros((len(texts), 32), np.float32)
    for i, t in enumerate(texts):
        out[i, _WORDS.setdefault(t.split()[0], len(_WORDS) % 32)] = 1.0
    return out


def tracker(tmp_path, name="fam.json"):
    return families.FamilyTracker(path=tmp_path / name, embed_fn=fake_embed)


def rec(q, model="claude-haiku-5-5", tin=200, tout=1500, t=0.0, **kw):
    return backlog.make_record(q, "answer", model, input_tokens=tin, output_tokens=tout, timestamp=t, **kw)


def questions_until_build(tmp_path, model, build_cost, limit=500):
    tr = tracker(tmp_path, f"{model}.json")
    for i in range(limit):
        tr.observe(rec(f"mortgage payment {i}", model=model, t=i * 60.0))
        if tr.readiness(build_cost=build_cost)[0].verdict == "build":
            return i + 1
    return None


# ---- the threshold is not a fixed count ----

def test_number_of_questions_to_break_even_depends_on_the_production_model(tmp_path):
    # 200 in / 1500 out tokens. Saving per question = answer cost - $0.0001.
    # Haiku $0.00077 -> 0.058 / 0.00067 = 86.6 -> 87; Sonnet $0.0154 -> 3.8 -> 4;
    # Opus $0.0308 -> 1.9, Fable $0.0616 -> 0.9: money says 2 and 1, the 3-sighting floor wins.
    need = {m: questions_until_build(tmp_path, m, 0.058)
            for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2")}
    assert need == {"claude-haiku-5-5": 87, "claude-sonnet-5-5": 4, "claude-opus-5-5": 3, "claude-fable-5-2": 3}


def test_longer_answers_on_the_same_model_reach_break_even_sooner(tmp_path):
    tr = tracker(tmp_path)
    for i in range(10):
        tr.observe(rec(f"short {i}", tout=500, t=float(i)))
        tr.observe(rec(f"long {i}", tout=5000, t=float(i)))
    rows = {r.label.split()[0]: r for r in tr.readiness(build_cost=0.034)}
    assert rows["long"].questions_to_go < rows["short"].questions_to_go


def test_min_sightings_floor_blocks_a_build_that_already_pays(tmp_path):
    tr = tracker(tmp_path)
    tr.observe(rec("contract deadline", model="claude-fable-5-2", tout=20000))
    tr.observe(rec("contract deadline again", model="claude-fable-5-2", tout=20000))
    row = tr.readiness(build_cost=0.058)[0]
    assert row.meter >= row.threshold and row.verdict == "wait" and "recurs" in row.why
    tr.observe(rec("contract deadline third", model="claude-fable-5-2", tout=20000))
    assert tr.readiness(build_cost=0.058)[0].verdict == "build"


def test_cheaper_cache_answer_means_more_saving(tmp_path, monkeypatch):
    tr = tracker(tmp_path)
    tr.observe(rec("loan a"))
    before = tr.readiness(build_cost=1)[0].saving_per_question
    monkeypatch.setattr(config, "CACHE_ANSWER_COST", 0.0)
    assert tr.readiness(build_cost=1)[0].saving_per_question > before


def test_cache_dearer_than_production_never_adds_negative_saving(tmp_path):
    tr = tracker(tmp_path)
    tr.observe(rec("tiny", tin=1, tout=1))
    row = tr.readiness(build_cost=1)[0]
    assert row.meter == 0.0 and row.verdict == "wait" and "saves nothing" in row.why


# ---- recency ----

def test_old_popularity_decays_away(tmp_path):
    tr = tracker(tmp_path)
    for i in range(60):
        tr.observe(rec(f"gas pressure {i}", t=i * 60.0))
    assert tr.readiness(now=3600.0, build_cost=0.034)[0].verdict == "build"
    assert tr.readiness(now=365 * DAY, build_cost=0.034)[0].verdict == "wait"


def test_half_life_halves_and_late_records_count_less(tmp_path):
    tr = tracker(tmp_path)
    tr.observe(rec("gas a", t=100 * DAY))
    tr.observe(rec("gas b", t=70 * DAY))           # arrives late, 30 days old
    assert tr.families["f0001"].sightings == pytest.approx(1.5)
    assert tr.readiness(now=130 * DAY, build_cost=1)[0].sightings == pytest.approx(0.75)


def test_report_defaults_to_the_newest_record_time(tmp_path):
    tr = tracker(tmp_path)
    for i in range(3):
        tr.observe(rec(f"fx {i}", model="claude-opus-5-5", t=float(i)))   # timestamps near 1970
    assert tr.readiness(build_cost=0.034)[0].verdict == "build"


# ---- pricing gaps and serve rate ----

def test_unpriced_model_counts_sightings_but_no_saving(tmp_path):
    tr = tracker(tmp_path)
    for i in range(5):
        tr.observe(backlog.make_record(f"tax bracket {i}", "a", "in-house-llm", timestamp=float(i)))
    row = tr.readiness(build_cost=0.034)[0]
    assert row.unpriced == 5 and row.meter == 0 and row.sightings == pytest.approx(5, abs=1e-3)
    assert row.verdict == "wait" and "in-house-llm" in row.why


def test_record_cost_field_prices_any_provider(tmp_path):
    tr = tracker(tmp_path)
    for i in range(3):
        tr.observe(backlog.make_record(f"refund policy {i}", "a", "gpt-x", cost=0.05, timestamp=float(i)))
    assert tr.readiness(build_cost=0.058)[0].verdict == "build"


def test_serve_rate_scales_the_meter(tmp_path):
    tr = tracker(tmp_path)
    for i in range(100):
        tr.observe(rec(f"loan {i}", t=float(i)))
    full = tr.readiness(build_cost=0.034)[0].meter
    tr.set_serve_rate("f0001", 0.5)
    assert tr.readiness(build_cost=0.034)[0].meter == pytest.approx(full / 2, rel=1e-4)
    tr.set_serve_rate("f0001", 7)
    assert tr.families["f0001"].serve_rate == 1.0


# ---- grouping ----

def test_numbers_are_masked_before_grouping():
    assert families.mask_numbers("10.0 mol at 27°C, 1,200 kg, 3e-5 m, -4") == "● mol at ●°C, ● kg, ● m, ●"


def test_close_questions_share_a_family_and_far_ones_do_not(tmp_path):
    tr = tracker(tmp_path)
    a, b, c = (tr.observe(rec(q)) for q in ("mortgage x", "mortgage y", "dosage z"))
    assert a.id == b.id != c.id and a.size == 2


def test_caller_given_family_skips_embedding(tmp_path):
    tr = families.FamilyTracker(path=tmp_path / "f.json", embed_fn=lambda t: pytest.fail("embedded"))
    f = tr.observe(rec("anything", family="billing"))
    assert f.id == "billing" and tr.observe(rec("other", family="billing")).size == 2


def test_duplicate_record_ids_are_counted_once(tmp_path):
    tr = tracker(tmp_path)
    r = rec("mortgage x")
    assert tr.observe(r) is not None and tr.observe(r) is None
    assert tr.families["f0001"].size == 1


# ---- after building: usage, complaints, staleness, rejection ----

def test_cache_answers_count_as_hits_not_savings(tmp_path):
    tr = tracker(tmp_path)
    for i in range(80):
        tr.observe(rec(f"bayes {i}", t=float(i)))
    tr.mark_built("f0001", "bayes_ppv")
    meter = tr.readiness()[0].meter
    f = tr.observe(rec("bayes hit", t=100.0, served_by="bayes_ppv"))
    assert f.id == "f0001" and f.cache_hits_total == 1
    row = tr.readiness()[0]
    assert row.verdict == "built" and row.meter == pytest.approx(meter, rel=1e-4)


def test_negative_feedback_on_cache_answers_flags_complaints(tmp_path):
    tr = tracker(tmp_path)
    tr.observe(rec("bayes", t=0.0))
    tr.mark_built("f0001", "bayes_ppv")
    for i in range(10):
        tr.observe(rec(f"bayes hit {i}", t=1.0 + i, served_by="bayes_ppv",
                       feedback="thumbs_down" if i < 3 else "thumbs_up"))
    assert tr.readiness()[0].verdict == "complaints"


def test_unused_entry_goes_stale(tmp_path):
    tr = tracker(tmp_path)
    tr.observe(rec("fx rate", t=0.0, served_by="fx_entry"))
    assert tr.families["entry:fx_entry"].status == "built"
    assert tr.readiness(now=10 * DAY)[0].verdict == "built"
    assert tr.readiness(now=200 * DAY)[0].verdict == "stale"


def test_rejected_family_stops_triggering(tmp_path):
    tr = tracker(tmp_path)
    for i in range(100):
        tr.observe(rec(f"drug dose {i}", t=float(i)))
    tr.mark_rejected("f0001", "certificate disagreed")
    row = tr.readiness(build_cost=0.034)[0]
    assert row.verdict == "rejected" and "certificate" in row.why


def test_report_puts_build_first(tmp_path):
    tr = tracker(tmp_path)
    tr.observe(rec("once only"))
    for i in range(3):
        tr.observe(rec(f"popular {i}", model="claude-opus-5-5", t=float(i)))
    rows = tr.readiness(build_cost=0.034)
    assert [r.verdict for r in rows] == ["build", "wait"]
    text = families.format_report(rows, families.effective_build_cost())
    assert "build: 1" in text and "popular" in text


# ---- persistence and incremental ingest ----

def test_ingest_reads_only_new_lines_and_survives_reload(tmp_path):
    path = tmp_path / "backlog.jsonl"
    for i in range(3):
        backlog.log_interaction(f"interest rate {i}", "a", "claude-opus-5-5", path=path,
                                input_tokens=200, output_tokens=1500, timestamp=float(i), id=f"r{i}")
    tr = tracker(tmp_path, "persist.json")
    assert tr.ingest(path) == 3
    tr.mark_built("f0001", "interest_entry")
    tr.save()
    backlog.log_interaction("interest rate 9", "a", "claude-opus-5-5", path=path, timestamp=9.0, id="r9")
    with open(path, "a") as f:
        f.write("not json\n")
    tr = tracker(tmp_path, "persist.json")
    assert tr.families["f0001"].status == "built"
    assert tr.ingest(path) == 1 and tr.ingest(path) == 0
    assert tr.families["f0001"].size == 4


def test_members_of_returns_every_record_in_backlog_order(tmp_path):
    path = tmp_path / "backlog.jsonl"
    for i in range(5):
        backlog.log_interaction(f"mortgage {i}", f"a{i}", "claude-opus-5-5", path=path, timestamp=float(i), id=f"m{i}")
        backlog.log_interaction(f"dosage {i}", "a", "claude-opus-5-5", path=path, timestamp=float(i), id=f"d{i}")
    tr = tracker(tmp_path)
    tr.ingest(path)
    fid = tr.observe(rec("mortgage probe", id="probe")).id   # not in the backlog file: not returned
    tr.mark_built(fid, "mortgage_entry")
    backlog.log_interaction("mortgage hit", "a", "claude-opus-5-5", path=path, timestamp=9.0, id="mh",
                            served_by="mortgage_entry")   # the cache can only answer after the build
    tr.ingest(path)
    assert [r["id"] for r in tr.members_of(fid, path)] == ["m0", "m1", "m2", "m3", "m4", "mh"]
    assert [r["response"] for r in tr.members_of(fid, path, served=False)] == ["a0", "a1", "a2", "a3", "a4"]
    assert [r["id"] for r in tr.members_of(fid, path, served=True)] == ["mh"]


def test_families_saved_before_member_ids_still_load(tmp_path):
    tr = tracker(tmp_path, "old.json")
    tr.observe(rec("mortgage x"))
    tr.save()
    raw = json.loads((tmp_path / "old.json").read_text())
    for f in raw["families"]:
        del f["member_ids"]
    (tmp_path / "old.json").write_text(json.dumps(raw))
    assert tracker(tmp_path, "old.json").families["f0001"].member_ids == []


def test_ingest_of_missing_backlog_is_a_no_op(tmp_path):
    assert tracker(tmp_path).ingest(tmp_path / "none.jsonl") == 0


# ---- measured build cost ----

def test_build_cost_uses_measured_defaults_until_three_builds():
    e = families.effective_build_cost()
    assert e["effective"] == pytest.approx(config.BUILD_COST_DEFAULT / config.BUILD_ACCEPT_RATE_DEFAULT)
    assert "defaults" in e["source"]


def test_build_cost_is_spend_per_accepted_entry():
    for pid, c, ok in [("a", 0.03, True), ("b", 0.04, False), ("c", 0.02, True), ("d", 0.03, True)]:
        families.record_build_cost(pid, c, ok)
    e = families.effective_build_cost()
    assert e["per_attempt"] == pytest.approx(0.03) and e["accept_rate"] == 0.75
    assert e["effective"] == pytest.approx(0.04) and "measured over 4" in e["source"]


def test_all_builds_rejected_falls_back_to_defaults():
    for pid in "abc":
        families.record_build_cost(pid, 0.03, False)
    assert "defaults" in families.effective_build_cost()["source"]


# ---- CLI wiring ----

def test_cli_families_reports_from_a_backlog(tmp_path, monkeypatch, capsys):
    from arc import __main__ as cli
    path = tmp_path / "b.jsonl"
    for i in range(3):
        backlog.log_interaction(f"popular {i}", "a", "claude-opus-5-5", path=path, input_tokens=200,
                                output_tokens=1500, timestamp=float(i))
    monkeypatch.setattr(families.FamilyTracker, "_embed", lambda self, texts: fake_embed(texts))
    args = argparse.Namespace(backlog=str(path), reset=False, now=None, limit=10, json=True)
    cli.cmd_families(args)
    out = json.loads(capsys.readouterr().out)
    assert out["families"][0]["verdict"] == "build" and (tmp_path / "families.json").exists()
    cli.cmd_families(argparse.Namespace(**{**vars(args), "json": False}))
    assert "Read 0 new records" in capsys.readouterr().out
    cli.cmd_families(argparse.Namespace(**{**vars(args), "json": False, "reset": True}))
    assert "Read 3 new records" in capsys.readouterr().out


def test_build_command_logs_its_measured_cost(tmp_path, monkeypatch):
    from arc import __main__ as cli, llm
    monkeypatch.setattr(llm, "LEDGER", llm.Ledger())
    llm.LEDGER.add(llm.Record("builder", "claude-haiku-5-5", 10, 10, 0, 0, 0.02))
    cli._record_build("gas_law", True)
    row = json.loads((tmp_path / "build_costs.jsonl").read_text())
    assert row["problem"] == "gas_law" and row["cost"] == 0.02 and row["parts"] == {"builder": 0.02}


# ---- real embeddings (local model) ----

@pytest.mark.models
def test_real_embeddings_group_variants_and_split_unrelated_tasks(tmp_path):
    tr = families.FamilyTracker(path=tmp_path / "real.json")
    a = tr.observe(rec("Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure "
                       "exerted by the ethane from the perfect gas equation."))
    b = tr.observe(rec("Suppose that 3.5 mol CO2(g) is confined to 12.0 dm3 at 80°C. Predict the pressure "
                       "exerted by the carbon dioxide from the perfect gas equation."))
    c = tr.observe(rec("A patient is 70 years old, weighs 60 kg and has serum creatinine 1.2 mg/dL. Estimate "
                       "creatinine clearance with Cockcroft-Gault."))
    assert a.id == b.id != c.id


# ---- build orchestration (stub builder: no API) ----

class _Result:
    def __init__(self, accepted, entry_id=None, reason="", cost=0.03, parts=None, serve_rate=None):
        self.accepted, self.entry_id, self.reason, self.cost = accepted, entry_id, reason, cost
        self.parts, self.serve_rate = parts or {"build": cost}, serve_rate


def _ready_tracker(tmp_path):
    path = tmp_path / "backlog.jsonl"
    for word in ("mortgage", "dosage", "seating"):
        for i in range(3):
            backlog.log_interaction(f"{word} {i}", f"{word}-a{i}", "claude-opus-5-5", path=path,
                                    input_tokens=200, output_tokens=1500, timestamp=float(i), id=f"{word}{i}")
    backlog.log_interaction("rare thing", "a", "claude-haiku-5-5", path=path, timestamp=5.0, id="rare")
    tr = tracker(tmp_path)
    tr.ingest(path)
    return tr, path


def test_build_ready_applies_each_result_and_logs_cost(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    seen = {}

    def stub(fid, records):
        word = records[0]["question"].split()[0]
        seen[word] = [r["response"] for r in records]
        if word == "mortgage":
            return _Result(True, "mortgage_entry", serve_rate=0.8)
        if word == "dosage":
            return _Result(False, reason="certificate: 2 of 6 disagreed")
        raise RuntimeError("builder crashed")

    done = families.build_ready(tr, stub, path=path)
    assert len(done) == 3 and "rare" not in seen                       # only "build" verdicts
    assert seen["mortgage"] == ["mortgage-a0", "mortgage-a1", "mortgage-a2"]
    by_word = {f.label.split()[0]: f for f in tr.families.values()}
    assert by_word["mortgage"].status == "built" and by_word["mortgage"].built_entry == "mortgage_entry"
    assert by_word["mortgage"].serve_rate == 0.8
    assert by_word["dosage"].status == "rejected" and "disagreed" in by_word["dosage"].note
    assert by_word["seating"].status == "rejected" and "crashed" in by_word["seating"].note
    logged = [json.loads(l) for l in (tmp_path / "build_costs.jsonl").read_text().splitlines()]
    assert [(r["problem"], r["accepted"]) for r in logged] == [("mortgage_entry", True), (by_word["dosage"].id, False)]
    reloaded = tracker(tmp_path)                                      # saved after each build
    assert {f.status for f in reloaded.families.values()} >= {"built", "rejected"}
    assert families.build_ready(reloaded, stub, path=path) == []      # nothing left to build


def test_build_ready_respects_max_builds(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    calls = []
    families.build_ready(tr, lambda fid, recs: calls.append(fid) or _Result(True, f"e_{fid}"), max_builds=1,
                         path=path)
    assert len(calls) == 1 and sum(f.status == "built" for f in tr.families.values()) == 1


def test_need_more_data_leaves_the_family_watching(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    r = _Result(False, "fam_x", reason="4 of 10 certification questions", cost=0.02)
    r.status = "need_more_data"
    free = _Result(False, None, reason="fewer records than the minimum", cost=0.0)
    free.status = "need_more_data"
    results = iter([r, free, free])
    families.build_ready(tr, lambda fid, recs: next(results), path=path)
    assert {f.status for f in tr.families.values()} == {"watching"}
    assert sum("pending" in f.note for f in tr.families.values()) == 3
    logged = [json.loads(l) for l in (tmp_path / "build_costs.jsonl").read_text().splitlines()]
    assert len(logged) == 1 and logged[0]["pending"] and not logged[0]["accepted"]   # zero-cost not logged
    assert [x.verdict for x in tr.readiness()].count("build") == 3                   # retried next time


def test_explicit_status_wins_over_accepted_flag(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    r = _Result(True, "fam_x")
    r.status = "rejected"
    families.build_ready(tr, lambda fid, recs: r, max_builds=1, path=path)
    assert sum(f.status == "rejected" for f in tr.families.values()) == 1


def test_pending_spend_counts_as_cost_but_not_as_an_attempt():
    families.record_build_cost("a", 0.03, True)
    families.record_build_cost("b", 0.03, False)
    families.record_build_cost("c", 0.03, True)
    families.record_build_cost("c", 0.03, False, pending=True)
    e = families.effective_build_cost()
    assert e["accept_rate"] == pytest.approx(2 / 3)
    assert e["effective"] == pytest.approx(0.12 / 2)            # all spend / entries cached


# ---- simple flow: "mismatch" (re-solve didn't match the production answer) ----

def _mismatch(record_id, cost=0.004):
    r = _Result(False, None, reason="caching tried but answer didn't match", cost=cost)
    r.status, r.record_id = "mismatch", record_id
    return r


def test_mismatch_keeps_watching_notes_the_record_and_waits_for_a_newer_question(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    calls = []

    def build(fid, recs):
        calls.append(fid)
        return _mismatch(recs[-1]["id"])
    families.build_ready(tr, build, path=path)
    assert len(calls) == 3
    fam = next(f for f in tr.families.values() if f.label.startswith("mortgage"))
    assert fam.status == "watching" and fam.tried_record == "mortgage2"
    assert fam.record_notes == {"mortgage2": families.MISMATCH_NOTE} and "didn't match" in fam.note
    row = next(r for r in tr.readiness() if r.family == fam.id)
    assert row.verdict == "wait" and "newer question" in row.why
    families.build_ready(tr, build, path=path)
    assert len(calls) == 3                                           # no retry without new data
    backlog.log_interaction("mortgage 9", "a", "claude-opus-5-5", path=path, input_tokens=200,
                            output_tokens=1500, timestamp=50.0, id="mortgage9")
    tr.ingest(path)
    families.build_ready(tr, build, path=path)
    assert calls[3:] == [fam.id]                                     # retried once a newer one came
    assert tr.families[fam.id].tried_record == "mortgage9"


def test_mismatch_spend_counts_as_a_failed_attempt(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    families.build_ready(tr, lambda fid, recs: _mismatch(recs[-1]["id"]), max_builds=1, path=path)
    [row] = [json.loads(l) for l in (tmp_path / "build_costs.jsonl").read_text().splitlines()]
    assert row["accepted"] is False and not row["pending"] and row["cost"] == 0.004


def test_a_late_arriving_older_record_does_not_count_as_newer(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    families.build_ready(tr, lambda fid, recs: _mismatch(recs[-1]["id"]), path=path)
    backlog.log_interaction("mortgage old", "a", "claude-opus-5-5", path=path, timestamp=-5.0, id="old")
    tr.ingest(path)
    fam = next(f for f in tr.families.values() if f.label.startswith("mortgage"))
    assert fam.last_source_id == "mortgage2"
    assert next(r for r in tr.readiness() if r.family == fam.id).verdict == "wait"


def test_success_after_a_mismatch_clears_the_retry_marker(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    families.build_ready(tr, lambda fid, recs: _mismatch(recs[-1]["id"]), max_builds=1, path=path)
    fid = next(f.id for f in tr.families.values() if f.tried_record)
    tr.mark_built(fid, "fam_x")
    assert tr.families[fid].tried_record is None
    assert next(r for r in tr.readiness() if r.family == fid).verdict == "built"


# ---- FAMILY_CHECK="clef": Clef confirms every addition ----

@pytest.fixture
def clef_mode(monkeypatch):
    monkeypatch.setattr(config, "FAMILY_CHECK", "clef")
    monkeypatch.setattr(config, "FAMILY_CLEF_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "FAMILY_CLEF_TOP_K", 3)
    monkeypatch.setattr(config, "FAMILY_CLEF_ANCHORS", 1)   # one comparison per family unless a test says so
    monkeypatch.setattr(config, "FAMILY_CLEF_SIMILARITY", 0.75)
    calls = []

    def decide(state, questions, *, model=None, tag=None):
        calls.append({"state": state, "questions": questions, "tag": tag})
        # stand-in for Clef: a question fits an anchor when both ask for the same last word
        return {k: (0.95 if state.split()[-1] == v.split("<<<")[1].split(">>>")[0].split()[-1] else 0.05)
                for k, v in questions.items()}
    from arc import llm
    monkeypatch.setattr(llm, "decide", decide)
    return calls


def test_clef_confirms_a_true_variant_against_the_anchor(tmp_path, clef_mode):
    tr = tracker(tmp_path)
    a = tr.observe(rec("gas: pressure of 2 mol at 300 K, find pressure", id="a"))
    b = tr.observe(rec("gas: 5 mol in 3 L at 40 C, find pressure", id="b"))
    assert a.id == b.id and a.confirmed_ids == ["a", "b"]
    [call] = clef_mode                                   # the founder needs no check
    assert call["state"].endswith("find pressure") and "<<<gas: pressure of 2 mol" in call["questions"][f"{a.id}#0"]
    assert tr.assignments["b"]["joined_by"] == "clef" and tr.assignments["b"]["clef"][a.id] == 0.95


def test_clef_keeps_a_near_miss_out_of_the_family(tmp_path, clef_mode):
    tr = tracker(tmp_path)
    a = tr.observe(rec("gas: 10 mol at 27 C in 4.86 L, find pressure", id="a"))
    nm = tr.observe(rec("gas: 10 mol at 27 C exerts 50.7 atm, find volume", id="nm"))   # embeds close, other task
    assert nm.id != a.id and tr.families[a.id].size == 1
    assert tr.assignments["nm"]["joined_by"] == "new: clef said no"
    again = tr.observe(rec("gas: 3 mol at 50 C exerts 2 atm, find volume", id="nm2"))
    assert again.id == nm.id                              # near-misses form their own family


def test_no_clef_call_when_nothing_is_close(tmp_path, clef_mode):
    tr = tracker(tmp_path)
    tr.observe(rec("mortgage payment", id="a"))
    tr.observe(rec("dosage for a child", id="b"))
    assert clef_mode == [] and tr.assignments["b"]["joined_by"] == "new: nothing close"


def test_one_clef_call_covers_the_top_candidates(tmp_path, clef_mode, monkeypatch):
    monkeypatch.setattr(config, "FAMILY_CLEF_SIMILARITY", -1.0)    # everything is a candidate
    monkeypatch.setattr(config, "FAMILY_CLEF_TOP_K", 2)
    tr = tracker(tmp_path)
    for i, q in enumerate(["alpha find x", "beta find y", "gamma find z"]):
        tr.observe(rec(q, id=f"s{i}"))
    clef_mode.clear()
    tr.observe(rec("delta find y", id="new"))
    assert len(clef_mode) == 1 and len(clef_mode[0]["questions"]) == 2


def test_clef_error_counts_the_question_but_never_builds_from_it(tmp_path, clef_mode, monkeypatch):
    from arc import llm
    tr = tracker(tmp_path)
    a = tr.observe(rec("gas pressure one, find pressure", id="a"))

    def broken(*a, **k):
        raise llm.LLMError("Cloudflare HTTP 503")
    monkeypatch.setattr(llm, "decide", broken)
    b = tr.observe(rec("gas pressure two, find pressure", id="b"))
    assert b.id == a.id and b.size == 2 and b.confirmed_ids == ["a"]
    assert "unconfirmed" in tr.assignments["b"]["joined_by"] and "503" in tr.assignments["b"]["error"]


def test_builds_only_receive_confirmed_members(tmp_path, clef_mode, monkeypatch):
    from arc import llm
    path = tmp_path / "backlog.jsonl"
    for i in range(3):
        backlog.log_interaction(f"loan {i}, find payment", "a", "claude-opus-5-5", path=path, input_tokens=200,
                                output_tokens=1500, timestamp=float(i), id=f"ok{i}")
    tr = tracker(tmp_path)
    tr.ingest(path)
    monkeypatch.setattr(llm, "decide", lambda *a, **k: (_ for _ in ()).throw(llm.LLMError("down")))
    backlog.log_interaction("loan 9, find payment", "a", "claude-opus-5-5", path=path, input_tokens=200,
                            output_tokens=1500, timestamp=9.0, id="unconfirmed")
    tr.ingest(path)
    got = []
    families.build_ready(tr, lambda fid, recs: got.append([r["id"] for r in recs]) or _Result(True, "e"), path=path)
    assert got == [["ok0", "ok1", "ok2"]]


def test_anchor_moves_to_the_most_typical_confirmed_member(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FAMILY_CHECK", "embed")
    vecs = {"far": [1.0, 0.0], "mid": [0.8, 0.6], "midb": [0.6, 0.8]}   # no digits: numbers get masked
    tr = families.FamilyTracker(path=tmp_path / "a.json",
                                embed_fn=lambda ts: np.array([vecs[t.split()[0]] for t in ts], np.float32))
    monkeypatch.setattr(config, "FAMILY_SIMILARITY", 0.0)
    tr.observe(rec("far x", id="far"))
    tr.observe(rec("midb x", id="midb"))
    assert tr.families["f0001"].anchor_id == "far"          # a tie keeps the current anchor
    tr.observe(rec("mid x", id="mid"))                     # more typical than the anchor: replaces it
    fam = tr.families["f0001"]
    assert fam.anchor_id == "mid" and fam.anchor_text == "mid x"


def test_assignments_survive_save_and_load(tmp_path, clef_mode):
    tr = tracker(tmp_path, "s.json")
    tr.observe(rec("gas a, find pressure", id="a"))
    tr.observe(rec("gas b, find pressure", id="b"))
    tr.save()
    again = tracker(tmp_path, "s.json")
    assert again.assignments["b"]["joined_by"] == "clef" and again.families["f0001"].anchor_id == "a"


def test_mismatch_note_carries_the_builders_reason(tmp_path):
    tr, path = _ready_tracker(tmp_path)
    r = _mismatch("mortgage2")
    r.reason = "built entry doesn't fit its family: 1 of 5 fit"
    families.build_ready(tr, lambda fid, recs: r, max_builds=1, path=path)
    fam = next(f for f in tr.families.values() if f.tried_record)
    assert "1 of 5 fit" in fam.note and fam.record_notes[fam.tried_record].startswith("built entry doesn't fit")


def test_joining_uses_its_own_cutoff_not_the_serving_one(tmp_path, clef_mode, monkeypatch):
    from arc import llm
    monkeypatch.setattr(llm, "decide", lambda state, qs, **k: {q: 0.6 for q in qs})
    monkeypatch.setattr(config, "CLEF_THRESHOLD", 0.99)          # serving cutoff: irrelevant here
    tr = tracker(tmp_path)
    a = tr.observe(rec("gas one, find pressure", id="a"))
    assert tr.observe(rec("gas two, find pressure", id="b")).id != a.id     # 0.6 < 0.8
    monkeypatch.setattr(config, "FAMILY_CLEF_THRESHOLD", 0.5)
    assert tr.observe(rec("gas three, find pressure", id="c")).id == a.id   # 0.6 >= 0.5


def test_a_question_joins_if_any_recent_confirmed_member_fits(tmp_path, clef_mode, monkeypatch):
    monkeypatch.setattr(config, "FAMILY_CLEF_ANCHORS", 3)
    monkeypatch.setattr(config, "FAMILY_CHECK", "embed")
    tr = tracker(tmp_path)
    for i, q in enumerate(["bayes test, find ppv", "bayes spam, find share", "bayes widgets, find chance"]):
        tr.observe(rec(q, id=f"m{i}"))                    # confirmed by the embed rule
    monkeypatch.setattr(config, "FAMILY_CHECK", "clef")
    clef_mode.clear()
    joined = tr.observe(rec("bayes email, find share", id="new"))    # fits only the 2nd member's wording
    assert joined.id == "f0001" and tr.assignments["new"]["clef"] == {"f0001": 0.95}
    [call] = clef_mode
    assert len(call["questions"]) == 3 and set(call["questions"]) == {"f0001#0", "f0001#1", "f0001#2"}


def test_recent_texts_keep_only_the_latest_confirmed_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FAMILY_CLEF_ANCHORS", 2)
    tr = tracker(tmp_path)
    for i in range(4):
        tr.observe(rec(f"loan {i}", id=f"l{i}"))
    assert tr.families["f0001"].recent_texts == ["loan 2", "loan 3"]


def test_clef_mode_uses_its_own_candidate_cosine(tmp_path, clef_mode, monkeypatch):
    monkeypatch.setattr(config, "FAMILY_SIMILARITY", 0.99)        # embed-mode cut: ignored in clef mode
    monkeypatch.setattr(config, "FAMILY_CLEF_SIMILARITY", 0.5)
    tr = tracker(tmp_path)
    a = tr.observe(rec("gas one, find pressure", id="a"))
    assert tr.observe(rec("gas two, find pressure", id="b")).id == a.id


def test_defaults_are_the_tuned_values():
    import importlib
    import os
    for k in ("ARC_FAMILY_CHECK", "ARC_FAMILY_CLEF_THRESHOLD", "ARC_FAMILY_CLEF_ANCHORS", "ARC_FAMILY_CLEF_SIMILARITY"):
        assert k not in os.environ
    src = (ROOT_DIR / "arc/config.py").read_text()
    for line in ('FAMILY_CHECK = os.environ.get("ARC_FAMILY_CHECK", "clef")',
                 'os.environ.get("ARC_FAMILY_CLEF_THRESHOLD", 0.5)', 'os.environ.get("ARC_FAMILY_CLEF_ANCHORS", 3)',
                 'os.environ.get("ARC_FAMILY_CLEF_SIMILARITY", 0.65)'):
        assert line in src, line
