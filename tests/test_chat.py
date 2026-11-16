"""The chatbot loop (arc/chat.py): logging, cache hits, fallback, learning. Stubs only."""

import json

import numpy as np
import pytest

from arc import backlog, chat, config, families, llm, serve
from arc.cache import Cache, EntryPoint

from test_families import fake_embed


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "caches")
    monkeypatch.setattr(config, "FAMILY_MIN_SIGHTINGS", 3.0)
    monkeypatch.setattr(config, "GATE", "v2")
    monkeypatch.setattr(config, "FAMILY_CHECK", "embed")
    monkeypatch.setattr(config, "FAMILY_BUILD_CHECK", "off")
    monkeypatch.setattr(config, "BUILD_COST_DEFAULT", 0.034)
    monkeypatch.setattr(families.FamilyTracker, "_embed", lambda self, texts: fake_embed(texts))


Q = "A loan of 200000 dollars at 6 percent over 30 years: what is the monthly payment?"


def production(text="Working... Final answer: $1,199.10", tin=300, tout=900, model="claude-haiku-5-5"):
    calls = []

    def answer(message):
        calls.append(message)
        return llm.LLMResult(text=text, model=model, cost=0.001, input_tokens=tin, output_tokens=tout)
    answer.calls = calls
    return answer


def bot(prod=None):
    return chat.Chatbot(production=prod or production())


def rows(tmp_path):
    return [json.loads(l) for l in (tmp_path / "backlog.jsonl").read_text().splitlines()]


def test_reusable_filter():
    assert chat.is_reusable(Q)
    assert not chat.is_reusable("hi there, how are you doing today my friend?")   # no number
    assert not chat.is_reusable("what is 2+2")                                     # too short


def test_miss_answers_with_the_model_and_logs_tokens(tmp_path):
    b = bot()
    r = b.chat(Q, session="user-1")
    assert not r.from_cache and r.answer.endswith("$1,199.10") and r.reason == "cache is empty"
    [row] = rows(tmp_path)
    assert row["input_tokens"] == 300 and row["output_tokens"] == 900 and row["model"] == "claude-haiku-5-5"
    assert row["served_by"] is None and row["reusable"] is True and row["session"].startswith("h:")


def test_chit_chat_is_answered_logged_and_never_forms_a_family(tmp_path):
    b = bot()
    for _ in range(5):
        b.chat("thanks so much, that was really helpful!")
    assert all(r["reusable"] is False for r in rows(tmp_path))
    b.tracker.ingest(b.backlog_path)
    assert b.tracker.families == {} and b.tracker.skipped == 5


def _cache_with_entry():
    c = Cache("master")
    c.entry_points.append(EntryPoint(Q, "original", "root", [], "fam_f0001"))
    return c


def _hit(entry="fam_f0001"):
    ep = EntryPoint(Q, "original", "root", [], entry)
    return serve.CacheResult({"monthly_payment": "$1,199.10"}, "", ep, 0.9, {}, 0.0001)


def test_cache_hit_skips_the_model_and_logs_served_by(tmp_path, monkeypatch):
    prod = production()
    b = bot(prod)
    b.cache = _cache_with_entry()
    monkeypatch.setattr(serve, "use", lambda cache, msg, tag=None, trace=None: _hit())
    r = b.chat(Q)
    assert r.from_cache and r.entry == "fam_f0001" and "1,199.10" in r.answer and prod.calls == []
    [row] = rows(tmp_path)
    assert row["served_by"] == "fam_f0001" and row["cost"] == 0.0001


def test_gate_rejection_falls_back_to_the_model_with_a_reason(tmp_path, monkeypatch):
    prod = production()
    b = bot(prod)
    b.cache = _cache_with_entry()

    def rejected(cache, msg, tag=None, trace=None):
        trace.update(stage="gate_rejected", rejected_by=["method"])
        return None
    monkeypatch.setattr(serve, "use", rejected)
    r = b.chat(Q)
    assert not r.from_cache and prod.calls == [Q] and "method" in r.reason


def test_cache_error_falls_back_to_the_model(tmp_path, monkeypatch):
    prod = production()
    b = bot(prod)
    b.cache = _cache_with_entry()

    def broken(cache, msg, tag=None, trace=None):
        raise serve.CacheInvocationError("ZeroDivisionError: boom")
    monkeypatch.setattr(serve, "use", broken)
    r = b.chat(Q)
    assert not r.from_cache and "cache error" in r.reason and prod.calls == [Q]


def test_learn_builds_ready_families_and_reloads_the_cache(tmp_path, monkeypatch):
    b = bot(production(model="claude-opus-5-5", tout=1500))   # $0.031 each at Opus prices: ready after 3
    for i in range(3):
        b.chat(f"mortgage question number {i} with 200000 dollars at 6 percent")
    b.chat("thanks, that is all I needed for today")
    built = []

    def stub_build(fid, records):
        built.append((fid, len(records)))
        _cache_with_entry().save(config.cache_dir("master"))        # what the real builder does
        return type("R", (), {"accepted": True, "entry_id": "fam_f0001", "reason": "", "cost": 0.05,
                              "parts": {}, "status": "accepted", "serve_rate": None})()
    results = b.learn(build_fn=stub_build)
    assert built == [("f0001", 3)] and len(results) == 1
    assert b.tracker.families["f0001"].status == "built"
    assert [e.problem_id for e in b.cache.entry_points] == ["fam_f0001"]               # reloaded
    monkeypatch.setattr(serve, "use", lambda cache, msg, tag=None, trace=None: _hit())
    assert b.chat("mortgage question number 9 with 300000 dollars at 5 percent").from_cache
    b.learn(build_fn=stub_build)
    assert built == [("f0001", 3)]                                                      # not rebuilt
    assert b.tracker.families["f0001"].cache_hits_total == 1


def test_llm_result_carries_token_counts():
    r = llm.LLMResult(text="x")
    assert r.input_tokens == 0 and r.output_tokens == 0


def test_learning_with_no_build_budget_still_groups_questions(tmp_path):
    # regression (chat run 2026-11-15): a zero build budget skipped grouping too, so no family formed
    b = bot(production(model="claude-opus-5-5", tout=1500))
    for i in range(3):
        b.chat(f"mortgage question number {i} with 200000 dollars at 6 percent")
    calls = []
    assert b.learn(build_fn=lambda fid, recs: calls.append(fid), max_builds=0) == []
    assert calls == [] and len(b.tracker.families) == 1 and b.tracker.families["f0001"].size == 3
    assert (tmp_path / "families.json").exists()
