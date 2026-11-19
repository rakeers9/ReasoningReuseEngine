"""Hit guards stay present and current (arc/guards.py), and the reference cross-check
(arc/reference.py) behaves. Offline: no model calls (writers are faked)."""

import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from arc import config, guards, outchecks, reference
from arc.cache import Cache

DEV = Path(__file__).resolve().parent.parent / "caches" / "chat_loop" / "master"
LOAN, DDB = "fam_f0003", "fam_f0002"


@pytest.fixture
def cache(tmp_path):
    shutil.copytree(DEV, tmp_path / "c")
    return Cache.load(tmp_path / "c")


def _touch(cache, description_prefix):
    rec = next(r for r in cache.functions.values() if r.description.startswith(description_prefix))
    cache.add_function(rec.description, rec.source + "\n# rebuilt\n", rec.inputs, rec.outputs, replace_id=rec.id)


def test_fingerprint_follows_the_code_the_entry_calls(cache):
    fns = guards.used_functions(cache, LOAN)
    assert len(fns) > 1                                       # root plus the functions do() reached
    before = guards.fingerprint(cache, LOAN, fns)
    _touch(cache, "Iterate double-declining-balance")         # another entry's function
    assert guards.fingerprint(cache, LOAN, fns) == before
    _touch(cache, "Compute the periodic interest rate")       # one the loan entry calls
    assert guards.fingerprint(cache, LOAN, fns) != before


def test_stale_guard_is_ignored_and_reported(cache, capsys):
    guards._WARNED.clear()
    for kind in guards.KINDS:
        guards.stamp(cache, LOAN, getattr(cache, kind)[LOAN])
    assert guards.report(cache)[LOAN] == {"checks": "ok", "references": "ok"}
    _touch(cache, "Compute the periodic interest rate")
    assert guards.report(cache)[LOAN] == {"checks": "stale", "references": "stale"}
    rec, why = guards.usable(cache, LOAN, "checks")
    assert rec is None and "stale" in why
    assert outchecks.run(cache, LOAN, {}, {"monthly payment in dollars": 1.0}) is None
    assert "served without checks" in capsys.readouterr().err


def test_ensure_writes_missing_and_stale_but_does_not_retry_a_failure(cache, monkeypatch, tmp_path):
    calls = []

    def fake(kind, status):
        def write(c, entry_id, tag=""):
            calls.append((kind, entry_id))
            rec = {"source": "def check(i, o):\n    return []\n", "status": status}
            getattr(c, kind)[entry_id] = rec
            return rec
        return write

    monkeypatch.setattr(outchecks, "write", fake("checks", "ok"))
    monkeypatch.setattr(reference, "write", fake("references", "disagrees_with_verified"))
    cache.checks.pop(LOAN)
    cache.references.pop(LOAN)
    for kind in guards.KINDS:                                 # DDB's guards: current
        guards.stamp(cache, DDB, getattr(cache, kind)[DDB])
    rows = guards.ensure(cache, tmp_path)
    assert sorted(calls) == [("checks", LOAN), ("references", LOAN)]
    assert {r["status"] for r in rows} == {"ok", "disagrees_with_verified"}
    assert (tmp_path / "checks.json").exists() and (tmp_path / "references.json").exists()
    calls.clear()
    assert guards.ensure(cache, tmp_path) == [] and calls == []     # failed on the same code: no retry
    _touch(cache, "Compute the periodic interest rate")             # rebuilt: both are stale now
    guards.ensure(cache, tmp_path)
    assert sorted(calls) == [("checks", LOAN), ("references", LOAN)]


def test_learn_writes_guards_after_an_accepted_build(monkeypatch, tmp_path):
    from arc import chat, families
    seen = []
    monkeypatch.setattr(families, "build_ready", lambda *a, **k: [("f1", type("R", (), {"accepted": True})())])
    monkeypatch.setattr(families.FamilyTracker, "ingest", lambda self, path=None: 0)
    monkeypatch.setattr(guards, "ensure", lambda c, folder, tag="": seen.append(folder) or [])
    bot = chat.Chatbot(cache_name="test_guards_tmp", backlog_path=tmp_path / "b.jsonl",
                       tracker_path=tmp_path / "t.json", production=lambda m: None)
    bot.learn(build_fn=lambda *a: None)
    assert seen == [config.cache_dir("test_guards_tmp")]
    monkeypatch.setattr(families, "build_ready", lambda *a, **k: [("f1", type("R", (), {"accepted": False})())])
    seen.clear()
    bot.learn(build_fn=lambda *a: None)
    assert seen == []                                         # nothing new built: nothing written


# ---------------------------------------------------------------- reference cross-check

REF = '''
def reference(inputs):
    return {"total": number(inputs["a"]) + number(inputs["b"])}
'''


def _ref_cache(source=REF):
    c = Cache("t")
    c.references["e"] = {"source": source, "status": "ok"}
    c.outputs["e"] = {"total": {"kind": "money", "unit": "USD", "decimals": 2}}
    return c


def test_reference_agrees_at_shown_precision():
    c = _ref_cache()
    assert reference.run(c, "e", {"a": "$1.004", "b": "$2"}, {"total": Decimal("3.001")}) == []
    d = reference.run(c, "e", {"a": "$1", "b": "$2"}, {"total": Decimal("3.01")})
    assert d and "total" in d[0]


def test_reference_crash_or_empty_counts_as_disagreement():
    assert reference.run(_ref_cache("def reference(inputs):\n    return 1 / 0\n"), "e", {}, {"total": 1})[0] \
        .startswith("reference crashed")
    assert reference.run(_ref_cache("def reference(inputs):\n    return {}\n"), "e", {}, {"total": 1}) == \
        ["reference returned no comparable output"]


def test_format_combos_include_zero_edge_cases(cache):
    combos = outchecks.format_combos(cache, LOAN)
    labels = [c["question"] for c in combos]
    assert any(l.startswith("(edge case") for l in labels) and len(combos) > 4
    assert any(c["inputs"]["the annual nominal interest rate as a percentage"] == "0%" for c in combos)
