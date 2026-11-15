"""eval/labelfree/run_eval.py: the free parts of the harness (scoring, noise, mutants, memo)."""

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "labelfree"))
_s = importlib.util.spec_from_file_location("lf_run_eval", ROOT / "eval" / "labelfree" / "run_eval.py")
ev = importlib.util.module_from_spec(_s)
sys.modules["lf_run_eval"] = ev
_s.loader.exec_module(ev)

from arc.cache import Cache  # noqa: E402


def test_right_accepts_any_listed_form():
    truth = ["50.68 atm", "5135 kPa", "51.35 bar"]
    assert ev.right("pressure: 5.135e6 Pa", truth)          # kPa form, rescaled
    assert ev.right("about 50.7 atm", truth)
    assert not ev.right("pressure: 42 atm", truth)
    assert ev.right("2155.01 per month, interest 425803.67", ["2155.01 and 425803.67"])
    assert not ev.right("2155.01 per month", ["2155.01 and 425803.67"])   # both numbers needed


def test_noise_only_touches_positives_and_is_deterministic():
    items = [{"id": f"i{k}", "family": "f", "kind": "positive" if k < 20 else "near_miss",
              "question": f"q{k}", "truth": [f"{k + 10}"], "timestamp": k} for k in range(25)]
    prod = {x["id"]: {"response": f"answer {x['truth'][0]}"} for x in items}
    clean = ev.backlog_for("clean", "f", items, prod)
    a = ev.backlog_for("noisy", "f", items, prod)
    b = ev.backlog_for("noisy", "f", items, prod)
    assert len(clean) == 20 and a == b
    changed = [x for x, y in zip(clean, a) if x["response"] != y["response"]]
    assert 0 < len(changed) < 20
    assert len(ev.backlog_for("near_miss", "f", items, prod)) == 25


def test_mutator_makes_exactly_one_change():
    src = "def semantic_function(ctx):\n    return {'y': 2 * 3 + 4}\n"
    m = ev._Mutator(0)
    import ast
    out = ast.unparse(m.visit(ast.parse(src)))
    assert out != ast.unparse(ast.parse(src)) and m.what


def test_mutants_of_a_real_entry_compile_and_differ():
    cache = Cache.load(ROOT / "caches" / "corpus_cert" / "perfect_gas_pressure")
    muts = ev.mutants(cache, 3, random.Random(1))
    assert len(muts) == 3
    for label, mc in muts:
        changed = [f for f in mc.functions if mc.functions[f].source != cache.functions[f].source]
        assert len(changed) == 1 and label
    # the original cache is untouched
    assert all(cache.functions[f].source == Cache.load(ROOT / "caches" / "corpus_cert" / "perfect_gas_pressure")
               .functions[f].source for f in cache.functions)


def test_memo_round_trip(tmp_path):
    m = ev.Memo(tmp_path / "memo.json")
    k = m.key("compare", "q", "a", "b")
    m.put(k, {"agree": True})
    m.save()
    assert ev.Memo(tmp_path / "memo.json").get(k) == {"agree": True}


def test_memoized_blind_solves_are_shared_across_entries(tmp_path, monkeypatch):
    from arc import explain, labelfree, serve
    calls = []
    monkeypatch.setattr(explain, "_blind_solve",
                        lambda q, tag: calls.append(tag) or explain.Worked(working="", final_answer="6"))
    monkeypatch.setattr(serve, "extract_parameters", lambda *a, **k: {})
    monkeypatch.setattr(labelfree, "_compare", lambda *a, **k: None)
    monkeypatch.setattr(labelfree, "_program_solve", labelfree._program_solve)   # restored after install_memo
    memo = ev.Memo(tmp_path / "m.json")
    ev.install_memo(memo)
    explain._blind_solve("double 3", "entryA:lf:r1:adj1")
    explain._blind_solve("double 3", "mutant7:lf:r1:adj1")   # same question, same solve number
    explain._blind_solve("double 3", "entryA:lf:r1:adj2")
    assert len(calls) == 2


def test_right_is_exact_to_the_written_precision():
    assert ev.right("monthly_payment: 5053.0678", ["$5,053.07"])
    assert not ev.right("monthly_payment: 5053.27", ["$5,053.07"])
    assert not ev.right("monthly_payment: 0.92113", ["$921.13"])          # "$150k" read as 150
    assert ev.right("about 50.7 atm", ["50.68 atm"])                       # coarser side decides
    assert not ev.right("pressure: 50.5 atm", ["50.68 atm"])                # was "right" under 1%


def test_memoized_program_solves_are_shared_across_entries(tmp_path, monkeypatch):
    from arc import explain, labelfree, serve
    calls = []
    monkeypatch.setattr(labelfree, "_program_solve",
                        lambda q, tag, conventions=None: calls.append(tag) or ("6", ""))
    monkeypatch.setattr(explain, "_blind_solve", lambda q, tag: explain.Worked(working="", final_answer="6"))
    monkeypatch.setattr(serve, "extract_parameters", lambda *a, **k: {})
    monkeypatch.setattr(labelfree, "_compare", lambda *a, **k: None)
    ev.install_memo(ev.Memo(tmp_path / "m.json"))
    assert labelfree._program_solve("double 3", "entryA:lf:r1:adj1") == ("6", "")
    labelfree._program_solve("double 3", "mutant7:lf:r1:adj1")
    labelfree._program_solve("double 3", "entryA:lf:r1:adj1", ["round to the cent"])   # other conventions
    assert len(calls) == 2
