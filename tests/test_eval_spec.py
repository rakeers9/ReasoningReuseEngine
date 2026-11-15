"""eval/spec/run_spec.py bookkeeping: stale specs are ignored, metrics are right."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_spec", ROOT / "eval" / "spec" / "run_spec.py")
run_spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_spec)


def test_specs_from_an_older_prompt_are_ignored(tmp_path, monkeypatch):
    f = tmp_path / "specs.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"key": "a", "prompt": "oldhash", "spec": {}},
        {"key": "b", "prompt": run_spec.PROMPT_HASH, "spec": {}},
        {"key": "c", "spec": {}},  # written before hashes existed
    ]))
    monkeypatch.setattr(run_spec, "SPECS", f)
    assert set(run_spec.load_specs()) == {"b"}


def test_auc():
    assert run_spec.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert run_spec.auc([0.1], [0.9]) == 0.0
    assert run_spec.auc([0.5], [0.5]) == 0.5
    assert run_spec.auc([], [0.5]) is None


def test_summarize_finds_zero_false_hit_threshold():
    scores = [0.9, 0.7, 0.6, 0.3]
    labels = [True, True, False, False]
    m = run_spec.summarize(scores, labels, ["p", "p", "near", "near"], default_t=0.5)
    assert m["auc"] == 1.0
    assert m["at_default"]["false_hits"] == 1
    assert m["zero_false_hits"]["recall"] == 1.0
    assert m["negatives_accepted_by_type"] == {"near": "1/2"}
