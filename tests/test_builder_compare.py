"""eval/builder_compare.py: the backlog and the scoring set are disjoint and well formed."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import builder_compare as bc  # noqa: E402


def test_each_family_gets_disjoint_backlog_and_held_out_items():
    rows = [json.loads(l) for l in (ROOT / "eval/labelfree/corpus.jsonl").read_text().splitlines() if l.strip()]
    for family in bc.FAMILIES:
        records, held = bc.family_split(rows, family)
        assert len(records) == bc.N_BACKLOG and len(held) == bc.N_SCORE
        assert not {r["id"] for r in records} & {h["id"] for h in held}
        assert all(r["response"] and r["model"] == "claude-haiku-5-5" for r in records)
        assert [r["timestamp"] for r in records] == sorted(r["timestamp"] for r in records)
        assert all(h["kind"] == "positive" for h in held)


def test_each_arm_runs_a_single_builder():
    src = (ROOT / "eval/builder_compare.py").read_text()
    assert "config.LABELFREE_BUILDERS = [model]" in src
