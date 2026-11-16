"""eval/decision/run_decision.py: prompt v2 uses the live gate's prompt and its own run files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "decision"))

import run_decision as rd  # noqa: E402
from arc import clef  # noqa: E402

SEEDS = {"perfect_gas_pressure": "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure."}


def test_v1_pair_text_is_unchanged():
    assert rd.pair_text(SEEDS, "perfect_gas_pressure", raw=False) == rd.PROMPT.format(seed=SEEDS["perfect_gas_pressure"])
    assert rd.pair_text(SEEDS, "perfect_gas_pressure", raw=False) == clef.PROMPT_V1.format(seed=SEEDS["perfect_gas_pressure"])


def test_v2_pair_text_is_the_live_gate_prompt_with_the_seed_profile():
    text = rd.pair_text(SEEDS, "perfect_gas_pressure", raw=False, prompt="v2")
    spec = rd.seed_specs()["perfect_gas_pressure"]
    assert text == clef.PROMPT_V2.format(seed=SEEDS["perfect_gas_pressure"], profile=clef._profile(spec))
    assert f"What it computes: {spec['deliverable']}." in text and spec["method"] in text


def test_every_seed_has_a_stored_structure():
    import json
    corpus = [json.loads(l) for l in (ROOT / "eval/corpus/cache_eval_corpus.jsonl").read_text().splitlines()]
    seeds = {r["seed_entry"] for r in corpus if r["split"] == "seed"}
    assert seeds <= set(rd.seed_specs())


def test_v2_runs_never_overwrite_v1_runs():
    assert rd.run_path("clef", "dev", "top3", "raw") != rd.run_path("clef", "dev", "top3", "raw", "v2")
    assert rd.run_path("clef", "dev", "top3", "raw", "v2").name == "dev-top3-pv2.jsonl"


def test_v3_pair_text_is_the_live_v3_prompt_with_the_seed_profile():
    text = rd.pair_text(SEEDS, "perfect_gas_pressure", raw=False, prompt="v3")
    spec = rd.seed_specs()["perfect_gas_pressure"]
    assert text == clef.PROMPT_V3.format(seed=SEEDS["perfect_gas_pressure"], profile=clef._profile(spec))
    assert text == rd.pair_text(SEEDS, "perfect_gas_pressure", raw=False, prompt="v2") + clef.CAREFUL_READING


def test_v3_runs_have_their_own_files():
    paths = {rd.run_path("clef", "dev", m, "raw", p) for m in ("own", "top3") for p in ("v1", "v2", "v3")}
    assert len(paths) == 6
    assert rd.run_path("clef", "dev", "top3", "raw", "v3").name == "dev-top3-pv3.jsonl"


def test_v4_pair_text_is_the_live_gate_v4_prompt():
    text = rd.pair_text(SEEDS, "perfect_gas_pressure", raw=False, prompt="v4")
    spec = rd.seed_specs()["perfect_gas_pressure"]
    assert text == clef.PROMPT_V4.format(seed=SEEDS["perfect_gas_pressure"], profile=clef._profile(spec))
    assert rd.run_path("clef", "dev", "top3", "raw", "v4").name == "dev-top3-pv4.jsonl"


def test_family_seeds_never_include_the_test_split_or_originals():
    import json
    corpus = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/corpus/cache_eval_corpus.jsonl").read_text().splitlines()}
    extra = rd.family_seeds("dev")
    for entry, items in extra.items():
        for rid, q in items:
            r = corpus[rid]
            assert r["split"] in ("seed", "dev") and r["should_hit"] and r["expected_entry"] == entry
            assert r["variant_type"] != "original"


def test_multi_seed_rows_take_the_best_seed_and_skip_the_row_itself(monkeypatch):
    calls = []

    class B:
        def decide(self, state, questions):
            calls.append(questions)
            return {k: (0.9 if k.endswith("#1") else 0.3) for k in questions}, {}
    monkeypatch.setattr(rd, "N_SEEDS", 3)
    extra = rd.family_seeds("dev")
    entry = next(iter(extra))
    first_id, first_q = extra[entry][0]
    row = {"id": first_id, "question": first_q, "candidates": [{"entry": entry}]}
    probs, _, _ = rd._decide_row(B(), False, "raw", row, {entry: "seed text"}, {}, None, "v6")
    assert probs == {entry: 0.9}
    assert len(calls[0]) == 3 and all(f"<<<{first_q}>>>" not in v for v in calls[0].values())   # never itself
    assert rd.run_path("clef", "dev", "own", "raw", "v6", seeds=3).name == "dev-own-pv6-s3.jsonl"


def test_separate_seeds_make_one_call_per_seed(monkeypatch):
    calls = []

    class B:
        def decide(self, state, questions):
            calls.append(list(questions))
            return {k: 0.5 for k in questions}, {"input_tokens": 10}
    monkeypatch.setattr(rd, "N_SEEDS", 3)
    monkeypatch.setattr(rd, "SEPARATE", True)
    entry = next(iter(rd.family_seeds("dev")))
    probs, _, usage = rd._decide_row(B(), False, "raw", {"id": "x", "question": "q", "candidates": [{"entry": entry}]},
                                     {entry: "seed"}, {}, None, "v6")
    assert len(calls) == 3 and all(len(c) == 1 for c in calls) and usage["input_tokens"] == 30
    assert rd.run_path("clef", "dev", "own", "raw", "v6", seeds=3).name == "dev-own-pv6-s3-sep.jsonl"
