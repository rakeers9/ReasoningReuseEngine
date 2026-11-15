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
