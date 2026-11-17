"""The second hidden chat-cache test set (eval/hidden2): schema, counts, no overlap with the other
corpora or the first hidden set, and that every solver reproduces a hand-checked example.
Free and offline (no model is ever called).

These tests check the generator's code and the file's shape. They deliberately never print or
assert on the wording of the questions, so the set stays out of anyone's tuning loop.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import chat_loop as cl  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h2 = _load("build_hidden2", ROOT / "eval" / "hidden2" / "build_hidden2.py")
h1 = _load("build_hidden_for_h2", ROOT / "eval" / "hidden" / "build_hidden.py")

HIDDEN2 = ROOT / "eval" / "hidden2" / "chat_hidden2.jsonl"
KEYS = ["kind", "family", "question", "answer", "answer_value", "answer_unit", "should_hit",
        "acceptable_either", "category"]
FAMILY_IDS = [h2.NC_ID, h2.IV_ID, h2.LN_ID, h2.EQ_ID]


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line) for line in HIDDEN2.read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- the file

def test_file_is_what_the_generator_writes():
    assert HIDDEN2.read_text(encoding="utf-8") == h2.dumps(h2.build())


def test_generator_is_deterministic_and_seeded():
    assert h2.build() == h2.build()
    assert h2.build(seed=h2.SEED + 1) != h2.build()
    assert h2.SEED != h1.SEED                                  # a different seed from the first set


def test_schema(rows):
    for r in rows:
        assert list(r) == KEYS
        assert r["kind"] in {"variant", "near_miss", "one_off", "chitchat"}
        assert isinstance(r["question"], str) and r["question"].strip()
        assert isinstance(r["answer"], str) and isinstance(r["answer_unit"], str)
        assert r["answer_value"] is None or isinstance(r["answer_value"], float)
        assert isinstance(r["should_hit"], bool) and isinstance(r["acceptable_either"], bool)
        assert r["family"] is None or isinstance(r["family"], str)
        assert r["category"] is None or isinstance(r["category"], str)


def test_chat_loop_accepts_the_file():
    assert len(cl.load_messages(HIDDEN2)) == len(h2.build())


def test_counts(rows):
    kinds = Counter(r["kind"] for r in rows)
    assert kinds == {"variant": 80, "near_miss": 24, "one_off": 8, "chitchat": 6}
    assert 100 <= len(rows) <= 120
    for fam in FAMILY_IDS:
        assert sum(r["kind"] == "variant" and r["family"] == fam for r in rows) == 20
        assert 4 <= sum(r["kind"] == "near_miss" and r["family"] == fam for r in rows) <= 6


def test_no_duplicate_questions(rows):
    qs = [_norm(r["question"]) for r in rows]
    assert len(qs) == len(set(qs))


def test_no_unfilled_templates(rows):
    for r in rows:
        assert "{" not in r["question"] and "}" not in r["question"]
        assert "None" not in r["question"] and "nan" not in r["question"].lower().split()
        assert "1 hours" not in r["question"] and "one hours" not in r["question"]


def test_set_is_shuffled_not_grouped(rows):
    assert sum(a["family"] != b["family"] for a, b in zip(rows, rows[1:])) > len(rows) // 2


def test_no_overlap_with_the_other_corpora_or_the_first_hidden_set(rows):
    existing_ids = set(h1.FAMILIES) | {r["family"] for r in map(json.loads, (ROOT / "eval/hidden/chat_hidden.jsonl").read_text(encoding="utf-8").splitlines()) if r["family"]}
    existing_qs = {_norm(r["question"]) for r in map(json.loads, (ROOT / "eval/hidden/chat_hidden.jsonl").read_text(encoding="utf-8").splitlines())}
    for path in (ROOT / "eval/corpus/cache_eval_corpus.jsonl", ROOT / "eval/labelfree/corpus.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            existing_ids |= {v for k, v in rec.items() if k in ("seed_entry", "family", "task") and v}
            existing_qs.add(_norm(rec["question"]))
    existing_ids |= set(cl.FAMILIES)
    ours = {r["family"] for r in rows if r["family"]}
    assert not ours & existing_ids
    assert not {_norm(r["question"]) for r in rows} & existing_qs


# ----------------------------------------------------------------------------- conventions

def test_flags_by_kind(rows):
    for r in rows:
        if r["kind"] == "variant":
            assert r["should_hit"] is True and r["category"] is None and r["acceptable_either"] is False
            assert r["family"] in FAMILY_IDS and r["answer_value"] is not None and r["answer"]
        elif r["kind"] == "near_miss":
            assert r["should_hit"] is False and r["family"] in FAMILY_IDS and r["category"]
            assert r["acceptable_either"] == (r["category"] == "benign_extra_info")
        elif r["kind"] == "one_off":
            assert r["should_hit"] is True and r["family"] not in FAMILY_IDS
            assert r["family"].startswith("oneoff_") and r["answer_value"] is not None
        else:
            assert r["family"] is None and r["should_hit"] is False
            assert r["answer"] == "" and r["answer_value"] is None and r["answer_unit"] == ""


def test_every_family_has_the_main_near_miss_types(rows):
    for fam in FAMILY_IDS:
        cats = Counter(r["category"] for r in rows if r["kind"] == "near_miss" and r["family"] == fam)
        assert cats["missing_given"] == 1 and cats["benign_extra_info"] == 1
        assert len(cats) >= 5


def test_missing_given_has_no_numeric_answer(rows):
    for r in rows:
        if r.get("category") == "missing_given":
            assert r["answer_value"] is None and r["answer"].startswith("Cannot be determined")
        elif r["kind"] in ("variant", "near_miss", "one_off"):
            assert r["answer_value"] is not None


def test_answer_display_follows_the_rounding_policy(rows):
    for r in rows:
        v, unit, shown, q = r["answer_value"], r["answer_unit"], r["answer"], r["question"]
        if v is None:
            continue
        if unit == "$":
            assert shown == f"${h2.cents(v):,.2f}" and v == float(h2.cents(v))       # to the cent
        elif "nearest whole" in q:
            assert v == int(v) and shown == f"{int(v)} {unit}"                      # the question's own rounding
        elif r["family"] == "oneoff_committee_count":
            assert v == int(v) and shown == str(int(v))                             # counts exact
        else:
            assert shown == f"{h2.sig(v)} {unit}".strip()                           # 4 significant figures


def test_answer_values_are_clean_floats(rows):
    for r in rows:
        if r["answer_value"] is not None:
            assert r["answer_value"] == float(f"{r['answer_value']:.12g}")


def test_same_kind_answers_use_the_units_of_the_inputs(rows):
    # temperature: the unit written in the question; distance: the unit written in the question
    for r in rows:
        if r["kind"] == "variant" and r["family"] == h2.NC_ID:
            assert r["answer_unit"] in ("°C", "°F", "K")
            assert r["answer_unit"] == ("K" if " K" in r["question"] else "°F" if "°F" in r["question"] else "°C")
        if r["kind"] == "variant" and r["family"] == h2.LN_ID:
            assert r["answer_unit"] in ("cm", "mm", "m")
            assert f" {r['answer_unit']}" in r["question"]
            other = {"cm", "mm", "m"} - {r["answer_unit"]}
            assert not any(f" {u} " in r["question"] or f" {u}," in r["question"] or f" {u}." in r["question"]
                           or f" {u}?" in r["question"] for u in other)


def test_units_vary_within_each_family(rows):
    var = {f: [r for r in rows if r["family"] == f and r["kind"] == "variant"] for f in FAMILY_IDS}
    assert {r["answer_unit"] for r in var[h2.NC_ID]} == {"°C", "°F", "K"}
    assert {r["answer_unit"] for r in var[h2.LN_ID]} == {"cm", "mm", "m"}
    iv = " ".join(r["question"] for r in var[h2.IV_ID])
    assert " mL" in iv and " L " in iv and "hours" in iv and "min" in iv
    assert any("per month" in r["question"] for r in var[h2.EQ_ID])
    assert any("per year" in r["question"] for r in var[h2.EQ_ID])
    assert any("a week" in r["question"] for r in var[h2.EQ_ID])


def test_iv_family_has_the_rounded_to_whole_drop_variants(rows):
    iv = [r for r in rows if r["family"] == h2.IV_ID and r["kind"] == "variant"]
    whole = [r for r in iv if "nearest whole" in r["question"]]
    assert 1 <= len(whole) < len(iv)
    assert all(r["answer_value"] == int(r["answer_value"]) for r in whole)


def test_money_answers_are_not_on_a_half_cent_boundary(rows):
    for r in rows:
        if r["answer_unit"] == "$":
            assert h2.cents_are_safe(r["answer_value"])


# ----------------------------------------------------------------------------- hand-checked solvers

def test_newton_cooling_by_hand():
    # the excess halves every half-life: 90 -> 20 with the room at 20 and a 10-minute half-life
    k = math.log(2) / 10
    assert h2.newton_cooling(20, 90, k, 10) == pytest.approx(55.0, abs=1e-12)
    assert h2.newton_cooling(20, 90, k, 20) == pytest.approx(37.5, abs=1e-12)
    assert h2.newton_cooling(20, 90, 0.05, 0) == 90
    assert h2.time_to_reach(20, 90, 55, k) == pytest.approx(10.0, abs=1e-12)


def test_newton_cooling_matches_a_numerical_integration():
    rng = random.Random(4)
    for _ in range(40):
        ta, t0, k, t = rng.uniform(15, 30), rng.uniform(50, 95), rng.uniform(0.01, 0.1), rng.uniform(5, 90)
        steps, temp = 200_000, t0
        dt = t / steps
        for _ in range(steps):
            temp += -k * (temp - ta) * dt
        assert h2.newton_cooling(ta, t0, k, t) == pytest.approx(temp, abs=0.02)


def test_newton_cooling_is_scale_free_across_c_f_and_k():
    # the same physical cooling gives the same answer in any of the three scales
    c = h2.newton_cooling(20.0, 80.0, 0.04, 25)
    f = h2.newton_cooling(68.0, 176.0, 0.04, 25)
    kel = h2.newton_cooling(293.15, 353.15, 0.04, 25)
    assert f == pytest.approx(c * 9 / 5 + 32, abs=1e-9) and kel == pytest.approx(c + 273.15, abs=1e-9)


def test_newton_scenarios_in_all_three_scales_agree():
    rng = random.Random(12)
    for _ in range(300):
        s = h2.nc_scenario(rng)
        assert 0.25 <= s["x"] <= 3.5
        assert s["T"] > s["Ta"]
        back = {"C": s["T"], "F": (s["T"] - 32) * 5 / 9, "K": s["T"] - 273.15}[s["u"]]
        assert back == pytest.approx(s["tc"], abs=1e-9)


def test_iv_drip_by_hand():
    # 1000 mL over 8 h at 15 gtt/mL: 1000 * 15 / 480 = 31.25
    assert h2.drip_rate(1000, 480, 15) == pytest.approx(31.25, abs=1e-12)
    assert h2.drip_rate(100, 60, 60) == 100.0                        # microdrip: gtt/min == mL/h
    assert h2.pump_rate_ml_per_h(1500, 240) == 375.0
    assert h2.hours_to_empty(1000, 10, 50) == pytest.approx(1000 * 10 / 50 / 60, abs=1e-12)
    # 5 mcg/kg/min x 70 kg = 350 mcg/min = 21 mg/h; bag 1.6 mg/mL -> 13.125 mL/h
    assert h2.weight_based_pump_rate(5, 70, 400, 250) == pytest.approx(13.125, abs=1e-12)


def test_iv_pump_and_drip_rates_are_consistent():
    rng = random.Random(8)
    for _ in range(100):
        v, m, df = rng.choice(h2.IV_VOLS), rng.choice(h2.IV_MINS), rng.choice(h2.IV_DF)
        assert h2.drip_rate(v, m, df) == pytest.approx(h2.pump_rate_ml_per_h(v, m) * df / 60, rel=1e-12)


def test_half_up_whole_drops():
    assert h2.half_up_int(62.5) == 63 and h2.half_up_int(10.4) == 10 and h2.half_up_int(188.0) == 188


def test_thin_lens_by_hand():
    # f = 12, d_o = 30: d_i = 12 * 30 / 18 = 20; the object at 2f images at 2f
    assert h2.image_distance(12, 30) == pytest.approx(20.0, abs=1e-12)
    assert h2.image_distance(10, 20) == pytest.approx(20.0, abs=1e-12)
    assert h2.magnification(10, 20) == pytest.approx(1.0, abs=1e-12)
    assert h2.object_distance(12, 20) == pytest.approx(30.0, abs=1e-12)
    assert h2.diverging_image_distance(30, 45) == pytest.approx(18.0, abs=1e-12)
    assert h2.lenses_in_contact(40, 10) == pytest.approx(8.0, abs=1e-12)


def test_thin_lens_obeys_the_lens_equation():
    rng = random.Random(6)
    for _ in range(200):
        f = rng.uniform(2, 60)
        d_o = f * rng.uniform(1.1, 6)
        d_i = h2.image_distance(f, d_o)
        assert 1 / f == pytest.approx(1 / d_o + 1 / d_i, rel=1e-12)
        assert h2.object_distance(f, d_i) == pytest.approx(d_o, rel=1e-9)
        assert 1 / d_o - 1 / h2.diverging_image_distance(f, d_o) == pytest.approx(-1 / f, rel=1e-12)


def test_lens_scenarios_are_in_one_unit_and_scale_correctly():
    rng = random.Random(10)
    for _ in range(300):
        s = h2.ln_scenario(rng)
        assert s["d"] / s["f"] >= 1.2
        in_cm = h2.image_distance(s["f"], s["d"]) * s["scale"]
        assert s["di"] == pytest.approx(in_cm, rel=1e-12)


def test_eoq_by_hand():
    # D = 10,000, S = 50, H = 2.5: Q* = sqrt(2*10000*50/2.5) = 632.46; cost = sqrt(2*10000*50*2.5) = 1581.14
    assert h2.eoq(10000, 50, 2.5) == pytest.approx(632.455532, abs=1e-6)
    assert h2.eoq_total_cost(10000, 50, 2.5) == pytest.approx(1581.138830, abs=1e-6)
    # at the EOQ, ordering cost equals holding cost: each is half of the total
    q = h2.eoq(10000, 50, 2.5)
    assert 10000 / q * 50 == pytest.approx(q / 2 * 2.5, rel=1e-12)
    assert h2.order_cost_from_eoq(h2.eoq(10000, 50, 2.5), 10000, 2.5) == pytest.approx(50.0, abs=1e-9)
    # production rate 4x the demand: the cost shrinks by sqrt(3/4)
    assert h2.epq_total_cost(10000, 50, 2.5, 40000) == pytest.approx(1581.138830 * math.sqrt(0.75), abs=1e-6)


def test_eoq_is_the_minimum_of_ordering_plus_holding():
    rng = random.Random(2)
    for _ in range(100):
        d, s, h = rng.choice([1200, 8000, 36000]), rng.choice([25, 80, 150]), rng.choice([0.8, 2.5, 6.0])
        q = h2.eoq(d, s, h)

        def cost(x):
            return d / x * s + x / 2 * h

        assert cost(q) == pytest.approx(h2.eoq_total_cost(d, s, h), rel=1e-12)
        assert cost(q) <= cost(q * 0.99) and cost(q) <= cost(q * 1.01)


def test_eoq_scenarios_have_clean_money():
    rng = random.Random(30)
    for _ in range(200):
        s = h2.eq_scenario(rng)
        assert h2.cents_are_safe(s["total"])
        assert s["value"] == float(h2.cents(s["total"]))


def test_one_off_solvers_by_hand():
    assert h2.continuous_growth(100, 0, 5) == 100
    assert h2.continuous_growth(1, 100, 1) == pytest.approx(math.e, abs=1e-12)
    assert h2.rc_time_constant_ms(1, 1) == pytest.approx(1.0, abs=1e-12)             # 1 kΩ x 1 µF = 1 ms
    assert h2.rc_time_constant_ms(4.7, 22) == pytest.approx(103.4, abs=1e-9)
    assert h2.average_speed((100, 50), (100, 100)) == pytest.approx(200 / 3, abs=1e-12)  # 2 h + 1 h
    assert h2.average_speed((150, 60), (90, 90)) == pytest.approx(240 / 3.5, abs=1e-12)
    # 5 men, 4 women, choose 4, at least 2 women: 6*10 + 4*5 + 1 = 81
    assert h2.committee_count(5, 4, 4, 2) == 81
    assert h2.committee_count(5, 4, 4, 0) == math.comb(9, 4)
    assert h2.poisson_pmf(1.0, 0) == pytest.approx(math.exp(-1), abs=1e-12)
    assert sum(h2.poisson_pmf(3.2, k) for k in range(80)) == pytest.approx(1.0, abs=1e-12)
    assert h2.kinetic_energy_kj(1200, 90) == pytest.approx(375.0, abs=1e-9)           # 0.5 * 1200 * 25^2
    assert h2.bmi(80, 2.0) == 20.0
    assert h2.ladder_height(5, 3) == 4.0


def test_one_off_rows_hold_the_hand_checked_values(rows):
    by_id = {r["family"]: r for r in rows if r["kind"] == "one_off"}
    assert by_id["oneoff_committee_count"]["answer"] == "81"
    assert by_id["oneoff_kinetic_energy"]["answer_unit"] == "kJ" and by_id["oneoff_kinetic_energy"]["answer_value"] == 375.0
    assert by_id["oneoff_rc_time_constant"]["answer_unit"] == "ms"            # the question asks for ms
    assert by_id["oneoff_average_speed"]["answer_unit"] == "km/h"             # same kind as the inputs
    assert by_id["oneoff_ladder_height"]["answer_unit"] == "m"
    assert by_id["oneoff_continuous_growth"]["answer"] == "$6,255.36"          # 5000 * e^0.224 = 6255.3566...


# ----------------------------------------------------------------------------- formatting helpers

def test_hours_text_handles_the_singular():
    assert h2.hours_text(60) == "1 hour" and h2.hours_text(90) == "1.5 hours" and h2.hours_text(120) == "2 hours"
    assert h2.hours_text(60, spelled=True) == "one hour" and h2.hours_text(240, spelled=True) == "four hours"


def test_cents_are_safe_flags_half_cent_boundaries():
    assert not h2.cents_are_safe(0.005) and not h2.cents_are_safe(1234.565)
    assert h2.cents_are_safe(12.34) and h2.cents_are_safe(1581.13883)
