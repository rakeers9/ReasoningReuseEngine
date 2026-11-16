"""The hidden chat-cache test set (eval/hidden): schema, counts, no duplicates, and that every
solver reproduces a hand-checked example. Free and offline (no model is ever called).

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

_spec = importlib.util.spec_from_file_location("build_hidden", ROOT / "eval" / "hidden" / "build_hidden.py")
bh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bh)

HIDDEN = ROOT / "eval" / "hidden" / "chat_hidden.jsonl"
KEYS = ["kind", "family", "question", "answer", "answer_value", "answer_unit", "should_hit",
        "acceptable_either", "category"]
FAMILY_IDS = [bh.CAL_ID, bh.FV_ID, bh.ST_ID, bh.CH_ID]


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line) for line in HIDDEN.read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- the file

def test_file_is_what_the_generator_writes():
    # the committed file never drifts from the seeded generator
    assert HIDDEN.read_text(encoding="utf-8") == bh.dumps(bh.build())


def test_generator_is_deterministic_and_seeded():
    assert bh.build() == bh.build()
    assert bh.build(seed=bh.SEED + 1) != bh.build()


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
    msgs = cl.load_messages(HIDDEN)
    assert len(msgs) == len(bh.build())


def test_counts(rows):
    kinds = Counter(r["kind"] for r in rows)
    assert kinds == {"variant": 80, "near_miss": 24, "one_off": 8, "chitchat": 6}
    assert 100 <= len(rows) <= 120
    for fam in FAMILY_IDS:
        assert sum(r["kind"] == "variant" and r["family"] == fam for r in rows) == 20
        assert 4 <= sum(r["kind"] == "near_miss" and r["family"] == fam for r in rows) <= 6


def test_no_duplicate_questions(rows):
    qs = [" ".join(r["question"].lower().split()) for r in rows]
    assert len(qs) == len(set(qs))


def test_no_unfilled_templates(rows):
    for r in rows:
        assert "{" not in r["question"] and "}" not in r["question"]
        assert "None" not in r["question"] and "nan" not in r["question"].lower().split()


def test_set_is_shuffled_not_grouped(rows):
    # families and kinds are interleaved, not written in blocks
    runs = sum(a["family"] != b["family"] for a, b in zip(rows, rows[1:]))
    assert runs > len(rows) // 2


def test_new_families_are_not_in_the_existing_corpora(rows):
    existing_ids, existing_qs = set(), set()
    for path in (ROOT / "eval/corpus/cache_eval_corpus.jsonl", ROOT / "eval/labelfree/corpus.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            existing_ids |= {v for k, v in rec.items() if k in ("seed_entry", "family", "task") and v}
            existing_qs.add(" ".join(rec["question"].lower().split()))
    assert not set(FAMILY_IDS) & existing_ids
    assert not {" ".join(r["question"].lower().split()) for r in rows} & existing_qs


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
        assert len(cats) >= 5          # different targets, rules, conditions, not six of one kind


def test_missing_given_has_no_numeric_answer(rows):
    for r in rows:
        if r.get("category") == "missing_given":
            assert r["answer_value"] is None and r["answer"].startswith("Cannot be determined")
        elif r["kind"] in ("variant", "near_miss", "one_off"):
            assert r["answer_value"] is not None


def test_answer_display_follows_the_rounding_policy(rows):
    for r in rows:
        if r["answer_value"] is None:
            continue
        v, unit, shown = r["answer_value"], r["answer_unit"], r["answer"]
        if unit == "$":
            assert shown == f"${bh.cents(v):,.2f}" and v == float(bh.cents(v))      # to the cent
        elif unit in ("", "months") and v == int(v):
            assert shown == str(int(v))                                             # counts exact (n, months)
        else:
            assert shown == f"{bh.sig(v)} {unit}".strip()                           # 4 significant figures


def test_answer_values_are_clean_floats(rows):
    for r in rows:
        v = r["answer_value"]
        if v is not None:
            assert v == float(f"{v:.12g}")


def test_money_variants_use_dollar_shorthand_somewhere(rows):
    fv = [r["question"] for r in rows if r["family"] == bh.FV_ID and r["kind"] == "variant"]
    assert any("k" in q and "$" in q for q in fv) or any("million" in q for q in fv)


def test_units_vary_within_each_family(rows):
    q = {f: " ".join(r["question"] for r in rows if r["family"] == f and r["kind"] == "variant")
         for f in FAMILY_IDS}
    assert "kg" in q[bh.CAL_ID] and "°C" in q[bh.CAL_ID] and "mL" in q[bh.CAL_ID]
    assert "months" in q[bh.FV_ID] and "years" in q[bh.FV_ID]
    assert "mmol" in q[bh.CH_ID] or " mM" in q[bh.CH_ID]
    assert "mL" in q[bh.CH_ID] and " L" in q[bh.CH_ID]


# ----------------------------------------------------------------------------- hand-checked solvers

def test_calorimetry_by_hand():
    # equal masses, equal c, 100 °C into 0 °C: the midpoint
    assert bh.mix_final(100, 1.0, 100, 100, 1.0, 0) == 50.0
    # (25*90 + 100*10) / (25 + 100) = 3250 / 125 = 26
    assert bh.mix_final(50, 0.5, 90, 100, 1.0, 10) == 26.0
    # a cup of 25 J/°C adds 25 to the water's side: (2250 + 125*10) / 150 = 3500/150
    assert bh.mix_final(50, 0.5, 90, 100, 1.0, 10, c_cal=25) == pytest.approx(3500 / 150, abs=1e-12)


def test_calorimetry_heat_balance_on_random_inputs():
    rng = random.Random(3)
    for _ in range(200):
        m1, c1, t1 = rng.uniform(20, 500), rng.uniform(0.1, 1.0), rng.uniform(60, 300)
        m2, c2, t2 = rng.uniform(50, 600), rng.uniform(4.0, 4.2), rng.uniform(2, 40)
        cal = rng.uniform(0, 80)
        tf = bh.mix_final(m1, c1, t1, m2, c2, t2, c_cal=cal)
        assert m1 * c1 * (t1 - tf) == pytest.approx(m2 * c2 * (tf - t2) + cal * (tf - t2), rel=1e-12)


def test_future_value_by_hand():
    # 1% a month for 2 months: 100 * ((1.01)^2 - 1) / 0.01 = 201.00
    assert bh.fv_ordinary(0, 100, 12, 2) == pytest.approx(201.0, abs=1e-9)
    assert bh.fv_ordinary(0, 100, 12, 2, due=True) == pytest.approx(203.01, abs=1e-9)
    assert bh.fv_ordinary(1000, 100, 12, 2) == pytest.approx(1000 * 1.0201 + 201, abs=1e-9)
    # one month at 1%/month then one month at 2%/month: 100 * 1.02 + 100
    assert bh.fv_two_rates(0, 100, 12, 1, 24, 1) == pytest.approx(202.0, abs=1e-9)
    assert bh.required_deposit(201.0, 0, 12, 2) == pytest.approx(100.0, abs=1e-9)
    months, margin = bh.months_to_exceed(1000, 0, 100, 12)
    assert months == 10 and margin == pytest.approx(100 * (1.01 ** 10 - 1) / 0.01 - 1000, abs=1e-9)   # 46.22 over


def test_future_value_matches_a_month_by_month_simulation():
    rng = random.Random(5)
    for _ in range(100):
        p0, d, r, n = rng.choice([0, 5000, 150000]), rng.choice([150, 800, 2000]), rng.choice([3.0, 5.25, 8.0]), rng.randrange(12, 400)
        for due in (False, True):
            bal, i = float(p0), r / 1200
            for _ in range(n):
                bal = (bal + d) * (1 + i) if due else bal * (1 + i) + d
            assert bh.fv_ordinary(p0, d, r, n, due=due) == pytest.approx(bal, rel=1e-9)
        t = bh.fv_ordinary(p0, d, r, n)
        assert bh.required_deposit(t, p0, r, n) == pytest.approx(d, rel=1e-9)


def test_z_scores():
    assert bh.z_two(95) == pytest.approx(1.959964, abs=1e-6)
    assert bh.z_two(90) == pytest.approx(1.644854, abs=1e-6)
    assert bh.z_two(99) == pytest.approx(2.575829, abs=1e-6)
    assert bh.z_one(95) == pytest.approx(1.644854, abs=1e-6)


def test_sample_size_by_hand():
    # sigma 15, E 3, 95%: (1.96 * 15 / 3)^2 = 9.8^2 = 96.04 -> 97 (exact z gives 96.036 -> 97 too)
    assert bh.n_mean(1.96, 15, 3) == pytest.approx(96.04, abs=1e-9)
    assert bh.n_mean_ceil(15, 3, bh.z_two(95)) == 97
    # finite population N = 200: 96.04 / (1 + 95.04 / 200) = 65.1 -> 66 (rounding n0 up first agrees)
    assert bh.n_finite(15, 3, bh.z_two(95), 200) == 66
    # the textbook proportion case: +-5 points at 95%, p = 0.5 -> 385
    assert bh.n_proportion(0.05, bh.z_two(95)) == 385
    # one-sided 95%: (1.645 * 10 / 2)^2 = 67.65 -> 68
    assert bh.n_mean_ceil(10, 2, bh.z_one(95)) == 68


def test_sample_size_is_the_smallest_n_that_meets_the_margin():
    rng = random.Random(9)
    for _ in range(100):
        sigma, e, cl = rng.choice([2, 5, 12.5]), rng.choice([0.5, 1, 2]), rng.choice([90, 95, 99])
        z = bh.z_two(cl)
        n = bh.n_mean_ceil(sigma, e, z)
        if n is None:
            continue
        assert z * sigma / math.sqrt(n) <= e < z * sigma / math.sqrt(n - 1)


def test_values_that_depend_on_z_rounding_are_refused():
    # candidates straddling an integer, or sitting on one, give no single answer
    assert bh.robust_ceil([100.2, 99.9]) is None
    assert bh.robust_ceil([97.0, 97.0]) is None
    assert bh.robust_ceil([96.04, 96.036]) == 97


def test_generated_sample_sizes_do_not_depend_on_z_rounding():
    rng = random.Random(21)
    for _ in range(200):
        s = bh.st_scenario(rng)
        for z in bh.z_candidates(s["z0"]):
            assert math.ceil(bh.n_mean(z, s["sg"], s["e"])) == s["n"]


def test_solution_prep_by_hand():
    # 0.100 mol/L x 0.500 L = 0.0500 mol; x 58.44 g/mol = 2.922 g
    assert bh.mass_needed(0.100, 500, 58.44) == pytest.approx(2.922, abs=1e-12)
    assert bh.molarity_from_mass(2.922, 500, 58.44) == pytest.approx(0.100, abs=1e-12)
    # C1V1 = C2V2: 0.5 M x 250 mL / 6 M
    assert bh.stock_volume_ml(6.0, 0.5, 250) == pytest.approx(250 / 12, abs=1e-12)


def test_one_off_solvers_by_hand():
    assert bh.parallel_resistance(12, 12) == pytest.approx(6.0, abs=1e-12)
    assert bh.parallel_resistance(60, 30, 20) == pytest.approx(10.0, abs=1e-12)       # 1/60+2/60+3/60
    assert bh.trip_fuel_cost(100, 10, 2) == pytest.approx(20.0)
    assert bh.doubling_time_years(100) == pytest.approx(1.0)
    assert bh.free_fall_time(4.905) == pytest.approx(1.0)                             # 2 * 4.905 / 9.81 = 1
    assert bh.distinct_arrangements("AAB") == 3 and bh.distinct_arrangements("ABC") == 6
    assert bh.distinct_arrangements("ALABAMA") == 210                                  # 7! / 4!
    assert bh.distinct_arrangements("MISSISSIPPI") == 34650
    assert bh.binomial_pmf(2, 1, 0.5) == 0.5
    assert bh.binomial_pmf(8, 3, 0.6) == pytest.approx(56 * 0.216 * 0.01024, abs=1e-12)   # 0.12386304
    assert bh.dose_each(18, 12.5, 3) == 75.0
    assert bh.cylinder_litres(1, 1) == pytest.approx(1000 * math.pi)


def test_one_off_rows_hold_the_hand_checked_values(rows):
    by_id = {r["family"]: r for r in rows if r["kind"] == "one_off"}
    assert by_id["oneoff_letter_arrangements"]["answer_value"] == 210.0
    assert by_id["oneoff_letter_arrangements"]["answer"] == "210"
    assert by_id["oneoff_pediatric_dose"]["answer_value"] == 75.0
    assert by_id["oneoff_trip_fuel_cost"]["answer"] == "$91.85"                        # 640 * 0.078 * 1.84 = 91.8528
    assert by_id["oneoff_binomial_heads"]["answer_value"] == pytest.approx(0.12386304, abs=1e-12)


# ----------------------------------------------------------------------------- formatting helpers

def test_sig_rounds_half_up_to_four_figures():
    assert bh.sig(2.0058750000000005) == "2.006"
    assert bh.sig(51.345) == "51.35"            # exact half goes up
    assert bh.sig(15833.6) == "15,830"
    assert bh.sig(0.12386304) == "0.1239"
    assert bh.sig(0.5) == "0.5000"


def test_money_to_the_cent_half_up():
    assert bh.money(91.8528) == "$91.85"
    assert bh.money(1105994.265) == "$1,105,994.27"
    assert bh.money(0.005) == "$0.01"


def test_clean_removes_float_noise_only():
    assert bh.clean(51.345000000000006) == 51.345
    assert bh.clean(0.12386304) == 0.12386304
    assert bh.clean(None) is None


def test_number_words():
    assert [bh.words(n) for n in (5, 15, 20, 25, 99)] == ["five", "fifteen", "twenty", "twenty-five", "ninety-nine"]
