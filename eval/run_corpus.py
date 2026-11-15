"""Run the cache eval corpus through the cache path and score it.

    .venv/bin/python eval/run_corpus.py --tag baseline --split dev
    .venv/bin/python eval/run_corpus.py --tag gate-v1 --split dev --gate v1
    .venv/bin/python eval/run_corpus.py --tag verify-v2 --split dev --gate v2 --index cards --floor 0.65
    .venv/bin/python eval/run_corpus.py --compare baseline gate-v1 verify-v2
    .venv/bin/python eval/run_corpus.py --ablate verify-v2     (free: re-scores a saved run)

Only the cache path runs (entry search + Haiku extraction + cached code). The
source-model fallback is never called, so a miss costs nothing here; the count
of misses is reported instead. Cost is a few hundredths of a cent per row.

Runs use shadow mode: the gate and the cached code run on every row's top match,
even below the floor or when the gate rejects, and what the code WOULD have
answered is saved. The live decision is scored as usual, and --ablate can then
score other floors and check combinations from the same run at no cost.

Each run is saved under eval/runs/<tag>/ (results.jsonl, metrics.json) so every
change can be compared with the runs before it. Tune on dev; run test once,
when the fixes are frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus" / "cache_eval_corpus.jsonl"
RUNS = ROOT / "eval" / "runs"
# Answers must be right to the precision they are WRITTEN in (CLAUDE.md, production rule):
# a served number counts only if it is within half a unit of its own last stated digit of the
# true value ("18.6 atm" for 18.571 is right, "$5,053.27" for $5,053.07 is wrong). Money must
# be right to the cent whatever is written. No percentage tolerance.
MONEY_UNITS = {"$", "£", "€", "¥", "₹"}
EPS = 1e-9

# ----------------------------------------------------------------------------- scoring

NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?")
MONTHS = "january february march april may june july august september october november december".split()


def _numbers(text: str) -> list[tuple[float, bool]]:
    """Every number in the text, with whether a % sign follows it."""
    return [(v, pct) for v, pct, _ in _numbers_with_precision(text)]


def _numbers_with_precision(text: str) -> list[tuple[float, bool, float]]:
    """(value, followed by %, half a unit in its last stated digit) for every number."""
    out = []
    for m in NUM.finditer(text):
        raw = m.group().replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        mant, _, exp = raw.lower().partition("e")
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        half = 0.5 * 10 ** -decimals * (10 ** int(exp) if exp else 1)
        out.append((v, text[m.end():m.end() + 2].lstrip().startswith("%"), half))
    return out


def _agrees(served: float, half: float, truth: float, money: bool) -> bool:
    tol = 0.005 if money else half      # money: to the cent, whatever precision is written
    return abs(served - truth) <= tol + EPS * max(1.0, abs(truth))


def _dates(text: str) -> set[tuple[int, int, int]]:
    t = text.lower()
    found = set()
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", t):
        found.add((int(m[1]), int(m[2]), int(m[3])))
    for i, name in enumerate(MONTHS, 1):
        for m in re.finditer(rf"{name}\s+(\d{{1,2}}),?\s+(\d{{4}})", t):
            found.add((int(m[2]), i, int(m[1])))
        for m in re.finditer(rf"(\d{{1,2}})\s+{name},?\s+(\d{{4}})", t):
            found.add((int(m[2]), i, int(m[1])))
    return found


def score(row: dict, served_text: str) -> tuple[bool | None, str]:
    """Is the served answer right? Returns (correct, how). None = could not score."""
    v = row["answer_value"]
    if v is not None:
        money = (row.get("answer_unit") or "").strip() in MONEY_UNITS
        nums = _numbers_with_precision(served_text)
        if any(_agrees(x, h, v, money) for x, _, h in nums):
            return True, "number"
        if any(pct and _agrees(x / 100, h / 100, v, False) for x, pct, h in nums):
            return True, "percent"
        # A value 1000x off with no unit saying so is WRONG (it hid a "$150k" read as $150 bug).
        # Report it separately so it is easy to find, but never count it as right.
        if any(_agrees(x * f, h * f, v, False) for x, _, h in nums for f in (1e3, 1e-3, 1e6, 1e-6)):
            return False, "scaled_without_unit"
        return False, "no_match"
    want = _dates(row["answer"])
    if want:
        return bool(want & _dates(served_text)), "date"
    if not row["should_hit"]:
        return False, "no_answer_exists"  # "cannot be determined" or an explanation was wanted
    return None, "unscorable"


# ----------------------------------------------------------------------------- running

def run(args) -> None:
    os.environ["ARC_PROFILE"] = args.profile
    os.environ["ARC_NAMESPACE"] = args.namespace
    os.environ["ARC_GATE"] = args.gate
    os.environ["ARC_INDEX"] = args.index
    if args.floor is not None:
        os.environ["ARC_ENTRY_THRESHOLD"] = str(args.floor)
    sys.path.insert(0, str(ROOT))
    from arc import config, embed, llm
    from arc.cache import Cache
    from arc.runtime import CacheInvocationError
    from arc.serve import format_output, use
    from arc.verify import rank_entries
    # Set on the module too, in case arc.config was imported before the env vars were set.
    config.GATE, config.INDEX = args.gate, args.index
    if args.floor is not None:
        config.ENTRY_THRESHOLD = args.floor

    out_dir = RUNS / args.tag
    if out_dir.exists() and not args.overwrite:
        sys.exit(f"{out_dir} exists; pick a new --tag or pass --overwrite")
    if args.split == "test" and not args.i_am_done_tuning:
        sys.exit("test split is for the final, frozen run; pass --i-am-done-tuning to confirm")

    cache = Cache.load(config.cache_dir(args.cache))
    built = {e.problem_id for e in cache.entry_points}
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids.split(",")]
    if args.limit:
        rows = rows[: args.limit]
    if args.gate == "v2":
        missing_cards = sorted(built - set(cache.cards))
        if missing_cards:
            print(f"warning: no gate card for {missing_cards}; those entries always turn away")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out_dir / "results.jsonl", "w") as f:
        for n, r in enumerate(rows, 1):
            if llm.LEDGER.cost() >= args.budget:
                print(f"budget ${args.budget:.2f} reached after {n - 1} rows, stopping")
                break
            top = [(ep.problem_id, round(sim, 4), text[:80]) for ep, sim, text in rank_entries(cache, r["question"], 3)]
            trace: dict = {}
            before = llm.LEDGER.cost()
            served_text, error = None, None
            try:
                res = use(cache, r["question"], tag=f"corpus:{args.tag}", trace=trace, shadow=True)
                if res is not None:
                    served_text = format_output(res.output)
            except CacheInvocationError as e:
                error = str(e).splitlines()[0][:300]
                trace["stage"] = "crash"
            served = served_text is not None
            correct, how = score(r, served_text) if served else (None, "")
            would = served_text or trace.get("shadow_output")
            would_correct, _ = score(r, would) if would else (None, "")
            rec = {
                "id": r["id"], "split": r["split"], "seed_entry": r["seed_entry"],
                "variant_type": r["variant_type"], "near_miss_category": r["near_miss_category"],
                "should_hit": r["should_hit"], "acceptable_either": r["acceptable_either"],
                "expected_entry": r["expected_entry"], "seed_built": r["seed_entry"] in built,
                "question": r["question"], "answer": r["answer"], "answer_value": r["answer_value"],
                "top3": top, "matched_entry": trace.get("entry"), "matched_kind": trace.get("entry_kind"),
                "similarity": trace.get("similarity"), "stage": trace.get("stage"),
                "served": served, "served_answer": served_text, "correct": correct, "scored_by": how,
                "slots": trace.get("values"), "verdict": trace.get("verdict"),
                "rejected_by": trace.get("rejected_by"), "error": error,
                "checks": trace.get("checks"), "uncovered_numbers": trace.get("uncovered_numbers"),
                "form": trace.get("form"), "would_answer": would, "would_correct": would_correct,
                "shadow_error": trace.get("shadow_error"),
                "cost": llm.LEDGER.cost() - before,
            }
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = ("HIT " + ("ok" if correct else "WRONG" if correct is False else "?")) if served \
                else (trace.get("stage") or "miss")
            print(f"[{n:3d}/{len(rows)}] {r['id']} {r['variant_type']:16s} should_hit={r['should_hit']!s:5s} "
                  f"{mark:16s} sim={trace.get('similarity') or 0:.3f} {r['question'][:60]}", flush=True)

    metrics = compute_metrics(results, floor=config.ENTRY_THRESHOLD)
    metrics["run"] = {
        "tag": args.tag, "split": args.split, "cache": str(config.cache_dir(args.cache).relative_to(ROOT)),
        "entries_in_cache": sorted(built), "time": datetime.now().isoformat(timespec="seconds"),
        "settings": config.snapshot(), "note": args.note, "rows_run": len(results),
        "gate": args.gate, "index": args.index, "floor": config.ENTRY_THRESHOLD,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print_metrics(metrics)
    print(f"\nsaved to {out_dir.relative_to(ROOT)}/")


# ----------------------------------------------------------------------------- metrics

def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def compute_metrics(results: list[dict], floor: float | None = None) -> dict:
    missing = sorted({r["seed_entry"] for r in results if not r["seed_built"]})
    rs = [r for r in results if r["seed_built"]]  # rows whose seed failed to build are left out
    miss_rows = [r for r in rs if not r["should_hit"] and not r["acceptable_either"]]
    hit_rows = [r for r in rs if r["should_hit"]]
    false_hit = [r for r in miss_rows if r["served"] and r["correct"] is not True]
    false_miss = [r for r in hit_rows if not r["served"]]
    served_hits = [r for r in hit_rows if r["served"]]
    wrong_entry = [r for r in served_hits if r["matched_entry"] != r["expected_entry"]]
    served = [r for r in rs if r["served"] and not r["acceptable_either"]]

    by_cat = defaultdict(lambda: [0, 0])
    for r in miss_rows:
        c = r["near_miss_category"] or r["variant_type"]
        by_cat[c][1] += 1
        by_cat[c][0] += r in false_hit
    by_type = defaultdict(lambda: [0, 0])
    for r in hit_rows:
        by_type[r["variant_type"]][1] += 1
        by_type[r["variant_type"]][0] += r in false_miss
    by_entry = defaultdict(lambda: {"false_hits": 0, "false_misses": 0})
    for r in false_hit:
        by_entry[r["seed_entry"]]["false_hits"] += 1
    for r in false_miss:
        by_entry[r["seed_entry"]]["false_misses"] += 1

    # Shadow mode calls Haiku on every row; in production only rows above the floor would.
    live = [r for r in results if floor is None or (r["similarity"] or -1) >= floor]
    cost = sum(r["cost"] for r in live)
    return {
        "false_hit_rate": _rate(len(false_hit), len(miss_rows)),
        "false_miss_rate": _rate(len(false_miss), len(hit_rows)),
        "wrong_entry_rate": _rate(len(wrong_entry), len(served_hits)),
        "served_accuracy": _rate(sum(r["correct"] is True for r in served), len(served)),
        "should_hit_answered_right": _rate(sum(r["served"] and r["correct"] is True for r in hit_rows), len(hit_rows)),
        "false_hits_by_category": {k: _rate(*v) for k, v in sorted(by_cat.items())},
        "false_misses_by_type": {k: _rate(*v) for k, v in sorted(by_type.items())},
        "false_miss_reasons": dict(Counter(
            (r["stage"] or "?") + (":" + "+".join(r["rejected_by"]) if r.get("rejected_by") else "")
            for r in false_miss)),
        "by_entry": dict(sorted(by_entry.items())),
        "scored_by": dict(Counter(r["scored_by"] for r in rs if r["served"])),
        "either_rows": {"n": sum(r["acceptable_either"] for r in rs),
                        "served": sum(r["acceptable_either"] and r["served"] for r in rs)},
        "rows_left_out_seed_not_built": {"entries": missing, "rows": len(results) - len(rs)},
        "fallbacks_needed": sum(not r["served"] for r in rs),
        "cache_path_cost": round(cost, 5),
        "cache_path_cost_per_row": round(cost / len(results), 6) if results else None,
        "eval_spend_incl_shadow": round(sum(r["cost"] for r in results), 5),
    }


def _fmt(m: dict) -> str:
    return f"{m['k']}/{m['n']} = {m['rate'] * 100:.1f}%" if m["n"] else "n/a"


def print_metrics(m: dict) -> None:
    print("\n" + "=" * 70)
    for k in ["false_hit_rate", "false_miss_rate", "wrong_entry_rate", "served_accuracy",
              "should_hit_answered_right"]:
        print(f"  {k:28s} {_fmt(m[k])}")
    print(f"  {'cache path cost':28s} ${m['cache_path_cost']:.4f} (${m['cache_path_cost_per_row']:.6f}/row)")
    print(f"  {'fallbacks needed':28s} {m['fallbacks_needed']}")
    if m["rows_left_out_seed_not_built"]["rows"]:
        print(f"  left out (seed not built): {m['rows_left_out_seed_not_built']}")
    print("  false hits by category:")
    for k, v in m["false_hits_by_category"].items():
        if v["n"]:
            print(f"    {k:24s} {_fmt(v)}")
    print("  false misses by type:", {k: _fmt(v) for k, v in m["false_misses_by_type"].items()})
    print("  false miss reasons:", m["false_miss_reasons"])
    print("  answers scored by:", m["scored_by"])


def compare(tags: list[str]) -> None:
    ms = [json.loads((RUNS / t / "metrics.json").read_text()) for t in tags]
    keys = ["false_hit_rate", "false_miss_rate", "wrong_entry_rate", "served_accuracy",
            "should_hit_answered_right"]
    print(f"{'':28s}" + "".join(f"{t:>22s}" for t in tags))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{_fmt(m[k]):>22s}" for m in ms))
    print(f"{'cache cost per row':28s}" + "".join(f"{'$%.6f' % m['cache_path_cost_per_row']:>22s}" for m in ms))
    cats = sorted({c for m in ms for c in m["false_hits_by_category"]})
    print("\nfalse hits by category")
    for c in cats:
        print(f"  {c:26s}" + "".join(f"{_fmt(m['false_hits_by_category'].get(c, {'n': 0})):>22s}" for m in ms))


def policy_view(results: list[dict], floor: float, checks: list[str] | None) -> list[dict]:
    """Re-decide every row under another floor / check set, using what the shadow run saved.
    checks=None keeps the run's own gate decision (for off and v1 runs)."""
    out = []
    for r in results:
        r = dict(r)
        above = (r["similarity"] or -1) >= floor
        if checks is None:
            gate_ok = not r.get("rejected_by")
        else:
            gate_ok = r.get("checks") is not None and all(r["checks"].get(c, True) for c in checks)
        r["served"] = bool(above and gate_ok and r.get("would_answer"))
        r["correct"] = r["would_correct"] if r["served"] else None
        r["stage"] = "served" if r["served"] else "below_threshold" if not above else \
            "gate_rejected" if not gate_ok else "crash"
        r["rejected_by"] = [] if checks is None or r.get("checks") is None else \
            [c for c in checks if not r["checks"].get(c, True)]
        out.append(r)
    return out


def ablate(tag: str, floors: list[float]) -> None:
    results = [json.loads(l) for l in (RUNS / tag / "results.jsonl").read_text().splitlines()]
    run_info = json.loads((RUNS / tag / "metrics.json").read_text())["run"]
    base_floor = run_info.get("floor", 0.8)
    all_checks = ["asked_for", "method", "out_of_scope", "no_unused", "stated", "grounded", "numbers", "unit"]
    variants: list[tuple[str, float, list[str] | None]] = []
    if run_info.get("gate") == "v2":
        variants.append(("all checks", base_floor, all_checks))
        variants.append(("no checks", base_floor, []))
        variants += [(f"drop {c}", base_floor, [x for x in all_checks if x != c]) for c in all_checks]
        variants += [(f"only {c}", base_floor, [c]) for c in all_checks]
        variants += [(f"all checks, floor {f}", f, all_checks) for f in floors]
    else:
        variants += [(f"floor {f}", f, None) for f in floors]
    print(f"{tag}: re-scored offline from saved shadow results (no API calls)\n")
    print(f"{'variant':32s}{'false hits':>18s}{'false misses':>18s}{'wrong entry':>16s}{'served acc':>18s}")
    for name, floor, checks in variants:
        m = compute_metrics(policy_view(results, floor, checks))
        print(f"{name:32s}{_fmt(m['false_hit_rate']):>18s}{_fmt(m['false_miss_rate']):>18s}"
              f"{_fmt(m['wrong_entry_rate']):>16s}{_fmt(m['served_accuracy']):>18s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="name for this run, e.g. baseline or gate-v1")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "seed", "all"])
    ap.add_argument("--gate", default="off", choices=["off", "v1", "v2"])
    ap.add_argument("--index", default="paper", choices=["paper", "cards"])
    ap.add_argument("--floor", type=float, default=None, help="entry similarity floor (default: config)")
    ap.add_argument("--profile", default="haiku", choices=["haiku", "paper"])
    ap.add_argument("--namespace", default="corpus")
    ap.add_argument("--cache", default="master")
    ap.add_argument("--budget", type=float, default=0.25, help="stop past this spend ($)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--i-am-done-tuning", action="store_true")
    ap.add_argument("--compare", nargs="+", metavar="TAG")
    ap.add_argument("--ablate", metavar="TAG")
    ap.add_argument("--floors", default="0.5,0.6,0.65,0.7,0.75,0.8")
    args = ap.parse_args()
    if args.ablate:
        ablate(args.ablate, [float(x) for x in args.floors.split(",")])
    elif args.compare:
        compare(args.compare)
    elif args.tag:
        run(args)
    else:
        ap.error("give --tag to run, --compare TAG TAG ..., or --ablate TAG")


if __name__ == "__main__":
    main()
