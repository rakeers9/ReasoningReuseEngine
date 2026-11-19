"""Do output checks (arc/outchecks.py) stop wrong answers without blocking right ones?

    .venv/Scripts/python eval/outcheck_mutants.py --write     # PAID once (~$0.05): write the checks
    .venv/Scripts/python eval/outcheck_mutants.py             # FREE: score them

The dev set gives the live entries no wrong answers to catch (baseline: 0 of 47 wrong), so we
break the code on purpose, the way eval/cert_mutants.py does. Each mutant is a realistic builder
bug (a dropped /100, a missing salvage floor, a term counted in months as years). For every
question the baseline served, the mutated entry reruns on the SAME extracted values (extraction
doesn't depend on the code), so scoring is free and every mutant sees identical inputs.

Per mutant: how many answers come out wrong, and how many of those the checks stop. On the
clean entries: how many right answers the checks wrongly block (false alarms).

A mutant whose wrong answers look plausible (straight-line rate instead of double-declining) is
expected to slip through: output checks only catch impossible answers. Those are reported, not
hidden, because they are what a second implementation (step 4) is for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

CACHE = ROOT / "caches" / "chat_loop" / "master"
DDB_ITER = "Iterate double-declining-balance yearly"
LOAN_PCT = "Parse a percentage rate from raw text"
LOAN_RATE = "Compute the periodic interest rate"
LOAN_PMT = "Compute the growth factor"
LOAN_ROUND = "Round the periodic payment"

# (name, entry, function description prefix, old text, new text, what the bug looks like)
MUTANTS = [
    ("ddb_straight_line_rate", "fam_f0002", DDB_ITER, "rate = Decimal(2) / life", "rate = Decimal(1) / life",
     "plausible: half the depreciation rate"),
    ("ddb_extra_year", "fam_f0002", DDB_ITER, "range(1, years + 1)", "range(1, years + 2)",
     "plausible: off by one year"),
    ("ddb_no_salvage_floor", "fam_f0002", DDB_ITER, "capped = dep > room", "capped = False",
     "impossible once the floor binds: book value below salvage"),
    ("ddb_cost_in_thousands", "fam_f0002", DDB_ITER, "        bv = cost\n", "        bv = cost / 1000\n",
     "impossible: book value far below salvage"),
    ("ddb_sign_flip", "fam_f0002", DDB_ITER, "bv = bv - dep", "bv = bv + dep",
     "impossible: book value above cost"),
    ("loan_rate_not_percent", "fam_f0003", LOAN_PCT, "fraction = number / Decimal(100)", "fraction = number",
     "impossible: 7.5 read as 750%"),
    ("loan_rate_not_per_period", "fam_f0003", LOAN_RATE, "rate = annual / periods", "rate = annual",
     "impossible: yearly rate charged monthly"),
    ("loan_interest_no_principal", "fam_f0003", LOAN_ROUND, "interest = mp * Decimal(n) - principal",
     "interest = mp * Decimal(n)", "inconsistent: interest includes the principal"),
    ("loan_months_as_years", "fam_f0003", LOAN_RATE, "count = periods * years", "count = periods * years * 12",
     "impossible: 360-year loan"),
    ("loan_principal_x1000", "fam_f0003", LOAN_PMT, 'principal = to_dec(principal_raw, "principal")',
     'principal = to_dec(principal_raw, "principal") * 1000', "impossible: payment far above principal"),
    ("loan_payment_off_by_cent", "fam_f0003", LOAN_ROUND, "mp = pmt.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)",
     "mp = pmt.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) + Decimal('0.01')",
     "plausible: one cent high"),
]


def latest_baseline() -> Path:
    runs = sorted((ROOT / "runs" / "hit_replay").glob("dev_baseline_*.jsonl"))
    if not runs:
        sys.exit("no dev baseline yet: run eval/hit_replay.py --set dev --tag baseline first")
    return runs[-1]


def mutate(cache, prefix: str, old: str, new: str) -> None:
    """Break one function. The guards were validated on the clean code, so their freshness check
    would (rightly, in production) switch them off here; main() turns it off for this test."""
    rec = next(r for r in cache.functions.values() if r.description.startswith(prefix))
    assert old in rec.source, (prefix, old)
    cache.add_function(rec.description, rec.source.replace(old, new), rec.inputs, rec.outputs,
                       origin="mutant", replace_id=rec.id)


def answer(cache, ep, row: dict):
    """(raw output, presented text) for this row's extracted values, or (None, error)."""
    from arc import present, serve
    try:
        raw, _ = serve.dispatch_root(cache, ep.root_id, row["values"])
        shown = present.apply(raw, (row.get("extraction") or {}).get("output_format"),
                              (cache.outputs or {}).get(ep.problem_id))
        return raw, serve.format_output(shown)
    except Exception as e:   # noqa: BLE001 - a crash is a safe fallback, not a served answer
        return None, f"{type(e).__name__}: {e}"


GUARDS = ("checks", "reference", "either")


def score_cache(cache, rows: list[dict], clean_texts: dict | None = None, entry: str | None = None) -> dict:
    """Run the cache on every row, then the output checks and the reference on every answer.

    clean_texts None (the clean entries): every answer is the entry's own correct behaviour, so
    any block is a false alarm. clean_texts = {question: clean answer text} (a mutant): an answer
    is wrong when the bug changed it, and the question is whether a guard stopped it. Judging
    against the clean entry, not the expected answers, keeps the entry's own conventions (the
    loan's interest from the rounded payment) out of the mutant counts."""
    from arc import outchecks, reference
    eps = {e.problem_id: e for e in cache.entry_points if e.kind == "original"}
    t = {"answers": 0, "crashed": 0, "right": 0, "wrong": 0, "texts": {}, "examples": [],
         **{f"wrong_stopped_{g}": 0 for g in GUARDS}, **{f"right_blocked_{g}": 0 for g in GUARDS}}
    for row in rows:
        if entry and row["entry"] != entry:
            continue
        ep = eps[row["entry"]]
        raw, text = answer(cache, ep, row)
        t["answers"] += 1
        if raw is None:
            t["crashed"] += 1          # the code raised: the live path falls back to the model
            continue
        t["texts"][row["question"]] = text
        ok = clean_texts is None or clean_texts.get(row["question"]) == text
        fmt = (row.get("extraction") or {}).get("output_format")
        why = {"checks": outchecks.run(cache, ep.problem_id, row["values"], raw) or [],
               "reference": reference.run(cache, ep.problem_id, row["values"], raw, fmt) or []}
        why["either"] = why["checks"] + why["reference"]
        kind = "right_blocked" if ok else "wrong_stopped"
        t["right" if ok else "wrong"] += 1
        for g in GUARDS:
            t[f"{kind}_{g}"] += bool(why[g])
        if ok and why["either"] and len(t["examples"]) < 3:
            t["examples"].append({"false_alarm": why["either"][:2], "question": row["question"][:120]})
        if not ok and not why["either"] and len(t["examples"]) < 2:
            t["examples"].append({"slipped": text[:100], "clean": clean_texts.get(row["question"], "")[:100]})
    return t


def write(what: str) -> None:
    from arc import llm, outchecks, reference
    from arc.cache import Cache
    mod, attr, fname = {"checks": (outchecks, "checks", "checks.json"),
                        "references": (reference, "references", "references.json")}[what]
    cache = Cache.load(CACHE)
    for entry in sorted({e.problem_id for e in cache.entry_points}):
        rec = mod.write(cache, entry)
        from arc import guards
        guards.stamp(cache, entry, rec)
        print(f"{entry}: {rec['status']} ({rec['examples']} examples)")
        for r in rec.get("rules", []) or [rec.get("approach", "")]:
            print(f"   - {r}")
        for pr in rec.get("problems", []) if rec["status"] != "ok" else []:
            print(f"   ! {pr}")
    (CACHE / fname).write_text(json.dumps(getattr(cache, attr), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {(CACHE / fname).relative_to(ROOT)}; cost ${llm.LEDGER.cost():.4f}")


def main() -> None:
    if not sys.flags.utf8_mode:          # see eval/hit_replay.py
        import subprocess
        sys.exit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", choices=["checks", "references"], help="PAID: write them for every entry (overwrites)")
    ap.add_argument("--rows", default="", help="hit_replay rows (default: the latest dev baseline)")
    args = ap.parse_args()
    if args.write:
        write(args.write)
        return

    from arc import guards
    from arc.cache import Cache
    guards.CHECK_FRESHNESS = False        # see mutate()
    path = Path(args.rows) if args.rows else latest_baseline()
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["served"] and r.get("values")]
    print(f"rows: {path.relative_to(ROOT)} ({len(rows)} served answers)")

    clean = Cache.load(CACHE)
    print(f"guards present: checks for {sorted(clean.checks)}, references for "
          f"{sorted(k for k, v in clean.references.items() if v.get('status') == 'ok')}")
    t = score_cache(clean, rows)
    print(f"\nclean entries, {t['answers']} answers. False alarms: "
          + ", ".join(f"{g} {t['right_blocked_' + g]}" for g in GUARDS))
    for ex in t["examples"]:
        print("   ", ex)

    print(f"\n{'mutant':28s} {'wrong':>6s} {'checks':>7s} {'ref':>5s} {'either':>7s}  bug")
    totals = {g: 0 for g in GUARDS}
    total_wrong = 0
    results = {"clean": {k: v for k, v in t.items() if k != "texts"}, "mutants": {}}
    for name, entry, prefix, old, new, what in MUTANTS:
        cache = Cache.load(CACHE)
        mutate(cache, prefix, old, new)
        m = score_cache(cache, rows, clean_texts=t["texts"], entry=entry)
        results["mutants"][name] = {k: v for k, v in m.items() if k != "texts"}
        total_wrong += m["wrong"]
        for g in GUARDS:
            totals[g] += m["wrong_stopped_" + g]
        print(f"{name:28s} {m['wrong']:6d} {m['wrong_stopped_checks']:7d} {m['wrong_stopped_reference']:5d} "
              f"{m['wrong_stopped_either']:7d}  {what}")
    if total_wrong:
        print("\nwrong answers stopped: " + ", ".join(
            f"{g} {totals[g]} of {total_wrong} ({100 * totals[g] / total_wrong:.0f}%)" for g in GUARDS))
    out = path.with_name(path.stem + "_guards.json")
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"details: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
