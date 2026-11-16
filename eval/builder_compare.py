"""Haiku vs Sonnet as the builder, on the 4 question types Haiku-built entries got wrong. PAID.

    .venv/bin/python eval/builder_compare.py            # both builders, all 4 families
    .venv/bin/python eval/builder_compare.py --families ddb_book_value --models claude-sonnet-5-5

Families (eval/labelfree/corpus.jsonl, answers computed by hand-written Python): gas
pressure (dm3, mL, mmol, °C/K), mortgage ("k" / "million" amounts), creatinine clearance
(umol/L, lb), DDB book value ("k" / "million"). For each family and builder model:

  1. backlog: the family's first 6 positives, each with its CORRECT answer as the
     production response, so only the builder is being tested (not the chatbot's maths).
  2. build: labelfree.build_family, the simple flow exactly as in production (re-solve the
     newest question with a program, build, conventions + output specs, pre-publish checks).
     Only the builder role changes model; every other role stays Haiku.
  3. score: if the entry goes live, run it on the next 12 positives (forced entry, the
     serving gate and code as in production) and score each answer strictly against the
     computed truth (arc.checks.all_values_match).

Results: runs/builder_compare/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ["perfect_gas_pressure", "loan_payment_total_interest", "cockcroft_gault_crcl", "ddb_book_value"]
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
N_BACKLOG, N_SCORE = 6, 12


def family_split(rows: list[dict], family: str) -> tuple[list[dict], list[dict]]:
    """(backlog records, held-out scoring items) for one family, disjoint, in corpus order."""
    pos = [r for r in rows if r["family"] == family and r["kind"] == "positive"]
    records = [{"id": r["id"], "question": r["question"], "response": r["truth"][0], "model": "claude-haiku-5-5",
                "timestamp": float(i), "served_by": None} for i, r in enumerate(pos[:N_BACKLOG])]
    return records, pos[N_BACKLOG:N_BACKLOG + N_SCORE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    os.environ["ARC_PROFILE"] = "haiku"
    os.environ["ARC_GATE"] = "clef"
    sys.path.insert(0, str(ROOT))
    from arc import config, labelfree, llm, serve
    from arc.cache import Cache
    from arc.checks import all_values_match

    rows = [json.loads(l) for l in (ROOT / "eval/labelfree/corpus.jsonl").read_text().splitlines() if l.strip()]
    out_dir = ROOT / "runs" / "builder_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models.split(","):
        tag = model.split("-")[1]            # haiku / sonnet
        config.BUILDER_MODEL = model
        config.LABELFREE_BUILDERS = [model]   # one builder per arm (build_family follows this list)
        config.WORK_DIR = ROOT / "runs" / f"builder_compare_{tag}"
        config.CACHE_ROOT = ROOT / "caches" / f"builder_compare_{tag}"
        for family in args.families.split(","):
            records, held_out = family_split(rows, family)
            before, t0 = llm.LEDGER.cost(), time.time()
            res = labelfree.build_family(family, records)
            row = {"model": model, "family": family, "status": res.status, "reason": (res.reason or "")[:400],
                   "build_cost": round(res.cost or 0.0, 5), "parts": res.parts, "seconds": round(time.time() - t0)}
            if res.status == "accepted":
                cache = Cache.load(config.cache_dir("master"))
                ep = next(e for e in cache.entry_points if e.problem_id == res.entry_id and e.kind == "original")
                counts = {"right": 0, "wrong": 0, "declined": 0}
                detail = []
                for item in held_out:
                    try:
                        hit = serve.use(cache, item["question"], entry_point=ep, tag=f"cmp:{tag}:{item['id']}")
                    except serve.CacheInvocationError as e:
                        hit, why = None, str(e)[:120]
                    else:
                        why = "" if hit else "gate turned it away"
                    if hit is None:
                        counts["declined"] += 1
                        detail.append({"id": item["id"], "form": item["form"], "outcome": "declined", "why": why})
                        continue
                    answer = serve.format_output(hit.output)
                    ok = any(all_values_match(t, answer) for t in item["truth"])
                    counts["right" if ok else "wrong"] += 1
                    detail.append({"id": item["id"], "form": item["form"], "outcome": "right" if ok else "WRONG",
                                   "answer": answer[:200], "truth": item["truth"]})
                row.update(score=counts, detail=detail)
            row["total_cost"] = round(llm.LEDGER.cost() - before, 5)
            results.append(row)
            print(f"{tag:6s} {family:30s} {row['status']:9s} build ${row['build_cost']:.4f}  "
                  f"{row.get('score', '')}  {row['reason'][:90]}", flush=True)
            (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"total ${llm.LEDGER.cost():.4f}; saved runs/builder_compare/results.json")


if __name__ == "__main__":
    main()
