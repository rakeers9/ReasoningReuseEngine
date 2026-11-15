"""Test the explainer with Haiku (PAID, about $0.15).

    .venv/bin/python eval/explain_experiment.py

1. Production: Haiku answers each of the 17 corpus seeds with ONLY the final answer
   (no working), like a model that shows no reasoning. These become backlog-style
   records {id, question, response}.
2. Planted errors: 4 extra records whose production answer is deliberately wrong.
3. The explainer (Haiku) runs on all 21 records.
4. Diagnostics only: a judge compares the production answer and the explainer's
   transcript with the golden label, to score every decision.
5. End to end: 3 explained seeds are built WITHOUT the golden label from the
   explainer's transcript, and the cache is scored on the original and on its dev
   should-hit corpus rows.

Results: runs/explain/explain_experiment.json
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["ARC_PROFILE"] = "haiku"
os.environ["ARC_NAMESPACE"] = "explain"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from arc import config, explain, judges, llm  # noqa: E402

PRODUCTION_SYSTEM = "Reply with only the final answer. Do not show any working, steps or reasoning."
# Deliberately wrong production answers (correct ones in comments), written as a model would.
WRONG = {
    "perfect_gas_pressure": "65.9 atm",                                          # 50.68 atm
    "loan_payment_total_interest": "Payment $2,401.37/month; total interest $514,493.20",  # $2,155.01; $425,803.67
    "cockcroft_gault_crcl": "44.3 mL/min",                                       # 37.64 mL/min
    "hardy_weinberg_carriers": "784 people",                                     # 1,008 people
}
BUILD = ["perfect_gas_pressure", "bayes_ppv", "ddb_book_value"]


def production_answer(problem: dict) -> str:
    r = llm.call("production", problem["question"], model="claude-haiku-5-5", system=PRODUCTION_SYSTEM,
                 effort="medium", max_tokens=2000, tag=f"{problem['id']}:production")
    return r.text.strip()


def correct_per_golden(question: str, golden: str, answer: str, tag: str) -> bool:
    v = judges.judge(f"## Question\n{question}\n\n## Golden label\n{golden}\n\n## Answer\n{answer[-3000:]}\n\n"
                     "Is the answer's final result correct per the golden label (up to rounding and units)?",
                     role="diagnostic_judge", tag=tag)
    return v.passed


def build_and_score(pid: str) -> dict:
    """Build one entry from its explainer transcript (no golden label), score it on corpus rows."""
    os.environ["ARC_PROFILE"] = "haiku"
    os.environ["ARC_NAMESPACE"] = "explain"
    from arc import source
    from arc.builder import Builder, BuildError
    from arc.serve import dispatch_root, extract_parameters, format_output
    from run_corpus import score
    problem = json.loads((config.PROBLEMS_DIR / "corpus" / f"{pid}.json").read_text())
    ex = json.loads((config.problem_dir(pid) / "explanation.json").read_text())
    t = source.Transcript(**ex["transcript"])
    source.save(t)
    before = llm.LEDGER.cost()
    b = Builder(problem, t, use_golden=False)
    try:
        cache = b.build()
    except BuildError as e:
        b.save_report()
        return {"entry": pid, "built": False, "error": str(e)[:200], "cost": llm.LEDGER.cost() - before}
    b.save_report()
    cache.save(config.CACHE_ROOT / pid)
    ep = next(e for e in cache.entry_points if e.kind == "original")
    corpus = [json.loads(l) for l in (ROOT / "eval/corpus/cache_eval_corpus.jsonl").read_text().splitlines()]
    rows = [r for r in corpus if r["expected_entry"] == pid and r["should_hit"] and r["split"] in ("seed", "dev")]
    counts = {"right": 0, "wrong": 0, "crash": 0, "unscored": 0}
    for r in rows:
        try:
            out = format_output(dispatch_root(cache, ep.root_id, extract_parameters(r["question"], ep.params))[0])
            ok, _ = score(r, out)
            counts["right" if ok else "wrong" if ok is False else "unscored"] += 1
        except Exception:
            counts["crash"] += 1
    return {"entry": pid, "built": True, "rows": len(rows), **counts, "cost": llm.LEDGER.cost() - before}


def main() -> None:
    problems = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
                for p in sorted((config.PROBLEMS_DIR / "corpus").glob("*.json"))}

    # 1-2. backlog-style records
    with ThreadPoolExecutor(6) as pool:
        answers = dict(zip(problems, pool.map(production_answer, problems.values())))
    records = [{"id": pid, "question": p["question"], "response": answers[pid], "planted_wrong": False}
               for pid, p in problems.items()]
    records += [{"id": f"{pid}__wrong", "question": problems[pid]["question"], "response": wrong,
                 "planted_wrong": True} for pid, wrong in WRONG.items()]
    stage1 = llm.LEDGER.cost()
    print(f"production answers written: ${stage1:.4f}")

    # 3. explainer
    def run(rec: dict) -> dict:
        ex = explain.explain(rec)
        explain.save(ex)
        pid = rec["id"].split("__")[0]
        golden = problems[pid]["golden_label"]
        prod_ok = correct_per_golden(rec["question"], golden, rec["response"], f"{rec['id']}:diag:prod")
        expl_ok = correct_per_golden(rec["question"], golden, ex.transcript["solution"], f"{rec['id']}:diag:expl") \
            if ex.transcript else None
        return {"id": rec["id"], "planted_wrong": rec["planted_wrong"], "production": rec["response"][:160],
                "production_correct": prod_ok, "decision": ex.decision, "reason": ex.reason,
                "transcript_correct": expl_ok, "blind_answers": [a.final_answer[:100] for a in ex.attempts],
                "diagnosis": (ex.diagnosis or {}).get("verdict"),
                # by tag, not ex.cost: threads share the ledger, so position-based spans overlap
                "cost": sum(x.cost for x in list(llm.LEDGER.records) if (x.tag or "").startswith(rec["id"] + ":"))}
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(run, records))
    for r in rows:
        print(f"{r['id']:38s} prod_ok={r['production_correct']!s:5s} -> {r['decision']:13s} "
              f"transcript_ok={r['transcript_correct']!s:5s} | {r['reason'][:80]}")
    stage3 = llm.LEDGER.cost()

    # decision quality
    def cell(prod_ok, explained):
        return sum(r["production_correct"] is prod_ok and (r["decision"] == "explained") is explained for r in rows)
    matrix = {"production right, explained (good)": cell(True, True),
              "production right, not explained (lost entry)": cell(True, False),
              "production wrong, not explained (good)": cell(False, False),
              "production wrong, explained (BAD)": cell(False, True),
              "explained but transcript wrong (BAD)": sum(r["decision"] == "explained" and r["transcript_correct"] is False for r in rows)}
    print(json.dumps(matrix, indent=2))

    # 5. end to end
    todo = [pid for pid in BUILD if next(r for r in rows if r["id"] == pid)["decision"] == "explained"]
    with ProcessPoolExecutor(3) as pool:
        builds = list(pool.map(build_and_score, todo))
    for b in builds:
        print("build", b)
    build_cost = sum(b["cost"] for b in builds)

    out = config.WORK_DIR / "explain_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"matrix": matrix, "rows": rows, "builds": builds,
                               "cost": {"production": round(stage1, 4), "explainer_and_diagnostics": round(stage3 - stage1, 4),
                                        "builds": round(build_cost, 4)}}, indent=2, default=str))
    print(f"total ${stage3 + build_cost:.4f}; saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
