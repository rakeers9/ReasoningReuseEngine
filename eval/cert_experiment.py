"""Test the agreement certificate (label-free builds). PAID: calls the API.

    .venv/bin/python eval/cert_experiment.py faults   # broken gas-law transcripts + 1 clean control
    .venv/bin/python eval/cert_experiment.py seeds    # the 17 corpus seeds, no golden label

Every problem: source solves (Haiku profile) -> builder WITHOUT the golden label ->
certificate. Afterwards, and only for scoring the experiment, a judge compares the
transcript and the cache with the golden label, so we can tell whether each accept /
reject decision was right. The label never influences the build or the certificate.

Results: runs/<namespace>/cert_experiment_<set>.json, plus each problem's
build_report.json and certificate.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each fault is a reasoning error rewritten consistently into a correct transcript.
FAULTS = {
    "kelvin": "Forget to convert the temperature to kelvin: use T = 27 (the Celsius number) directly "
              "as if it were in kelvin in pV = nRT.",
    "gas_constant": "Use the wrong gas constant R = 0.08314 J K-1 mol-1 (100 times too small) in the SI "
                    "calculation.",
    "volume": "Convert the volume wrongly: treat 1 dm3 as 1e-6 m3 (it is really 1e-3 m3).",
    "final_slip": "Do every step correctly, but make an arithmetic slip in the very last division so the "
                  "final pressure is stated as 57.3 atm (and the matching wrong value in other units).",
}


def _setup(profile: str, namespace: str) -> None:
    os.environ["ARC_PROFILE"] = profile
    os.environ["ARC_NAMESPACE"] = namespace
    sys.path.insert(0, str(ROOT))


def _diagnose(llm_mod, judges, question: str, golden: str, answer: str, tag: str) -> bool | None:
    """Scoring only: is `answer` correct per the golden label?"""
    if not answer:
        return None
    v = judges.judge(f"## Question\n{question}\n\n## Golden label\n{golden}\n\n## Answer\n{answer[-3000:]}\n\n"
                     "Is the answer's final result correct per the golden label (up to rounding and units)?",
                     role="diagnostic_judge", tag=tag)
    return v.passed


def run_one(problem: dict, profile: str, namespace: str, transcript_override: dict | None = None) -> dict:
    """Solve (or use the given transcript), build without golden, certify, then diagnose."""
    _setup(profile, namespace)
    from arc import certify, config, judges, llm, source
    from arc.builder import Builder, BuildError

    pid = problem["id"]
    t0 = time.time()
    row: dict = {"problem": pid}
    folder = config.problem_dir(pid)
    try:
        if transcript_override:
            t = source.Transcript(**transcript_override)
            source.save(t)
        elif (folder / "transcript.json").exists():
            t = source.load(pid)
        else:
            t = source.solve(problem)
            source.save(t)
        b = Builder(problem, t, use_golden=False)
        try:
            cache = b.build()
            b.save_report()
            cache.save(config.CACHE_ROOT / pid)
            row["built"] = True
        except BuildError as e:
            b.save_report()
            row.update(built=False, build_error=str(e)[:300])
            cache = None
        if cache is not None:
            cert = certify.certify(cache, problem, t)
            certify.save(cert)
            row.update(accepted=cert.accepted, reason=cert.reason, cert_cost=cert.cost,
                       comparisons=[{"name": c.name, "agree": c.agree, "judge": c.judge_said,
                                     "numbers": c.number_status, "fresh": c.reference[:120],
                                     "candidate": c.candidate[-160:]} for c in cert.comparisons],
                       instances=cert.instances, cache_original=cert.cache_original[:300],
                       fresh_original=cert.fresh_original[:200])
        else:
            row.update(accepted=False, reason="build failed")
        # ---- diagnostics only (golden label), after every decision is made ----
        g = problem["golden_label"]
        row["transcript_correct"] = _diagnose(llm, judges, t.question, g, t.solution, f"{pid}:diag:transcript")
        row["cache_correct"] = _diagnose(llm, judges, t.question, g, row.get("cache_original", ""),
                                         f"{pid}:diag:cache")
        row["fresh_correct"] = _diagnose(llm, judges, t.question, g, row.get("fresh_original", ""),
                                         f"{pid}:diag:fresh")
    except Exception as e:  # one broken problem must not stop the batch
        row.update(accepted=False, reason=f"error: {type(e).__name__}: {e}"[:300])
    row["cost"] = llm.LEDGER.cost()
    row["seconds"] = round(time.time() - t0)
    return row


def make_faults(profile: str, namespace: str) -> list[tuple[dict, dict | None]]:
    """Gas-law problem copies: one clean control plus one per fault, transcripts rewritten by the generator."""
    _setup(profile, namespace)
    from arc import config, llm
    base_problem = json.loads((config.PROBLEMS_DIR / "gas_law.json").read_text())
    base_t = json.loads((ROOT / "runs" / "haiku" / "gas_law" / "transcript.json").read_text())
    jobs = [({**base_problem, "id": "gas_law_clean"}, {**base_t, "problem_id": "gas_law_clean"})]
    for name, fault in FAULTS.items():
        pid = f"gas_law_fault_{name}"
        cached = config.problem_dir(pid) / "transcript.json"
        if cached.exists():
            jobs.append(({**base_problem, "id": pid}, None))
            continue
        r = llm.call("generator",
                     "Rewrite this worked solution so that it contains exactly the following error, "
                     "carried through consistently to the final answer. Keep the same structure and "
                     "style, and do not mention or hint that anything is wrong.\n\n"
                     f"## Error to introduce\n{fault}\n\n## Worked solution\n{base_t['solution']}",
                     model=config.GENERATOR_MODEL, effort="low", max_tokens=8000, tag=f"{pid}:fault")
        jobs.append(({**base_problem, "id": pid}, {**base_t, "problem_id": pid, "solution": r.text,
                                                   "thinking_summary": f"FAULT INJECTED: {fault}"}))
    print(f"fault transcripts written, ${llm.LEDGER.cost():.4f}")
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["faults", "seeds"])
    ap.add_argument("--profile", default="haiku")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    namespace = {"faults": "cert_faults", "seeds": "corpus_cert"}[args.set]

    if args.set == "faults":
        jobs = make_faults(args.profile, namespace)
    else:
        jobs = [(json.loads(p.read_text()), None) for p in sorted((ROOT / "problems" / "corpus").glob("*.json"))]
    if args.only:
        jobs = [j for j in jobs if j[0]["id"] in args.only.split(",")]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, p, args.profile, namespace, t): p["id"] for p, t in jobs}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{r['problem']:34s} built={r.get('built')!s:5s} accepted={r.get('accepted')!s:5s} "
                  f"transcript_ok={r.get('transcript_correct')!s:5s} cache_ok={r.get('cache_correct')!s:5s} "
                  f"${r['cost']:.4f} {r['seconds']}s  {r.get('reason', '')[:70]}", flush=True)

    _setup(args.profile, namespace)
    from arc import config
    from arc.cache import Cache
    master = Cache("master")
    for r in rows:   # merge accepted entries only, in one process (no write races)
        if r.get("accepted"):
            master.merge(Cache.load(config.CACHE_ROOT / r["problem"]))
    if master.entry_points:
        master.save(config.CACHE_ROOT / "master")

    acc = [r for r in rows if r.get("accepted")]
    rej = [r for r in rows if not r.get("accepted")]
    summary = {
        "problems": len(rows), "accepted": len(acc), "rejected": len(rej),
        "accepted_and_cache_correct": sum(r.get("cache_correct") is True for r in acc),
        "accepted_but_cache_wrong": [r["problem"] for r in acc if r.get("cache_correct") is False],
        "rejected_but_cache_correct": [r["problem"] for r in rej if r.get("cache_correct") is True],
        "transcript_wrong": [r["problem"] for r in rows if r.get("transcript_correct") is False],
        "total_cost": round(sum(r["cost"] for r in rows), 4),
    }
    out = config.WORK_DIR / f"cert_experiment_{args.set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": sorted(rows, key=lambda r: r["problem"])},
                              indent=2, default=str))
    print("\n" + json.dumps(summary, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
