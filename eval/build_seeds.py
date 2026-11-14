"""Solve and build every corpus seed into one cache (PAID: calls the API).

    .venv/bin/python eval/build_seeds.py [--profile haiku] [--budget 1.00] [--only id1,id2]

Writes to its own namespace (caches/corpus/, runs/corpus/) so it never mixes
with the gas-law caches. Each seed gets merged into caches/corpus/master/.
A seed that fails to build is logged and skipped; the harness reports which
entries are missing. Seeds already built are skipped unless --rebuild.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="haiku", choices=["haiku", "paper"])
    ap.add_argument("--namespace", default="corpus")
    ap.add_argument("--budget", type=float, default=1.00, help="stop starting new seeds past this spend ($)")
    ap.add_argument("--only", default="", help="comma-separated entry ids")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-golden", action="store_true")
    ap.add_argument("--no-cards", action="store_true", help="skip writing gate cards")
    ap.add_argument("--cards-only", action="store_true", help="(re)write gate cards for built entries only")
    args = ap.parse_args()

    os.environ["ARC_PROFILE"] = args.profile
    os.environ["ARC_NAMESPACE"] = args.namespace
    sys.path.insert(0, str(ROOT))
    from arc import config, llm
    from arc import source
    from arc.__main__ import cmd_build, cmd_solve, load_problem
    from arc.cache import Cache
    from arc.gatecard import make_card

    def write_card(pid: str) -> None:
        """Gate card for one built entry, saved in its own cache and in master."""
        problem = load_problem(f"corpus/{pid}")
        own = Cache.load(config.CACHE_ROOT / pid)
        own.cards[pid] = make_card(own, problem, source.load(pid), use_golden=not args.no_golden)
        own.save(config.CACHE_ROOT / pid)
        master = Cache.load(config.CACHE_ROOT / "master")
        master.cards[pid] = own.cards[pid]
        master.save(config.CACHE_ROOT / "master")

    ids = sorted(p.stem for p in (config.PROBLEMS_DIR / "corpus").glob("*.json"))
    if args.only:
        ids = [i for i in ids if i in args.only.split(",")]
    log = []
    for pid in ids:
        if llm.LEDGER.cost() >= args.budget:
            print(f"\nbudget ${args.budget:.2f} reached, stopping before {pid}")
            break
        built = (config.CACHE_ROOT / pid / "registry.json").exists()
        if args.cards_only:
            if built:
                before = llm.LEDGER.cost()
                write_card(pid)
                print(f"card {pid}  ${llm.LEDGER.cost() - before:.4f}")
            continue
        if not args.rebuild and built:
            print(f"skip {pid} (already built)")
            continue
        before, t0 = llm.LEDGER.cost(), time.time()
        status = "built"
        try:
            ns = argparse.Namespace(problem=f"corpus/{pid}", no_golden=args.no_golden)
            if not (config.problem_dir(pid) / "transcript.json").exists():
                cmd_solve(ns)
            cmd_build(ns)
            if not args.no_cards:
                write_card(pid)
        except SystemExit:
            status = "build_failed"
        except Exception as e:  # keep going; one bad seed should not stop the batch
            status = f"error: {type(e).__name__}: {e}"
        row = {"entry": pid, "status": status, "cost": llm.LEDGER.cost() - before,
               "seconds": round(time.time() - t0)}
        log.append(row)
        print(f"\n>>> {pid}: {status}  ${row['cost']:.4f}  {row['seconds']}s  (total ${llm.LEDGER.cost():.4f})")

    if args.cards_only:
        print(f"cards done, ${llm.LEDGER.cost():.4f}")
        return
    out = config.WORK_DIR / "build_seeds_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(old + log, indent=2))
    print(f"\n{sum(r['status'] == 'built' for r in log)}/{len(log)} built, ${llm.LEDGER.cost():.4f} spent; log: {out}")


if __name__ == "__main__":
    main()
