"""Command line for the ARC replication.

    python -m arc paper   gas_law        Step 1: run the paper's own published functions
    python -m arc ask     <cache> "..."  Step 2: serve a question (cache first, LLM fallback)
    python -m arc solve   gas_law        Step 3: source model writes a worked solution
    python -m arc build   gas_law        Step 4: builder turns the transcript into a cache entry
    python -m arc eval    gas_law        Step 5: correctness, perturbations, costs
    python -m arc compile gas_law        Step 6: static compilation to plain Python
    python -m arc all     gas_law        Steps 3-6 in order
    python -m arc costs                  Cost by role for this process (printed at the end)
    python -m arc families               Which backlog task families are worth building (free)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
import traceback

from . import config, llm
from .cache import Cache


def load_problem(pid: str) -> dict:
    return json.loads((config.PROBLEMS_DIR / f"{pid}.json").read_text())


def _rel(path) -> str:
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:   # outside the project (custom paths)
        return str(path)


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def print_costs(since: int = 0) -> None:
    rows = llm.LEDGER.by_role(since)
    if not rows:
        return
    print("\nCost by role (this run):")
    for role, r in rows.items():
        print(f"  {role:12s} {int(r['calls']):4d} calls  in {int(r['input']):8d}  out {int(r['output']):7d}"
              f"  cache-read {int(r['cache_read']):8d}  ${r['cost']:.4f}")
    print(f"  {'total':12s} ${llm.LEDGER.cost(since=since):.4f}")


# ----------------------------------------------------------------------------

def cmd_paper(args) -> None:
    from . import paper_entries
    from .serve import dispatch_root
    problem = load_problem(args.problem)
    header(f"Step 1: the paper's published cache entry for {problem['id']}")
    cache = paper_entries.build_paper_cache(problem)
    ep = next(e for e in cache.entry_points if e.kind == "original")
    values = {p.description: p.original_value for p in ep.params}
    print("Parameters placed in the first context layer:")
    for k, v in values.items():
        print(f"  {k!r}: {v!r}")
    output, ctx = dispatch_root(cache, ep.root_id, values)
    print("\nOutput dictionary:")
    for k, v in output.items():
        print(f"  {k!r}: {v!r}")
    print("\nNarration trace:\n" + "\n".join("  " + line for line in ctx.narration().splitlines()))
    print("\nMeaning-based lookups (query -> matched, similarity):")
    for l in ctx.lookups:
        print(f"  {l.kind:11s} {l.similarity:.3f}  {l.query[:55]!r} -> {l.matched[:55]!r}")
    cache.save(config.CACHE_ROOT / f"paper_{problem['id']}")
    print(f"\nSaved to {_rel(config.CACHE_ROOT / f'paper_{problem['id']}')}/")


def cmd_ask(args) -> None:
    from .serve import serve
    cache = Cache.load(config.cache_dir(args.cache))
    header(f"Step 2: serving through cache '{args.cache}'")
    print(f"Question: {args.question}\n")
    started = time.time()
    served = serve(cache, args.question)
    print(f"Path: {'CACHE' if served.from_cache else 'FULL LLM'} ({served.reason}) "
          f"in {time.time() - started:.1f}s, ${served.cost:.5f}")
    r = served.cache_result
    if r:
        print(f"Entry point ({r.entry_point.kind}, similarity {r.similarity:.3f}): {r.entry_point.text[:90]}")
        print("Extracted parameters:")
        for k, v in r.parameters.items():
            print(f"  {k!r}: {v!r}")
        print("\nNarration:\n" + "\n".join("  " + line for line in r.narration.splitlines()))
    print("\nAnswer:\n" + served.answer)


def cmd_solve(args) -> None:
    from . import source
    problem = load_problem(args.problem)
    header(f"Step 3: source model solves {problem['id']}")
    t = source.solve(problem)
    source.save(t)
    print(t.solution)
    print(f"\n(thinking summary saved separately; ${t.cost:.4f}) -> {_rel(config.problem_dir(problem['id']))}/transcript.md")


def _record_build(problem_id: str, accepted: bool) -> None:
    """Log this build's measured cost for the cache-worthiness threshold (arc/families.py)."""
    from . import families
    families.record_build_cost(problem_id, llm.LEDGER.cost(), accepted,
                               {role: round(r["cost"], 6) for role, r in llm.LEDGER.by_role().items()})


def cmd_families(args) -> None:
    from . import backlog, families
    from pathlib import Path
    tracker = families.FamilyTracker()
    path = Path(args.backlog) if args.backlog else backlog.default_path()
    if args.reset:
        tracker = families.FamilyTracker(path=tracker.path)
        tracker.families, tracker.ingested_lines, tracker.seen_ids, tracker.latest = {}, 0, set(), 0.0
    added = tracker.ingest(path)
    tracker.save()
    cost = families.effective_build_cost()
    rows = tracker.readiness(now=args.now)
    if args.json:
        from dataclasses import asdict
        print(json.dumps({"build_cost": cost, "families": [asdict(r) for r in rows]}, indent=2, default=str))
        return
    print(f"Read {added} new records from {path}; {len(tracker.families)} families "
          f"(state: {_rel(tracker.path)})\n")
    print(families.format_report(rows, cost, limit=args.limit))


def cmd_build(args) -> None:
    from . import source
    from .builder import Builder, BuildError
    problem = load_problem(args.problem)
    header(f"Step 4: building the cache entry for {problem['id']}")
    t = source.load(problem["id"])
    verify = getattr(args, "verify", "none")
    # --verify agreement replaces the golden label with the agreement certificate (DEVIATION).
    b = Builder(problem, t, use_golden=not args.no_golden and verify != "agreement")
    try:
        cache = b.build()
    except BuildError as e:
        b.save_report()
        _record_build(problem["id"], False)
        print(f"\nBUILD FAILED: {e}\n(report: {_rel(config.problem_dir(problem['id']))}/build_report.json)")
        sys.exit(1)
    b.save_report()
    cache.save(config.CACHE_ROOT / problem["id"])
    if verify == "agreement":
        from . import certify
        header(f"Agreement certificate for {problem['id']} (no golden label)")
        cert = certify.certify(cache, problem, t)
        certify.save(cert)
        for c in cert.comparisons:
            print(f"  {'agree' if c.agree else 'DISAGREE':8s} {c.name}: fresh={c.reference[:60]!r} "
                  f"vs {c.candidate[-60:]!r} (judge {c.judge_said}, numbers {c.number_status})")
        print(f"\nCertificate: {'ACCEPTED' if cert.accepted else 'REJECTED'} ({cert.reason}); ${cert.cost:.4f}")
        print(f"Diagnostics only (golden label): {cert.golden_diagnostic}")
        _record_build(problem["id"], cert.accepted)
        if not cert.accepted:
            print("Entry NOT merged into master: questions like this keep going to the source model.")
            sys.exit(2)
    if verify != "agreement":
        _record_build(problem["id"], True)
    master_dir = config.CACHE_ROOT / "master"
    master = Cache.load(master_dir) if (master_dir / "registry.json").exists() else Cache("master")
    master.merge(cache)
    master.save(master_dir)
    print(f"\nBuilt {len(cache.functions)} functions, {len(cache.entry_points)} entry points; "
          f"builder cost ${b.report.cost:.4f}")
    print(f"Saved to {_rel(config.CACHE_ROOT / problem['id'])}/ and merged into {_rel(master_dir)}/")


def cmd_eval(args) -> None:
    from . import source
    from .evaluate import Evaluator, pass_rates
    problem = load_problem(args.problem)
    header(f"Step 5: evaluating the cache on {problem['id']}")
    t = source.load(problem["id"])
    build = json.loads((config.problem_dir(problem["id"]) / "build_report.json").read_text())
    cache = Cache.load(config.cache_dir(args.cache))
    ev = Evaluator(problem, t, cache, build["cost"])
    report = ev.run()
    ev.save()
    c = report.costs
    rates = pass_rates(report)
    print("\nResults (paper Table 2 columns):")
    print(f"  S Acc {report.source_correct} | C Acc {report.cache_correct} | Fidelity {report.fidelity}")
    print(f"  Phrasal {rates['phrasal']} | Parametric {rates['parametric']} | Combined {rates['combined']}")
    print(f"  Source ${c['source_cost']:.4f} | Builder ${c['builder_cost']:.4f} | "
          f"Cache ${c['cache_cost_per_invocation']:.5f} per use")
    if c["build_multiplier"]:
        print(f"  Build x {c['build_multiplier']:.1f} | Savings x "
              f"{c['savings_multiplier'] or float('nan'):.1f} | break-even after "
              f"{c['break_even_invocations']} uses")
    print(f"(report: {_rel(config.problem_dir(problem['id']))}/eval_report.json)")


def cmd_compile(args) -> None:
    from .compiler import compile_cache
    from .serve import dispatch_root, extract_parameters
    problem = load_problem(args.problem)
    header(f"Step 6: compiling the cache entry for {problem['id']}")
    cache = Cache.load(config.cache_dir(args.cache))
    ep = next(e for e in cache.entry_points if e.problem_id == problem["id"] and e.kind == "original")
    code = compile_cache(cache, ep)
    out_path = config.cache_dir(args.cache) / f"compiled_{problem['id']}.py"
    out_path.write_text(code)
    print(code)
    print(f"\nWritten to {out_path.relative_to(config.ROOT)}")

    spec = importlib.util.spec_from_file_location("compiled_cache", out_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = extract_parameters(problem["question"], ep.params) if args.extract else \
        {p.description: p.original_value for p in ep.params}
    t0 = time.time()
    dynamic, _ = dispatch_root(cache, ep.root_id, values)
    t1 = time.time()
    compiled, _ = module.run(values)
    t2 = time.time()
    print("\nSame inputs through both versions:")
    for k in compiled:
        print(f"  {k!r}: compiled={compiled[k]!r}  uncompiled={dynamic.get(k)!r}")
    print(f"  match: {compiled == dynamic} | uncompiled {t1 - t0:.2f}s, compiled {t2 - t1:.2f}s")


def cmd_all(args) -> None:
    for step in (cmd_solve, cmd_build, cmd_eval, cmd_compile):
        step(args)


def stop_on_ctrl_break() -> None:
    """Windows: the web UI stops a run with Ctrl+Break (it can't send Ctrl+C to another
    process). Raise KeyboardInterrupt for it, as for Ctrl+C, so the run still records its
    result and costs. No-op where SIGBREAK doesn't exist."""
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)


def main() -> None:
    stop_on_ctrl_break()
    parser = argparse.ArgumentParser(prog="python -m arc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [("paper", cmd_paper), ("solve", cmd_solve), ("build", cmd_build),
                     ("eval", cmd_eval), ("compile", cmd_compile), ("all", cmd_all)]:
        p = sub.add_parser(name)
        p.add_argument("problem")
        p.set_defaults(fn=fn)
        if name in ("build", "all"):
            p.add_argument("--no-golden", action="store_true", help="build without the answer key")
            p.add_argument("--verify", default="none", choices=["none", "agreement"],
                           help="agreement: no golden label; merge only if the certificate passes")
        if name in ("eval", "compile", "all"):
            p.add_argument("--cache", default="master")
        if name in ("compile", "all"):
            p.add_argument("--extract", action="store_true", help="extract parameters with the target model")
    p = sub.add_parser("families", help="cache-worthiness report from the backlog (no API calls)")
    p.add_argument("--backlog", default="", help="backlog .jsonl (default runs/<namespace>/backlog.jsonl)")
    p.add_argument("--reset", action="store_true", help="forget saved families and re-read the whole backlog")
    p.add_argument("--now", type=float, default=None, help="unix time to evaluate at (default: newest record)")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_families)
    p = sub.add_parser("ask")
    p.add_argument("cache")
    p.add_argument("question")
    p.set_defaults(fn=cmd_ask)
    args = parser.parse_args()
    folder, started = start_experiment(args)
    print(f"[profile {config.PROFILE}] source {config.SOURCE_MODEL} · builder {config.BUILDER_MODEL} · "
          f"judges {config.JUDGE_MODEL} · generator {config.GENERATOR_MODEL} · target {config.TARGET_MODEL} · "
          f"outputs in {_rel(config.WORK_DIR)}/ and {_rel(config.CACHE_ROOT)}/", flush=True)
    status, code, error = "ok", 0, None
    try:
        args.fn(args)
        print_costs()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        status = "ok" if code == 0 else "failed"
        print_costs()
        raise
    except KeyboardInterrupt:
        status, code, error = "stopped", 130, "stopped by user"
        print_costs()
        raise
    except BaseException as e:
        status, code, error = "error", 1, "".join(traceback.format_exception_only(type(e), e)).strip()
        print_costs()
        raise
    finally:
        finish_experiment(folder, args, started, status, code, error)


# ----------------------------------------------------------------------------
# Experiment bookkeeping: each run gets runs/experiments/<run_id>/ with its
# settings, every model call (written by llm.py), and copies of the report
# files it changed. Nothing here affects what the pipeline does.
# ----------------------------------------------------------------------------

def start_experiment(args) -> tuple:
    run_id = os.environ.get("ARC_RUN_ID") or f"cli-{time.strftime('%Y%m%d-%H%M%S')}-{args.cmd}"
    os.environ["ARC_RUN_ID"] = run_id
    folder = config.RUNS_DIR / "experiments" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (folder / "settings.json").write_text(json.dumps({
        "run_id": run_id, "command": args.cmd,
        "args": {k: v for k, v in vars(args).items() if k not in ("fn", "cmd")},
        "argv": sys.argv[1:], "started": started, "settings": config.snapshot(),
    }, indent=2, default=str))
    return folder, started


def finish_experiment(folder, args, started: float, status: str, code: int, error: str | None) -> None:
    copied = []
    problem = getattr(args, "problem", None)
    run_dir = config.problem_dir(problem) if problem else None
    if run_dir and run_dir.is_dir():
        for f in sorted(run_dir.iterdir()):
            if f.is_file() and f.suffix in (".json", ".md") and f.stat().st_mtime >= started:
                (folder / "files").mkdir(exist_ok=True)
                shutil.copy2(f, folder / "files" / f.name)
                copied.append(f.name)
    if problem and args.cmd in ("compile", "all"):
        compiled = config.cache_dir(getattr(args, "cache", "master")) / f"compiled_{problem}.py"
        if compiled.exists() and compiled.stat().st_mtime >= started:
            (folder / "files").mkdir(exist_ok=True)
            shutil.copy2(compiled, folder / "files" / compiled.name)
            copied.append(compiled.name)
    (folder / "result.json").write_text(json.dumps({
        "finished": time.time(), "seconds": round(time.time() - started, 2), "status": status,
        "exit_code": code, "error": error, "files": copied,
        "ledger_cost": llm.LEDGER.cost(), "ledger_by_role": llm.LEDGER.by_role(),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
