"""Does label-free certification accept good entries and reject bad ones?

    .venv/bin/python eval/labelfree/run_eval.py make                     # free: corpus.jsonl
    .venv/bin/python eval/labelfree/run_eval.py production --max-cost 0.2  # PAID: production answers
    .venv/bin/python eval/labelfree/run_eval.py run --max-cost 1.0       # PAID: every condition
    .venv/bin/python eval/labelfree/run_eval.py report                   # free: tables from saved results

Ground truth comes from hand-written reference solvers (families.py), never a model.
The entries under test are the 13 existing no-golden builds in caches/corpus_cert/.

Conditions (each certifies every entry on its family's backlog, gate off):
  clean       positives only, real production answers
  noisy       clean, but NOISE_RATE of production answers replaced with wrong ones
  near_miss   positives + near-misses mixed (no gate, so near-misses get served)
  mutants     clean backlog, against MUTANTS_PER_ENTRY code mutants of each entry

For each entry/mutant, the TRUE error rate is measured by running it on every
positive and scoring against the reference answer. An accept is a FALSE ACCEPT if the
entry served any wrong answer on the questions it would serve. That's the safety number.

Every model call is memoized in memo.json (extraction, judge, blind solve,
production answer), so a rerun or a new condition only pays for new work.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

CORPUS = HERE / "corpus.jsonl"
PRODUCTION = HERE / "production.jsonl"
MEMO = HERE / "memo.json"
RESULTS = HERE / "results"
ENTRIES = ROOT / "caches" / "corpus_cert"

POSITIVES, NEAR_MISSES = 30, 6
NOISE_RATE = 0.2
MUTANTS_PER_ENTRY = 3
SEED = 20261114


# ----------------------------------------------------------------------------- make (free)

def cmd_make(args) -> None:
    import families as fam
    rng = random.Random(SEED)
    items = []
    for name, gen in sorted(fam.GENERATORS.items()):
        fam_items = [gen(rng, "positive") for _ in range(POSITIVES)] + \
                    [gen(rng, "near_miss") for _ in range(NEAR_MISSES)]
        rng.shuffle(fam_items)
        for i, it in enumerate(fam_items):
            items.append({"id": f"{name[:10]}_{i:02d}", "family": name, "kind": it.kind, "form": it.form,
                          "question": it.question, "truth": it.truth, "timestamp": i})
    CORPUS.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n")
    print(f"wrote {len(items)} items ({len(fam.GENERATORS)} families) to {CORPUS.relative_to(ROOT)}")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ----------------------------------------------------------------------------- memo

class Memo:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.data = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        with self.lock:
            self.data[k] = v

    def save(self):
        with self.lock:
            self.path.write_text(json.dumps(self.data))


def install_memo(memo: Memo) -> None:
    """Memoize the model calls certification makes, keyed by their inputs."""
    from arc import explain, labelfree, serve
    from arc.certify import Comparison

    real_extract, real_compare, real_solve = serve.extract_parameters, labelfree._compare, explain._blind_solve

    def extract(request, params, tag=None, allow_implied=True):
        k = memo.key("extract", request, [p.description for p in params], allow_implied)
        hit = memo.get(k)
        if hit is None:
            hit = real_extract(request, params, tag=tag, allow_implied=allow_implied)
            memo.put(k, hit)
        return hit

    def compare(name, question, reference, candidate, tag):
        k = memo.key("compare", question, reference, candidate)
        hit = memo.get(k)
        if hit is None:
            c = real_compare(name, question, reference, candidate, tag)
            hit = {"agree": c.agree, "judge_said": c.judge_said, "number_status": c.number_status,
                   "critique": c.critique}
            memo.put(k, hit)
        return Comparison(name, question, reference, candidate, hit["agree"], hit["judge_said"],
                          hit["number_status"], hit["critique"])

    def solve(question, tag):
        # adjudicate tags solves "...:adj1", "...:adj2": the n-th solve of a question is reused
        # by every entry/mutant that adjudicates that question
        i = tag.rsplit(":adj", 1)[-1] if ":adj" in tag else tag
        k = memo.key("blind", question, i)
        hit = memo.get(k)
        if hit is None:
            w = real_solve(question, tag)
            hit = {"working": w.working[:2000], "final_answer": w.final_answer}
            memo.put(k, hit)
        return explain.Worked(**hit)

    real_program = labelfree._program_solve

    def program(question, tag, conventions=None):
        # the n-th program solve of a question (with the same conventions) is reused too
        i = tag.rsplit(":adj", 1)[-1] if ":adj" in tag else tag
        k = memo.key("program", question, list(conventions or []), i)
        hit = memo.get(k)
        if hit is None:
            ans, why = real_program(question, tag, conventions)
            hit = {"answer": ans, "why": why}
            memo.put(k, hit)
        return hit["answer"], hit["why"]

    serve.extract_parameters = extract
    labelfree._compare = compare
    explain._blind_solve = solve
    labelfree._program_solve = program


# ----------------------------------------------------------------------------- production (paid)

PRODUCTION_PROMPT = "Answer the following question. Show brief working, then state the final answer clearly.\n\n{q}"


def cmd_production(args) -> None:
    from arc import config, llm
    items = load_jsonl(CORPUS)
    done = {r["id"] for r in load_jsonl(PRODUCTION)}
    todo = [x for x in items if x["id"] not in done]
    print(f"{len(items)} items, {len(done)} answered, {len(todo)} to go, cap ${args.max_cost}")
    lock, stop = threading.Lock(), threading.Event()

    def one(x):
        if stop.is_set() or llm.LEDGER.cost() >= args.max_cost:
            stop.set()
            return
        r = llm.call("source", PRODUCTION_PROMPT.format(q=x["question"]), model=config.SOURCE_MODEL,
                     effort="low", max_tokens=3000, tag=f"lf-prod:{x['id']}")
        with lock, PRODUCTION.open("a") as f:
            f.write(json.dumps({"id": x["id"], "response": r.text, "model": config.SOURCE_MODEL,
                                "cost": r.cost}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    n, c = len(llm.LEDGER.records), llm.LEDGER.cost()
    print(f"{n} answers this run, ${c:.4f} (${c / max(n, 1):.5f} each)" + ("; STOPPED at cap" if stop.is_set() else ""))


# ----------------------------------------------------------------------------- scoring helpers

def right(answer: str, truth: list[str]) -> bool:
    """Right if some listed form of the truth is stated exactly, to the precision written
    (owner rule 2026-11-15; there is no percentage tolerance)."""
    from arc.checks import all_values_match
    return any(all_values_match(t, answer or "") for t in truth)


def backlog_for(condition: str, family: str, items: list[dict], prod: dict) -> list[dict]:
    rows = [x for x in items if x["family"] == family and (condition == "near_miss" or x["kind"] == "positive")]
    out = []
    rng = random.Random(f"{SEED}:{family}")
    for x in rows:
        resp = prod.get(x["id"], {}).get("response")
        if not resp:
            continue
        if condition == "noisy" and x["kind"] == "positive" and rng.random() < NOISE_RATE:
            from arc.checks import numbers
            v = (numbers(x["truth"][0]) or [1.0])[0]
            resp = f"Working omitted.\n\nFinal answer: {v * rng.choice([1.15, 1.3, 0.8, 1.6]):.4g}"
        out.append({"id": x["id"], "question": x["question"], "response": resp, "timestamp": x["timestamp"]})
    return out


# ----------------------------------------------------------------------------- mutants

class _Mutator(ast.NodeTransformer):
    """Make exactly one change: the target-th arithmetic operator or numeric constant."""
    SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}

    def __init__(self, target: int):
        self.target, self.i, self.what = target, -1, None

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAP:
            self.i += 1
            if self.i == self.target:
                self.what = f"{type(node.op).__name__} -> {self.SWAP[type(node.op)].__name__}"
                node.op = self.SWAP[type(node.op)]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and node.value not in (0, 1):
            self.i += 1
            if self.i == self.target:
                self.what = f"constant {node.value} -> {node.value * 1.5:g}"
                return ast.copy_location(ast.Constant(node.value * 1.5), node)
        return node


def mutants(cache, n: int, rng: random.Random) -> list[tuple[str, object]]:
    from arc import sandbox
    from arc.runtime import PRIMITIVES
    sites = []
    for fid, rec in cache.functions.items():
        if "while" in rec.source:
            continue
        tree = ast.parse(rec.source)
        counter = _Mutator(10 ** 9)
        counter.visit(copy.deepcopy(tree))
        sites += [(fid, k) for k in range(counter.i + 1)]
    out = []
    for fid, k in rng.sample(sites, len(sites)):   # random order, deterministic per seed
        if len(out) >= n:
            break
        m = _Mutator(k)
        src = ast.unparse(m.visit(ast.parse(cache.functions[fid].source)))
        try:
            fn = sandbox.load_function(src, PRIMITIVES)
        except Exception:
            continue
        c = copy.copy(cache)
        c.functions = dict(cache.functions)
        rec = copy.copy(cache.functions[fid])
        rec.source, rec.fn = src, fn
        c.functions[fid] = rec
        out.append((f"{fid[:30]}: {m.what}", c))
    return out


# ----------------------------------------------------------------------------- run (paid)

def truth_pass(cache, entry_id: str, items: list[dict]) -> dict:
    """Run the entry on every item through the serving path and score against the truth."""
    from arc import labelfree
    from arc.serve import format_output
    rows = {}
    for x in items:
        result, served, why = labelfree._run_serving(cache, x["question"], f"{entry_id}:truth:{x['id']}")
        ans = format_output(result.output) if result else ""
        rows[x["id"]] = {"kind": x["kind"], "served": served, "answer": ans,
                         "right": right(ans, x["truth"]) if served == "served" else None}
    served = [r for r in rows.values() if r["served"] == "served"]
    wrong = [r for r in served if not r["right"]]
    return {"rows": rows, "served": len(served), "wrong": len(wrong),
            "wrong_positive": sum(r["kind"] == "positive" for r in wrong),
            "wrong_near_miss": sum(r["kind"] == "near_miss" for r in wrong)}


def certify_one(label, cache, entry_id, family, backlog, items_by_id, fail_fast) -> dict:
    from arc import labelfree
    cert = labelfree.certify(cache, entry_id, family, backlog, fail_fast=fail_fast)
    s = cert.summary()
    # how good were the adjudications? (truth used for scoring only)
    adj = {"cache_right_correct": 0, "cache_right_wrong": 0, "cache_wrong_correct": 0, "cache_wrong_wrong": 0}
    for c in cert.checks:
        if c["outcome"] in ("cache_right", "cache_wrong"):
            truly = right(c["cache_answer"], items_by_id[c["record_id"]]["truth"])
            ok = truly if c["outcome"] == "cache_right" else not truly
            adj[f"{c['outcome']}_{'correct' if ok else 'wrong'}"] += 1
    return {"label": label, "status": cert.status, "reason": cert.reason, "summary": s, "adjudication": adj}


def cmd_run(args) -> None:
    from arc import config, llm
    from arc.cache import Cache
    config.GATE = "off"
    memo = Memo(MEMO)
    install_memo(memo)
    items = load_jsonl(CORPUS)
    by_id = {x["id"]: x for x in items}
    prod = {r["id"]: r for r in load_jsonl(PRODUCTION)}
    families = sorted({x["family"] for x in items} if not args.families else args.families)
    RESULTS.mkdir(exist_ok=True)
    stop = threading.Event()

    def per_family(family: str) -> dict:
        if stop.is_set():
            return {"family": family, "skipped": "cost cap"}
        cache = Cache.load(ENTRIES / family)
        fam_items = [x for x in items if x["family"] == family]
        out = {"family": family, "truth": truth_pass(cache, family, fam_items), "conditions": {}}
        for cond in args.conditions:
            if llm.LEDGER.cost() >= args.max_cost:
                stop.set()
                break
            backlog = backlog_for(cond if cond != "mutants" else "clean", family, items, prod)
            if cond != "mutants":
                out["conditions"][cond] = certify_one(cond, cache, family, family, backlog, by_id, fail_fast=False)
                continue
            rows = []
            for label, mc in mutants(cache, MUTANTS_PER_ENTRY, random.Random(f"{SEED}:{family}:mut")):
                t = truth_pass(mc, family, fam_items)
                r = certify_one(label, mc, family, family, backlog, by_id, fail_fast=True)
                r["truth"] = {k: v for k, v in t.items() if k != "rows"}
                rows.append(r)
            out["conditions"]["mutants"] = rows
        memo.save()
        print(f"  {family}: done, total spend so far ${llm.LEDGER.cost():.4f}", flush=True)
        return out

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(per_family, families))
    memo.save()
    tag = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"run-{tag}.json"
    path.write_text(json.dumps({"conditions": args.conditions, "results": results,
                                "cost": llm.LEDGER.cost(), "seconds": round(time.time() - t0)}, indent=1))
    print(f"saved {path.relative_to(ROOT)}; spend ${llm.LEDGER.cost():.4f}" + ("; STOPPED at cap" if stop.is_set() else ""))
    report(path)


# ----------------------------------------------------------------------------- report (free)

def report(path: Path | None = None) -> None:
    runs = sorted(RESULTS.glob("run-*.json"))
    path = path or (runs[-1] if runs else None)
    if not path:
        print("no results yet")
        return
    d = json.loads(path.read_text())
    items = load_jsonl(CORPUS)
    prod = {r["id"]: r for r in load_jsonl(PRODUCTION)}
    pr = [right(prod[x["id"]]["response"], x["truth"]) for x in items if x["id"] in prod]
    print(f"\nproduction (Haiku) accuracy vs reference: {sum(pr)}/{len(pr)} = {sum(pr) / max(len(pr), 1):.1%}")
    print(f"run cost ${d['cost']:.4f}, {d['seconds']}s\n")
    for cond in d["conditions"]:
        print(f"== {cond}")
        rows = []
        for r in d["results"]:
            if "conditions" not in r or cond not in r["conditions"]:
                continue
            if cond == "mutants":
                for m in r["conditions"]["mutants"]:
                    bad = m["truth"]["wrong_positive"] > 0
                    rows.append((r["family"], m["label"], bad, m["status"], m["summary"], m["adjudication"]))
                continue
            c = r["conditions"][cond]
            t = r["truth"]
            bad = t["wrong_positive"] > 0 or (cond == "near_miss" and t["wrong_near_miss"] > 0)
            rows.append((r["family"], "entry", bad, c["status"], c["summary"], c["adjudication"]))
        fa = [x for x in rows if x[2] and x[3] == "accepted"]
        fr = [x for x in rows if not x[2] and x[3] == "rejected"]
        good = [x for x in rows if not x[2]]
        bad = [x for x in rows if x[2]]
        print(f"   truly bad: {len(bad)}  rejected {sum(x[3] == 'rejected' for x in bad)}, "
              f"need_more_data {sum(x[3] == 'need_more_data' for x in bad)}, FALSE ACCEPT {len(fa)}")
        print(f"   truly good: {len(good)}  accepted {sum(x[3] == 'accepted' for x in good)}, "
              f"need_more_data {sum(x[3] == 'need_more_data' for x in good)}, false reject {len(fr)}")
        adj = {}
        for x in rows:
            for k, v in x[5].items():
                adj[k] = adj.get(k, 0) + v
        print(f"   adjudications: {adj}")
        for fam_, label, b, status, s, _ in rows:
            if cond == "mutants" and not (b and status == "accepted"):
                continue
            print(f"   {fam_[:28]:<28} {label[:44]:<44} {'BAD ' if b else 'good'} {status:<15} "
                  f"verified {s['verified']:>2} wrong {s['cache_wrong']} unres {s['unresolved']} "
                  f"prod_err {s['production_errors_found']} viol {s['relations_violated']}")
        print()


def cmd_report(args) -> None:
    report(Path(args.path) if args.path else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    p = sub.add_parser("production")
    p.add_argument("--max-cost", type=float, default=0.2)
    r = sub.add_parser("run")
    r.add_argument("--max-cost", type=float, default=1.0)
    r.add_argument("--conditions", nargs="+", default=["clean", "noisy", "near_miss", "mutants"])
    r.add_argument("--families", nargs="*")
    r.add_argument("--workers", type=int, default=6)
    rp = sub.add_parser("report")
    rp.add_argument("--path")
    args = ap.parse_args()
    {"make": cmd_make, "production": cmd_production, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
