"""Does rewriting questions into a fixed structure separate hits from near-misses?

    .venv/bin/python eval/spec/run_spec.py write --max-cost 0.15   # PAID: Haiku specs for seeds + dev
    .venv/bin/python eval/spec/run_spec.py score                   # free: every metric, both modes

Uses the same dev pairs as eval/decision (read-only), so numbers line up with the
decision-model runs there. The test split is never touched.

write  one Haiku call per seed question and per dev question (arc/spec.py, same prompt
       on both sides). Saved to eval/spec/specs.jsonl, resumable, stops at --max-cost.
       That file is also the structured input for eval/decision models.
score  for every (question, candidate entry) pair:
         raw_sim     EmbeddingGemma on the raw question vs seed question (paper baseline)
         spec_sim    EmbeddingGemma on the value-free structure texts
         gate_<d>    code checks + min field score with decider d (embed, nli)
       plus per-field AUC and the best field subsets, so the gate can be tuned offline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
PAIRS = ROOT / "eval" / "decision" / "pairs"
SPECS = HERE / "specs.jsonl"
RUNS = HERE / "runs"

from arc import llm, spec  # noqa: E402

import hashlib  # noqa: E402

# Specs written by an older prompt are ignored (and rewritten), never deleted.
PROMPT_HASH = hashlib.sha1(spec.PROMPT.encode()).hexdigest()[:10]


def load_pairs(mode: str) -> dict:
    return json.loads((PAIRS / f"dev-{mode}.json").read_text())


def load_specs() -> dict[str, dict]:
    if not SPECS.exists():
        return {}
    out = {}
    for line in SPECS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("prompt") == PROMPT_HASH:
                out[r["key"]] = r
    return out


# ----------------------------------------------------------------------------- write

def cmd_write(args) -> None:
    data = load_pairs("own")
    jobs = [(f"seed:{name}", q) for name, q in data["seeds"].items()]
    jobs += [(r["id"], r["question"]) for r in data["rows"]]
    done = load_specs()
    todo = [(k, t) for k, t in jobs if k not in done]
    print(f"{len(jobs)} texts, {len(done)} already written, {len(todo)} to go, cap ${args.max_cost}")
    lock, stop = threading.Lock(), threading.Event()

    def one(job):
        key, text = job
        if stop.is_set():
            return
        if llm.LEDGER.cost() >= args.max_cost:
            stop.set()
            return
        try:
            s = spec.write_spec(text, tag=f"spec-eval:{key}")
        except Exception as e:  # keep going; a rerun retries the failures
            print(f"  ! {key}: {type(e).__name__}: {e}"[:200])
            return
        rec = {"key": key, "prompt": PROMPT_HASH, "text": text, "spec": s, "spec_text": spec.spec_text(s)}
        with lock, SPECS.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    have = len(load_specs())
    calls = len(llm.LEDGER.records)
    cost = llm.LEDGER.cost()
    print(f"wrote {calls} specs this run, cost ${cost:.4f} (${cost / max(calls, 1):.5f} per call); "
          f"{have}/{len(jobs)} present" + ("; STOPPED at cost cap" if stop.is_set() else ""))


# ----------------------------------------------------------------------------- score

def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def summarize(scores, labels, cats, default_t):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]

    def at(t):
        fh = sum(s >= t for s in neg)
        return {"threshold": round(t, 4), "false_hits": fh, "false_hit_rate": fh / len(neg),
                "recall": sum(s >= t for s in pos) / len(pos)}

    cuts = sorted(set(scores))
    zero = [at(t) for t in cuts if at(t)["false_hits"] == 0]
    le2 = [at(t) for t in cuts if at(t)["false_hit_rate"] <= 0.02]
    by_type = defaultdict(lambda: [0, 0])
    for s, l, c in zip(scores, labels, cats):
        if not l:
            by_type[c][1] += 1
            by_type[c][0] += s >= default_t
    return {"auc": auc(pos, neg), "at_default": at(default_t),
            "zero_false_hits": max(zero, key=lambda m: m["recall"]) if zero else None,
            "false_hit_le_2pct": max(le2, key=lambda m: m["recall"]) if le2 else None,
            "negatives_accepted_by_type": {k: f"{a}/{n}" for k, (a, n) in sorted(by_type.items())}}


def cmd_score(args) -> None:
    from arc import embed
    specs = load_specs()
    RUNS.mkdir(exist_ok=True)
    for mode in args.modes:
        data = load_pairs(mode)
        pairs = []
        for r in data["rows"]:
            if r["acceptable_either"] or r["id"] not in specs:
                continue
            for c in r["candidates"]:
                if f"seed:{c['entry']}" not in specs:
                    continue
                cat = (r["near_miss_category"] or r["variant_type"]) if c["entry"] == r["seed_entry"] else "wrong_entry"
                pairs.append((r, c, cat))
        if not pairs:
            print(f"{mode}: no pairs with specs yet; run `write` first")
            continue
        labels = [c["label"] for _, c, _ in pairs]
        cats = [cat for _, _, cat in pairs]
        q_specs = [specs[r["id"]]["spec"] for r, _, _ in pairs]
        e_specs = [specs[f"seed:{c['entry']}"]["spec"] for _, c, _ in pairs]
        questions = [r["question"] for r, _, _ in pairs]

        results = {"raw_sim": summarize([c["retrieval_sim"] for _, c, _ in pairs], labels, cats, 0.80)}
        texts = sorted({spec.spec_text(s) for s in q_specs + e_specs})
        vecs = dict(zip(texts, embed.embedder().encode(texts)))
        spec_sim = [float(vecs[spec.spec_text(q)] @ vecs[spec.spec_text(e)]) for q, e in zip(q_specs, e_specs)]
        results["spec_sim"] = summarize(spec_sim, labels, cats, 0.80)

        checks = [spec.code_checks(q, t) for q, t in zip(q_specs, questions)]
        check_fail = {c: sum(not ch[c] for ch, l in zip(checks, labels) if l) for c in spec.CODE_CHECKS}
        field_detail = {}
        for d in args.deciders:
            fields = spec.compare_many(list(zip(q_specs, e_specs)), d)
            ok = [all(ch.values()) for ch in checks]
            gate = [min(f.values()) if o else 0.0 for f, o in zip(fields, ok)]
            results[f"gate_{d}"] = summarize(gate, labels, cats, 0.5)
            per_field = {f: auc([x[f] for x, l in zip(fields, labels) if l],
                                [x[f] for x, l in zip(fields, labels) if not l]) for f in spec.FIELDS}
            subsets = []
            for k in range(1, len(spec.FIELDS) + 1):
                for sub in itertools.combinations(spec.FIELDS, k):
                    sc = [min(f[x] for x in sub) if o else 0.0 for f, o in zip(fields, ok)]
                    m = summarize(sc, labels, cats, 0.5)
                    subsets.append({"fields": sub, "auc": m["auc"],
                                    "recall_fh_le_2pct": (m["false_hit_le_2pct"] or {}).get("recall", 0.0)})
            subsets.sort(key=lambda x: (-x["recall_fh_le_2pct"], -x["auc"]))
            field_detail[d] = {"per_field_auc": per_field, "best_subsets": subsets[:5]}
            # keep per-pair rows for inspection
            with (RUNS / f"dev-{mode}-{d}.jsonl").open("w") as f:
                for (r, c, cat), q, e, fs, ch, sv in zip(pairs, q_specs, e_specs, fields, checks, spec_sim):
                    f.write(json.dumps({"id": r["id"], "entry": c["entry"], "label": c["label"], "type": cat,
                                        "raw_sim": c["retrieval_sim"], "spec_sim": round(sv, 4),
                                        "fields": {k: round(v, 4) for k, v in fs.items()}, "checks": ch,
                                        "q_spec": spec.spec_text(q), "e_spec": spec.spec_text(e)},
                                       ensure_ascii=False) + "\n")

        out = {"mode": mode, "pairs": len(pairs), "positives": sum(labels), "negatives": len(labels) - sum(labels),
               "positives_failing_code_checks": check_fail, "results": results, "fields": field_detail}
        (RUNS / f"dev-{mode}.metrics.json").write_text(json.dumps(out, indent=1))

        print(f"\n== dev-{mode}: {len(pairs)} pairs ({sum(labels)} should hit)")
        print(f"   positives failing code checks: {check_fail}")
        print(f"   {'scorer':<11} {'AUC':>6} {'FH@default':>11} {'recall@default':>15} "
              f"{'recall@0FH':>11} {'recall@FH<=2%':>14}")
        for name, m in results.items():
            z = m["zero_false_hits"]["recall"] if m["zero_false_hits"] else 0.0
            l2 = m["false_hit_le_2pct"]["recall"] if m["false_hit_le_2pct"] else 0.0
            print(f"   {name:<11} {m['auc']:>6.3f} {m['at_default']['false_hit_rate']:>10.0%} "
                  f"{m['at_default']['recall']:>15.0%} {z:>11.0%} {l2:>14.0%}")
        for d, fd in field_detail.items():
            print(f"   {d} per-field AUC: " + ", ".join(f"{k} {v:.2f}" for k, v in fd["per_field_auc"].items()))
            for s in fd["best_subsets"][:3]:
                print(f"      best subset {'+'.join(s['fields'])}: AUC {s['auc']:.3f}, "
                      f"recall@FH<=2% {s['recall_fh_le_2pct']:.0%}")
        for name in [f"gate_{d}" for d in args.deciders]:
            print(f"   {name} near-misses accepted at 0.5: {results[name]['negatives_accepted_by_type']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--max-cost", type=float, default=0.15)
    w.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--modes", nargs="+", default=["own", "top3"])
    s.add_argument("--deciders", nargs="+", default=["embed", "nli"])
    args = ap.parse_args()
    {"write": cmd_write, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
