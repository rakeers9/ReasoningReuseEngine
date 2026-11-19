"""Replay harness for the cache-HIT path (parts 1-6: filter, retrieve, gate, extract, run, output).
PAID but cheap: one Clef call per reusable question, plus one Haiku extraction when the gate
passes or shadow mode runs a turned-away question (about $0.0005 per question). No builds.

    .venv/Scripts/python eval/hit_replay.py --set dev                 # tune on this
    .venv/Scripts/python eval/hit_replay.py --set hidden --i-am-measuring   # measure only, never tune

Sets and the cache each one replays against:
  dev     eval/chat_loop.py script(25): DDB book value and loan payment have entries
          (caches/chat_loop/master); perfect-gas pressure has none, so its variants must not be served.
  hidden  eval/hidden/chat_hidden.jsonl vs caches/chat_loop_hidden/master (calorimetry entry).
          The hidden set is for measuring only (eval/hidden/build_hidden.py): never tune on it.

Every question runs through the live serving path (serve.use) in shadow mode: the gate and
the cached code run even when the question would be turned away, so one run shows both what
the cache served and what it would have answered at each turn-away. Nothing is served to
anyone; nothing is written to any backlog or cache.

Each row records the stage it stopped at, the Clef P, the extracted values and the output.
Outcomes (only `served` decides what a user would have seen):
  served_correct     served, right answer
  served_wrong       served, wrong answer (the number we drive to zero)
  missed             a covered variant the cache could have answered but didn't (serve-rate loss)
  declined_ok        correctly not served (near-miss, uncovered family, one-off, chit-chat)
A near-miss marked acceptable_either counts as served_correct when its answer is right.

Coverage: a family is covered when an entry was built from one of its questions (the entry's
original question appears in the set). Override with --covered fam1,fam2.

Output: runs/hit_replay/<set>_<timestamp>.jsonl (one row per question) and a .json summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

SETS = {
    "dev": {"messages": None, "cache": ROOT / "caches" / "chat_loop" / "master"},
    "hidden": {"messages": ROOT / "eval" / "hidden" / "chat_hidden.jsonl",
               "cache": ROOT / "caches" / "chat_loop_hidden" / "master"},
}
MEASURE_ONLY = {"hidden"}


def load_set(name: str) -> list[dict]:
    import chat_loop
    spec = SETS[name]
    return chat_loop.load_messages(spec["messages"]) if spec["messages"] else chat_loop.script(25)


def covered_families(cache, msgs: list[dict]) -> dict[str, str]:
    """entry id -> family, from each entry's original question appearing in the set."""
    by_q = {m["question"].strip(): m["family"] for m in msgs if m.get("family")}
    out = {}
    for ep in cache.entry_points:
        if ep.kind == "original" and ep.text.strip() in by_q:
            out[ep.problem_id] = by_q[ep.text.strip()]
    return out


_MONEY = None


def score(m: dict, text: str) -> tuple[bool | None, str]:
    """chat_loop.score_message, plus: every OTHER dollar amount in the expected answer must appear
    too. The scorer only checks answer_value (for the loan family, the monthly payment), so a
    wrong total interest scored as right."""
    import re
    import chat_loop
    global _MONEY
    ok, how = chat_loop.score_message(m, text)
    if not ok or m.get("kind") == "chitchat":
        return ok, how
    _MONEY = _MONEY or re.compile(r"\$\s?(-?[\d,]+\.\d{2})")
    want = [float(x.replace(",", "")) for x in _MONEY.findall(m.get("answer") or "")]
    if len(want) < 2:
        return ok, how
    got = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
           if x not in ("-", ".")]
    for w in want[1:]:
        if not any(abs(g - w) <= 0.011 for g in got):
            return False, f"missing expected amount {w:.2f}"
    return ok, how


def rescore(path: Path) -> dict:
    """Re-score a saved run with the current scorer (free) and rewrite its summary."""
    msgs = {m["question"]: m for name in SETS for m in load_set(name)}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        m = msgs[r["question"]]
        for key, out_key in (("output", "correct"), ("shadow_output", "shadow_correct")):
            if r.get(key):
                r[out_key], r[out_key + "_how"] = score(m, r[key])
    old = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    covered = set(old["covered_families"])
    summary = summarize(rows, covered)
    summary.update({k: old[k] for k in ("set", "gate", "embed_model", "extract_form", "clef_threshold",
                                         "entries", "time") if k in old})
    path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def outcome(m: dict, covered: set[str], served: bool, correct: bool | None) -> str:
    if served:
        if m["kind"] == "near_miss":
            return "served_correct" if (m.get("acceptable_either") and correct) else "served_wrong"
        return "served_correct" if correct else "served_wrong"
    if m["kind"] == "variant" and m["family"] in covered:
        return "missed"
    return "declined_ok"


def replay_one(cache, m: dict) -> dict:
    from arc import chat, llm, serve
    row = {k: m.get(k) for k in ("kind", "family", "category", "acceptable_either", "question", "answer")}
    before = llm.LEDGER.cost()
    t0 = time.time()
    if not chat.is_reusable(m["question"]):
        row.update(stage="filtered", served=False)
    else:
        trace: dict = {}
        try:
            hit = serve.use(cache, m["question"], tag="hit_replay", trace=trace, shadow=True)
        except (serve.CacheInvocationError, llm.LLMError) as e:
            # a shadow run of a turned-away question can fail too: it stays a gate turn-away
            kind = ("extraction_error" if isinstance(e, serve.ExtractionError) else
                    "code_error" if isinstance(e, serve.CacheInvocationError) else "api_error")
            hit = None
            trace["error"] = f"{kind}: {str(e).splitlines()[0][:300]}"
            if trace.get("stage") != "gate_rejected":
                trace["stage"] = kind
        row.update(stage=trace.get("stage"), served=hit is not None, entry=trace.get("entry"),
                   similarity=trace.get("similarity"), clef_p=trace.get("clef_p"), clef=trace.get("clef"),
                   rejected_by=trace.get("rejected_by"), values=trace.get("values"),
                   extraction=trace.get("extraction"), error=trace.get("error") or trace.get("shadow_error"),
                   output_check=trace.get("output_check"), output_violations=trace.get("output_violations"),
                   reference_check=trace.get("reference_check"),
                   reference_disagreements=trace.get("reference_disagreements"))
        if hit is not None:
            row["output"] = serve.format_output(hit.output)
        elif trace.get("shadow_output"):
            row["shadow_output"] = trace["shadow_output"]
    row["cost"] = round(llm.LEDGER.cost() - before, 6)
    row["seconds"] = round(time.time() - t0, 2)
    for key, out_key in (("output", "correct"), ("shadow_output", "shadow_correct")):
        if row.get(key):
            ok, how = score(m, row[key])
            row[out_key], row[out_key + "_how"] = ok, how
    return row


def summarize(rows: list[dict], covered: set[str]) -> dict:
    def pct(a, b):
        return round(100 * a / b, 1) if b else None
    for r in rows:
        r["outcome"] = outcome(r, covered, r["served"], r.get("correct"))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    cov = [r for r in rows if r["kind"] == "variant" and r["family"] in covered]
    served = [r for r in rows if r["served"]]
    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"n": len(rs), "served": sum(r["served"] for r in rs),
                         "served_wrong": sum(r["outcome"] == "served_wrong" for r in rs)}
    # where covered variants stopped (serve-rate losses), and what the cache would have said there
    lost = {}
    for r in cov:
        if not r["served"]:
            s = r.get("stage") or "?"
            d = lost.setdefault(s, {"n": 0, "shadow_correct": 0, "shadow_wrong": 0})
            d["n"] += 1
            if r.get("shadow_correct") is True:
                d["shadow_correct"] += 1
            elif r.get("shadow_correct") is False:
                d["shadow_wrong"] += 1
    # Clef P by what actually happened: the raw material for part 3 (confidence bands)
    def ps(rs):
        v = sorted(r["clef_p"] for r in rs if r.get("clef_p") is not None)
        return {"n": len(v), "min": v[0], "median": round(statistics.median(v), 4), "max": v[-1]} if v else {"n": 0}
    clef = {
        "served_correct": ps([r for r in rows if r["outcome"] == "served_correct"]),
        "served_wrong": ps([r for r in rows if r["outcome"] == "served_wrong"]),
        "rejected_shadow_correct": ps([r for r in rows if not r["served"] and r.get("shadow_correct") is True]),
        "rejected_shadow_wrong": ps([r for r in rows if not r["served"] and r.get("shadow_correct") is False]),
    }
    cost = sum(r["cost"] for r in rows)
    return {
        "questions": len(rows),
        "outcomes": counts,
        "wrong_served": counts.get("served_wrong", 0),
        "wrong_served_pct_of_served": pct(counts.get("served_wrong", 0), len(served)),
        "serve_rate_covered_pct": pct(sum(r["served"] and r["outcome"] == "served_correct" for r in cov), len(cov)),
        "covered_variants": len(cov),
        "covered_families": sorted(covered),
        "by_kind": by_kind,
        "covered_losses_by_stage": lost,
        "clef_p": clef,
        "cost_total": round(cost, 4),
        "cost_per_served": round(sum(r["cost"] for r in served) / len(served), 6) if served else None,
        "wrong_rows": [{k: r.get(k) for k in ("kind", "family", "question", "answer", "output", "clef_p",
                                                "values", "correct_how")}
                       for r in rows if r["outcome"] == "served_wrong"],
    }


def print_summary(s: dict) -> None:
    print(f"\n=== {s['questions']} questions, covered families: {', '.join(s['covered_families']) or 'none'}")
    print(f"wrong answers served : {s['wrong_served']}  ({s['wrong_served_pct_of_served']}% of served)")
    print(f"serve rate (covered) : {s['serve_rate_covered_pct']}% of {s['covered_variants']} covered variants")
    print(f"cost                 : ${s['cost_total']} total, ${s['cost_per_served']} per served answer")
    print("by kind              : " + "; ".join(f"{k} {v['served']}/{v['n']} served, {v['served_wrong']} wrong"
                                            for k, v in s["by_kind"].items()))
    if s["covered_losses_by_stage"]:
        print("covered variants lost: " + "; ".join(
            f"{k} {v['n']} (would have been right {v['shadow_correct']}, wrong {v['shadow_wrong']})"
            for k, v in s["covered_losses_by_stage"].items()))
    print("clef P               : " + "; ".join(f"{k} {v}" for k, v in s["clef_p"].items()))
    for r in s["wrong_rows"]:
        print(f"  WRONG {r['kind']:9s} {str(r['family'])[:24]:24s} P={r['clef_p']} expected {r['answer']!r}"
              f" got {str(r['output'])[:80]!r}")


def main() -> None:
    # Windows reads text as cp1252 unless told otherwise, and arc/ reads cache files (which hold
    # "°C") without an encoding. UTF-8 mode makes every default read/write UTF-8.
    if not sys.flags.utf8_mode:
        import subprocess
        sys.exit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="dev")
    ap.add_argument("--i-am-measuring", action="store_true",
                    help="required for measure-only sets: confirms the results won't be used for tuning")
    ap.add_argument("--covered", default="", help="comma list of covered families (default: from the cache)")
    ap.add_argument("--limit", type=int, default=0, help="replay only the first N questions")
    ap.add_argument("--max-cost", type=float, default=0.50, help="stop past this spend ($)")
    ap.add_argument("--tag", default="", help="label added to the output file name")
    ap.add_argument("--rescore", default="", help="re-score a saved rows file with the current scorer (free)")
    args = ap.parse_args()
    if args.rescore:
        print_summary(rescore(Path(args.rescore)))
        return
    if args.set in MEASURE_ONLY and not args.i_am_measuring:
        ap.error(f"--set {args.set} is for measuring only; pass --i-am-measuring to confirm")

    os.environ.setdefault("ARC_PROFILE", "haiku")
    os.environ["ARC_GATE"] = "clef"
    from arc import config, llm
    from arc.cache import Cache
    if not (os.environ.get("CLOUDFLARE_ACCOUNT_ID") and os.environ.get("CLOUDFLARE_API_TOKEN")):
        sys.exit("Clef needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env")

    msgs = load_set(args.set)
    if args.limit:
        msgs = msgs[:args.limit]
    cache = Cache.load(SETS[args.set]["cache"])
    entry_map = covered_families(cache, load_set(args.set))
    covered = set(filter(None, args.covered.split(","))) if args.covered else set(entry_map.values())
    from arc import embed
    embed.embedder().one("load")                       # loads lazily; the fallback may replace the default
    embed_model = embed.embedder().name or config.EMBED_MODEL   # the one actually loaded
    print(f"set {args.set}: {len(msgs)} questions; cache {SETS[args.set]['cache'].relative_to(ROOT)}; "
          f"entries -> families {entry_map}; gate {config.GATE}; embed {embed_model}")

    rows = []
    for i, m in enumerate(msgs, 1):
        if llm.LEDGER.cost() > args.max_cost:
            print(f"stopping at question {i}: spent ${llm.LEDGER.cost():.3f} > --max-cost {args.max_cost}")
            break
        r = replay_one(cache, m)
        rows.append(r)
        flag = "SERVED" if r["served"] else str(r.get("stage"))
        print(f"{i:3d} {m['kind']:9s} {str(m['family'])[:24]:24s} {flag:16s} P={r.get('clef_p')} "
              f"correct={r.get('correct', r.get('shadow_correct'))}", flush=True)

    summary = summarize(rows, covered)
    summary.update(set=args.set, gate=config.GATE, embed_model=embed_model,
                   extract_form=config.EXTRACT_FORM, clef_threshold=config.CLEF_THRESHOLD,
                   entries=entry_map, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    out = ROOT / "runs" / "hit_replay"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.set}{'_' + args.tag if args.tag else ''}_{time.strftime('%Y%m%d_%H%M%S')}"
    (out / f"{stem}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print_summary(summary)
    print(f"\nrows: {(out / (stem + '.jsonl')).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
