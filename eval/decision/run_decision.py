"""Can a decision model tell whether a cached entry applies to a question?

    .venv/bin/python eval/decision/run_decision.py --model embed-gemma            # free baseline
    .venv/bin/python eval/decision/run_decision.py --model clef-flash --mode top3
    .venv/bin/python eval/decision/run_decision.py --model jevos-1b --limit 10   # smoke test
    .venv/bin/python eval/decision/run_decision.py --compare                      # table of every run

No cache is built and no Anthropic call is made. The "description" of each entry is just
its seed question (the only thing we have for every entry before a build), wrapped in one
fixed, task-agnostic yes/no prompt (PROMPT below). Every model sees identical pairs.

Modes
    own    each row against its own seed entry. Out-of-domain rows (no seed entry) go
           against the seed EmbeddingGemma ranks first, as retrieval would hand them over.
           Positive = the row should hit that entry.
    top3   each row against the 3 seeds EmbeddingGemma ranks highest, all in one request.
           Positive = candidate is the row's expected entry and the row should hit.
           Adds wrong-entry candidates, the case top-1 retrieval cannot recover from.

Rows marked acceptable_either are scored separately, never as hits or misses.
Tune on dev. The test split needs --final.

Outputs: eval/decision/runs/<model>/<split>-<mode>.jsonl (one row per corpus row, resumable)
         eval/decision/runs/<model>/<split>-<mode>.metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
CORPUS = ROOT / "eval" / "corpus" / "cache_eval_corpus.jsonl"
RUNS = HERE / "runs"
PAIRS = HERE / "pairs"

from models import MODELS  # noqa: E402

PROMPT_VERSION = "v1"
PROMPT = (
    "A cached solver was built from this example question:\n"
    "<<<{seed}>>>\n"
    "It re-runs exactly the same procedure on new input values. Answer yes only if running that "
    "procedure on the values given in the question above answers that question exactly: it asks "
    "for the same result, needs the same method and assumptions, gives every input the procedure "
    "needs, and adds no condition, step or item the procedure would ignore. Different numbers, "
    "units, names or wording are fine."
)


# ----------------------------------------------------------------------------- pairs

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]


def build_pairs(split: str, mode: str, structs: dict | None = None) -> list[dict]:
    """Rows with their candidate entries. Cached on disk so every model sees the same pairs.

    With structs, retrieval embeds the rendered structures (values stripped) instead of the
    raw questions (ARC_INDEX=spec), so candidates can differ."""
    path = PAIRS / f"{split}-{mode}{'-struct' if structs else ''}.json"
    if path.exists():
        return json.loads(path.read_text())
    from backends import Embed
    from structured import render
    rows = load_corpus()
    seed_rows = {r["seed_entry"]: r for r in rows if r["split"] == "seed"}
    seeds = {n: r["question"] for n, r in seed_rows.items()}
    text = (lambda r: render(structs[r["id"]])) if structs else (lambda r: r["question"])
    if structs:
        missing = [r["id"] for r in rows if r["split"] in ("seed", split) and r["id"] not in structs]
        if missing:
            raise SystemExit(f"structures file lacks {len(missing)} rows, e.g. {missing[:5]}")
    names = list(seeds)
    e = Embed("embed-gemma", {}).e
    seed_vecs = e.encode([text(seed_rows[n]) for n in names])
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        sims = seed_vecs @ e.one(text(r))
        ranked = sorted(zip(names, map(float, sims)), key=lambda x: -x[1])
        if mode == "own":
            cands = [r["seed_entry"]] if r["seed_entry"] else [ranked[0][0]]
        else:
            cands = [n for n, _ in ranked[:3]]
        sim = dict(ranked)
        out.append({
            "id": r["id"], "question": r["question"], "variant_type": r["variant_type"],
            "near_miss_category": r["near_miss_category"], "domain": r["domain"],
            "seed_entry": r["seed_entry"], "expected_entry": r["expected_entry"],
            "should_hit": r["should_hit"], "acceptable_either": r["acceptable_either"],
            "candidates": [{"entry": c, "retrieval_sim": round(sim[c], 4),
                            "label": bool(r["should_hit"] and c == r["expected_entry"])} for c in cands],
        })
    PAIRS.mkdir(parents=True, exist_ok=True)
    seed_ids = {n: r["id"] for n, r in seed_rows.items()}
    path.write_text(json.dumps({"seeds": seeds, "seed_ids": seed_ids, "rows": out}, indent=1,
                               ensure_ascii=False))
    return json.loads(path.read_text())


SEED_SPECS = ROOT / "eval" / "spec" / "specs.jsonl"   # Haiku structures, keyed "seed:<entry>"


def seed_specs() -> dict[str, dict]:
    out = {}
    for line in SEED_SPECS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("key", "").startswith("seed:"):
                out[r["key"][5:]] = r["spec"]
    return out


def pair_text(seeds: dict, entry: str, raw: bool, prompt: str = "v1") -> str:
    """v1: the prompt below. v2 / v3: the live gate's prompts (arc/clef.py PROMPT_V2 / V3),
    with the entry's profile taken from its seed question's stored structure."""
    if raw:
        return seeds[entry]
    if prompt in ("v2", "v3", "v4", "v5"):
        sys.path.insert(0, str(ROOT))
        from arc.clef import PROMPT_V2, PROMPT_V3, PROMPT_V4, PROMPT_V5, _profile
        template = {"v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4, "v5": PROMPT_V5}[prompt]
        return template.format(seed=seeds[entry], profile=_profile(seed_specs().get(entry)))
    return PROMPT.format(seed=seeds[entry])


# ----------------------------------------------------------------------------- run

def run_path(model: str, split: str, mode: str, input_kind: str, prompt: str = "v1") -> Path:
    suffix = ("" if input_kind == "raw" else f"-{input_kind}") + ("" if prompt == "v1" else f"-p{prompt}")
    return RUNS / model / f"{split}-{mode}{suffix}.jsonl"


def _decide_row(b, raw: bool, input_kind: str, r: dict, seeds: dict, seed_ids: dict, structs: dict | None,
                prompt: str = "v1"):
    """One corpus row -> ({entry: P}, {entry: {field: P}}, usage). Raises on backend errors."""
    import structured as st
    entries = [c["entry"] for c in r["candidates"]]
    if input_kind == "raw":
        probs, usage = b.decide(r["question"], {e: pair_text(seeds, e, raw, prompt) for e in entries})
        return probs, {}, usage
    q = structs[r["id"]]
    sd = {e: structs[seed_ids[e]] for e in entries}
    if input_kind == "struct":
        qs = {e: (st.render(sd[e]) if raw else st.STRUCT_PROMPT.format(seed=st.render(sd[e]))) for e in entries}
        probs, usage = b.decide(st.render(q), qs)
        return probs, {}, usage
    # fields: a field the seed leaves unstated constrains nothing (p = 1)
    fp: dict[str, dict[str, float]] = {e: {} for e in entries}
    usage: dict = {}
    asks = {(e, f) for e in entries for f in st.COMPARE if not st.unstated(st.field_text(sd[e], f))}
    for e in entries:
        for f in st.COMPARE:
            if (e, f) not in asks:
                fp[e][f] = 1.0
            elif f == "method" and st.unstated(st.field_text(q, f)):
                fp[e][f] = 1.0   # an unstated method in the request is allowed
    asks = {(e, f) for (e, f) in asks if f not in fp[e]}
    if raw:   # pair models: compare the two field texts directly, one field at a time
        for f in st.COMPARE:
            es = [e for e in entries if (e, f) in asks]
            if es:
                p, _ = b.decide(st.field_text(q, f), {e: st.field_text(sd[e], f) for e in es})
                for e in es:
                    fp[e][f] = p[e]
    elif asks:  # decision models: one request, state = the request's whole structure
        qs = {f"{e}..{f}": st.FIELD_PROMPT.format(label=st.FIELD_LABEL[f], seed=st.field_text(sd[e], f),
                                                  extra=st.FIELD_EXTRA.get(f, "")) for (e, f) in asks}
        p, usage = b.decide(st.render(q), qs)
        for k, v in p.items():
            e, f = k.split("..")
            fp[e][f] = v
    return {e: min(fp[e].values()) for e in entries}, fp, usage


def run(model: str, split: str, mode: str, limit: int | None, gpu: bool, remote: bool,
        input_kind: str = "raw", structs: dict | None = None, prompt: str = "v1") -> Path:
    from backends import RAW_SEED, make_backend
    import structured as st
    data = build_pairs(split, mode, structs if input_kind != "raw" else None)
    seeds, seed_ids = data["seeds"], data.get("seed_ids", {})
    rows = data["rows"][:limit] if limit else data["rows"]
    path = run_path(model, split, mode, input_kind, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        # rows that errored (e.g. a rate limit) are dropped and retried; good rows are kept
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        good = [r for r in recs if not r.get("error")]
        if len(good) < len(recs):
            path.write_text("".join(json.dumps(r) + "\n" for r in good))
        done = {r["id"] for r in good}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{model}: {len(todo)} rows to score ({len(done)} already done) -> {_rel(path)}")
    if not todo:
        return path
    raw = MODELS[model]["kind"] in RAW_SEED
    with make_backend(model, gpu=gpu, allow_remote_code=remote) as b, path.open("a") as f:
        b.plain = input_kind != "raw"   # NLI: plain two-way entailment on structure text
        for i, r in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                probs, field_p, usage = _decide_row(b, raw, input_kind, r, seeds, seed_ids, structs, prompt)
                err = None
            except Exception as ex:  # keep going; the row is reported as an error
                probs, field_p, usage, err = {}, {}, {}, f"{type(ex).__name__}: {ex}"[:300]
            ms = (time.perf_counter() - t0) * 1000
            cc = st.code_check(structs[r["id"]], r["question"]) if structs else None
            rec = {**{k: r[k] for k in ("id", "variant_type", "near_miss_category", "domain", "seed_entry",
                                        "expected_entry", "should_hit", "acceptable_either")},
                   "candidates": [{**c, "p": probs.get(c["entry"]),
                                   **({"field_p": field_p[c["entry"]]} if field_p else {})}
                                  for c in r["candidates"]],
                   "code_check": cc, "input": input_kind,
                   "ms": round(ms, 1), "usage": usage, "error": err, "prompt": prompt}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            ps = " ".join(f"{c['entry'][:18]}={probs.get(c['entry'], float('nan')):.2f}" for c in r["candidates"])
            flag = "ERR " + err[:80] if err else ""
            print(f"[{i:3}/{len(todo)}] {r['id']} {r['variant_type']:<16} hit={str(r['should_hit']):<5} "
                  f"{ms:6.0f}ms {ps} {flag}")
            first_errors = (first_errors if i > 1 else 0) + bool(err)
            if i == 3 and first_errors == 3:
                raise SystemExit("The first 3 rows all failed; stopping. See the errors above.")
    return path


# ----------------------------------------------------------------------------- score

def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score(path: Path, model: str, structs: dict | None = None, code_checks: bool = False) -> dict:
    """code_checks: reject (p = 0) every candidate of a row whose structure fails the code
    checks (input with no value, evidence not in the question). Free; needs structures."""
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cc_rejected = 0
    if code_checks:
        import structured as st
        if not structs:
            raise SystemExit("--code-checks needs --structures")
        qtext = {r["id"]: r["question"] for r in load_corpus()}
        for r in recs:
            why = r.get("code_check")
            if why is None:
                why = st.code_check(structs[r["id"]], qtext[r["id"]])
            r["code_check"] = why
            if why:
                cc_rejected += 1
                for c in r["candidates"]:
                    if c["p"] is not None:
                        c["p"] = 0.0
    pairs = [(r, c) for r in recs if not r["error"] and not r["acceptable_either"] for c in r["candidates"]
             if c["p"] is not None]
    pos = [c["p"] for r, c in pairs if c["label"]]
    neg = [c["p"] for r, c in pairs if not c["label"]]
    # default cut: P(yes) >= 0.5 for decision models; the paper's ENTRY_THRESHOLD for cosine
    th = 0.80 if MODELS[model]["kind"] == "embed" else 0.5

    def at(t):
        fh = sum(p >= t for p in neg)
        tp = sum(p >= t for p in pos)
        return {"threshold": round(t, 4), "false_hits": fh, "false_hit_rate": fh / len(neg) if neg else None,
                "recall": tp / len(pos) if pos else None}

    cuts = sorted(set(pos + neg))
    zero_fh = min((t for t in cuts if sum(p >= t for p in neg) == 0), default=None)
    fh2 = min((t for t in cuts if neg and sum(p >= t for p in neg) / len(neg) <= 0.02), default=None)
    r90 = max((t for t in cuts if pos and sum(p >= t for p in pos) / len(pos) >= 0.90), default=None)

    by_cat = defaultdict(list)
    for r, c in pairs:
        if not c["label"]:
            key = (r["near_miss_category"] or r["variant_type"]) if c["entry"] == r["seed_entry"] else "wrong_entry"
            by_cat[key].append(c["p"])
    by_pos = defaultdict(list)
    for r, c in pairs:
        if c["label"]:
            by_pos[r["variant_type"]].append(c["p"])

    # top-3 selection: does the argmax candidate (if >= th) equal the expected entry?
    # right = served the expected entry; served_wrong = served a near-miss, the wrong entry, or an
    # out-of-domain row; none = nothing cleared the threshold (falls back to the full model)
    sel = {"right": 0, "served_wrong": 0, "none": 0}
    for r in recs:
        if r["error"] or r["acceptable_either"] or len(r["candidates"]) < 2:
            continue
        best = max(r["candidates"], key=lambda c: c["p"] if c["p"] is not None else -1)
        if best["p"] is None or best["p"] < th:
            sel["none"] += 1
        elif best["label"]:
            sel["right"] += 1
        else:
            sel["served_wrong"] += 1

    ms = [r["ms"] for r in recs if not r["error"]]
    toks = sum((r["usage"] or {}).get("input_tokens", 0) or (r["usage"] or {}).get("prompt_tokens", 0) for r in recs)
    price = MODELS[model].get("price_per_mtok")
    m = {
        "model": model, "file": _rel(path), "rows": len(recs),
        "code_checks": code_checks, "rows_rejected_by_code_checks": cc_rejected,
        "errors": sum(bool(r["error"]) for r in recs),
        "either_rows_skipped": sum(r["acceptable_either"] for r in recs),
        "positives": len(pos), "negatives": len(neg), "auc": _auc(pos, neg),
        "at_default": at(th),
        "zero_false_hits": at(zero_fh) if zero_fh is not None else None,
        "false_hit_le_2pct": at(fh2) if fh2 is not None else None,
        "recall_ge_90pct": at(r90) if r90 is not None else None,
        "negatives_accepted_by_type": {k: f"{sum(p >= th for p in v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "negatives_mean_p_by_type": {k: round(statistics.mean(v), 3) for k, v in sorted(by_cat.items())},
        "positives_accepted_by_type": {k: f"{sum(p >= th for p in v)}/{len(v)}" for k, v in sorted(by_pos.items())},
        "top3_selection": sel if any(sel.values()) else None,
        "latency_ms_median": statistics.median(ms) if ms else None,
        "latency_ms_p90": sorted(ms)[int(0.9 * (len(ms) - 1))] if ms else None,
        "input_tokens": toks,
        "cost_usd": toks * price / 1e6 if price and toks else (0.0 if not price else None),
    }
    out = path.with_suffix(".cc.metrics.json" if code_checks else ".metrics.json")
    out.write_text(json.dumps(m, indent=2))
    return m


def show(m: dict) -> None:
    f = lambda x: "n/a" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
    cc = f"  [+code checks: {m['rows_rejected_by_code_checks']} rows rejected]" if m.get("code_checks") else ""
    print(f"\n== {m['model']}  ({m['file']}){cc}")
    print(f"rows {m['rows']}  errors {m['errors']}  positives {m['positives']}  negatives {m['negatives']}"
          f"  AUC {f(m['auc'])}")
    for key, label in [("at_default", "default threshold"), ("zero_false_hits", "zero false hits"),
                       ("false_hit_le_2pct", "false-hit rate <= 2%"), ("recall_ge_90pct", "recall >= 90%")]:
        a = m[key]
        if a:
            print(f"  {label:<22} t={a['threshold']:<8} false hits {a['false_hits']:>3} "
                  f"({f(a['false_hit_rate'])})  recall {f(a['recall'])}")
    print("  negatives accepted at default, by type:")
    for k, v in m["negatives_accepted_by_type"].items():
        print(f"    {k:<24} {v:>7}   mean p {m['negatives_mean_p_by_type'][k]}")
    print("  positives accepted at default, by type:", m["positives_accepted_by_type"])
    if m["top3_selection"]:
        print("  top-3 pick (argmax >= default):", m["top3_selection"])
    print(f"  latency median {f(m['latency_ms_median'])} ms, p90 {f(m['latency_ms_p90'])} ms;"
          f"  input tokens {m['input_tokens']}  cost ${(m['cost_usd'] or 0):.5f}")


def compare() -> None:
    rows = []
    for p in sorted(RUNS.glob("*/*.metrics.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        print("no runs yet")
        return
    pct = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}"
    print(f"{'model':<22}{'run':<20}{'AUC':>6} | {'@default FH%':>12}{'recall%':>8} | {'@0 FH recall%':>14}"
          f" | {'@FH<=2% recall%':>16} | {'ms':>7} {'$':>8}")
    for m in rows:
        run_name = Path(m["file"]).name.replace(".jsonl", "") + ("+cc" if m.get("code_checks") else "")
        d, z, t = m["at_default"], m["zero_false_hits"], m["false_hit_le_2pct"]
        print(f"{m['model']:<22}{run_name:<20}{(m['auc'] or 0):6.3f} | {pct(d['false_hit_rate']):>12}"
              f"{pct(d['recall']):>8} | {pct(z['recall'] if z else None):>14} | {pct(t['recall'] if t else None):>16}"
              f" | {(m['latency_ms_median'] or 0):7.0f} {(m['cost_usd'] or 0):8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--mode", default="own", choices=["own", "top3"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu", action="store_true", help="llama.cpp: use Metal instead of CPU only")
    ap.add_argument("--allow-remote-code", action="store_true")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--input", default="raw", choices=["raw", "struct", "fields"],
                    help="raw text, whole structure, or field-by-field structure comparison")
    ap.add_argument("--structures", help="structures file, e.g. eval/spec/specs.jsonl (needed unless --input raw)")
    ap.add_argument("--code-checks", action="store_true", help="also score with the code checks layered on")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2", "v3", "v4", "v5"],
                    help="v1 = the prompt below; v2 = the live gate's richer prompt; "
                         "v3 = v2 + read-every-word rule (arc/clef.py)")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        return compare()
    if not a.model:
        ap.error("--model is required")
    if a.split == "test" and not a.final:
        ap.error("the test split is held out; pass --final once the gate is frozen")
    if (a.input != "raw" or a.code_checks) and not a.structures:
        ap.error("--input struct/fields and --code-checks need --structures")
    import structured
    structs = structured.load(a.structures) if a.structures else None
    if a.prompt in ("v2", "v3", "v4", "v5") and a.input != "raw":
        ap.error(f"--prompt {a.prompt} applies to --input raw only")
    path = run_path(a.model, a.split, a.mode, a.input, a.prompt)
    if not a.score_only:
        path = run(a.model, a.split, a.mode, a.limit, a.gpu, a.allow_remote_code, a.input, structs, a.prompt)
    show(score(path, a.model))
    if a.code_checks:
        show(score(path, a.model, structs, code_checks=True))


if __name__ == "__main__":
    main()
