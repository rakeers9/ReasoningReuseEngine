"""Which model can tell "same meaning" from "near-miss" on short spec phrases?

Free, local test (no Anthropic calls). Each model scores every pair in pairs.jsonl;
we then measure how cleanly it separates same-meaning pairs from near-misses.

Model specs (one per CLI arg):
    embed:<hf id>   bi-encoder, cosine of the two embeddings (what the cache uses today)
    cross:<hf id>   cross-encoder with one output (rerankers, paraphrase/QQP/PAWS models)
    nli:<hf id>     NLI cross-encoder; score = min(P(a entails b), P(b entails a))
    cf:<model>      Cloudflare Workers AI reranker, e.g. cf:@cf/baai/bge-reranker-base
                    (uses CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN from .env)

Usage:
    .venv/bin/python eval/match_models/compare.py embed:google/embeddinggemma-300m \
        cross:BAAI/bge-reranker-base nli:cross-encoder/nli-deberta-v3-base

Metrics per model:
    AUC       share of (same, near-miss) pairs where the same pair scores higher. 1.0 = perfect.
    best_acc  accuracy at the single best threshold.
    gap       lowest same score minus highest near-miss score. > 0 means a clean threshold exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_embed(model_id: str, pairs: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device="cpu")
    prompt = "STS" if "STS" in (getattr(m, "prompts", None) or {}) else None
    texts = [t for p in pairs for t in (p["a"], p["b"])]
    v = m.encode(texts, prompt_name=prompt, normalize_embeddings=True, convert_to_numpy=True)
    return np.array([float(v[2 * i] @ v[2 * i + 1]) for i in range(len(pairs))])


def score_cross(model_id: str, pairs: list[dict]) -> np.ndarray:
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(model_id, device="cpu")
    ab = m.predict([(p["a"], p["b"]) for p in pairs])
    ba = m.predict([(p["b"], p["a"]) for p in pairs])
    ab, ba = np.asarray(ab, dtype=float), np.asarray(ba, dtype=float)
    if ab.ndim > 1:  # two-label classifier: take the "same / duplicate" column
        ab, ba = ab[:, -1], ba[:, -1]
    return (ab + ba) / 2  # order-independent


def score_nli(model_id: str, pairs: list[dict]) -> np.ndarray:
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(model_id, device="cpu")
    labels = {v.lower(): int(k) for k, v in m.model.config.id2label.items()}
    ent = labels["entailment"]

    def p_entail(xs):
        logits = np.asarray(m.predict(xs), dtype=float)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, ent]

    ab = p_entail([(p["a"], p["b"]) for p in pairs])
    ba = p_entail([(p["b"], p["a"]) for p in pairs])
    return np.minimum(ab, ba)  # equivalence = entailment both ways


def score_cf(model: str, pairs: list[dict]) -> np.ndarray:
    import urllib.request
    from arc import config  # noqa: F401  (loads .env)
    acct, token = os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}"
    out = []
    for p in pairs:
        body = json.dumps({"query": p["a"], "contexts": [{"text": p["b"]}]}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)["result"]["response"]
        out.append(float(res[0]["score"]))
    return np.array(out)


KINDS = {"embed": score_embed, "cross": score_cross, "nli": score_nli, "cf": score_cf}


def metrics(scores: np.ndarray, same: np.ndarray) -> dict:
    pos, neg = scores[same], scores[~same]
    auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
    cuts = np.unique(scores)
    best_acc, best_t = max((float(np.mean((scores >= t) == same)), float(t)) for t in cuts)
    return {"auc": auc, "best_acc": best_acc, "best_threshold": best_t,
            "gap": float(pos.min() - neg.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="kind:model_id, see module docstring")
    ap.add_argument("--pairs", default=str(HERE / "pairs.jsonl"))
    ap.add_argument("--worst", type=int, default=5, help="show the N most confusing pairs per model")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    same = np.array([p["same"] for p in pairs])
    fields = sorted({p["field"] for p in pairs})
    results = []
    for spec in args.models:
        kind, model_id = spec.split(":", 1)
        t0 = time.time()
        scores = KINDS[kind](model_id, pairs)
        m = metrics(scores, same)
        m["per_field_auc"] = {f: metrics(scores[mask], same[mask])["auc"]
                              for f in fields if (mask := np.array([p["field"] == f for p in pairs])).any()}
        m.update(spec=spec, seconds=round(time.time() - t0, 1), scores=scores.tolist())
        results.append(m)

        # most confusing: near-misses scoring high, same pairs scoring low
        norm = (scores - scores.min()) / (np.ptp(scores) or 1)
        badness = np.where(same, 1 - norm, norm)
        print(f"\n== {spec}  ({m['seconds']}s)")
        print(f"   AUC {m['auc']:.3f}   best_acc {m['best_acc']:.3f} @ {m['best_threshold']:.3f}   gap {m['gap']:+.3f}")
        print("   per field AUC: " + ", ".join(f"{f} {a:.2f}" for f, a in m["per_field_auc"].items()))
        for i in np.argsort(-badness)[: args.worst]:
            p = pairs[i]
            print(f"   {scores[i]:7.3f}  {'SAME' if p['same'] else 'DIFF'}  {p['a']!r} vs {p['b']!r}")

    print("\n== summary (sorted by AUC)")
    for m in sorted(results, key=lambda r: -r["auc"]):
        print(f"   {m['auc']:.3f}  acc {m['best_acc']:.3f}  gap {m['gap']:+.3f}  {m['spec']}")

    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"pairs": args.pairs, "results": results}, indent=1))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
