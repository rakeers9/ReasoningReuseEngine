"""Ground truth for the certificate's decisions (PAID, tiny: one Haiku extraction per row).

The certificate accepts or rejects each label-free entry. Was it right? The corpus has
should-hit rows for every seed with answers computed in code (new numbers, units,
wording). Run each built entry, accepted or not, on its own DEV should-hit rows,
straight through extraction + its code (no retrieval, no gate), and score them.
A good certificate accepts entries that answer these right and rejects ones that don't.
The test split is never touched.

    .venv/bin/python eval/cert_truth.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["ARC_PROFILE"] = "haiku"
os.environ["ARC_NAMESPACE"] = "corpus_cert"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from arc import config, llm  # noqa: E402
from arc.cache import Cache  # noqa: E402
from arc.serve import dispatch_root, extract_parameters, format_output  # noqa: E402
from run_corpus import score  # noqa: E402


def main() -> None:
    results = {r["problem"]: r for r in map(json.loads, (config.WORK_DIR / "cert_experiment_seeds.rows.jsonl")
                                           .read_text().splitlines())}
    corpus = [json.loads(l) for l in (ROOT / "eval/corpus/cache_eval_corpus.jsonl").read_text().splitlines()]
    out = []
    for pid, r in sorted(results.items()):
        if not r.get("built"):
            out.append({"entry": pid, "accepted": False, "built": False})
            continue
        cache = Cache.load(config.CACHE_ROOT / pid)
        ep = next(e for e in cache.entry_points if e.kind == "original")
        rows = [c for c in corpus if c["split"] == "dev" and c["should_hit"] and c["expected_entry"] == pid]
        detail = []
        for c in rows:
            try:
                values = extract_parameters(c["question"], ep.params, tag=f"cert_truth:{pid}")
                ans = format_output(dispatch_root(cache, ep.root_id, values)[0])
                ok, how = score(c, ans)
                status = "right" if ok else "WRONG" if ok is False else "unscored"
            except Exception as e:
                ans, status = f"{type(e).__name__}: {e}"[:160], "crash"
            detail.append({"id": c["id"], "type": c["variant_type"], "status": status, "notes": c["notes"],
                           "question": c["question"][:140], "expected": c["answer"], "got": ans[-160:]})
        counts = {s: sum(d["status"] == s for d in detail) for s in ("right", "WRONG", "crash", "unscored")}
        out.append({"entry": pid, "built": True, "accepted": r.get("accepted"), "reason": r.get("reason"),
                    "rows": len(detail), **counts, "detail": detail})
        print(f"{pid:32s} accepted={r.get('accepted')!s:5s} right {counts['right']}/{len(detail)}  "
              f"wrong {counts['WRONG']}  crash {counts['crash']}  unscored {counts['unscored']}", flush=True)

    def tally(group):
        g = [o for o in out if o.get("built") and o["accepted"] is group]
        return {k: sum(o[k] for o in g) for k in ("rows", "right", "WRONG", "crash", "unscored")} | {"entries": len(g)}
    summary = {"accepted": tally(True), "rejected": tally(False), "not_built": [o["entry"] for o in out if not o.get("built")],
               "cost": round(llm.LEDGER.cost(), 4)}
    (config.WORK_DIR / "cert_truth.json").write_text(json.dumps({"summary": summary, "entries": out}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
