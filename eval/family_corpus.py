"""How well does family grouping find the real task families? (FREE: local embeddings only.)

    .venv/bin/python eval/family_corpus.py

The 383-row cache-eval corpus has 17 true families (the seeds) plus 179 near-miss /
unrelated questions. We feed every row, in shuffled order, through the family
assigner at several similarity thresholds and measure:
  recall      share of a true family's questions that land in its biggest group
              (low = one task split into several meters, each filling slowly)
  purity      share of a group's should-hit questions that come from one true family
              (low = two tasks share one meter)
  near-miss absorbed   share of should-miss questions that join a group built around a
              seed (inflates that seed's count; the serve rate corrects it later)
This tunes FAMILY_SIMILARITY, so it uses the seed + dev splits only. Pass --test to
run the held-out test split once, after the threshold is frozen.

Results: runs/family_corpus.json
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ARC_PROFILE", "haiku")
sys.path.insert(0, str(ROOT))

from arc import backlog, config, embed, families  # noqa: E402

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92]
# --check clef: each question must also be confirmed by Clef against the family's anchor
# (config.FAMILY_CHECK="clef"). PAID: about one Clef call per question; run at one threshold.


def main() -> None:
    splits = ("test",) if "--test" in sys.argv else ("seed", "dev")
    check = "clef" if "--check" in sys.argv and sys.argv[sys.argv.index("--check") + 1] == "clef" else "embed"
    config.FAMILY_CHECK = check
    if "--cutoff" in sys.argv:
        config.FAMILY_CLEF_THRESHOLD = float(sys.argv[sys.argv.index("--cutoff") + 1])
    thresholds = [config.FAMILY_SIMILARITY] if check == "clef" else THRESHOLDS
    from arc import llm
    rows = [json.loads(l) for l in (ROOT / "eval/corpus/cache_eval_corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] in splits]
    random.Random(7).shuffle(rows)
    vecs = embed.embedder().encode([families.mask_numbers(r["question"]) for r in rows])
    print(f"embedded {len(rows)} questions with {embed.embedder().name}")
    results = []
    for th in thresholds:
        spent_before = llm.LEDGER.cost()
        config.FAMILY_SIMILARITY = th
        tr = families.FamilyTracker(path=ROOT / "runs" / "_family_corpus_tmp.json", embed_fn=lambda t: None)
        tr.families, tr.seen_ids, tr.assignments = {}, set(), {}
        group_of = {}
        for i, (r, v) in enumerate(zip(rows, vecs)):
            rec = backlog.make_record(r["question"], r["answer"], "claude-haiku-5-5", input_tokens=200,
                                      output_tokens=1500, timestamp=float(i), id=r["id"])
            group_of[r["id"]] = tr.observe(rec, v).id
        hits = [r for r in rows if r["should_hit"]]
        misses = [r for r in rows if not r["should_hit"]]
        by_true = defaultdict(Counter)
        by_group = defaultdict(Counter)
        for r in hits:
            by_true[r["expected_entry"]][group_of[r["id"]]] += 1
            by_group[group_of[r["id"]]][r["expected_entry"]] += 1
        recall = {e: c.most_common(1)[0][1] / sum(c.values()) for e, c in by_true.items()}
        purity_num = sum(c.most_common(1)[0][1] for c in by_group.values())
        seed_groups = {g for g, c in by_group.items()}
        absorbed = [r for r in misses if group_of[r["id"]] in seed_groups]
        absorbed_same_seed = [r for r in absorbed if by_group[group_of[r["id"]]].most_common(1)[0][0] == r["seed_entry"]]
        res = {"threshold": th, "groups": len(tr.families),
               "groups_with_hits": len(by_group),
               "mean_recall": round(sum(recall.values()) / len(recall), 3),
               "worst_recall": sorted(((round(v, 2), e) for e, v in recall.items()))[:3],
               "purity": round(purity_num / len(hits), 3),
               "merged_pairs": [sorted(c) for c in by_group.values() if len(c) > 1],
               "near_miss_absorbed": round(len(absorbed) / len(misses), 3),
               "near_miss_absorbed_into_own_seed": round(len(absorbed_same_seed) / len(misses), 3),
               "absorbed_by_category": dict(Counter(r["near_miss_category"] for r in absorbed).most_common(6)),
               "check": check, "family_clef_cutoff": config.FAMILY_CLEF_THRESHOLD, "clef_prompt": config.CLEF_PROMPT, "clef_cost": round(llm.LEDGER.cost() - spent_before, 5),
               "clef_errors": sum("error" in a for a in tr.assignments.values()),
               "split_off": [{"id": r["id"], "entry": r["expected_entry"], "q": r["question"][:120],
                              "how": tr.assignments.get(r["id"], {})}
                             for e, c in by_true.items() for r in hits
                             if r["expected_entry"] == e and group_of[r["id"]] != c.most_common(1)[0][0]],
               "assignments": {r["id"]: {**tr.assignments.get(r["id"], {}), "should_hit": r["should_hit"],
                                          "expected_entry": r["expected_entry"], "seed_entry": r["seed_entry"],
                                          "category": r["near_miss_category"], "either": r["acceptable_either"],
                                          "q": r["question"][:160]} for r in rows},
               "absorbed": [{"id": r["id"], "seed": r["seed_entry"], "category": r["near_miss_category"],
                             "q": r["question"][:120], "how": tr.assignments.get(r["id"], {})} for r in absorbed]}
        results.append(res)
        print(f"th {th:.2f}: {res['groups']:3d} groups ({res['groups_with_hits']} hold seed questions) | "
              f"recall {res['mean_recall']:.2f} (worst {res['worst_recall'][0]}) | purity {res['purity']:.2f} | "
              f"near-miss absorbed {res['near_miss_absorbed']:.2f} | check {check} ${res['clef_cost']:.4f}"
              f" errors {res['clef_errors']}")
    (ROOT / "runs" / "_family_corpus_tmp.json").unlink(missing_ok=True)
    out = ROOT / "runs" / f"family_corpus_{'_'.join(splits)}{'' if check == 'embed' else f'_clef{config.FAMILY_CLEF_THRESHOLD:g}'}.json"
    out.write_text(json.dumps({"embedder": embed.embedder().name, "splits": splits, "rows": len(rows), "results": results}, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
