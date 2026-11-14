"""Turn the 17 seed rows of the corpus into problem files the pipeline can build from.

    .venv/bin/python eval/make_problems.py      -> problems/corpus/<entry>.json

Only seed rows are written, so the builder never sees any other corpus row.
The problem id is the row's expected_entry, which is how the harness checks
that a hit went to the right entry.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval" / "corpus" / "cache_eval_corpus.jsonl"
OUT = ROOT / "problems" / "corpus"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [r for r in map(json.loads, CORPUS.read_text().splitlines()) if r["split"] == "seed"]
    for r in seeds:
        problem = {
            "id": r["expected_entry"],
            "source": f"cache eval corpus seed {r['id']} ({r['domain']})",
            "question": r["question"],
            "golden_label": r["answer"],
        }
        (OUT / f"{r['expected_entry']}.json").write_text(json.dumps(problem, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(seeds)} problems to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
