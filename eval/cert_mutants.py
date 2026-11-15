"""Mutation test for the agreement certificate (PAID, Haiku: about a cent per mutant).

The fault-injection run showed the builder refuses broken gas-law transcripts on its
own, so the certificate never ran on them. Here we break the CODE of the clean,
accepted gas-law entry instead and certify each mutant. Mutants that are right at
the original values but wrong elsewhere are the cases only the new instances catch.

    .venv/bin/python eval/cert_mutants.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["ARC_PROFILE"] = "haiku"
os.environ["ARC_NAMESPACE"] = "cert_faults"
sys.path.insert(0, str(ROOT))

from arc import certify, config, llm, source  # noqa: E402
from arc.cache import Cache  # noqa: E402

# (name, which function (description prefix), old text, new text, right at the original values?)
MUTANTS = [
    ("hardcoded_temperature", "Convert a temperature", "    return {'temperature in K': kelvin}",
     "    kelvin = 300.15\n    return {'temperature in K': kelvin}", True),
    ("hardcoded_amount", "Compute the perfect gas pressure", "    nRT = n * R * T",
     "    n = 10.0\n    nRT = n * R * T", True),
    ("litre_factor_bug", "Convert a volume with unit", "'l': 1e-3", "'l': 1e-6", True),
    ("wrong_R", "Compute the perfect gas pressure", "    R = 8.314", "    R = 0.08314", False),
    ("celsius_as_kelvin", "Convert a temperature", "        kelvin = value + 273.15", "        kelvin = value", False),
]


def mutate(cache: Cache, prefix: str, old: str, new: str) -> None:
    rec = next(r for r in cache.functions.values() if r.description.startswith(prefix))
    assert old in rec.source, (prefix, old)
    cache.add_function(rec.description, rec.source.replace(old, new), rec.inputs, rec.outputs,
                       origin="mutant", replace_id=rec.id)


def main() -> None:
    problem = {**json.loads((config.PROBLEMS_DIR / "gas_law.json").read_text()), "id": "gas_law_clean"}
    t = source.load("gas_law_clean")
    slip = source.load("gas_law_fault_final_slip")
    jobs = [(name, (prefix, old, new), t, right) for name, prefix, old, new, right in MUTANTS]
    # Correct code + a transcript whose final number is wrong: only the transcript check can see it.
    jobs.append(("correct_code_wrong_transcript", None, source.Transcript(**{**slip.__dict__, "problem_id": "gas_law_clean"}), None))
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    jobs = [j for j in jobs if not only or j[0] in only]
    rows = []
    for name, mut, transcript, right_at_original in jobs:
        cache = Cache.load(config.CACHE_ROOT / "gas_law_clean")
        if mut:
            mutate(cache, *mut)
        before = llm.LEDGER.cost()
        cert = certify.certify(cache, problem, transcript)
        first_fail = next((c.name for c in cert.comparisons if not c.agree), cert.reason)
        rows.append({"mutant": name, "right_at_original": right_at_original, "accepted": cert.accepted,
                     "caught_by": None if cert.accepted else first_fail, "instances": cert.instances,
                     "comparisons": [{"name": c.name, "agree": c.agree, "fresh": c.reference[:80],
                                      "cache_or_transcript": c.candidate[-120:]} for c in cert.comparisons],
                     "cost": llm.LEDGER.cost() - before})
        print(f"{name:32s} accepted={cert.accepted!s:5s} caught_by={rows[-1]['caught_by']}  ${rows[-1]['cost']:.4f}", flush=True)
        for q in cert.instances:
            print("      instance:", q[:130])
    out = config.WORK_DIR / ("cert_mutants.json" if not only else f"cert_mutants_{'_'.join(only)}.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\ncaught {sum(not r['accepted'] for r in rows)}/{len(rows)}, ${llm.LEDGER.cost():.4f}; saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
