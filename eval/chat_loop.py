"""End-to-end loop test: a chatbot wrapped in the cache, starting empty. PAID (Haiku only).

    .venv/bin/python eval/chat_loop.py --dry-run     # free: stub model + stub builder, checks plumbing
    .venv/bin/python eval/chat_loop.py --fresh       # the real run (asks nothing; get approval first)

A scripted user sends ~105 standalone messages, in shuffled order:
  - 3 recurring task families, 25 variants each, values and answers computed in code
    (formulas copied from eval/corpus/build_corpus.py): gas pressure, mortgage payment,
    double-declining-balance book value
  - every seed+dev near-miss question for those 3 families (should NOT be served)
  - 6 one-off questions from other families, and 6 chit-chat messages
Every 10 messages the bot learns: backlog -> families -> build_ready (arc/labelfree.py,
simple flow: newest record re-solved by its own production model, built if the answers
match) -> reload the live cache.

Production model: Haiku (project rule). The break-even meter prices its answers as if
they were Opus (ARC_PRICES_FILE), so families become worth building after a few
questions instead of ~87. No Opus call is made: the records say Haiku, so the re-solve
is Haiku too. Gate: ARC_GATE=clef (needs CLOUDFLARE_* in .env).

Scored for free afterwards: every answer (cache or model) is checked against the
computed answer with eval/run_corpus.score.

Results: runs/chat_loop/chat_loop.json and the backlog / families / build logs beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NS = "chat_loop"
PRICES = ROOT / "runs" / NS / "_prices_as_opus.json"


def configure() -> None:
    """Experiment settings. Called before any arc import when run as a script, never on
    import (tests import the scoring helpers and must keep their own settings)."""
    os.environ["ARC_PROFILE"] = "haiku"
    os.environ["ARC_NAMESPACE"] = NS
    os.environ["ARC_GATE"] = "clef"          # the chosen gate: structure retrieval + Clef decides
    os.environ["ARC_PRICES_FILE"] = str(PRICES)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

# ---------------------------------------------------------------- variants (answers in code)
R = 0.082057366
GASES = [("N2", "nitrogen"), ("O2", "oxygen"), ("CO2", "carbon dioxide"), ("CH4", "methane"), ("Ar", "argon"),
         ("He", "helium"), ("C2H6", "ethane"), ("NH3", "ammonia")]


def sig(v, n=4):
    d = n - int(math.floor(math.log10(abs(v)))) - 1
    return f"{round(v, d):,.{max(d, 0)}f}"


def money(v, cur="$"):
    return f"{cur}{v:,.2f}"


def gas(rng):
    f, name = rng.choice(GASES)
    n = round(rng.uniform(0.5, 12), 2)
    V = round(rng.uniform(1.5, 30), 2)
    c = rng.randint(-20, 150)
    p = n * R * (c + 273.15) / V
    q = rng.choice([
        f"Suppose that {n} mol {f}(g) is confined to {V} dm3 at {c}°C. Predict the pressure exerted by the {name} from the perfect gas equation.",
        f"What pressure do {n} moles of {name} exert in a {V} litre container at {c} °C, treating it as a perfect gas?",
        f"{n} mol of {name} gas is held in a {V} dm3 vessel at {c}°C. Assuming ideal behaviour, calculate the pressure.",
        f"A {V} L flask holds {n} mol of {name} at {c + 273.15:.2f} K. Assuming perfect-gas behaviour, what pressure does the gas exert?",
        f"Calculate the pressure of {n} mol of {name} in a {V * 1000:.0f} cm3 container at {c} °C, assuming it behaves as a perfect gas.",
    ])
    return q, p, "atm", f"{sig(p)} atm"


def pmt(P, rate, years):
    r, n = rate / 12, years * 12
    return P * r / (1 - (1 + r) ** -n)


def loan(rng):
    P = rng.randrange(80, 900) * 1000
    rate = rng.choice([3.5, 4.25, 4.75, 5.0, 5.5, 5.875, 6.25, 6.5, 6.875, 7.0, 7.25, 7.5])
    yrs = rng.choice([10, 15, 20, 25, 30])
    m = pmt(P, rate / 100, yrs)
    q = rng.choice([
        f"A ${P:,} mortgage has a {rate}% annual interest rate, compounded monthly, over {yrs} years. What is the monthly payment, and how much total interest is paid over the life of the loan?",
        f"What are the monthly payments on a {yrs}-year, ${P:,} home loan at {rate}% per year (monthly compounding), and what is the total interest over the full term?",
        f"I'm taking out ${P:,} at {rate}% for {yrs} years, paid monthly. How much will I pay each month and how much of that ends up being interest overall?",
        f"Calculate the monthly installment and lifetime interest for a ${P // 1000}k mortgage at {rate}% annual interest compounded monthly, repaid over {yrs} years.",
    ])
    return q, m, "$", f"Payment {money(m)}/month; total interest {money(m * yrs * 12 - P)}"


def ddb(cost, salv, life, year):
    bv = cost
    for _ in range(year):
        bv -= max(min(bv * 2 / life, bv - salv), 0)
    return bv


def depreciation(rng):
    cost = rng.randrange(10, 400) * 1000
    salv = round(cost * rng.uniform(0.05, 0.2), -2)
    life = rng.randint(4, 12)
    year = rng.randint(1, life - 1)
    v = ddb(cost, salv, life, year)
    q = rng.choice([
        f"Equipment costs ${cost:,}, has a salvage value of ${salv:,.0f}, and a useful life of {life} years. Using double-declining-balance depreciation, what is its book value at the end of year {year}?",
        f"Using the double-declining-balance method, find the book value after {year} years of an asset that cost ${cost:,}, has a {life}-year life and a ${salv:,.0f} salvage value.",
        f"A ${cost:,} machine with ${salv:,.0f} residual value is depreciated over {life} years by double-declining balance. What is its carrying amount at the end of year {year}?",
        f"Cost ${cost:,}; salvage ${salv:,.0f}; life {life} years; method DDB. What is the net book value at the close of year {year}?",
    ])
    return q, v, "$", money(v)


FAMILIES = {"perfect_gas_pressure": gas, "loan_payment_total_interest": loan, "ddb_book_value": depreciation}
CHITCHAT = ["hi!", "thanks, that helps a lot", "what kinds of questions can you answer?",
            "ok cool", "can you explain that more simply please?", "great, bye for now"]


def script(per_family: int, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    msgs = []
    for fam, gen in FAMILIES.items():
        seen = set()
        while sum(m["family"] == fam for m in msgs) < per_family:
            q, v, unit, ans = gen(rng)
            if q not in seen:
                seen.add(q)
                msgs.append({"kind": "variant", "family": fam, "question": q, "answer": ans, "answer_value": v,
                             "answer_unit": unit, "should_hit": True})
    corpus = [json.loads(l) for l in (ROOT / "eval/corpus/cache_eval_corpus.jsonl").read_text().splitlines()]
    dev = [r for r in corpus if r["split"] in ("seed", "dev")]
    for r in dev:
        if r["seed_entry"] in FAMILIES and not r["should_hit"]:
            msgs.append({"kind": "near_miss", "family": r["seed_entry"], "question": r["question"],
                         "answer": r["answer"], "answer_value": r["answer_value"], "answer_unit": r["answer_unit"],
                         "should_hit": False, "acceptable_either": r["acceptable_either"],
                         "category": r["near_miss_category"]})
    others = [r for r in dev if r["seed_entry"] not in FAMILIES and r["variant_type"] == "original"]
    for r in rng.sample(others, 6):
        msgs.append({"kind": "one_off", "family": r["seed_entry"], "question": r["question"], "answer": r["answer"],
                     "answer_value": r["answer_value"], "answer_unit": r["answer_unit"], "should_hit": True})
    msgs += [{"kind": "chitchat", "family": None, "question": c, "answer": "", "answer_value": None,
              "answer_unit": "", "should_hit": False} for c in CHITCHAT]
    rng.shuffle(msgs)
    return msgs


# ---------------------------------------------------------------- scoring

# A pressure is right in any of these units (score() only tries powers of 1000).
UNIT_FORMS = {"atm": {"atm": 1.0, "bar": 1.01325, "kPa": 101.325, "Pa": 101325.0, "mmHg": 760.0}}


def score_message(m: dict, text: str):
    """(correct, how) for one answer: the computed answer in its own unit or an equivalent one."""
    from run_corpus import score
    if m["kind"] == "chitchat":
        return None, "chitchat"
    ok, how = score(m, text)
    if ok or m.get("answer_value") is None:
        return ok, how
    import re
    from run_corpus import _numbers_with_precision, _agrees
    for unit, factor in UNIT_FORMS.get(m.get("answer_unit"), {}).items():
        if factor == 1.0:
            continue
        # only numbers actually written in that unit: "526 kPa", or a label before the number
        # such as "pressure in bar: 7.7589" (never "18.8 atm" read as bar)
        num = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        u = re.escape(unit)
        patterns = [rf"{num}\s*{u}(?![A-Za-z])", rf"(?<![A-Za-z]){u}\)?\s*[:=]\s*{num}"]
        for pat in patterns:
            for hit in re.finditer(pat, text):
                pre = text[max(0, hit.start(1) - 1):hit.start(1)]
                for x, _, h in _numbers_with_precision(hit.group(1)):
                    if _agrees(x, h, m["answer_value"] * factor, False) and not pre.isalpha():
                        return True, f"number in {unit}"
    return ok, how


def rescore(work: Path) -> dict:
    """Re-score a finished run from its saved files (free): full answers come from the backlog."""
    data = json.loads((work / "chat_loop.json").read_text())
    full = {}
    for line in (work / "backlog.jsonl").read_text().splitlines():
        r = json.loads(line)
        full.setdefault(r["question"], []).append(r["response"])
    msgs = {m["question"][:160]: m for m in script(data["settings"]["per_family"])}
    for row in data["messages"]:
        m = msgs.get(row["question"])
        answers = [a for q, a in full.items() if q[:160] == row["question"]]
        text = answers[0][0] if answers else row["answer"]
        if m is not None:
            row["correct"], row["how"] = score_message(m, text)
    return data


# ---------------------------------------------------------------- the run

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stub model and builder: free plumbing check")
    ap.add_argument("--fresh", action="store_true", help="delete this experiment's earlier backlog and caches")
    ap.add_argument("--per-family", type=int, default=25)
    ap.add_argument("--learn-every", type=int, default=10)
    ap.add_argument("--max-cost", type=float, default=0.60, help="stop sending messages past this spend ($)")
    args = ap.parse_args()

    configure()
    from arc import config, llm
    if args.dry_run:
        config.WORK_DIR = ROOT / "runs" / f"{NS}_dry"
        config.CACHE_ROOT = ROOT / "caches" / f"{NS}_dry"
    work, caches = config.WORK_DIR, config.CACHE_ROOT
    if args.fresh or args.dry_run:
        for d in (work, caches):     # only this experiment's own folders
            if d.exists() and d.name.startswith(NS):
                shutil.rmtree(d)
    elif (work / "backlog.jsonl").exists():
        sys.exit(f"{work.relative_to(ROOT)}/backlog.jsonl exists; pass --fresh to start the loop from empty")
    work.mkdir(parents=True, exist_ok=True)
    PRICES.parent.mkdir(parents=True, exist_ok=True)
    PRICES.write_text(json.dumps({"claude-haiku-5-5": [4.0, 20.0]}))   # meter prices Haiku as Opus 5.5

    from arc import chat
    msgs = script(args.per_family)

    production = build_fn = None
    if args.dry_run:
        def production(message):   # pretend model: echoes the right answer when it knows it
            m = next((x for x in msgs if x["question"] == message), None)
            return llm.LLMResult(text=f"Final answer: {m['answer'] if m else 'ok'}", model="claude-haiku-5-5",
                                 input_tokens=200, output_tokens=1500)

        def build_fn(fid, records):
            return type("R", (), {"accepted": False, "entry_id": None, "status": "need_more_data", "cost": 0.0,
                                  "reason": f"dry run ({len(records)} records)", "parts": {}, "serve_rate": None})()
    else:
        from arc.labelfree import build_family as build_fn

    bot = chat.Chatbot(production=production)
    log, builds, t0 = [], [], time.time()
    for i, m in enumerate(msgs, 1):
        spent = llm.LEDGER.cost()
        if spent > args.max_cost:
            print(f"stopping at message {i}: spent ${spent:.3f} > --max-cost {args.max_cost}")
            break
        before = llm.LEDGER.cost()
        r = bot.chat(m["question"], session=f"user-{i % 7}")
        ok, how = score_message(m, r.answer)
        log.append({"n": i, **{k: m.get(k) for k in ("kind", "family", "category", "acceptable_either")},
                    "question": m["question"][:160], "expected": m["answer"], "from_cache": r.from_cache,
                    "entry": r.entry, "reason": r.reason, "correct": ok, "how": how,
                    "answer": r.answer[-200:], "cost": round(llm.LEDGER.cost() - before, 6)})
        flag = "CACHE" if r.from_cache else "model"
        print(f"{i:3d} {m['kind']:9s} {str(m['family'])[:22]:22s} {flag:5s} correct={ok!s:5s} {r.reason[:60]}",
              flush=True)
        if i % args.learn_every == 0 or i == len(msgs):
            before = llm.LEDGER.cost()
            # each build_family call is hard-capped at LABELFREE_MAX_COST, so only start as many
            # builds as the remaining budget can cover in the worst case
            room = int(max(0.0, args.max_cost - before) // max(config.LABELFREE_MAX_COST, 1e-9)) \
                if not args.dry_run else None
            if room == 0:   # still group questions into families; only builds wait for budget
                print(f"    no builds this step: ${before:.3f} spent leaves no room for a "
                      f"${config.LABELFREE_MAX_COST:.2f} worst-case build under --max-cost {args.max_cost}",
                      flush=True)
            for fid, res in bot.learn(build_fn=build_fn, max_builds=room):
                row = {"after_message": i, "family": fid,
                       "label": bot.tracker.families[fid].label[:80],
                       "status": getattr(res, "status", None) if res else "error",
                       "entry": getattr(res, "entry_id", None) if res else None,
                       "reason": (getattr(res, "reason", "") if res else bot.tracker.families[fid].note)[:200],
                       "cost": round(getattr(res, "cost", 0.0) or 0.0, 5) if res else 0.0}
                builds.append(row)
                if row["status"] != "need_more_data" or row["cost"]:
                    print(f"    build {fid}: {row['status']} {row['entry'] or ''} ${row['cost']:.4f}  {row['reason'][:70]}", flush=True)
            spent_learning = llm.LEDGER.cost() - before
            if spent_learning:
                print(f"    learning step cost ${spent_learning:.4f}; live entries: "
                      f"{sorted({e.problem_id for e in bot.cache.entry_points})}")

    summary = summarize(log, builds, bot, llm.LEDGER.cost(), time.time() - t0)
    out = work / "chat_loop.json"
    out.write_text(json.dumps({"summary": summary, "builds": builds, "messages": log,
                               "settings": {"per_family": args.per_family, "learn_every": args.learn_every,
                                            "dry_run": args.dry_run}}, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"saved {out.relative_to(ROOT)}")


def summarize(log, builds, bot, cost, seconds) -> dict:
    def count(**want):
        return sum(all(r.get(k) == v for k, v in want.items()) for r in log)
    first_hit = {}
    for r in log:
        if r["from_cache"] and r["family"] not in first_hit:
            first_hit[r["family"]] = r["n"]
    served_nm = [r for r in log if r["kind"] == "near_miss" and r["from_cache"]]
    return {
        "messages": len(log),
        "cache_answers": count(from_cache=True),
        "cache_right": count(from_cache=True, correct=True),
        "cache_wrong_on_variants": count(from_cache=True, kind="variant", correct=False),
        "near_miss_served": len(served_nm),
        "near_miss_served_and_wrong": sum(r["correct"] is not True and not r["acceptable_either"] for r in served_nm),
        "variants_after_entry_live": {f: sum(r["family"] == f and r["kind"] == "variant" and r["n"] > n for r in log)
                                      for f, n in first_hit.items()},
        "variant_hit_rate_after_live": {f: round(sum(r["family"] == f and r["kind"] == "variant" and r["n"] >= n
                                                     and r["from_cache"] for r in log)
                                                 / max(1, sum(r["family"] == f and r["kind"] == "variant" and r["n"] >= n
                                                              for r in log)), 2)
                                        for f, n in first_hit.items()},
        "first_cache_hit_at_message": first_hit,
        "model_answers_right": count(from_cache=False, correct=True),
        "model_answers_scored": sum(not r["from_cache"] and r["correct"] is not None for r in log),
        "builds": {s: sum(b["status"] == s for b in builds) for s in ("accepted", "mismatch", "not_ready", "rejected", "need_more_data", "error")},
        "build_spend": round(sum(b["cost"] for b in builds), 4),
        "chitchat_skipped_by_families": bot.tracker.skipped,
        "families_seen": len(bot.tracker.families),
        "total_cost": round(cost, 4),
        "seconds": round(seconds),
    }


if __name__ == "__main__":
    main()
