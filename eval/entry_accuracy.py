"""Is the cached CODE right? FREE: no model calls.

Runs each built entry's functions directly on generated inputs whose answers are computed here
in plain Python, so extraction is out of the picture: any wrong answer is the builder's. Inputs
are written the ways users write them (the entry's own format examples and more: "$150k",
"twelve years", "0.45 kg", "98.6 °F", "897 J/(kg·K)") and include edge cases (salvage floor
binding, year past the useful life, a 0% loan). Each category says whether the entry claims to
handle it (its format examples or conventions mention it) or not.

Also runs the hit guards (output checks, reference) on every answer: do they stop the code's
wrong answers, and do they block right ones?

    .venv/Scripts/python eval/entry_accuracy.py
"""

from __future__ import annotations

import json
import random
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEV = ROOT / "caches" / "chat_loop" / "master"
HID = ROOT / "caches" / "chat_loop_hidden" / "master"
N = 25   # cases per category

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty", 25: "twenty-five", 30: "thirty"}


def money(x: float) -> str:
    return f"${x:,.0f}" if x == int(x) else f"${x:,.2f}"


def cents(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ----------------------------------------------------------------------------- DDB book value

DDB_KEYS = ("the number of years elapsed since the asset was acquired, as a whole number of years",
            "the original cost of the asset in dollars with currency unit",
            "the useful life of the asset in years",
            "the estimated salvage value of the asset in dollars with currency unit")


def ddb(cost, salv, life, year):
    bv = cost
    for _ in range(year):
        bv -= max(min(bv * 2 / life, bv - salv), 0)
    return bv


def ddb_cases(rng):
    def case(cat, claimed, cost, salv, life, year, cost_s=None, salv_s=None, life_s=None, year_s=None):
        return {"cat": cat, "claimed": claimed,
                "values": dict(zip(DDB_KEYS, (year_s or f"{year} years", cost_s or money(cost),
                                              life_s or f"{life} years", salv_s or money(salv)))),
                "expected": {"book value after the elapsed years in dollars": ddb(cost, salv, life, year)}}
    out = []
    for _ in range(N):
        cost = rng.randrange(10, 900) * 1000; salv = round(cost * rng.uniform(0.02, 0.2), -2)
        life = rng.randint(3, 15); year = rng.randint(1, life - 1)
        out.append(case("plain digits", True, cost, salv, life, year))
        out.append(case("shorthand $k / million", True, cost, salv, life, year,
                        cost_s=rng.choice([f"${cost // 1000}k", f"${cost / 1e6:g} million"]),
                        salv_s=f"${salv / 1000:g}k"))
        y2 = rng.choice([y for y in WORDS if y < life] or [1])
        out.append(case("numbers as words", True, cost, salv, life, y2, year_s=f"{WORDS[y2]} years",
                        life_s=f"{WORDS.get(life, life)}-year life" if life in WORDS else f"a {life}-year useful life"))
        out.append(case("salvage floor binds (late years)", True, cost, cost * 0.4, life, life - 1))
        out.append(case("year at or past useful life", False, cost, salv, life, rng.randint(life, life + 3)))
        out.append(case("zero salvage", False, cost, 0, life, year, salv_s="$0"))
    return out


# ----------------------------------------------------------------------------- loan payment

LOAN_KEYS = ("the loan principal amount in dollars", "the annual nominal interest rate as a percentage",
             "the compounding and payment frequency description", "the loan term with unit")
FREQ = {"monthly": 12, "quarterly": 4, "semiannually": 2, "annually": 1, "biweekly": 26, "weekly": 52}


def loan_expected(P, rate_pct, per, years):
    r, n = rate_pct / 100 / per, round(per * years)
    pay = P / n if r == 0 else P * r / (1 - (1 + r) ** -n)
    mp = cents(pay)
    # the entry's stated convention: interest from the rounded payment (user: set rounding aside)
    return {"monthly payment in dollars": mp, "total interest in dollars": cents(mp * n - P)}


def loan_cases(rng):
    def case(cat, claimed, P, rate, freq, years, P_s=None, rate_s=None, term_s=None):
        return {"cat": cat, "claimed": claimed,
                "values": dict(zip(LOAN_KEYS, (P_s or money(P), rate_s or f"{rate}%", freq,
                                               term_s or f"{years} years"))),
                "expected": loan_expected(P, rate, FREQ[freq], years)}
    out = []
    for _ in range(N):
        P = rng.randrange(50, 1500) * 1000
        rate = rng.choice([3.5, 4.25, 5.0, 5.875, 6.5, 7.25, 8.0, 9.75])
        yrs = rng.choice([5, 10, 15, 20, 25, 30])
        out.append(case("plain digits, monthly", True, P, rate, "monthly", yrs))
        out.append(case("shorthand $k / million", True, P, rate, "monthly", yrs,
                        P_s=rng.choice([f"${P // 1000}k", f"${P / 1e6:g} million"])))
        out.append(case("'4 percent' wording", True, P, rate, "monthly", yrs, rate_s=f"{rate} percent"))
        out.append(case("term in months", True, P, rate, "monthly", yrs, term_s=f"{yrs * 12} months"))
        out.append(case("term in words", False, P, rate, "monthly", yrs,
                        term_s=f"{WORDS[yrs]} years" if yrs in WORDS else f"{yrs} years"))
        f = rng.choice(["quarterly", "semiannually", "annually"])
        out.append(case("other frequencies (quarterly...)", True, P, rate, f, yrs))
        out.append(case("biweekly / weekly", False, P, rate, rng.choice(["biweekly", "weekly"]), yrs))
        out.append(case("0% interest", False, P, 0.0, "monthly", yrs, rate_s="0%"))
    return out


# ----------------------------------------------------------------------------- calorimetry

CAL_KEYS = ("the mass of the water (the colder body) with unit", "the initial temperature of the water with unit",
            "the mass of the aluminum object (the hot body) with unit",
            "the initial temperature of the aluminum object with unit",
            "the specific heat of aluminum with unit", "the specific heat of water with unit")


def cal_cases(rng):
    def case(cat, claimed, mw, tw, mm, tm, cm, cw, mw_s=None, tw_s=None, mm_s=None, tm_s=None, cm_s=None, cw_s=None):
        T = (mw * cw * (tw + 273.15) + mm * cm * (tm + 273.15)) / (mw * cw + mm * cm)
        return {"cat": cat, "claimed": claimed,
                "values": dict(zip(CAL_KEYS, (mw_s or f"{mw:g} g", tw_s or f"{tw:g} °C", mm_s or f"{mm:g} g",
                                              tm_s or f"{tm:g} °C", cm_s or f"{cm:g} J/(g·°C)",
                                              cw_s or f"{cw:g} J/(g·°C)"))),
                "expected": {"equilibrium temperature in K": T}}
    out = []
    for _ in range(N):
        mw, mm = rng.randrange(100, 900), rng.randrange(50, 600)
        tw, tm = rng.randrange(5, 40), rng.randrange(60, 250)
        cm, cw = rng.choice([0.385, 0.449, 0.897, 0.129, 0.523]), rng.choice([4.18, 4.184, 4.186])
        out.append(case("grams and °C", True, mw, tw, mm, tm, cm, cw))
        out.append(case("kg masses", True, mw, tw, mm, tm, cm, cw, mw_s=f"{mw / 1000:g} kg", mm_s=f"{mm / 1000:g} kg"))
        out.append(case("kelvin temperatures", True, mw, tw, mm, tm, cm, cw,
                        tw_s=f"{tw + 273.15:.2f} K", tm_s=f"{tm + 273.15:.2f} K"))
        out.append(case("°F temperatures", True, mw, tw, mm, tm, cm, cw,
                        tw_s=f"{tw * 9 / 5 + 32:g} °F", tm_s=f"{tm * 9 / 5 + 32:g} °F"))
        out.append(case("J/(kg·K) specific heats", True, mw, tw, mm, tm, cm, cw,
                        cm_s=f"{cm * 1000:g} J/(kg·K)", cw_s=f"{cw * 1000:g} J/(kg·K)"))
        out.append(case("cal/(g·°C) specific heats", True, mw, tw, mm, tm, cm, cw,
                        cm_s=f"{cm / 4.184:.4g} cal/(g·°C)", cw_s="1.00 cal/(g·°C)")
                   | {"expected": {"equilibrium temperature in K":
                                   (mw * 4.184 * (tw + 273.15) + mm * float(f"{cm / 4.184:.4g}") * 4.184 * (tm + 273.15))
                                   / (mw * 4.184 + mm * float(f"{cm / 4.184:.4g}") * 4.184)}})
        out.append(case("water in mL (density 1 g/mL)", False, mw, tw, mm, tm, cm, cw, mw_s=f"{mw} mL"))
    return out


# ----------------------------------------------------------------------------- running

def close(got, want, money_out: bool) -> bool:
    try:
        g = float(str(got).replace(",", "").replace("$", ""))
    except ValueError:
        return False
    return abs(g - want) <= (0.011 if money_out else max(1e-6 * abs(want), 1e-9))


def run_entry(cache, entry_id: str, cases: list[dict]) -> dict:
    from arc import outchecks, reference, serve
    ep = next(e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original")
    specs = (cache.outputs or {}).get(entry_id) or {}
    cats: dict[str, dict] = {}
    for c in cases:
        t = cats.setdefault(c["cat"], {"claimed": c["claimed"], "n": 0, "right": 0, "wrong": 0, "crashed": 0,
                                       "wrong_stopped": 0, "right_blocked": 0, "examples": []})
        t["n"] += 1
        try:
            raw, _ = serve.dispatch_root(cache, ep.root_id, c["values"])
        except Exception as e:   # noqa: BLE001 - a crash falls back to the model: safe, but a lost hit
            t["crashed"] += 1
            if len(t["examples"]) < 2:
                t["examples"].append(f"crash: {type(e).__name__}: {str(e)[:110]}")
            continue
        ok = all(close(raw.get(k), v, (specs.get(k) or {}).get("kind") == "money") for k, v in c["expected"].items())
        guard = (outchecks.run(cache, entry_id, c["values"], raw) or []) + \
                (reference.run(cache, entry_id, c["values"], raw) or [])
        if ok:
            t["right"] += 1
            t["right_blocked"] += bool(guard)
            if guard and len(t["examples"]) < 2:
                t["examples"].append(f"false alarm: {guard[0][:110]}")
        else:
            t["wrong"] += 1
            t["wrong_stopped"] += bool(guard)
            if len(t["examples"]) < 2:
                got = {k: str(raw.get(k))[:14] for k in c["expected"]}
                want = {k: round(v, 4) for k, v in c["expected"].items()}
                t["examples"].append(f"wrong{' (guard stopped it)' if guard else ''}: {c['values']} got {got} want {want}")
    return cats


def main() -> None:
    if not sys.flags.utf8_mode:
        import subprocess
        sys.exit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)
    from arc.cache import Cache
    rng = random.Random(7)
    plan = [("DDB book value", DEV, "fam_f0002", ddb_cases(rng)),
            ("loan payment", DEV, "fam_f0003", loan_cases(rng)),
            ("calorimetry", HID, "fam_f0002", cal_cases(rng))]
    report = {}
    for name, folder, entry, cases in plan:
        cats = run_entry(Cache.load(folder), entry, cases)
        report[name] = cats
        print(f"\n=== {name} ({len(cases)} cases)")
        print(f"{'inputs':36s} {'claims':>6s} {'right':>6s} {'wrong':>6s} {'crash':>6s}   guards: wrong stopped / right blocked")
        for cat, t in cats.items():
            print(f"{cat:36s} {'yes' if t['claimed'] else 'no':>6s} {t['right']:6d} {t['wrong']:6d} {t['crashed']:6d}"
                  f"   {t['wrong_stopped']}/{t['wrong']} stopped, {t['right_blocked']} blocked")
            for ex in t["examples"]:
                print(f"      {ex[:230]}")
    out = ROOT / "runs" / "hit_replay" / "entry_accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\ndetails: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
