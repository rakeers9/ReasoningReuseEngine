"""Generator for the HIDDEN chatbot-cache test set (eval/hidden/chat_hidden.jsonl).

    .venv/bin/python eval/hidden/build_hidden.py          # rewrites chat_hidden.jsonl (free, no API)

The set is for measuring the chat loop, never for tuning it. Do not read the questions while
tuning prompts, gates or the builder. The schema is the one eval/chat_loop.py script() uses
(load it with chat_loop.load_messages):

    kind: variant | near_miss | one_off | chitchat      family: task id or null
    question, answer (display form), answer_value (float | null), answer_unit
    should_hit, acceptable_either, category (near-miss category | null)

Contents (fixed seed, shuffled):
  - 4 recurring task families, 20 variants each, none in eval/corpus or eval/labelfree:
      calorimetry_final_temperature   physics     final temperature when hot metal meets water
      fv_monthly_deposits             finance     future value of month-end deposits
      sample_size_mean_ci             statistics  sample size for a mean at a given margin
      solution_prep_mass              chemistry   mass of solid for a solution of given molarity
    Wording varies (full sentences, terse notes, background chatter, number words, "$150k" /
    "$2.5 million" shorthand) and so do units (g/kg/mL, degC/K, years/months, mol/L vs mmol/L,
    mL/L/cm3, g vs kg vs minutes vs hours).
  - 5 to 6 near-misses per family: same topic and style, different task.
  - 8 one-off calculations from other domains, and 6 chit-chat messages.

Every answer comes from the plain-Python solvers below, never a model. Nothing is rounded until
the end: answer_value is the full-precision float, except money, which is the value to the cent
(the final answer, rounded once, half up) and counts, which are exact integers. The display
form follows arc/present.ROUNDING_POLICY: money to the cent, counts exact, else 4 significant
figures. Values that depend on a rounded z-score are only generated when every common
rounding of z (1.96 vs 1.959964, 2.58 vs 2.5758, ...) gives the same answer.
"""

from __future__ import annotations

import json
import math
import random
from decimal import ROUND_HALF_UP, Decimal as D
from pathlib import Path
from statistics import NormalDist

SEED = 20261116
OUT = Path(__file__).with_name("chat_hidden.jsonl")
PER_FAMILY = 20


# ----------------------------------------------------------------------------- formatting

def clean(v):
    """Drop binary float noise (51.345000000000006 -> 51.345) at 12 significant digits, far
    below any precision a question states. Applied to the final answer only."""
    return v if v is None else float(f"{float(v):.12g}")


def _d(x) -> D:
    if isinstance(x, D):
        return x
    return D(repr(clean(x) if isinstance(x, float) else x))


def fd(x) -> str:
    """Plain decimal text without trailing zeros."""
    return format(_d(x).normalize(), "f")


def sig(v, n: int = 4) -> str:
    """n significant figures, half up, with thousands separators."""
    d = _d(v)
    if d == 0:
        return "0"
    r = d.quantize(D(1).scaleb(d.adjusted() - n + 1), rounding=ROUND_HALF_UP)
    return format(r, ",f")


def cents(v) -> D:
    return _d(v).quantize(D("0.01"), rounding=ROUND_HALF_UP)


def money(v) -> str:
    return f"${cents(v):,.2f}"


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
        80: "eighty", 90: "ninety"}


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n % 10 == 0:
        return TENS[n]
    return TENS[n // 10 * 10] + "-" + ONES[n % 10]


def msg(kind, family, question, answer, value, unit, should_hit, either=False, category=None) -> dict:
    return {"kind": kind, "family": family, "question": question, "answer": answer,
            "answer_value": clean(value), "answer_unit": unit, "should_hit": should_hit,
            "acceptable_either": either, "category": category}


def near(question, answer, value, unit, category, either=False) -> dict:
    return {"question": question, "answer": answer, "value": value, "unit": unit,
            "category": category, "either": either}


def cannot(what: str) -> str:
    return f"Cannot be determined: {what} is not given."


# ============================================================================ family 1: physics
CAL_ID = "calorimetry_final_temperature"
METALS = [("copper", "Cu", "0.385"), ("aluminium", "Al", "0.897"), ("aluminum", "Al", "0.897"),
          ("iron", "Fe", "0.449"), ("lead", "Pb", "0.128"), ("brass", "brass", "0.380"),
          ("zinc", "Zn", "0.388"), ("tin", "Sn", "0.227"), ("nickel", "Ni", "0.444"),
          ("silver", "Ag", "0.235"), ("titanium", "Ti", "0.523")]
CW_CHOICES = ["4.184", "4.18", "4.186"]            # J/(g·°C), always stated in the question
BACKGROUND = ["It was the third lab session of the week and the thermometer kept sticking.",
              "We only had 40 minutes left before the lab closed.",
              "This is trial 4 of 9 for the report.",
              "The hot plate in bay 6 had been running since the morning.",
              "My lab partner insists the cup is from a pack of 50, so ignore its size."]
LATENT_FUSION = "334"                              # J/g, stated in the ice question


def mix_final(m_m, c_m, t_m, m_w, c_w, t_w, c_cal=0.0) -> float:
    """Equilibrium temperature of a hot metal in water (no heat lost), optional calorimeter."""
    return (m_m * c_m * t_m + (m_w * c_w + c_cal) * t_w) / (m_m * c_m + m_w * c_w + c_cal)


def _mass(g: int, style: str) -> str:
    return f"{fd(g)} g" if style == "g" else f"{fd(D(g) / 1000)} kg"


def _temp(tc: float, kelvin: bool) -> str:
    return f"{_d(tc) + D('273.15'):.2f} K" if kelvin else f"{fd(tc)} °C"


def _cspec(c: str, style: str) -> str:
    return {"gC": f"{c} J/(g·°C)", "gK": f"{c} J/(g·K)", "kJ": f"{c} kJ/(kg·K)",
            "kg": f"{fd(D(c) * 1000)} J/(kg·K)"}[style]


def cal_scenario(rng: random.Random, allow_kelvin: bool = True) -> dict:
    while True:
        name, sym, cm = rng.choice(METALS)
        m_m = rng.randrange(50, 405, 5)
        m_w = rng.randrange(100, 505, 10)
        t_m = rng.randrange(70, 255, 5)
        t_w = rng.randrange(20, 80) / 2
        cw = rng.choice(CW_CHOICES)
        tf = mix_final(m_m, float(cm), t_m, m_w, float(cw), t_w)
        if tf < 90 and tf - t_w >= 3:
            break
    kelvin = allow_kelvin and rng.random() < 0.2
    mstyle, wstyle = rng.choice(["g", "kg"]), rng.choice(["g", "kg", "mL"])
    cstyle = rng.choice(["gC", "gC", "gK", "kJ", "kg"])
    core = {"g": f"{m_w} g", "kg": f"{fd(D(m_w) / 1000)} kg", "mL": f"{m_w} mL (density 1.00 g/mL)"}[wstyle]
    return {"name": name, "sym": sym, "m_s": _mass(m_m, mstyle), "w_core": core,
            "w_s": f"{core} of water", "Tm_s": _temp(t_m, kelvin), "Tw_s": _temp(t_w, kelvin),
            "cm_s": _cspec(cm, cstyle), "cw_s": _cspec(cw, cstyle), "bg": rng.choice(BACKGROUND),
            "kelvin": kelvin, "m_m": m_m, "cm": float(cm), "t_m": t_m, "m_w": m_w, "cw": float(cw),
            "t_w": t_w, "tf": tf}


CAL_TEMPLATES = [
    "A {m_s} piece of {name} at {Tm_s} is dropped into {w_s} at {Tw_s} in an insulated container. The specific heat of {name} is {cm_s} and that of water is {cw_s}. What is the final equilibrium temperature of the water and metal?",
    "I heated a {m_s} {name} sample to {Tm_s} and quickly moved it into {w_s} initially at {Tw_s} inside a well-insulated cup (no heat lost to the surroundings). Taking c({name}) = {cm_s} and c(water) = {cw_s}, what temperature will the water reach once they equilibrate?",
    "{sym} block {m_s} @ {Tm_s} -> {w_s} @ {Tw_s}, insulated, no losses. c_{sym} = {cm_s}, c_water = {cw_s}. T_final?",
    "{bg} Anyway: a {m_s} {name} cylinder, preheated to {Tm_s}, goes into {w_s} at {Tw_s} in a foam-cup calorimeter that loses no heat. Given specific heats of {cm_s} for {name} and {cw_s} for water, find the final temperature.",
    "Find the equilibrium temperature when {w_s} at {Tw_s} is mixed with a hot {m_s} {name} object at {Tm_s}. Specific heats: {name} {cm_s}, water {cw_s}. Assume the system is isolated.",
    "What final temperature results if a {name} ingot ({m_s}, {Tm_s}) is quenched in {w_s} ({Tw_s})? Use {cm_s} for {name} and {cw_s} for water, and assume a perfectly insulated system.",
    "calorimetry q: water {w_core} {Tw_s}; {name} {m_s} {Tm_s}; c_metal {cm_s}; c_water {cw_s}; thermally isolated; final temperature of the mixture?",
    "A student is preparing a quench test. They plan to drop a hot {name} part with mass {m_s}, currently at {Tm_s}, into a bath holding {w_s} at {Tw_s}. With {name}'s specific heat at {cm_s} and water's at {cw_s}, and no heat escaping, what will the final temperature be?",
]


def cal_answer(s: dict) -> tuple[float, str]:
    return (s["tf"] + 273.15, "K") if s["kelvin"] else (s["tf"], "°C")


def cal_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = cal_scenario(rng)
    v, unit = cal_answer(s)
    return CAL_TEMPLATES[tpl].format(**s), v, unit, f"{sig(v)} {unit}"


def cal_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = cal_scenario(rng, allow_kelvin=False)                       # heat absorbed by the water
    q = rng.choice([
        "A {m_s} piece of {name} at {Tm_s} is dropped into {w_s} at {Tw_s} in an insulated container. The specific heat of {name} is {cm_s} and that of water is {cw_s}. How much heat does the water absorb, in kilojoules?",
        "{bg} A {m_s} {name} sample at {Tm_s} goes into {w_s} at {Tw_s} (insulated, no heat lost). With c({name}) = {cm_s} and c(water) = {cw_s}, how many kJ of heat flow into the water by the time they equilibrate?",
    ]).format(**s)
    kj = s["m_w"] * s["cw"] * (s["tf"] - s["t_w"]) / 1000
    out.append(near(q, f"{sig(kj)} kJ", kj, "kJ", "different_target"))

    s = cal_scenario(rng, allow_kelvin=False)                       # the cup absorbs heat too
    ccal = rng.choice([20, 25, 30, 40, 50, 60])
    q = rng.choice([
        "A {m_s} piece of {name} at {Tm_s} is dropped into {w_s} at {Tw_s} in a calorimeter whose cup absorbs heat, with a heat capacity of {ccal} J/°C (the cup starts at the water's temperature). The specific heat of {name} is {cm_s} and that of water is {cw_s}. What is the final equilibrium temperature?",
        "{sym} block {m_s} @ {Tm_s} -> {w_s} @ {Tw_s}. The calorimeter cup is not negligible: C_cup = {ccal} J/°C, starting at {Tw_s}. c_{sym} = {cm_s}, c_water = {cw_s}. T_final?",
    ]).format(ccal=ccal, **s)
    tf = mix_final(s["m_m"], s["cm"], s["t_m"], s["m_w"], s["cw"], s["t_w"], c_cal=ccal)
    out.append(near(q, f"{sig(tf)} °C", tf, "°C", "extra_condition"))

    while True:                                                      # hot metal into ice: grams melted
        name, sym, cm = rng.choice(METALS)
        m_m, t_m, ice = rng.randrange(100, 410, 10), rng.randrange(100, 260, 10), rng.randrange(150, 410, 10)
        melted = m_m * float(cm) * t_m / float(LATENT_FUSION)
        if melted < 0.9 * ice:
            break
    q = rng.choice([
        f"A {m_m} g piece of {name} at {t_m} °C is dropped into {ice} g of ice at 0 °C in an insulated container. The specific heat of {name} is {cm} J/(g·°C) and the latent heat of fusion of ice is {LATENT_FUSION} J/g. How many grams of ice melt?",
        f"{sym} block {m_m} g @ {t_m} °C -> {ice} g ice @ 0 °C, insulated. c_{sym} = {cm} J/(g·°C), L_fusion = {LATENT_FUSION} J/g. Mass of ice melted?",
    ])
    out.append(near(q, f"{sig(melted)} g", melted, "g", "regime_change"))

    s = cal_scenario(rng, allow_kelvin=False)                       # specific heat of the metal missing
    q = rng.choice([
        "A {m_s} piece of an unidentified metal at {Tm_s} is dropped into {w_s} at {Tw_s} in an insulated container. The specific heat of water is {cw_s}. What is the final equilibrium temperature?",
        "{bg} A {m_s} chunk of some unknown metal, heated to {Tm_s}, goes into {w_s} at {Tw_s} in an insulated cup. c(water) = {cw_s}. What final temperature will they reach?",
    ]).format(**s)
    out.append(near(q, cannot("the specific heat of the metal"), None, "", "missing_given"))

    s = cal_scenario(rng, allow_kelvin=False)                       # inverse: needed initial metal temperature
    target = round(s["tf"])
    if target - s["t_w"] < 3:
        target = round(s["t_w"]) + 6
    need = (target * (s["m_m"] * s["cm"] + s["m_w"] * s["cw"]) - s["m_w"] * s["cw"] * s["t_w"]) / (s["m_m"] * s["cm"])
    q = rng.choice([
        "A {m_s} piece of {name} is heated and then dropped into {w_s} at {Tw_s} in an insulated container. The specific heat of {name} is {cm_s} and that of water is {cw_s}. To what temperature must the {name} be heated so that the final temperature of the water is {tgt} °C?",
        "{sym} block {m_s}, preheated to an unknown temperature, -> {w_s} @ {Tw_s}, insulated. c_{sym} = {cm_s}, c_water = {cw_s}. The mixture ends at {tgt} °C. What was the block's initial temperature?",
    ]).format(tgt=target, **s)
    out.append(near(q, f"{sig(need)} °C", need, "°C", "inverse_problem"))

    s = cal_scenario(rng, allow_kelvin=False)                       # harmless extra detail
    q = rng.choice([
        "A {m_s} piece of {name} at {Tm_s} is dropped into {w_s} at {Tw_s} in an insulated container with a plastic lid of negligible heat capacity. The specific heat of {name} is {cm_s} and that of water is {cw_s}. What is the final equilibrium temperature?",
        "{bg} A {m_s} {name} sample, heated to {Tm_s}, is moved into {w_s} at {Tw_s} in a foam cup (its heat capacity is negligible). With c({name}) = {cm_s} and c(water) = {cw_s}, what temperature will the water reach?",
    ]).format(**s)
    out.append(near(q, f"{sig(s['tf'])} °C", s["tf"], "°C", "benign_extra_info", either=True))
    return out


# ============================================================================ family 2: finance
FV_ID = "fv_monthly_deposits"
FV_P0 = [0, 0, 5000, 12000, 20000, 35000, 50000, 75000, 120000, 150000, 250000, 300000]
FV_D = [150, 200, 250, 300, 400, 450, 500, 600, 750, 800, 1000, 1200, 1500, 2000]
FV_R = [3.0, 3.5, 4.0, 4.25, 4.5, 5.0, 5.25, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
FV_N = [60, 72, 84, 90, 96, 120, 126, 144, 150, 180, 216, 240, 300, 360, 420]
AMT_WORDS = {200: "two hundred", 250: "two hundred fifty", 300: "three hundred", 400: "four hundred",
             500: "five hundred", 600: "six hundred", 750: "seven hundred fifty",
             800: "eight hundred", 1000: "one thousand", 1200: "twelve hundred",
             1500: "fifteen hundred", 2000: "two thousand"}


def fv_ordinary(p0, d, r_pct, n, due: bool = False) -> float:
    """Balance after n months: opening balance p0, deposit d each month, nominal annual rate
    r_pct compounded monthly. due=True puts each deposit at the start of its month."""
    i = r_pct / 1200
    g = (1 + i) ** n
    annuity = d * (g - 1) / i
    return p0 * g + (annuity * (1 + i) if due else annuity)


def fv_two_rates(p0, d, r1, n1, r2, n2) -> float:
    return fv_ordinary(fv_ordinary(p0, d, r1, n1), d, r2, n2)


def required_deposit(target, p0, r_pct, n) -> float:
    i = r_pct / 1200
    g = (1 + i) ** n
    return (target - p0 * g) * i / (g - 1)


def months_to_exceed(target, p0, d, r_pct, limit: int = 2000) -> tuple[int, float]:
    """First month-end on which the balance is strictly above target, and how far above."""
    i, bal = r_pct / 1200, float(p0)
    for k in range(1, limit + 1):
        bal = bal * (1 + i) + d
        if bal > target:
            return k, bal - target
    raise ValueError("target not reached")


def amount_text(x: int, rng: random.Random, terse: bool = False) -> str:
    opts = [f"${x:,}"]
    if x >= 1000 and x % 1000 == 0:
        opts.append(f"${x // 1000}k")
    if 1000 <= x < 10000 and x % 100 == 0:
        opts.append(f"${fd(D(x) / 1000)}k")
    if x >= 100000 and x % 10000 == 0:
        opts.append(f"${fd(D(x) / 1000000)} million")
    if not terse:
        opts.append(f"{x:,} dollars")
        if x >= 1_000_000 and x % 100000 == 0:
            opts.append(f"{fd(D(x) / 1000000)} million dollars")
        if x in AMT_WORDS:
            opts.append(f"{AMT_WORDS[x]} dollars")
    return rng.choice(opts)


def rate_text(r: float, rng: random.Random) -> str:
    opts = [f"{fd(r)}%", f"{fd(r)} percent"]
    whole, frac = int(r), round(r - int(r), 2)
    tail = {0.0: "", 0.5: " and a half", 0.25: " and a quarter", 0.75: " and three quarters"}.get(frac)
    if tail is not None and whole <= 20:
        opts.append(f"{words(whole)}{tail} percent")
    return rng.choice(opts)


def term_text(n: int, rng: random.Random, terse: bool = False) -> str:
    opts = [f"{n} months"]
    if n % 12 == 0:
        opts += [f"{n // 12} years", f"{n // 12} years"]
        if not terse:
            opts.append(f"{words(n // 12)} years")
    elif n % 6 == 0:
        opts.append(f"{fd(D(n) / 12)} years")
    return rng.choice(opts)


def fv_scenario(rng: random.Random) -> dict:
    p0, d, r, n = rng.choice(FV_P0), rng.choice(FV_D), rng.choice(FV_R), rng.choice(FV_N)
    s = {"p0": p0, "d_n": d, "r_n": r, "n": n}
    s["d"] = amount_text(d, rng)
    s["d_terse"] = amount_text(d, rng, terse=True)
    s["r"] = rate_text(r, rng)
    s["t"] = term_text(n, rng)
    s["t_terse"] = term_text(n, rng, terse=True)
    if p0:
        p = amount_text(p0, rng)
        s["init"] = rng.choice([f" The account opens with a balance of {p}.", f" It already holds {p}.",
                                f" I'm starting from a balance of {p}."])
        s["init_terse"] = f" Start: {amount_text(p0, rng, terse=True)}."
        s["init2"] = f" on top of the {p} already in the account,"
        s["p"] = p
    else:
        s["init"] = s["init_terse"] = s["init2"] = s["p"] = ""
    return s


FV_TEMPLATES = [
    "I put {d} into a savings account at the end of each month. The account earns {r} a year, compounded monthly.{init} What will the balance be after {t}?",
    "Retirement planning: {d} goes in every month-end, return is {r} APR compounded monthly.{init} Balance after {t}?",
    "Plan: {d_terse}/mo (month-end deposits), {r} APR, monthly comp, {t_terse}.{init_terse} FV?",
    "My employer's matching plan means I invest {d} at the close of every month. Assuming an average of {r} per year compounded monthly,{init2} what is the future value after {t}? (Unrelated, but the plan paperwork is 14 pages.)",
    "How large will a fund grow in {t} if it earns {r} nominal annual interest, compounded monthly, and receives a deposit of {d} at the end of each month?{init}",
    "Kid's college fund: we add {d} at the end of every month, interest is {r} per year with monthly compounding.{init} What's the total accumulated after {t}?",
    "Future value question. Monthly deposit: {d_terse} (paid at month end). Interest: {r} compounded monthly. Term: {t_terse}.{init_terse} Calculate the final balance.",
    "Quick one for my budget spreadsheet: an account compounding monthly at {r} APR, ordinary monthly contributions of {d}.{init} Where does it stand at the {t} mark?",
]


def fv_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = fv_scenario(rng)
    v = float(cents(fv_ordinary(s["p0"], s["d_n"], s["r_n"], s["n"])))
    return FV_TEMPLATES[tpl].format(**s), v, "$", money(v)


def _nice_target(x: float) -> int:
    step = 10 ** max(int(math.floor(math.log10(x))) - 1, 3)
    return int(math.ceil(x / step) * step)


def fv_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = fv_scenario(rng)                                             # deposits at the START of the month
    q = rng.choice([
        "I put {d} into a savings account at the start of each month (the first deposit goes in today). The account earns {r} a year, compounded monthly.{init} What will the balance be after {t}?",
        "Plan: {d_terse}/mo deposited at the beginning of each month, {r} APR, monthly comp, {t_terse}.{init_terse} FV?",
    ]).format(**s)
    v = float(cents(fv_ordinary(s["p0"], s["d_n"], s["r_n"], s["n"], due=True)))
    out.append(near(q, money(v), v, "$", "timing_change"))

    s = fv_scenario(rng)                                             # deposit needed for a target
    base = fv_ordinary(s["p0"], s["d_n"], s["r_n"], s["n"])
    target = _nice_target(base * 1.15)
    d_req = float(cents(required_deposit(target, s["p0"], s["r_n"], s["n"])))
    tt = amount_text(target, rng)
    q = rng.choice([
        "I want {tt} in an account after {t}. It earns {r} a year, compounded monthly, and I deposit the same amount at the end of every month.{init} How much must each monthly deposit be?",
        "Savings goal {tt} in {t_terse}. Account: {r} APR, monthly comp, month-end deposits.{init_terse} Required monthly deposit?",
    ]).format(tt=tt, **s)
    out.append(near(q, money(d_req), d_req, "$", "inverse_problem"))

    s = fv_scenario(rng)                                             # the interest rate is missing
    q = rng.choice([
        "I put {d} into a savings account at the end of each month, at whatever rate the bank is currently paying, compounded monthly.{init} What will the balance be after {t}?",
        "Plan: {d_terse}/mo (month-end deposits), monthly comp, {t_terse}, rate not announced yet.{init_terse} FV?",
    ]).format(**s)
    out.append(near(q, cannot("the interest rate"), None, "", "missing_given"))

    s = fv_scenario(rng)                                             # the rate changes part-way
    while True:
        r2 = rng.choice(FV_R)
        if r2 != s["r_n"]:
            break
    n1 = rng.choice(range(12, s["n"], 12))                           # first leg, in whole years
    v = float(cents(fv_two_rates(s["p0"], s["d_n"], s["r_n"], n1, r2, s["n"] - n1)))
    q = rng.choice([
        "I put {d} into a savings account at the end of each month. The account earns {r} a year, compounded monthly, for the first {n1} months and then {r2} a year, compounded monthly, for the rest of the {t}.{init} What will the balance be at the end?",
        "Plan: {d_terse}/mo (month-end deposits), {r} APR for {n1} months, then {r2} APR (monthly comp), {t_terse} in all.{init_terse} FV?",
    ]).format(n1=n1, r2=rate_text(r2, rng), **s)
    out.append(near(q, money(v), v, "$", "extra_condition"))

    while True:                                                      # how long until a bound is passed
        s = fv_scenario(rng)
        k = rng.randrange(24, 200)
        target = _nice_target(fv_ordinary(s["p0"], s["d_n"], s["r_n"], k))
        months, margin = months_to_exceed(target, s["p0"], s["d_n"], s["r_n"])
        if margin > 0.01 and months < 600:
            break
    tt = amount_text(target, rng)
    q = rng.choice([
        "I put {d} into a savings account at the end of each month. The account earns {r} a year, compounded monthly.{init} After how many months will the balance first exceed {tt}?",
        "Plan: {d_terse}/mo (month-end deposits), {r} APR, monthly comp.{init_terse} In which month does the balance first go above {tt}?",
    ]).format(tt=tt, **s)
    out.append(near(q, str(months), float(months), "months", "boundary_change"))

    s = fv_scenario(rng)                                             # harmless extra detail
    q = rng.choice([
        "I put {d} into a savings account at the end of each month. The account earns {r} a year, compounded monthly, with no fees and no taxes.{init} What will the balance be after {t}?",
        "{d} goes in every month-end, return is {r} APR compounded monthly (the bank charges nothing).{init} Balance after {t}?",
    ]).format(**s)
    v = float(cents(fv_ordinary(s["p0"], s["d_n"], s["r_n"], s["n"])))
    out.append(near(q, money(v), v, "$", "benign_extra_info", either=True))
    return out


# ============================================================================ family 3: statistics
ST_ID = "sample_size_mean_ci"
DOMAINS = [
    dict(qty="battery lifetime", unit="hours", sig=["2", "2.5", "3", "4", "6"], E=["0.2", "0.25", "0.5", "1"], alt=None,
         prop="proportion of batteries that fail within the first year"),
    dict(qty="fill weight of cereal boxes", unit="g", sig=["4", "5", "6", "8", "12"], E=["0.5", "1", "1.5", "2"],
         alt=("kg", 1000), prop="proportion of boxes that are underfilled"),
    dict(qty="daily commute time", unit="minutes", sig=["6", "8", "10", "12", "15"], E=["1.5", "3", "6", "9"],
         alt=("hours", 60), prop="proportion of commuters who arrive late"),
    dict(qty="shaft diameter", unit="mm", sig=["0.04", "0.05", "0.08", "0.12"], E=["0.01", "0.02", "0.025", "0.04"],
         alt=("cm", 10), prop="proportion of shafts outside tolerance"),
    dict(qty="systolic blood pressure", unit="mmHg", sig=["12", "14", "15", "18"], E=["2", "3", "4", "5"], alt=None,
         prop="proportion of patients with hypertension"),
    dict(qty="resting heart rate", unit="bpm", sig=["8", "9", "10", "12"], E=["1", "1.5", "2", "2.5"], alt=None,
         prop="proportion of athletes below 50 bpm"),
    dict(qty="volume dispensed by a filling machine", unit="mL", sig=["1.2", "2", "3", "5"], E=["0.25", "0.5", "1"],
         alt=("L", 1000), prop="proportion of bottles outside the fill limits"),
    dict(qty="monthly rainfall", unit="mm", sig=["20", "25", "30", "40"], E=["3", "5", "8", "10"], alt=("cm", 10),
         prop="proportion of months with flooding"),
]
CONF = [90, 95, 98, 99]


def z_two(cl: int) -> float:
    return NormalDist().inv_cdf(1 - (1 - cl / 100) / 2)


def z_one(cl: int) -> float:
    return NormalDist().inv_cdf(cl / 100)


def z_candidates(z0: float) -> list[float]:
    """The exact z and the roundings people actually use (3 decimals; 2 when that is close)."""
    z3, z2 = round(z0, 3), round(z0, 2)
    out = [z0, z3]
    if abs(z2 - z3) <= 0.0045:
        out.append(z2)
    return out


def n_mean(z, sigma, e) -> float:
    return (z * sigma / e) ** 2


def robust_ceil(values: list[float]) -> int | None:
    """ceil of every candidate value if they all agree and none sits on an integer boundary."""
    if any(abs(v - round(v)) < 1e-6 for v in values):
        return None
    cs = {math.ceil(v) for v in values}
    return cs.pop() if len(cs) == 1 else None


def n_mean_ceil(sigma: float, e: float, z0: float) -> int | None:
    return robust_ceil([n_mean(z, sigma, e) for z in z_candidates(z0)])


def n_finite(sigma: float, e: float, z0: float, pop: int) -> int | None:
    vals = []
    for z in z_candidates(z0):
        n0 = n_mean(z, sigma, e)
        vals += [n0 / (1 + (n0 - 1) / pop), math.ceil(n0) / (1 + (math.ceil(n0) - 1) / pop)]
    return robust_ceil(vals)


def n_proportion(e_frac: float, z0: float, p: float = 0.5) -> int | None:
    return robust_ceil([z * z * p * (1 - p) / e_frac ** 2 for z in z_candidates(z0)])


def st_scenario(rng: random.Random, kind: str = "two") -> dict:
    while True:
        dom = rng.choice(DOMAINS)
        sg, e = rng.choice(dom["sig"]), rng.choice(dom["E"])
        cl = rng.choice(CONF)
        z0 = z_two(cl) if kind == "two" else z_one(cl)
        n = n_mean_ceil(float(sg), float(e), z0)
        if n is not None and 20 <= n <= 2500:
            break
    e_s = f"{e} {dom['unit']}"
    if dom["alt"] and rng.random() < 0.35:
        alt_unit, factor = dom["alt"]
        e_alt = D(e) / factor
        if e_alt * factor == D(e):
            e_s = f"{fd(e_alt)} {alt_unit}"
    cl_s = rng.choice([f"{cl}%", f"{cl} percent", f"{words(cl)} percent"])
    z_stated = rng.random() < 0.4
    z_clause = f" Use z = {fd(round(z0, 3))}." if z_stated else ""
    return {"dom": dom, "qty": dom["qty"], "unit": dom["unit"], "sg": float(sg), "e": float(e), "cl": cl,
            "z0": z0, "z_used": round(z0, 3) if z_stated else z0, "n": n, "sigma_s": f"{sg} {dom['unit']}",
            "E_s": e_s, "var_s": f"{fd(D(sg) ** 2)} {dom['unit']}²", "cl_s": cl_s, "zc": z_clause, "prop": dom["prop"]}


ST_TEMPLATES = [
    "A researcher wants to estimate the mean {qty} and construct a {cl_s} confidence interval with a margin of error of at most {E_s}. From past studies the population standard deviation is {sigma_s}. What is the minimum sample size required?{zc}",
    "How many observations do I need to estimate the average {qty} to within ±{E_s} at {cl_s} confidence? Population SD (known): {sigma_s}.{zc}",
    "n for mean CI: {cl_s} conf, σ = {sigma_s}, E = {E_s}. Minimum n?{zc}",
    "Our quality team (there are 6 of us, plus the intern) needs to pin down the average {qty}. Historically σ is {sigma_s}. If we want a {cl_s} confidence interval whose half-width is no more than {E_s}, how large must the sample be?{zc}",
    "Sample size question: the {qty} has a known population variance of {var_s}. To get a {cl_s} confidence interval for the mean no wider than ±{E_s}, what's the smallest n?{zc}",
    "I'd like a {cl_s} confidence interval for the mean {qty} that's accurate to within {E_s}. The population standard deviation is {sigma_s}. Smallest sample that guarantees this?{zc}",
    "Planning a survey on {qty}: known σ = {sigma_s}; want the {cl_s} CI half-width ≤ {E_s}. Required number of samples?{zc}",
    "For estimating a population mean with known standard deviation {sigma_s} for {qty}, find the sample size needed for a {cl_s} confidence level and a maximum error of {E_s}.{zc}",
]


def st_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = st_scenario(rng)
    return ST_TEMPLATES[tpl].format(**s), float(s["n"]), "", str(s["n"])


def st_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = st_scenario(rng)                                             # margin of error for a given n
    n_given = rng.choice([25, 36, 40, 50, 64, 80, 100, 120, 150, 200, 250, 400])
    z = round(s["z0"], 3)
    q = rng.choice([
        "A researcher has already collected {n_obs} measurements of {qty}. The population standard deviation is {sigma_s}. What is the margin of error of a {cl_s} confidence interval for the mean? Use z = {zs}.",
        "{n_obs} observations of {qty}, known σ = {sigma_s}. Half-width of the {cl_s} CI for the mean? (z = {zs})",
    ]).format(n_obs=n_given, zs=fd(z), **s)
    e = z * s["sg"] / math.sqrt(n_given)
    out.append(near(q, f"{sig(e)} {s['unit']}", e, s["unit"], "different_target"))

    while True:                                                      # a proportion, not a mean
        s = st_scenario(rng)
        ep = rng.choice(["2", "2.5", "3", "4", "5"])
        n = n_proportion(float(ep) / 100, s["z0"])
        if n is not None:
            break
    q = rng.choice([
        "A researcher wants to estimate the {prop} and construct a {cl_s} confidence interval with a margin of error of at most {ep} percentage points. There is no prior estimate of the proportion, so use the conservative value 0.5. What is the minimum sample size required?",
        "How many observations do I need to estimate the {prop} to within ±{ep} percentage points at {cl_s} confidence? No prior estimate, so take p = 0.5.",
    ]).format(ep=ep, **s)
    out.append(near(q, str(n), float(n), "", "different_method"))

    s = st_scenario(rng)                                             # no standard deviation given
    q = rng.choice([
        "A researcher wants to estimate the mean {qty} and construct a {cl_s} confidence interval with a margin of error of at most {E_s}. The population standard deviation is not known and there is no pilot data. What is the minimum sample size required?",
        "How many observations do I need to estimate the average {qty} to within ±{E_s} at {cl_s} confidence? (Nobody has any idea of the spread yet.)",
    ]).format(**s)
    out.append(near(q, cannot("the population standard deviation"), None, "", "missing_given"))

    while True:                                                      # finite population correction
        s = st_scenario(rng)
        pop = rng.choice([500, 800, 1000, 1500, 2000, 5000])
        n = n_finite(s["sg"], s["e"], s["z0"], pop)
        if n is not None and s["n"] < pop / 1.2:
            break
    q = rng.choice([
        "A researcher wants to estimate the mean {qty} of a finite batch of N = {pop:,} items and construct a {cl_s} confidence interval with a margin of error of at most {E_s}. The population standard deviation is {sigma_s}. Apply the finite population correction. What is the minimum sample size required?",
        "Only N = {pop:,} units exist. Estimate the mean {qty}: {cl_s} CI, E = {E_s}, σ = {sigma_s}. Use the finite population correction. Minimum n?",
    ]).format(pop=pop, **s)
    out.append(near(q, str(n), float(n), "", "extra_condition"))

    while True:                                                      # one-sided bound
        s = st_scenario(rng, kind="one")
        if s["cl"] != 90:
            break
    q = rng.choice([
        "A researcher wants a {cl_s} one-sided upper confidence bound for the mean {qty} that lies no more than {E_s} above the sample mean. The population standard deviation is {sigma_s}. What is the minimum sample size required?",
        "One-sided {cl_s} upper bound on mean {qty}, allowed error {E_s}, σ = {sigma_s}. Minimum n?",
    ]).format(**s)
    out.append(near(q, str(s["n"]), float(s["n"]), "", "different_rule_branch"))

    s = st_scenario(rng)                                             # harmless extra detail
    q = rng.choice([
        "A researcher wants to estimate the mean {qty} and construct a {cl_s} confidence interval with a margin of error of at most {E_s}. From past studies the population standard deviation is {sigma_s}, and the population is approximately normally distributed. What is the minimum sample size required?{zc}",
        "How many observations do I need to estimate the average {qty} to within ±{E_s} at {cl_s} confidence if the data are roughly bell-shaped? Population SD (known): {sigma_s}.{zc}",
    ]).format(**s)
    out.append(near(q, str(s["n"]), float(s["n"]), "", "benign_extra_info", either=True))
    return out


# ============================================================================ family 4: chemistry
CH_ID = "solution_prep_mass"
COMPOUNDS = [("sodium chloride", "NaCl", "58.44"), ("potassium chloride", "KCl", "74.55"),
             ("sodium hydroxide", "NaOH", "40.00"), ("calcium chloride", "CaCl2", "110.98"),
             ("glucose", "C6H12O6", "180.16"), ("sodium carbonate", "Na2CO3", "105.99"),
             ("potassium nitrate", "KNO3", "101.10"), ("magnesium sulfate", "MgSO4", "120.37"),
             ("sodium bicarbonate", "NaHCO3", "84.01"), ("copper(II) sulfate", "CuSO4", "159.61"),
             ("ammonium chloride", "NH4Cl", "53.49"), ("potassium permanganate", "KMnO4", "158.03"),
             ("sucrose", "C12H22O11", "342.30"), ("sodium sulfate", "Na2SO4", "142.04")]
CONCS = ["0.0100", "0.0250", "0.0500", "0.0750", "0.100", "0.150", "0.200", "0.250", "0.500", "0.750",
         "1.00", "1.50", "2.00", "3.00"]
VOLS = [50, 100, 125, 150, 200, 250, 300, 500, 750, 1000, 1500, 2000, 2500]            # mL
VOL_WORDS = {250: "a quarter of a litre", 500: "half a litre", 1000: "one litre", 2000: "two litres"}


def mass_needed(conc_mol_l, vol_ml, molar_mass) -> float:
    """Grams of solute for vol_ml of a conc_mol_l solution (molar_mass in g/mol)."""
    return conc_mol_l * (vol_ml / 1000) * molar_mass


def molarity_from_mass(mass_g, vol_ml, molar_mass) -> float:
    return mass_g / molar_mass / (vol_ml / 1000)


def stock_volume_ml(c_stock, c_final, v_final_ml) -> float:
    return c_final * v_final_ml / c_stock


def conc_text(c: str, rng: random.Random) -> str:
    opts = [f"{c} M", f"{c} mol/L", f"{c} molar", f"{c} mol dm⁻³"]
    if D(c) < 1:
        mm = fd(D(c) * 1000)
        opts += [f"{mm} mmol/L", f"{mm} mM"]
    return rng.choice(opts)


def vol_text(v: int, rng: random.Random) -> str:
    opts = [f"{v} mL", f"{v} mL", f"{v} cm³", f"{fd(D(v) / 1000)} L"]
    if v % 1000 == 0:
        opts.append(f"{v // 1000} dm³")
    if v in VOL_WORDS:
        opts.append(VOL_WORDS[v])
    return rng.choice(opts)


def ch_scenario(rng: random.Random) -> dict:
    name, formula, mm = rng.choice(COMPOUNDS)
    c, v = rng.choice(CONCS), rng.choice(VOLS)
    grams = mass_needed(float(c), v, float(mm))
    milli = grams < 1 or rng.random() < 0.1
    value, unit = (grams * 1000, "mg") if milli else (grams, "g")
    return {"name": name, "formula": formula, "M": mm, "c": c, "v": v, "C_s": conc_text(c, rng),
            "V_s": vol_text(v, rng), "grams": grams, "value": value, "unit": unit,
            "mass_q": "How many milligrams" if milli else "How many grams",
            "hint": " Give the answer in milligrams." if milli else rng.choice(["", "", " Give the answer in grams."]),
            "hint_terse": " (in mg)" if milli else rng.choice(["", " (in g)"])}


CH_TEMPLATES = [
    "{mass_q} of {name} ({formula}, molar mass {M} g/mol) are needed to make {V_s} of a {C_s} solution?",
    "I need {V_s} of {C_s} {formula} solution. With a molar mass of {M} g/mol, what mass of solid should I weigh out?{hint}",
    "prep: {C_s} {formula}, {V_s} final volume, M = {M} g/mol. mass of solute?{hint_terse}",
    "Our teaching lab (room 214, the one with the broken fume hood) is making {V_s} of {C_s} {name} stock. {formula} has a molar mass of {M} g/mol. How much of the solid is needed?{hint}",
    "What mass of {formula} (M = {M} g/mol) must be dissolved and made up to {V_s} to give a concentration of {C_s}?{hint}",
    "Calculate the mass of {name} required to prepare {V_s} of solution with a concentration of {C_s}. Molar mass: {M} g/mol.{hint}",
    "recipe says {C_s} {name}, I'm making {V_s}; {formula} weighs {M} g per mole, so how much do I weigh out?{hint}",
    "Weighing question: {name}, {formula}, {M} g/mol. Target: {V_s} at {C_s}. Required mass of the solid?{hint}",
]


def ch_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = ch_scenario(rng)
    return CH_TEMPLATES[tpl].format(**s), s["value"], s["unit"], f"{sig(s['value'])} {s['unit']}"


def ch_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = ch_scenario(rng)                                             # moles, not mass
    q = rng.choice([
        "How many moles of {name} ({formula}, molar mass {M} g/mol) are needed to make {V_s} of a {C_s} solution?",
        "prep: {C_s} {formula}, {V_s} final volume, M = {M} g/mol. How many moles of solute go in?",
    ]).format(**s)
    mol = float(s["c"]) * s["v"] / 1000
    out.append(near(q, f"{sig(mol)} mol", mol, "mol", "different_target"))

    s = ch_scenario(rng)                                             # find the concentration
    m_g = rng.choice([2.5, 5, 7.5, 10, 12.5, 20, 25, 40, 50])
    q = rng.choice([
        "{m} g of {name} ({formula}, molar mass {M} g/mol) is dissolved in water and made up to {V_s}. What is the molarity of the solution?",
        "{m} g {formula} (M = {M} g/mol), final volume {V_s}. Concentration in mol/L?",
    ]).format(m=fd(m_g), **s)
    c = molarity_from_mass(m_g, s["v"], float(s["M"]))
    out.append(near(q, f"{sig(c)} mol/L", c, "mol/L", "inverse_problem"))

    s = ch_scenario(rng)                                             # impure reagent
    purity = rng.choice(["95.0", "97.0", "98.0", "99.0", "99.5"])
    q = rng.choice([
        "How many grams of {name} ({formula}, molar mass {M} g/mol) are needed to make {V_s} of a {C_s} solution if the reagent is only {pur}% pure by mass?",
        "I need {V_s} of {C_s} {formula} solution. The molar mass is {M} g/mol, but my solid is {pur}% pure (by mass). What mass of it should I weigh out, in grams?",
    ]).format(pur=purity, **s)
    g = s["grams"] / (float(purity) / 100)
    out.append(near(q, f"{sig(g)} g", g, "g", "extra_condition"))

    s = ch_scenario(rng)                                             # dilution of a stock
    stock = rng.choice(["2.00", "3.00", "5.00", "6.00", "10.0", "12.0"])
    while float(stock) <= float(s["c"]):
        stock = rng.choice(["2.00", "3.00", "5.00", "6.00", "10.0", "12.0"])
    q = rng.choice([
        "How many millilitres of a {stock} M stock solution of {name} are needed to prepare {V_s} of a {C_s} solution?",
        "Stock: {stock} mol/L {formula}. Target: {V_s} at {C_s}. Volume of stock to dilute, in mL?",
    ]).format(stock=stock, **s)
    v1 = stock_volume_ml(float(stock), float(s["c"]), s["v"])
    out.append(near(q, f"{sig(v1)} mL", v1, "mL", "different_setup"))

    s = ch_scenario(rng)                                             # compound and molar mass missing
    q = rng.choice([
        "What mass of an unknown salt is needed to make {V_s} of a {C_s} solution? (The identity of the salt isn't given.)",
        "prep: {C_s} of some salt, {V_s} final volume. Mass of solute?",
    ]).format(**s)
    out.append(near(q, cannot("the molar mass of the solute"), None, "", "missing_given"))

    s = ch_scenario(rng)                                             # harmless extra detail
    q = rng.choice([
        "{mass_q} of {name} ({formula}, molar mass {M} g/mol) are needed to make {V_s} of a {C_s} solution, prepared with deionised water at 25 °C?",
        "I need {V_s} of {C_s} {formula} solution, made up in deionised water at room temperature (25 °C). With a molar mass of {M} g/mol, what mass of solid should I weigh out?{hint}",
    ]).format(**s)
    out.append(near(q, f"{sig(s['value'])} {s['unit']}", s["value"], s["unit"], "benign_extra_info", either=True))
    return out


# ============================================================================ one-off calculations
def parallel_resistance(*ohms: float) -> float:
    return 1 / sum(1 / r for r in ohms)


def trip_fuel_cost(km: float, l_per_100km: float, price_per_litre: float) -> float:
    return km * l_per_100km / 100 * price_per_litre


def doubling_time_years(rate_pct: float) -> float:
    return math.log(2) / math.log(1 + rate_pct / 100)


def free_fall_time(height_m: float, g: float = 9.81) -> float:
    return math.sqrt(2 * height_m / g)


def distinct_arrangements(word: str) -> int:
    n = math.factorial(len(word))
    for ch in set(word):
        n //= math.factorial(word.count(ch))
    return n


def binomial_pmf(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def dose_each(weight_kg: float, mg_per_kg_day: float, doses_per_day: int) -> float:
    return weight_kg * mg_per_kg_day / doses_per_day


def cylinder_litres(radius_m: float, height_m: float) -> float:
    return math.pi * radius_m ** 2 * height_m * 1000


def one_offs() -> list[dict]:
    rows = [
        ("oneoff_parallel_resistors",
         "Three resistors of 47 Ω, 68 Ω and 100 Ω are connected in parallel. What is the equivalent resistance?",
         parallel_resistance(47, 68, 100), "Ω", False),
        ("oneoff_trip_fuel_cost",
         "A 640 km road trip in a car that uses 7.8 litres per 100 km, with fuel at $1.84 per litre. How much does the fuel cost?",
         trip_fuel_cost(640, 7.8, 1.84), "$", True),
        ("oneoff_doubling_time",
         "How many years does it take for money to double at 4.6% interest compounded annually?",
         doubling_time_years(4.6), "years", False),
        ("oneoff_free_fall_time",
         "A stone is dropped from a 156 m cliff. Ignoring air resistance and taking g = 9.81 m/s², how long does it take to hit the ground?",
         free_fall_time(156), "s", False),
        ("oneoff_letter_arrangements",
         "How many distinct arrangements are there of the letters in the word ALABAMA?",
         float(distinct_arrangements("ALABAMA")), "", None),
        ("oneoff_binomial_heads",
         "A biased coin lands heads with probability 0.6. What is the probability of getting exactly 3 heads in 8 flips?",
         binomial_pmf(8, 3, 0.6), "", False),
        ("oneoff_pediatric_dose",
         "A child weighs 18 kg and the prescribed dose is 12.5 mg/kg/day, split into 3 equal doses. How many mg is each dose?",
         dose_each(18, 12.5, 3), "mg", False),
        ("oneoff_tank_volume",
         "A cylindrical water tank has a radius of 1.2 m and a height of 3.5 m. How many litres does it hold when full?",
         cylinder_litres(1.2, 3.5), "L", False),
    ]
    out = []
    for fam, q, v, unit, kind in rows:
        if kind is True:
            v = float(cents(v))
            disp = money(v)
        elif kind is None:
            disp = str(int(v))
        else:
            disp = f"{sig(v)} {unit}".strip()
        out.append(msg("one_off", fam, q, disp, v, unit, True))
    return out


CHITCHAT = ["hey there, quick question coming up", "thanks a lot, that was really clear",
            "can you remind me what we covered earlier?", "ok that makes sense",
            "you're awesome", "talk later!"]


def chitchat() -> list[dict]:
    return [msg("chitchat", None, c, "", None, "", False) for c in CHITCHAT]


# ============================================================================ assembly
FAMILIES = {
    CAL_ID: (cal_variant, cal_near_misses, len(CAL_TEMPLATES)),
    FV_ID: (fv_variant, fv_near_misses, len(FV_TEMPLATES)),
    ST_ID: (st_variant, st_near_misses, len(ST_TEMPLATES)),
    CH_ID: (ch_variant, ch_near_misses, len(CH_TEMPLATES)),
}


def build(seed: int = SEED) -> list[dict]:
    """The whole message list, shuffled. Deterministic for a given seed."""
    msgs, seen = [], set()
    for fam, (variant_fn, near_fn, n_tpl) in FAMILIES.items():
        rng = random.Random(f"{seed}:{fam}")
        order = list(range(n_tpl))
        rng.shuffle(order)
        count = 0
        while count < PER_FAMILY:
            q, v, unit, disp = variant_fn(rng, order[count % n_tpl])
            if q in seen:
                continue
            seen.add(q)
            msgs.append(msg("variant", fam, q, disp, v, unit, True))
            count += 1
        for nm in near_fn(random.Random(f"{seed}:{fam}:near")):
            if nm["question"] in seen:
                raise RuntimeError(f"duplicate near-miss question in {fam}")
            seen.add(nm["question"])
            msgs.append(msg("near_miss", fam, nm["question"], nm["answer"], nm["value"], nm["unit"], False,
                            nm["either"], nm["category"]))
    msgs += one_offs() + chitchat()
    random.Random(f"{seed}:shuffle").shuffle(msgs)
    return msgs


def dumps(msgs: list[dict]) -> str:
    return "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in msgs)


if __name__ == "__main__":
    messages = build()
    OUT.write_text(dumps(messages), encoding="utf-8")
    kinds: dict[str, int] = {}
    for m in messages:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(f"wrote {OUT} ({len(messages)} messages: {kinds})")
