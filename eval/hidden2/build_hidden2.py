"""Generator for the SECOND hidden chatbot-cache test set (eval/hidden2/chat_hidden2.jsonl).

    .venv/bin/python eval/hidden2/build_hidden2.py        # rewrites chat_hidden2.jsonl (free, no API)

Same spec and schema as eval/hidden (load with chat_loop.load_messages). Never read the questions
while tuning prompts, gates or the builder: this set exists to measure, not to tune.

Contents (fixed seed, shuffled), none of it in eval/corpus, eval/labelfree, eval/chat_loop.py
or eval/hidden:
  - 4 recurring task families, 20 variants each:
      newton_cooling_temperature   physics      temperature of a cooling object after a time
      iv_drip_rate                 nursing      drops per minute for a gravity IV infusion
      thin_lens_image_distance     optics       image distance of a thin converging lens
      eoq_total_annual_cost        operations   annual ordering + holding cost at the EOQ
  - 6 near-misses per family: same topic and style, different task.
  - 8 one-off calculations from other domains, and 6 chit-chat messages.

Every answer comes from the plain-Python solvers below, never a model. Nothing is rounded until
the end: answer_value is the full-precision float, except money, which is the value to the cent
(rounded once, half up), and counts, which are exact integers. The display form follows
arc/present.ROUNDING_POLICY: the question's own rounding wins ("to the nearest whole drop"),
else money to the cent, counts exact, everything else 4 significant figures, halves up.

Units: a quantity of the same kind as an input is answered in the unit the question's inputs
use (a temperature in the question's degC / degF / K, a distance in its cm / mm / m); if the
question asks for a unit, that unit is used.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
from decimal import ROUND_HALF_UP, Decimal as D
from pathlib import Path

# The formatting helpers are shared with the first hidden set (same rounding policy, same schema).
_spec = importlib.util.spec_from_file_location("build_hidden", Path(__file__).parents[1] / "hidden" / "build_hidden.py")
h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1)
clean, fd, sig, cents, money, words, msg, near, cannot = (
    h1.clean, h1.fd, h1.sig, h1.cents, h1.money, h1.words, h1.msg, h1.near, h1.cannot)

SEED = 7310417
OUT = Path(__file__).with_name("chat_hidden2.jsonl")
PER_FAMILY = 20


def half_up_int(v: float) -> int:
    return int(h1._d(v).quantize(D(1), rounding=ROUND_HALF_UP))


def cents_are_safe(v: float) -> bool:
    """True when the value is not within 1e-6 of a half-cent boundary (rounding is unambiguous)."""
    return abs((v * 100) % 1 - 0.5) > 1e-6


# ============================================================================ family 1: physics
NC_ID = "newton_cooling_temperature"
NC_OBJECTS = ["cup of coffee", "bowl of soup", "mug of tea", "cast-iron pan", "steel rod", "pot of stew",
              "metal casting", "cup of cocoa"]
NC_BG = ["The kitchen timer has been going off all afternoon.",
         "We're repeating this demo for the third class today.",
         "I left the thermometer in a drawer, so I'm estimating.",
         "It's the one on the blue tray by the window.",
         "This is for a lab write-up due Friday."]
NC_TIMES = [5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120]                     # minutes
NC_K_MIN = ["0.02", "0.025", "0.03", "0.04", "0.05", "0.06", "0.08", "0.1"]  # per minute
NC_HALF = [5, 8, 10, 12, 15, 20, 25, 30, 40, 60]                            # minutes


def newton_cooling(t_amb: float, t0: float, k: float, t: float) -> float:
    """Temperature after time t (k and t in matching time units); works in any linear scale."""
    return t_amb + (t0 - t_amb) * math.exp(-k * t)


def time_to_reach(t_amb: float, t0: float, target: float, k: float) -> float:
    return math.log((t0 - t_amb) / (target - t_amb)) / k


def hours_text(minutes: int, spelled: bool = False) -> str:
    """"1 hour", "1.5 hours", "two hours" for a whole or half number of hours."""
    n = fd(D(minutes) / 60)
    if spelled:
        n = words(minutes // 60)
    return f"{n} hour" if n in ("1", "one") else f"{n} hours"


def _conv(c: D, u: str) -> D:
    return {"C": c, "F": c * 9 / 5 + 32, "K": c + D("273.15")}[u]


def _tstr(v: D, u: str) -> str:
    return f"{fd(v)} {'K' if u == 'K' else '°' + u}"


def nc_scenario(rng: random.Random, any_unit: bool = True) -> dict:
    u = rng.choices(["C", "F", "K"], [5, 3, 2])[0] if any_unit else "C"
    while True:
        t0c, tac = D(rng.randrange(120, 191)) / 2, D(rng.randrange(36, 55)) / 2
        style, t = rng.choice(["k_min", "k_min", "k_h", "half", "half"]), rng.choice(NC_TIMES)
        if style == "half":
            h = rng.choice(NC_HALF)
            k, kk = math.log(2) / h, None
        else:
            kk = rng.choice(NC_K_MIN)
            k, h = float(kk), None
        x = k * t
        if 0.25 <= x <= 3.5:
            break
    if style == "k_min":
        rate = rng.choice([f"k = {kk} min⁻¹", f"a cooling constant of {kk} per minute", f"cooling constant k = {kk}/min"])
    elif style == "k_h":
        kh = fd(D(kk) * 60)
        rate = rng.choice([f"k = {kh} h⁻¹", f"a cooling constant of {kh} per hour"])
    else:
        rate = rng.choice([f"a temperature-difference half-life of {h} minutes",
                           f"the temperature difference halving every {h} minutes"])
    tmode = rng.choice(["min", "min", "words", "hours"]) if t % 15 == 0 else rng.choice(["min", "words"])
    if t >= 100 and tmode == "words":
        tmode = "min"
    t_s = (f"{t} minutes" if tmode == "min" else f"{words(t)} minutes" if tmode == "words"
           else hours_text(t))
    T0, Ta = _conv(t0c, u), _conv(tac, u)
    return {"obj": rng.choice(NC_OBJECTS), "T0_s": _tstr(T0, u), "Ta_s": _tstr(Ta, u), "rate_s": rate, "t_s": t_s,
            "bg": rng.choice(NC_BG), "u": u, "unit": "K" if u == "K" else "°" + u, "T0": float(T0),
            "Ta": float(Ta), "k": k, "t": t, "x": x, "T": newton_cooling(float(Ta), float(T0), k, t),
            "t0c": float(t0c), "tac": float(tac), "tc": newton_cooling(float(tac), float(t0c), k, t)}


NC_TEMPLATES = [
    "A {obj} at {T0_s} is left in a room that stays at {Ta_s}. Assuming Newton's law of cooling with {rate_s}, what is its temperature after {t_s}?",
    "{bg} Anyway, my {obj} starts at {T0_s} and the air around it is {Ta_s}. It cools by Newton's law with {rate_s}. How hot is it after {t_s}?",
    "{obj}: T0 = {T0_s}, ambient = {Ta_s}, {rate_s}. Temperature at t = {t_s}?",
    "Use Newton's law of cooling to find the temperature of a {obj} {t_s} after it was at {T0_s}, if the surroundings are at {Ta_s} and the object cools with {rate_s}.",
    "How warm will a {obj} be {t_s} from now? It's currently {T0_s}, the ambient temperature is {Ta_s}, and it follows Newton's cooling law with {rate_s}.",
    "newton cooling q - {obj} {T0_s}, room {Ta_s}, {rate_s}, wait {t_s}, what's the temp then?",
    "A technician pulls a {obj} out of the oven at {T0_s} and sets it on a bench in a {Ta_s} room. Modelling the cooling with Newton's law ({rate_s}), what temperature will it have reached after {t_s}?",
    "Find the temperature after {t_s} of a {obj} initially at {T0_s} placed in surroundings at {Ta_s}, with exponential (Newton) cooling characterised by {rate_s}.",
]


def nc_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = nc_scenario(rng)
    return NC_TEMPLATES[tpl].format(**s), s["T"], s["unit"], f"{sig(s['T'])} {s['unit']}"


def nc_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = nc_scenario(rng, any_unit=False)                             # how long until a target temperature
    tgt = round(s["tc"])
    while tgt - s["tac"] < 1.5:
        s = nc_scenario(rng, any_unit=False)
        tgt = round(s["tc"])
    q = rng.choice([
        "A {obj} at {T0_s} is left in a room that stays at {Ta_s}. Assuming Newton's law of cooling with {rate_s}, how many minutes until it has cooled to {tgt} °C?",
        "{bg} My {obj} starts at {T0_s}, the room is {Ta_s}, and it cools by Newton's law with {rate_s}. How long, in minutes, before it reaches {tgt} °C?",
    ]).format(tgt=tgt, **s)
    mins = time_to_reach(s["tac"], s["t0c"], tgt, s["k"])
    out.append(near(q, f"{sig(mins)} min", mins, "min", "inverse_problem"))

    s = nc_scenario(rng, any_unit=False)                             # fraction of the excess that is left
    q = rng.choice([
        "A {obj} at {T0_s} is left in a room that stays at {Ta_s}. Assuming Newton's law of cooling with {rate_s}, what fraction of its initial temperature excess over the room remains after {t_s}?",
        "{obj}: T0 = {T0_s}, ambient = {Ta_s}, {rate_s}. After {t_s}, what fraction of the starting temperature difference is still there?",
    ]).format(**s)
    frac = math.exp(-s["x"])
    out.append(near(q, sig(frac), frac, "", "different_target"))

    s = nc_scenario(rng, any_unit=False)                             # the room is heated part-way through
    t1, t2 = rng.choice([10, 15, 20, 30]), rng.choice([10, 20, 30, 45])
    ta2 = s["tac"] + rng.choice([4, 6, 8, 10])
    q = rng.choice([
        "A {obj} at {T0_s} is left in a room at {Ta_s}; it cools by Newton's law with {rate_s}. After {t1} minutes the heating comes on and the room rises to {ta2} °C and stays there. What is the temperature of the {obj} {t2} minutes after that?",
        "{obj}: T0 = {T0_s}, room {Ta_s}, {rate_s}. At t = {t1} min the room is warmed to {ta2} °C and held there. Temperature at t = {t1} + {t2} min?",
    ]).format(t1=t1, t2=t2, ta2=fd(ta2), **s)
    mid = newton_cooling(s["tac"], s["t0c"], s["k"], t1)
    end = newton_cooling(ta2, mid, s["k"], t2)
    out.append(near(q, f"{sig(end)} °C", end, "°C", "extra_condition"))

    s = nc_scenario(rng, any_unit=False)                             # no cooling rate given
    q = rng.choice([
        "A {obj} at {T0_s} is left in a room that stays at {Ta_s}. Assuming Newton's law of cooling, what is its temperature after {t_s}?",
        "{bg} My {obj} starts at {T0_s} in a {Ta_s} room and cools by Newton's law. How hot is it after {t_s}?",
    ]).format(**s)
    out.append(near(q, cannot("the cooling constant (or half-life)"), None, "", "missing_given"))

    s = nc_scenario(rng, any_unit=False)                             # how fast it is cooling at the start
    q = rng.choice([
        "A {obj} at {T0_s} is left in a room that stays at {Ta_s}. With Newton's law of cooling and {rate_s}, how fast is its temperature falling at the very start, in °C per minute?",
        "{obj}: T0 = {T0_s}, ambient = {Ta_s}, {rate_s}. Initial rate of temperature drop in °C/min?",
    ]).format(**s)
    r0 = s["k"] * (s["t0c"] - s["tac"])
    out.append(near(q, f"{sig(r0)} °C/min", r0, "°C/min", "different_target"))

    s = nc_scenario(rng, any_unit=False)                             # harmless extra detail
    q = rng.choice([
        "A {obj} at {T0_s}, sitting on a wooden table with the lights off, is left in a room that stays at {Ta_s}. Assuming Newton's law of cooling with {rate_s}, what is its temperature after {t_s}?",
        "{bg} My {obj} (blue ceramic, if that matters) starts at {T0_s} and the air around it is {Ta_s}. It cools by Newton's law with {rate_s}. How hot is it after {t_s}?",
    ]).format(**s)
    out.append(near(q, f"{sig(s['tc'])} °C", s["tc"], "°C", "benign_extra_info", either=True))
    return out


# ============================================================================ family 2: nursing
IV_ID = "iv_drip_rate"
IV_VOLS = [50, 100, 125, 150, 250, 500, 750, 1000, 1500, 2000]               # mL
IV_MINS = [30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 720]                # minutes
IV_DF = [10, 15, 20, 60]                                                     # gtt/mL
IV_FLUIDS = ["normal saline", "lactated Ringer's", "5% dextrose in water", "half-normal saline"]
IV_BG = ["Bed 7 is waiting on this one and the pump is out of service.",
         "Shift change is in twenty minutes.",
         "The patient is a 72-year-old, no known allergies.",
         "It's a gravity set because the pump's battery died."]


def drip_rate(volume_ml: float, minutes: float, drop_factor: float) -> float:
    """Drops per minute for a gravity infusion."""
    return volume_ml * drop_factor / minutes


def pump_rate_ml_per_h(volume_ml: float, minutes: float) -> float:
    return volume_ml / (minutes / 60)


def hours_to_empty(volume_ml: float, drop_factor: float, gtt_per_min: float) -> float:
    return volume_ml * drop_factor / gtt_per_min / 60


def weight_based_pump_rate(mcg_kg_min: float, kg: float, drug_mg: float, bag_ml: float) -> float:
    """mL/h for a dose in mcg/kg/min from a bag holding drug_mg in bag_ml."""
    return mcg_kg_min * kg * 60 / 1000 / (drug_mg / bag_ml)


def iv_vol_text(v: int, rng: random.Random) -> str:
    opts = [f"{v} mL", f"{v} mL", f"{v} ml"]
    if v >= 250:
        opts.append(f"{fd(D(v) / 1000)} L")
    return rng.choice(opts)


def iv_time_text(m: int, rng: random.Random) -> str:
    opts = [f"{m} minutes", f"{m} min"]
    if m % 30 == 0:
        opts += [hours_text(m), f"{fd(D(m) / 60)} h"]
    if m % 60 == 0 and m <= 720:
        opts.append(hours_text(m, spelled=True))
    return rng.choice(opts)


def iv_df_text(df: int, rng: random.Random) -> str:
    return rng.choice([f"{df} gtt/mL", f"{df} drops/mL", f"{df} drops per mL"])


def iv_scenario(rng: random.Random) -> dict:
    while True:
        v, m, df = rng.choice(IV_VOLS), rng.choice(IV_MINS), rng.choice(IV_DF)
        r = drip_rate(v, m, df)
        if 5 <= r <= 200:
            break
    whole = rng.random() < 0.2
    value = float(half_up_int(r)) if whole else r
    return {"vol_s": iv_vol_text(v, rng), "time_s": iv_time_text(m, rng), "df_s": iv_df_text(df, rng),
            "fluid": rng.choice(IV_FLUIDS), "bg": rng.choice(IV_BG), "v": v, "m": m, "df": df, "r": r,
            "value": value, "whole": whole, "rnd": " Round to the nearest whole drop." if whole else ""}


IV_TEMPLATES = [
    "A nurse needs to infuse {vol_s} of {fluid} over {time_s}. The tubing has a drop factor of {df_s}. What drip rate, in drops per minute, should be set?{rnd}",
    "Order: {vol_s} IV fluid over {time_s}. Drop factor {df_s}. Drip rate (gtt/min)?{rnd}",
    "{bg} Please calculate the drops per minute for {vol_s} of {fluid} to run in over {time_s} with {df_s} tubing.{rnd}",
    "How many drops per minute are needed to deliver {vol_s} in {time_s} through a set calibrated at {df_s}?{rnd}",
    "iv rate q: {vol_s} / {time_s}, {df_s}. gtt/min = ?{rnd}",
    "The doctor ordered {vol_s} of {fluid} to be given over {time_s}. The gravity set has a drop factor of {df_s}. What should the flow rate be in gtt/min?{rnd}",
    "For a gravity infusion of {vol_s} across {time_s}, using tubing with a drop factor of {df_s}, work out the drip rate per minute.{rnd}",
    "Nursing calc: the patient is to receive {vol_s} in {time_s}; set specification: {df_s}. Find the rate in drops/min.{rnd}",
]


def iv_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = iv_scenario(rng)
    shown = f"{int(s['value'])} gtt/min" if s["whole"] else f"{sig(s['value'])} gtt/min"
    return IV_TEMPLATES[tpl].format(**s), s["value"], "gtt/min", shown


def iv_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = iv_scenario(rng)                                             # pump setting in mL/h
    q = rng.choice([
        "A nurse needs to infuse {vol_s} of {fluid} over {time_s}. What rate should the infusion pump be set to, in mL/h?",
        "Order: {vol_s} IV fluid over {time_s}. Pump rate in mL per hour?",
    ]).format(**s)
    ph = pump_rate_ml_per_h(s["v"], s["m"])
    out.append(near(q, f"{sig(ph)} mL/h", ph, "mL/h", "different_target"))

    v, df, r = rng.choice([250, 500, 1000]), rng.choice(IV_DF), rng.choice([20, 25, 30, 40, 50, 60, 75, 100])
    q = rng.choice([
        f"A {v} mL bag of normal saline is running at {r} gtt/min through tubing with a drop factor of {df} gtt/mL. How many hours until the bag is empty?",
        f"{v} mL IV bag, {df} drops/mL set, running at {r} drops per minute. Time to finish, in hours?",
    ])
    hrs = hours_to_empty(v, df, r)
    out.append(near(q, f"{sig(hrs)} h", hrs, "h", "inverse_problem"))

    s = iv_scenario(rng)                                             # part of the bag is already in
    done = max(10, int(s["v"] * rng.choice([0.2, 0.3, 0.4]) // 10 * 10))
    left_min = rng.choice([m for m in IV_MINS if m <= s["m"]] or [s["m"]])
    q = rng.choice([
        f"A {s['v']} mL bag of {s['fluid']} has already run in {done} mL. The rest must finish in {left_min} minutes through a set with a drop factor of {s['df']} gtt/mL. What drip rate, in drops per minute, is needed now?",
        f"{s['v']} mL ordered, {done} mL already infused, remaining volume to go in over {left_min} min, {s['df']} gtt/mL tubing. gtt/min for the rest?",
    ])
    rem = drip_rate(s["v"] - done, left_min, s["df"])
    out.append(near(q, f"{sig(rem)} gtt/min", rem, "gtt/min", "extra_condition"))

    s = iv_scenario(rng)                                             # no drop factor
    q = rng.choice([
        "A nurse needs to infuse {vol_s} of {fluid} over {time_s}. What drip rate, in drops per minute, should be set?",
        "{bg} Please calculate the drops per minute for {vol_s} of {fluid} to run in over {time_s}.",
    ]).format(**s)
    out.append(near(q, cannot("the tubing's drop factor"), None, "", "missing_given"))

    kg, dose = rng.choice([55, 60, 70, 75, 80, 90, 100]), rng.choice([2, 3, 5, 7.5, 10])
    mg, ml = rng.choice([(400, 250), (800, 500), (200, 250), (250, 250), (500, 250)])
    q = rng.choice([
        f"A patient weighing {kg} kg is ordered dopamine at {fd(dose)} mcg/kg/min. The bag holds {mg} mg in {ml} mL. What pump rate, in mL/h, delivers that dose?",
        f"dopamine {fd(dose)} mcg/kg/min, patient {kg} kg, bag {mg} mg/{ml} mL. Pump rate mL/h?",
    ])
    wb = weight_based_pump_rate(dose, kg, mg, ml)
    out.append(near(q, f"{sig(wb)} mL/h", wb, "mL/h", "different_setup"))

    s = iv_scenario(rng)                                             # harmless extra detail
    q = rng.choice([
        "A nurse needs to infuse {vol_s} of {fluid} over {time_s} through an 18-gauge catheter in the left forearm. The tubing has a drop factor of {df_s}. What drip rate, in drops per minute, should be set?{rnd}",
        "{bg} Order: {vol_s} over {time_s} via a peripheral line (20 gauge). Drop factor {df_s}. gtt/min?{rnd}",
    ]).format(**s)
    shown = f"{int(s['value'])} gtt/min" if s["whole"] else f"{sig(s['value'])} gtt/min"
    out.append(near(q, shown, s["value"], "gtt/min", "benign_extra_info", either=True))
    return out


# ============================================================================ family 3: optics
LN_ID = "thin_lens_image_distance"
LN_F_CM = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]
LN_RATIOS = [1.25, 1.5, 2, 2.5, 3, 4, 5]
LN_BG = ["We're setting this up on the optical bench in the back lab.",
         "The rail has a ruler taped along it, so every reading is in the same unit.",
         "I'm double-checking my lab sheet before class.",
         "This is the third lens we've tried today."]
LN_WHAT = ["candle", "slide", "specimen", "small bulb", "printed grid"]


def image_distance(f: float, d_o: float) -> float:
    """Thin lens, real object outside the focal point: 1/f = 1/d_o + 1/d_i."""
    return f * d_o / (d_o - f)


def object_distance(f: float, d_i: float) -> float:
    return f * d_i / (d_i - f)


def magnification(f: float, d_o: float) -> float:
    return image_distance(f, d_o) / d_o


def diverging_image_distance(f: float, d_o: float) -> float:
    """Distance (magnitude) of the virtual image of a diverging lens of focal length f > 0."""
    return f * d_o / (f + d_o)


def lenses_in_contact(f1: float, f2: float) -> float:
    return f1 * f2 / (f1 + f2)


def ln_unit(rng: random.Random) -> tuple[str, float]:
    return rng.choices([("cm", 1.0), ("mm", 10.0), ("m", 0.01)], [5, 2, 3])[0]


def ln_len(base_cm: int, unit: str, scale: float) -> str:
    return f"{fd(D(base_cm) * D(str(scale)))} {unit}"


def ln_scenario(rng: random.Random) -> dict:
    unit, scale = ln_unit(rng)
    while True:
        f = rng.choice(LN_F_CM)
        d = round(f * rng.choice(LN_RATIOS))
        if d / f >= 1.2:
            break
    fs, ds = f * scale, d * scale
    return {"f_s": ln_len(f, unit, scale), "do_s": ln_len(d, unit, scale), "unit": unit, "scale": scale,
            "f": f, "d": d, "bg": rng.choice(LN_BG), "what": rng.choice(LN_WHAT),
            "di": image_distance(fs, ds), "fs": fs, "ds": ds}


LN_TEMPLATES = [
    "An object is placed {do_s} in front of a thin converging lens with a focal length of {f_s}. How far from the lens is the image formed?",
    "A convex lens with a focal length of {f_s} forms an image of a {what} located {do_s} from the lens. Find the image distance.",
    "thin lens: f = {f_s}, object distance = {do_s}. image distance?",
    "{bg} Anyway, a slide sits {do_s} from a projector lens whose focal length is {f_s}. Where does the screen have to be placed (distance from the lens) to get a sharp image?",
    "Using the thin-lens equation, calculate the image distance for an object {do_s} away from a converging lens of focal length {f_s}.",
    "A camera lens with f = {f_s} is focused on a subject {do_s} away. How far behind the lens does the sensor need to be for a sharp image?",
    "My magnifier is a converging lens, focal length {f_s}. If I hold the {what} {do_s} from it, at what distance from the lens will the real image appear?",
    "Real image: a lens with {f_s} focal length, object at {do_s}. Distance of the image from the lens?",
]


def ln_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = ln_scenario(rng)
    return LN_TEMPLATES[tpl].format(**s), s["di"], s["unit"], f"{sig(s['di'])} {s['unit']}"


def ln_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = ln_scenario(rng)                                             # magnification
    q = rng.choice([
        "An object is placed {do_s} in front of a thin converging lens with a focal length of {f_s}. What is the magnitude of the lateral magnification?",
        "thin lens: f = {f_s}, object distance = {do_s}. How many times larger or smaller is the image (magnification, ignoring sign)?",
    ]).format(**s)
    mag = magnification(s["fs"], s["ds"])
    out.append(near(q, sig(mag), mag, "", "different_target"))

    s = ln_scenario(rng)                                             # where to put the object
    di_base = max(round(s["d"] * rng.choice([0.8, 1.0, 1.5])), s["f"] + 3)
    di_s = ln_len(di_base, s["unit"], s["scale"])
    q = rng.choice([
        "A thin converging lens has a focal length of {f_s}. How far from the lens must an object be placed so that its image forms {di_s} from the lens?",
        "thin lens: f = {f_s}, wanted image distance = {di_s}. Object distance?",
    ]).format(di_s=di_s, **s)
    need = object_distance(s["fs"], di_base * s["scale"])
    out.append(near(q, f"{sig(need)} {s['unit']}", need, s["unit"], "inverse_problem"))

    s = ln_scenario(rng)                                             # diverging lens
    q = rng.choice([
        "An object is placed {do_s} in front of a thin diverging lens whose focal length has magnitude {f_s}. How far from the lens does the (virtual) image appear?",
        "diverging lens, |f| = {f_s}, object distance = {do_s}. Where is the image, as a distance from the lens?",
    ]).format(**s)
    dv = diverging_image_distance(s["fs"], s["ds"])
    out.append(near(q, f"{sig(dv)} {s['unit']}", dv, s["unit"], "different_setup"))

    s = ln_scenario(rng)                                             # two lenses pressed together
    f2 = rng.choice([x for x in LN_F_CM if x != s["f"]])
    fc = lenses_in_contact(s["f"], f2)
    d = max(round(fc * rng.choice([1.5, 2, 3])), round(fc) + 3)
    q = rng.choice([
        "Two thin converging lenses with focal lengths {f_s} and {f2_s} are pressed together. An object sits {d_s} from the pair. How far from the lenses is the image?",
        "thin lenses in contact: f1 = {f_s}, f2 = {f2_s}. Object at {d_s}. Image distance from the pair?",
    ]).format(f_s=s["f_s"], f2_s=ln_len(f2, s["unit"], s["scale"]), d_s=ln_len(d, s["unit"], s["scale"]))
    comb = image_distance(fc * s["scale"], d * s["scale"])
    out.append(near(q, f"{sig(comb)} {s['unit']}", comb, s["unit"], "extra_condition"))

    s = ln_scenario(rng)                                             # no focal length
    q = rng.choice([
        "An object is placed {do_s} in front of a thin converging lens. How far from the lens is the image formed?",
        "A convex lens (focal length not marked) forms an image of a {what} located {do_s} from the lens. Find the image distance.",
    ]).format(**s)
    out.append(near(q, cannot("the focal length of the lens"), None, "", "missing_given"))

    s = ln_scenario(rng)                                             # harmless extra detail
    q = rng.choice([
        "An object is placed {do_s} in front of a thin converging lens (crown glass, 5 cm diameter) with a focal length of {f_s}. How far from the lens is the image formed?",
        "{bg} A convex lens of {f_s} focal length, made of crown glass, forms an image of a {what} located {do_s} from the lens. Find the image distance.",
    ]).format(**s)
    out.append(near(q, f"{sig(s['di'])} {s['unit']}", s["di"], s["unit"], "benign_extra_info", either=True))
    return out


# ============================================================================ family 4: operations
EQ_ID = "eoq_total_annual_cost"
EQ_ANNUAL = [1200, 2400, 3600, 4800, 6000, 8000, 10000, 12000, 15000, 20000, 24000, 36000, 50000]
EQ_MONTHLY = [100, 150, 200, 250, 300, 400, 500, 800, 1000, 1500, 2000, 3000]
EQ_WEEKLY = [50, 60, 80, 100, 120, 150, 200, 250, 300, 400]
EQ_S = [25, 40, 50, 60, 75, 80, 100, 120, 150, 200]
EQ_H = ["0.50", "0.80", "1.20", "1.50", "2.00", "2.50", "3.00", "4.00", "5.00", "6.00", "8.00"]
EQ_COST = [8, 10, 12, 15, 20, 25, 30, 40, 50, 80]
EQ_PCT = [10, 12, 15, 18, 20, 25]
EQ_BG = ["Our warehouse in Dayton is nearly full this quarter.",
         "The supplier changed their invoice template again.",
         "Finance wants this for the Monday review.",
         "This is the part that keeps stocking out."]


def eoq(demand: float, order_cost: float, holding: float) -> float:
    """Economic order quantity in units."""
    return math.sqrt(2 * demand * order_cost / holding)


def eoq_total_cost(demand: float, order_cost: float, holding: float) -> float:
    """Annual ordering + holding cost when ordering the EOQ (purchase price excluded)."""
    return math.sqrt(2 * demand * order_cost * holding)


def epq_total_cost(demand: float, order_cost: float, holding: float, production_rate: float) -> float:
    return math.sqrt(2 * demand * order_cost * holding * (1 - demand / production_rate))


def order_cost_from_eoq(q: float, demand: float, holding: float) -> float:
    return q * q * holding / (2 * demand)


def eq_scenario(rng: random.Random, plain_holding: bool = False) -> dict:
    while True:
        style = rng.choice(["year", "month", "week"])
        if style == "year":
            d = rng.choice(EQ_ANNUAL)
            dem = f"{d:,} units per year"
        elif style == "month":
            m = rng.choice(EQ_MONTHLY)
            d, dem = m * 12, f"{m:,} units per month"
        else:
            w = rng.choice(EQ_WEEKLY)
            d, dem = w * 52, f"{w:,} units a week (52 weeks a year)"
        s_cost = rng.choice(EQ_S)
        if plain_holding or rng.random() < 0.55:
            h = D(rng.choice(EQ_H))
            hold = f"${h:.2f} per unit per year"
        else:
            c, pct = rng.choice(EQ_COST), rng.choice(EQ_PCT)
            h = D(c) * D(pct) / 100
            hold = f"{pct}% of the unit cost per year (units cost ${c} each)"
        total = eoq_total_cost(d, s_cost, float(h))
        if cents_are_safe(total):
            break
    return {"dem_s": dem, "S": s_cost, "hold_s": hold, "bg": rng.choice(EQ_BG), "d": d, "h": float(h),
            "total": total, "value": float(cents(total))}


EQ_TEMPLATES = [
    "A distributor sells {dem_s}. Each order costs ${S} to place, and holding inventory costs {hold_s}. If it orders the economic order quantity each time, what is the total annual ordering and holding cost?",
    "Demand: {dem_s}. Ordering cost ${S}/order; holding: {hold_s}. Using EOQ, total yearly cost of ordering plus holding?",
    "{bg} We use about {dem_s}. Placing an order costs ${S} and it costs {hold_s} to carry stock. What's the minimum total annual inventory cost (ordering + holding) at the optimal order size?",
    "Compute the annual ordering-plus-holding cost at the economic order quantity for an item with demand of {dem_s}, a fixed cost of ${S} per order, and a carrying cost of {hold_s}.",
    "eoq: demand {dem_s}; S = ${S} per order; H = {hold_s}. Annual inventory cost (ordering + holding) at Q*?",
    "A retailer moves {dem_s}. Each purchase order costs ${S} in admin time, and storing a unit costs {hold_s}. Assuming the EOQ model, how much does it spend per year on ordering and holding combined?",
    "Under the EOQ model, what is the total annual cost of ordering and holding for a part with demand of {dem_s}, ${S} per order, and a holding cost of {hold_s}?",
    "Inventory problem: demand {dem_s}; order cost ${S}; holding cost {hold_s}. Ignoring purchase price, how much is the yearly cost of keeping and reordering at the optimal lot size?",
]


def eq_variant(rng: random.Random, tpl: int) -> tuple[str, float, str, str]:
    s = eq_scenario(rng)
    return EQ_TEMPLATES[tpl].format(**s), s["value"], "$", money(s["total"])


def eq_near_misses(rng: random.Random) -> list[dict]:
    out = []
    s = eq_scenario(rng)                                             # the quantity itself, to a whole unit
    q = rng.choice([
        "A distributor sells {dem_s}. Each order costs ${S} to place, and holding inventory costs {hold_s}. What is the economic order quantity, rounded to the nearest whole unit?",
        "Demand: {dem_s}. Ordering cost ${S}/order; holding: {hold_s}. EOQ in units, to the nearest whole unit?",
    ]).format(**s)
    qn = half_up_int(eoq(s["d"], s["S"], s["h"]))
    out.append(near(q, f"{qn} units", float(qn), "units", "different_target"))

    s = eq_scenario(rng, plain_holding=True)                         # recover the ordering cost
    qn = half_up_int(eoq(s["d"], s["S"], s["h"]))
    while True:
        cost = order_cost_from_eoq(qn, s["d"], s["h"])
        if cents_are_safe(cost):
            break
        qn += 1
    q = rng.choice([
        "A distributor sells {dem_s} and holding inventory costs {hold_s}. Its economic order quantity is {qn:,} units. What is the cost of placing one order?",
        "Demand: {dem_s}; holding: {hold_s}; EOQ = {qn:,} units. Ordering cost per order?",
    ]).format(qn=qn, **s)
    out.append(near(q, money(cost), float(cents(cost)), "$", "inverse_problem"))

    s = eq_scenario(rng, plain_holding=True)                         # purchase cost included
    while True:
        price = rng.choice([4, 5, 6, 8, 10, 12, 15, 20, 25, 40])
        total = s["d"] * price + s["total"]
        if cents_are_safe(total):
            break
    q = rng.choice([
        "A distributor sells {dem_s}. Each order costs ${S} to place, and holding inventory costs {hold_s}. Each unit costs ${price} to buy. Ordering the EOQ each time, what is the total annual cost including purchasing?",
        "Demand: {dem_s}. Ordering cost ${S}/order; holding: {hold_s}; unit price ${price}. Total annual cost (purchase + ordering + holding) at the EOQ?",
    ]).format(price=price, **s)
    out.append(near(q, money(total), float(cents(total)), "$", "extra_condition"))

    s = eq_scenario(rng, plain_holding=True)                         # production lot: finite production rate
    while True:
        mult = rng.choice([2, 3, 4, 5])
        total = epq_total_cost(s["d"], s["S"], s["h"], s["d"] * mult)
        if cents_are_safe(total):
            break
        s = eq_scenario(rng, plain_holding=True)
    q = rng.choice([
        "A plant makes its own part, with demand of {dem_s}. Each production run costs ${S} to set up and holding costs {hold_s}. Production runs at {p:,} units per year while the demand is being met. What is the total annual setup and holding cost at the optimal run size?",
        "Make-in-house item: demand {dem_s}; setup ${S}/run; holding {hold_s}; production rate {p:,} units/year (replenishes gradually). Annual setup + holding cost at the best lot size?",
    ]).format(p=s["d"] * mult, **s)
    out.append(near(q, money(total), float(cents(total)), "$", "different_setup"))

    s = eq_scenario(rng)                                             # holding cost missing
    q = rng.choice([
        "A distributor sells {dem_s}. Each order costs ${S} to place. If it orders the economic order quantity each time, what is the total annual ordering and holding cost?",
        "{bg} Demand: {dem_s}. Ordering cost ${S}/order. Using EOQ, total yearly cost of ordering plus holding?",
    ]).format(**s)
    out.append(near(q, cannot("the holding cost"), None, "", "missing_given"))

    s = eq_scenario(rng)                                             # harmless extra detail
    q = rng.choice([
        "A distributor in Ohio that ships by truck sells {dem_s}. Each order costs ${S} to place, and holding inventory costs {hold_s}. If it orders the economic order quantity each time, what is the total annual ordering and holding cost?",
        "{bg} Demand: {dem_s}. Ordering cost ${S}/order (the supplier is net-30); holding: {hold_s}. Using EOQ, total yearly cost of ordering plus holding?",
    ]).format(**s)
    out.append(near(q, money(s["total"]), s["value"], "$", "benign_extra_info", either=True))
    return out


# ============================================================================ one-off calculations
def continuous_growth(principal: float, rate_pct: float, years: float) -> float:
    return principal * math.exp(rate_pct / 100 * years)


def rc_time_constant_ms(r_kohm: float, c_uf: float) -> float:
    return r_kohm * 1e3 * c_uf * 1e-6 * 1e3


def average_speed(*legs: tuple[float, float]) -> float:
    """Each leg is (distance, speed); total distance over total time."""
    return sum(d for d, _ in legs) / sum(d / v for d, v in legs)


def committee_count(men: int, women: int, size: int, min_women: int) -> int:
    return sum(math.comb(women, w) * math.comb(men, size - w) for w in range(min_women, min(women, size) + 1))


def poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def kinetic_energy_kj(mass_kg: float, speed_kmh: float) -> float:
    return 0.5 * mass_kg * (speed_kmh / 3.6) ** 2 / 1000


def bmi(weight_kg: float, height_m: float) -> float:
    return weight_kg / height_m ** 2


def ladder_height(length: float, base: float) -> float:
    return math.sqrt(length ** 2 - base ** 2)


def one_offs() -> list[dict]:
    rows = [
        ("oneoff_continuous_growth",
         "If $5,000 grows with continuous compounding at 3.2% a year, how much is it worth after 7 years?",
         continuous_growth(5000, 3.2, 7), "$", True),
        ("oneoff_rc_time_constant",
         "A 4.7 kΩ resistor is in series with a 22 µF capacitor. What is the time constant of the circuit, in milliseconds?",
         rc_time_constant_ms(4.7, 22), "ms", False),
        ("oneoff_average_speed",
         "A driver covers 150 km at 60 km/h and then 90 km at 90 km/h. What is the average speed for the whole trip?",
         average_speed((150, 60), (90, 90)), "km/h", False),
        ("oneoff_committee_count",
         "How many ways can a committee of 4 be chosen from 5 men and 4 women if it must include at least 2 women?",
         float(committee_count(5, 4, 4, 2)), "", None),
        ("oneoff_poisson_calls",
         "A help desk receives on average 3.2 calls per hour. Assuming a Poisson process, what is the probability of exactly 5 calls in a given hour?",
         poisson_pmf(3.2, 5), "", False),
        ("oneoff_kinetic_energy",
         "How much kinetic energy, in kilojoules, does a 1,200 kg car have when travelling at 90 km/h?",
         kinetic_energy_kj(1200, 90), "kJ", False),
        ("oneoff_body_mass_index",
         "A person weighs 82 kg and is 1.78 m tall. What is their body mass index?",
         bmi(82, 1.78), "kg/m²", False),
        ("oneoff_ladder_height",
         "A 6.5 m ladder leans against a wall with its foot 1.8 m from the base of the wall. How high up the wall does it reach?",
         ladder_height(6.5, 1.8), "m", False),
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


CHITCHAT = ["morning! got a minute?", "perfect, thank you", "hold on, let me grab my notes",
            "sorry, ignore that last thing I said", "cheers, that's all for now",
            "interesting, how does this work under the hood?"]


def chitchat() -> list[dict]:
    return [msg("chitchat", None, c, "", None, "", False) for c in CHITCHAT]


# ============================================================================ assembly
FAMILIES = {
    NC_ID: (nc_variant, nc_near_misses, len(NC_TEMPLATES)),
    IV_ID: (iv_variant, iv_near_misses, len(IV_TEMPLATES)),
    LN_ID: (ln_variant, ln_near_misses, len(LN_TEMPLATES)),
    EQ_ID: (eq_variant, eq_near_misses, len(EQ_TEMPLATES)),
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
