"""Reference solvers + question generators for the label-free test corpus.

Every answer here comes from plain Python written by hand, never from a model, so the
corpus can judge model-based certification without sharing its blind spots. Each family
matches one of the 17 seed entries in eval/corpus (same task), and generates:

  positives    the same task with new values, varied phrasing and varied INPUT FORMS
               (units, "k"/"million", "1 in N", pKa vs Ka, ...): the forms that broke
               real cached entries (km/h, percentages, "€2.0 million").
  near-misses  same topic and wording style, different task (another target quantity,
               method, timing, logic or regime). Their truth is THEIR own correct answer.

tests/test_labelfree_corpus.py checks every solver against the seed's known answer.

A family's `truth` is a list of acceptable answer strings (e.g. the same pressure in atm,
kPa and bar). An answer is right if it states the numbers of ANY of them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Item:
    family: str
    kind: str                 # positive | near_miss
    form: str                 # which input forms / near-miss type was used
    question: str
    truth: list[str]          # acceptable answers (numbers are what's checked)
    values: dict = field(default_factory=dict)


def g(x: float, sig: int = 4) -> str:
    """Round to `sig` significant figures, no scientific notation for normal sizes."""
    if x == 0:
        return "0"
    d = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{d}f}"


def money(x: float, cents: bool = False) -> str:
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def money_forms(x: float, rng: random.Random) -> tuple[str, str]:
    """(written form, form tag): $350,000 | $350k | $0.35 million."""
    k = rng.choice(["plain", "k", "million"] if x >= 10_000 else ["plain"])
    if k == "k" and x % 1000 == 0:
        return f"${x / 1000:g}k", "k"
    if k == "million" and x % 10_000 == 0:
        return f"${x / 1e6:g} million", "million"
    return money(x), "plain"


# ============================================================================ chemistry

R = 8.314462618
GASES = [("ethane", "C2H6", 30.07, 5.562, 0.0638), ("nitrogen", "N2", 28.014, 1.370, 0.0387),
         ("carbon dioxide", "CO2", 44.01, 3.640, 0.04267), ("methane", "CH4", 16.04, 2.283, 0.04278),
         ("argon", "Ar", 39.948, 1.355, 0.03201), ("oxygen", "O2", 31.998, 1.382, 0.03186)]


def gas_pressure_pa(n_mol: float, v_m3: float, t_k: float) -> float:
    return n_mol * R * t_k / v_m3


def _p_truth(pa: float) -> list[str]:
    return [f"{g(pa / 101325)} atm", f"{g(pa / 1000)} kPa", f"{g(pa / 1e5)} bar"]


def gas(rng: random.Random, kind: str) -> Item:
    name, formula, mm_, a, b = rng.choice(GASES)
    n = round(rng.uniform(0.5, 20), 2)
    v_l = round(rng.uniform(1, 50), 3)
    t_c = round(rng.uniform(-20, 300))
    nf, nform = rng.choice([(f"{n} mol", "mol"), (f"{n * 1000:g} mmol", "mmol")])
    vf, vform = rng.choice([(f"{v_l} dm3", "dm3"), (f"{v_l} L", "L"), (f"{v_l * 1000:g} mL", "mL"),
                            (f"{v_l / 1000:g} m3", "m3")])
    tf, tform = rng.choice([(f"{t_c}°C", "C"), (f"{t_c + 273.15:g} K", "K")])
    t_k, v_m3 = t_c + 273.15, v_l / 1000
    pa = gas_pressure_pa(n, v_m3, t_k)
    form = f"{nform},{vform},{tform}"
    if kind == "positive":
        q = rng.choice([
            f"Suppose that {nf} {formula}(g) is confined to {vf} at {tf}. Predict the pressure exerted by the {name} from the perfect gas equation.",
            f"What pressure is exerted by {nf} of {name} in a {vf} container at {tf}, treating it as a perfect gas?",
            f"A {vf} vessel holds {nf} of {name} gas at {tf}. Assuming ideal behaviour, calculate the pressure.",
            f"Using pV = nRT, find the pressure of {nf} of {formula} occupying {vf} at {tf}.",
        ])
        return Item("perfect_gas_pressure", "positive", form, q, _p_truth(pa))
    nm = rng.choice(["volume", "temperature", "density", "two_states", "van_der_waals", "mixture"])
    if nm == "volume":
        p_atm = round(pa / 101325, 2)
        v = n * R * t_k / (p_atm * 101325) * 1000
        q = f"{nf} of {name} at {tf} exerts a pressure of {p_atm} atm. Treating it as a perfect gas, what volume does it occupy?"
        return Item("perfect_gas_pressure", "near_miss", "different_target", q, [f"{g(v)} L"])
    if nm == "temperature":
        p_atm = round(pa / 101325, 2)
        t = p_atm * 101325 * v_m3 / (n * R)
        q = f"{nf} of {name} is confined to {vf} at a pressure of {p_atm} atm. Using the perfect gas law, what is its temperature?"
        return Item("perfect_gas_pressure", "near_miss", "different_target", q, [f"{g(t)} K", f"{g(t - 273.15)} °C"])
    if nm == "density":
        rho = n * mm_ / v_l
        q = f"{nf} of {name} is confined to {vf} at {tf}. Treating it as a perfect gas, what is its density in g/L?"
        return Item("perfect_gas_pressure", "near_miss", "different_target", q, [f"{g(rho)} g/L"])
    if nm == "two_states":
        t2 = t_c + rng.choice([50, 100, 150])
        p2 = gas_pressure_pa(n, v_m3, t2 + 273.15)
        q = f"{nf} of {name} is sealed in a rigid {vf} vessel at {tf} and then heated to {t2}°C. Treating it as a perfect gas, what is the pressure after heating?"
        return Item("perfect_gas_pressure", "near_miss", "process_two_states", q, _p_truth(p2))
    if nm == "van_der_waals":
        p_bar = n * 0.08314462618 * t_k / (v_l - n * b) - a * n ** 2 / v_l ** 2
        q = (f"Suppose that {nf} {formula}(g) is confined to {vf} at {tf}. Predict the pressure using the van der "
             f"Waals equation with a = {a} L² bar mol⁻² and b = {b} L mol⁻¹.")
        return Item("perfect_gas_pressure", "near_miss", "different_method", q, _p_truth(p_bar * 1e5))
    n2 = round(rng.uniform(0.5, 10), 2)
    other = rng.choice([x for x in GASES if x[0] != name])
    p = gas_pressure_pa(n + n2, v_m3, t_k)
    q = f"A {vf} vessel at {tf} holds {nf} of {name} and {n2} mol of {other[0]}. Treating both as perfect gases, what is the total pressure?"
    return Item("perfect_gas_pressure", "near_miss", "mixture", q, _p_truth(p))


ACIDS = [("acetic acid", "sodium acetate", 1.8e-5), ("formic acid", "sodium formate", 1.8e-4),
         ("hydrofluoric acid", "sodium fluoride", 6.8e-4), ("benzoic acid", "sodium benzoate", 6.3e-5),
         ("propanoic acid", "sodium propanoate", 1.3e-5)]


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup}"


def buffer(rng: random.Random, kind: str) -> Item:
    acid, salt, ka = rng.choice(ACIDS)
    a, b = round(rng.uniform(0.05, 0.5), 3), round(rng.uniform(0.05, 0.5), 3)
    x = round(rng.uniform(0.005, 0.8 * min(a, b)), 3)
    vol = rng.choice([250, 500, 750, 1000, 1500])
    af, aform = rng.choice([(f"{a} mol", "mol"), (f"{a * 1000:g} mmol", "mmol")])
    bf = f"{b} mol" if aform == "mol" else f"{b * 1000:g} mmol"
    xf = f"{x} mol" if aform == "mol" else f"{x * 1000:g} mmol"
    vf, vform = rng.choice([(f"{vol} mL", "mL"), (f"{vol / 1000:g} L", "L")])
    kf, kform = rng.choice([(f"Ka for {acid} is {sci(ka)}", "Ka"), (f"the pKa of {acid} is {-math.log10(ka):.2f}", "pKa")])
    pka = -math.log10(ka) if kform == "Ka" else round(-math.log10(ka), 2)
    form = f"{aform},{vform},{kform}"
    if kind == "positive":
        ph = pka + math.log10((b - x) / (a + x))
        q = rng.choice([
            f"A buffer is made by dissolving {af} {acid} and {bf} {salt} in water to make {vf} of solution. {kf[0].upper() + kf[1:]}. What is the pH after {xf} HCl is added?",
            f"{af} of {acid} and {bf} of {salt} are dissolved to make {vf} of buffer ({kf}). Find the pH after adding {xf} of HCl.",
        ])
        return Item("buffer_ph_after_strong_acid", "positive", form, q, [f"{ph:.3f}"])
    nm = rng.choice(["base_added", "before_addition"])
    if nm == "base_added":
        ph = pka + math.log10((b + x) / (a - x))
        q = f"A buffer is made by dissolving {af} {acid} and {bf} {salt} in water to make {vf} of solution. {kf[0].upper() + kf[1:]}. What is the pH after {xf} NaOH is added?"
        return Item("buffer_ph_after_strong_acid", "near_miss", "sign_change", q, [f"{ph:.3f}"])
    ph = pka + math.log10(b / a)
    q = f"A buffer is made by dissolving {af} {acid} and {bf} {salt} in water to make {vf} of solution. {kf[0].upper() + kf[1:]}. What is its pH?"
    return Item("buffer_ph_after_strong_acid", "near_miss", "missing_step", q, [f"{ph:.3f}"])


# ============================================================================ physics / engineering

def projectile_range(h: float, v: float, theta_deg: float, gg: float = 9.81) -> float:
    th = math.radians(theta_deg)
    vx, vy = v * math.cos(th), v * math.sin(th)
    t = (vy + math.sqrt(vy * vy + 2 * gg * h)) / gg
    return vx * t


def projectile(rng: random.Random, kind: str) -> Item:
    h_m = round(rng.uniform(5, 100), 1)
    v = round(rng.uniform(10, 40), 1)
    th = rng.choice(range(15, 71, 5))
    hf, hform = rng.choice([(f"{h_m} m", "m"), (f"{round(h_m / 0.3048)} ft", "ft")])
    h = h_m if hform == "m" else round(h_m / 0.3048) * 0.3048
    vf, vform = rng.choice([(f"{v} m/s", "m/s"), (f"{v * 3.6:g} km/h", "km/h")])
    form = f"{hform},{vform}"
    gtxt = rng.choice(["Taking g = 9.81 m/s²", "Using g = 9.81 m/s^2 and ignoring air resistance"])
    if kind == "positive":
        x = projectile_range(h, v, th)
        q = rng.choice([
            f"A ball is launched from the top of a {hf} cliff at {vf}, {th}° above the horizontal. {gtxt}, how far from the base of the cliff does it land?",
            f"From a {hf}-high platform, a stone is thrown at {vf} at {th}° above horizontal. {gtxt}, find the horizontal distance to where it hits the ground.",
            f"{gtxt}: a projectile leaves a {hf} cliff edge at {vf}, angled {th}° upward. What horizontal range does it reach at ground level?",
        ])
        return Item("projectile_range_from_height", "positive", form, q, [f"{g(x)} m"])
    thr = math.radians(th)
    vy = v * math.sin(thr)
    nm = rng.choice(["max_height", "time", "impact_speed"])
    if nm == "max_height":
        y = h + vy * vy / (2 * 9.81)
        q = f"A ball is launched from the top of a {hf} cliff at {vf}, {th}° above the horizontal. {gtxt}, what maximum height above the base of the cliff does it reach?"
        return Item("projectile_range_from_height", "near_miss", "different_target", q, [f"{g(y)} m"])
    t = (vy + math.sqrt(vy * vy + 2 * 9.81 * h)) / 9.81
    if nm == "time":
        q = f"A ball is launched from the top of a {hf} cliff at {vf}, {th}° above the horizontal. {gtxt}, how long is it in the air?"
        return Item("projectile_range_from_height", "near_miss", "different_target", q, [f"{g(t)} s"])
    s = math.sqrt(v * v + 2 * 9.81 * h)
    q = f"A ball is launched from the top of a {hf} cliff at {vf}, {th}° above the horizontal. {gtxt}, how fast is it moving when it hits the ground?"
    return Item("projectile_range_from_height", "near_miss", "different_target", q, [f"{g(s)} m/s"])


def beam(rng: random.Random, kind: str) -> Item:
    L = round(rng.uniform(2, 12), 1)
    w = round(rng.uniform(2, 40), 1)
    E, mat = rng.choice([(200, "steel"), (70, "aluminium"), (12, "timber")])
    I = round(rng.uniform(1, 50), 1) * 1e-5
    Lf, lform = rng.choice([(f"{L} m", "m"), (f"{L * 1000:g} mm", "mm")])
    wf, wform = rng.choice([(f"{w} kN/m", "kN/m"), (f"{w * 1000:g} N/m", "N/m")])
    If, iform = rng.choice([(f"I = {I * 1e5:g} × 10⁻⁵ m⁴", "m4"), (f"I = {I * 1e8:g} cm⁴", "cm4")])
    form = f"{lform},{wform},{iform}"
    EI = E * 1e9 * I
    if kind == "positive":
        d = 5 * w * 1000 * L ** 4 / (384 * EI)
        q = rng.choice([
            f"A simply supported {mat} beam spans {Lf} and carries a uniform load of {wf}. With E = {E} GPa and {If}, what is the maximum deflection?",
            f"Find the midspan deflection of a simply supported beam of span {Lf} under a uniformly distributed load of {wf} (E = {E} GPa, {If}).",
        ])
        return Item("ss_beam_udl_max_deflection", "positive", form, q, [f"{g(d * 1000)} mm"])
    nm = rng.choice(["moment", "cantilever", "quarter_span"])
    if nm == "moment":
        m = w * L * L / 8
        q = f"A simply supported {mat} beam spans {Lf} and carries a uniform load of {wf}. With E = {E} GPa and {If}, what is the maximum bending moment?"
        return Item("ss_beam_udl_max_deflection", "near_miss", "different_target", q, [f"{g(m)} kN·m"])
    if nm == "cantilever":
        d = w * 1000 * L ** 4 / (8 * EI)
        q = f"A {mat} cantilever of length {Lf} carries a uniform load of {wf}. With E = {E} GPa and {If}, what is the maximum deflection?"
        return Item("ss_beam_udl_max_deflection", "near_miss", "different_setup", q, [f"{g(d * 1000)} mm"])
    x = L / 4
    d = w * 1000 * x * (L ** 3 - 2 * L * x * x + x ** 3) / (24 * EI)
    q = f"A simply supported {mat} beam spans {Lf} and carries a uniform load of {wf}. With E = {E} GPa and {If}, what is the deflection at quarter span?"
    return Item("ss_beam_udl_max_deflection", "near_miss", "different_target", q, [f"{g(d * 1000)} mm"])


# ============================================================================ medicine / biology / probability

def bayes(rng: random.Random, kind: str) -> Item:
    sens, spec = round(rng.uniform(80, 99.5), 1), round(rng.uniform(85, 99.9), 1)
    pf, pform = rng.choice(["pct", "one_in"]), None
    if pf == "pct":
        p = round(rng.uniform(0.2, 10), 1) / 100
        ptxt, pform = f"{p * 100:g}% of a population", "pct"
    else:
        nn = rng.choice([50, 100, 200, 250, 500, 1000])
        p, ptxt, pform = 1 / nn, f"1 in {nn} people", "one_in"
    se, sp = sens / 100, spec / 100
    if kind == "positive":
        ppv = se * p / (se * p + (1 - sp) * (1 - p))
        q = rng.choice([
            f"A disease affects {ptxt}. A test for it has {sens}% sensitivity and {spec}% specificity. If a randomly chosen person tests positive, what is the probability they actually have the disease?",
            f"Prevalence is {ptxt}; the screening test's sensitivity is {sens}% and specificity {spec}%. What is the positive predictive value?",
        ])
        return Item("bayes_ppv", "positive", pform, q, [f"{ppv:.4f}", f"{ppv * 100:.2f}%"])
    nm = rng.choice(["npv", "p_positive", "two_tests"])
    if nm == "npv":
        v = sp * (1 - p) / (sp * (1 - p) + (1 - se) * p)
        q = f"A disease affects {ptxt}. A test for it has {sens}% sensitivity and {spec}% specificity. If a randomly chosen person tests negative, what is the probability they are actually disease-free?"
    elif nm == "p_positive":
        v = se * p + (1 - sp) * (1 - p)
        q = f"A disease affects {ptxt}. A test for it has {sens}% sensitivity and {spec}% specificity. What is the probability that a randomly chosen person tests positive?"
    else:
        p1 = se * p / (se * p + (1 - sp) * (1 - p))
        v = se * p1 / (se * p1 + (1 - sp) * (1 - p1))
        q = f"A disease affects {ptxt}. A test for it has {sens}% sensitivity and {spec}% specificity. A person tests positive twice on independent runs of the test. What is the probability they have the disease?"
    return Item("bayes_ppv", "near_miss", "different_target" if nm != "two_tests" else "extra_condition",
                q, [f"{v:.4f}", f"{v * 100:.2f}%"])


def hardy(rng: random.Random, kind: str) -> Item:
    n = rng.choice([500, 1000, 2500, 4000, 5000, 10000, 20000])
    q2 = rng.choice([0.0004, 0.0016, 0.0025, 0.0049, 0.01, 0.0196, 0.0256, 0.0324, 0.04, 0.0784])
    affected = round(q2 * n)
    q2 = affected / n
    disease = rng.choice(["cystic fibrosis", "phenylketonuria", "albinism", "sickle-cell disease"])
    af, aform = rng.choice([(f"{affected} have {disease}", "count"), (f"{q2 * 100:g}% have {disease}", "pct")])
    qa = math.sqrt(q2)
    p = 1 - qa
    if kind == "positive":
        carriers = 2 * p * qa * n
        q = rng.choice([
            f"In a population of {n:,} people, {af}, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be carriers?",
            f"{af[0].upper() + af[1:]} out of {n:,} people (autosomal recessive). Under Hardy-Weinberg equilibrium, how many heterozygous carriers are expected?",
        ])
        return Item("hardy_weinberg_carriers", "positive", aform, q, [f"{g(carriers, 4)} people"])
    nm = rng.choice(["homozygous_dominant", "allele_q", "carrier_among_unaffected"])
    if nm == "homozygous_dominant":
        v, unit = p * p * n, " people"
        tail = "how many people are expected to be homozygous for the normal allele?"
    elif nm == "allele_q":
        v, unit = qa, ""
        tail = "what is the frequency of the recessive allele?"
    else:
        v, unit = 2 * p * qa / (1 - q2), ""
        tail = "what is the probability that an unaffected person is a carrier?"
    q = f"In a population of {n:,} people, {af}, a recessive condition. Assuming Hardy-Weinberg equilibrium, {tail}"
    return Item("hardy_weinberg_carriers", "near_miss", "different_target", q, [f"{g(v, 4)}{unit}"])


def crcl(rng: random.Random, kind: str) -> Item:
    age = rng.randint(20, 90)
    w = round(rng.uniform(40, 120), 1)
    s = round(rng.uniform(0.5, 3.0), 2)
    female = rng.random() < 0.5
    who = "woman" if female else "man"
    wf, wform = rng.choice([(f"{w} kg", "kg"), (f"{round(w * 2.20462)} lb", "lb")])
    wk = w if wform == "kg" else round(w * 2.20462) / 2.20462
    sf, sform = rng.choice([(f"{s} mg/dL", "mg/dL"), (f"{round(s * 88.4)} µmol/L", "umol/L")])
    sm = s if sform == "mg/dL" else round(s * 88.4) / 88.4
    c = (140 - age) * wk / (72 * sm) * (0.85 if female else 1.0)
    form = f"{wform},{sform}"
    if kind == "positive":
        q = rng.choice([
            f"Using the Cockcroft-Gault equation, estimate the creatinine clearance of a {age}-year-old {who} who weighs {wf} and has a serum creatinine of {sf}.",
            f"A {age}-year-old {who} weighing {wf} has serum creatinine {sf}. What is the estimated creatinine clearance by Cockcroft-Gault?",
        ])
        return Item("cockcroft_gault_crcl", "positive", form, q, [f"{g(c)} mL/min"])
    q = f"Using the Cockcroft-Gault equation, estimate the creatinine clearance in mL/s of a {age}-year-old {who} who weighs {wf} and has a serum creatinine of {sf}."
    return Item("cockcroft_gault_crcl", "near_miss", "output_unit_change", q, [f"{g(c / 60)} mL/s"])


# ============================================================================ finance / accounting

def loan_payment(p: float, annual_rate_pct: float, years: int) -> tuple[float, float]:
    r, n = annual_rate_pct / 100 / 12, years * 12
    m = p * r / (1 - (1 + r) ** -n)
    return m, m * n - p


def loan(rng: random.Random, kind: str) -> Item:
    p = rng.choice(range(50, 901, 10)) * 1000
    rate = round(rng.uniform(2.5, 9), 2)
    years = rng.choice([10, 15, 20, 25, 30])
    pf, pform = money_forms(p, rng)
    m, i = loan_payment(p, rate, years)
    if kind == "positive":
        q = rng.choice([
            f"A {pf} mortgage has a {rate}% annual interest rate, compounded monthly, over {years} years. What is the monthly payment, and how much total interest is paid over the life of the loan?",
            f"You borrow {pf} for {years} years at {rate}% APR (monthly compounding). Find the monthly payment and the total interest paid.",
        ])
        return Item("loan_payment_total_interest", "positive", pform, q, [f"{m:.2f} and {i:.2f}"])
    nm = rng.choice(["balance", "total_paid", "interest_only"])
    if nm == "balance":
        k = rng.choice([y for y in (5, 10, 15) if y < years])
        r = rate / 100 / 12
        bal = p * (1 + r) ** (k * 12) - m * ((1 + r) ** (k * 12) - 1) / r
        q = f"A {pf} mortgage has a {rate}% annual interest rate, compounded monthly, over {years} years. What is the remaining balance after {k} years of payments?"
        return Item("loan_payment_total_interest", "near_miss", "different_target", q, [f"{bal:.2f}"])
    if nm == "total_paid":
        q = f"A {pf} mortgage has a {rate}% annual interest rate, compounded monthly, over {years} years. What is the total amount repaid over the life of the loan, principal included?"
        return Item("loan_payment_total_interest", "near_miss", "different_target", q, [f"{m * years * 12:.2f}"])
    q = f"A {pf} interest-only loan has a {rate}% annual interest rate, compounded monthly. What is the monthly payment?"
    return Item("loan_payment_total_interest", "near_miss", "different_setup", q, [f"{p * rate / 100 / 12:.2f}"])


def npv_value(cost: float, cf: float, n: int, salvage: float, r: float, due: bool = False) -> float:
    ann = cf * (1 - (1 + r) ** -n) / r * ((1 + r) if due else 1)
    return -cost + ann + salvage / (1 + r) ** n


def npv(rng: random.Random, kind: str) -> Item:
    cost = rng.choice(range(50, 2001, 10)) * 1000
    n = rng.randint(3, 10)
    cf = round(cost * rng.uniform(0.15, 0.4) / 1000) * 1000
    salvage = round(cost * rng.uniform(0, 0.2) / 1000) * 1000
    r = rng.choice([4, 5, 6, 7, 8, 9, 10, 12, 15])
    cf_txt, cform = money_forms(cost, rng)
    v = npv_value(cost, cf, n, salvage, r / 100)
    if kind == "positive":
        q = rng.choice([
            f"A project costs {cf_txt} upfront and returns {money(cf)} at the end of each year for {n} years, plus a {money(salvage)} salvage value at the end of year {n}. At a discount rate of {r}%, what is the NPV, and should the project be accepted?",
            f"An investment of {cf_txt} today yields {money(cf)} at the end of every year for {n} years and {money(salvage)} of salvage at the end. Using a {r}% discount rate, compute the NPV and give the accept/reject decision.",
        ])
        return Item("npv_annuity_salvage", "positive", cform, q, [f"{v:.2f}"])
    nm = rng.choice(["annuity_due", "payback", "profitability_index"])
    if nm == "annuity_due":
        v2 = npv_value(cost, cf, n, salvage, r / 100, due=True)
        q = f"A project costs {cf_txt} upfront and returns {money(cf)} at the START of each year for {n} years, plus a {money(salvage)} salvage value at the end of year {n}. At a discount rate of {r}%, what is the NPV?"
        return Item("npv_annuity_salvage", "near_miss", "timing_change", q, [f"{v2:.2f}"])
    if nm == "payback":
        q = f"A project costs {cf_txt} upfront and returns {money(cf)} at the end of each year for {n} years. What is its simple (undiscounted) payback period in years?"
        return Item("npv_annuity_salvage", "near_miss", "different_target", q, [f"{cost / cf:.3f} years"])
    pi = (v + cost) / cost
    q = f"A project costs {cf_txt} upfront and returns {money(cf)} at the end of each year for {n} years, plus a {money(salvage)} salvage value at the end of year {n}. At a discount rate of {r}%, what is its profitability index?"
    return Item("npv_annuity_salvage", "near_miss", "different_target", q, [f"{pi:.4f}"])


def bond_price(face: float, coupon_pct: float, years: int, y_pct: float, m: int = 2) -> float:
    c, y, n = face * coupon_pct / 100 / m, y_pct / 100 / m, years * m
    return c * (1 - (1 + y) ** -n) / y + face * (1 + y) ** -n


def bond(rng: random.Random, kind: str) -> Item:
    face = rng.choice([1000, 1000, 500, 100])
    cpn = rng.choice([2, 3, 4, 4.5, 5, 6, 7, 8])
    years = rng.choice([2, 5, 7, 10, 15, 20, 30])
    y = round(rng.uniform(1, 12), 1)
    if kind == "positive":
        p = bond_price(face, cpn, years, y)
        q = rng.choice([
            f"A {years}-year bond with a {money(face)} face value pays a {cpn}% annual coupon in semiannual installments. If the market yield is {y}% (annual, compounded semiannually), what is the bond's price?",
            f"Price a {money(face)}-par bond maturing in {years} years with a {cpn}% coupon paid twice a year, when the yield to maturity is {y}% (semiannual compounding).",
        ])
        return Item("bond_price_semiannual", "positive", "plain", q, [f"{p:.2f}"])
    nm = rng.choice(["annual", "current_yield"])
    if nm == "annual":
        p = bond_price(face, cpn, years, y, m=1)
        q = f"A {years}-year bond with a {money(face)} face value pays a {cpn}% coupon once a year. If the market yield is {y}% (annual compounding), what is the bond's price?"
        return Item("bond_price_semiannual", "near_miss", "regime_change", q, [f"{p:.2f}"])
    price = round(bond_price(face, cpn, years, y), 2)
    q = f"A {years}-year bond with a {money(face)} face value pays a {cpn}% annual coupon in semiannual installments and trades at {money(price, True)}. What is its current yield?"
    return Item("bond_price_semiannual", "near_miss", "different_target", q,
                [f"{face * cpn / price:.3f}%", f"{face * cpn / price / 100:.5f}"])


def ddb_book(cost: float, salvage: float, life: int, year: int, factor: float = 2.0) -> float:
    bv = cost
    for _ in range(year):
        bv = max(salvage, bv - bv * factor / life)
    return bv


def ddb(rng: random.Random, kind: str) -> Item:
    cost = rng.choice(range(10, 501, 5)) * 1000
    salvage = round(cost * rng.uniform(0.05, 0.2) / 100) * 100
    life = rng.randint(4, 10)
    k = rng.randint(1, life - 1)
    cf, cform = money_forms(cost, rng)
    what = rng.choice(["Equipment", "A machine", "A delivery van"])
    if kind == "positive":
        bv = ddb_book(cost, salvage, life, k)
        q = rng.choice([
            f"{what} costs {cf}, has a salvage value of {money(salvage)}, and a useful life of {life} years. Using double-declining-balance depreciation, what is its book value at the end of year {k}?",
            f"Using the 200% declining-balance method, find the book value after {k} years of an asset costing {cf} with a {life}-year life and {money(salvage)} salvage value.",
        ])
        return Item("ddb_book_value", "positive", cform, q, [f"{bv:.2f}"])
    nm = rng.choice(["expense", "straight_line", "150db"])
    if nm == "expense":
        v = ddb_book(cost, salvage, life, k - 1) - ddb_book(cost, salvage, life, k)
        q = f"{what} costs {cf}, has a salvage value of {money(salvage)}, and a useful life of {life} years. Using double-declining-balance depreciation, what is the depreciation expense in year {k}?"
        return Item("ddb_book_value", "near_miss", "different_target", q, [f"{v:.2f}"])
    if nm == "straight_line":
        v = cost - (cost - salvage) / life * k
        q = f"{what} costs {cf}, has a salvage value of {money(salvage)}, and a useful life of {life} years. Using straight-line depreciation, what is its book value at the end of year {k}?"
        return Item("ddb_book_value", "near_miss", "different_method", q, [f"{v:.2f}"])
    v = ddb_book(cost, salvage, life, k, factor=1.5)
    q = f"{what} costs {cf}, has a salvage value of {money(salvage)}, and a useful life of {life} years. Using 150% declining-balance depreciation, what is its book value at the end of year {k}?"
    return Item("ddb_book_value", "near_miss", "different_method", q, [f"{v:.2f}"])


# ============================================================================ math

def count_div(limit: int, a: int, b: int, c: int, mode: str = "or_not") -> int:
    ks = range(1, limit)
    if mode == "or_not":
        return sum(1 for k in ks if (k % a == 0 or k % b == 0) and k % c != 0)
    if mode == "and_not":
        return sum(1 for k in ks if k % a == 0 and k % b == 0 and k % c != 0)
    return sum(1 for k in ks if k % a == 0 or k % b == 0)


def counting(rng: random.Random, kind: str) -> Item:
    a, b = sorted(rng.sample([2, 3, 4, 5, 6, 7, 9, 11], 2))
    c = rng.choice([x for x in [7, 11, 13, 15, 17] if x not in (a, b)])
    limit = rng.choice([100, 200, 500, 1000, 2000, 5000])
    lf, lform = rng.choice([(f"less than {limit:,}", "less_than"), (f"from 1 to {limit - 1:,}", "range"),
                            (f"below {limit:,}", "below")])
    if kind == "positive":
        n = count_div(limit, a, b, c)
        q = rng.choice([
            f"How many positive integers {lf} are divisible by {a} or {b} but not by {c}?",
            f"Count the integers {lf} that are multiples of {a} or of {b}, excluding multiples of {c}.",
        ])
        return Item("count_divisible_or_not", "positive", lform, q, [str(n)])
    nm = rng.choice(["and", "no_exclusion", "boundary"])
    if nm == "and":
        q = f"How many positive integers {lf} are divisible by both {a} and {b} but not by {c}?"
        return Item("count_divisible_or_not", "near_miss", "logic_change", q, [str(count_div(limit, a, b, c, "and_not"))])
    if nm == "no_exclusion":
        q = f"How many positive integers {lf} are divisible by {a} or {b}?"
        return Item("count_divisible_or_not", "near_miss", "missing_step", q, [str(count_div(limit, a, b, c, "or"))])
    n = count_div(limit + 1, a, b, c)
    q = f"How many positive integers less than or equal to {limit:,} are divisible by {a} or {b} but not by {c}?"
    return Item("count_divisible_or_not", "near_miss", "boundary_change", q, [str(n)])


def seating(rng: random.Random, kind: str) -> Item:
    n = rng.randint(5, 12)
    f = math.factorial
    who = rng.choice(["A and B", "Alice and Bob", "the two brothers", "Priya and Sam"])
    if kind == "positive":
        v = f(n - 1) - 2 * f(n - 2)
        q = rng.choice([
            f"In how many ways can {n} people be seated around a round table if two particular people, {who}, refuse to sit next to each other?",
            f"{n} guests sit at a circular table. Rotations count as the same arrangement. In how many arrangements are {who} not adjacent?",
        ])
        return Item("circular_seating_not_adjacent", "positive", "plain", q, [f"{v:,}"])
    nm = rng.choice(["row", "together"])
    if nm == "row":
        v = f(n) - 2 * f(n - 1)
        q = f"In how many ways can {n} people be seated in a straight row if two particular people, {who}, refuse to sit next to each other?"
        return Item("circular_seating_not_adjacent", "near_miss", "different_setup", q, [f"{v:,}"])
    v = 2 * f(n - 2)
    q = f"In how many ways can {n} people be seated around a round table if two particular people, {who}, must sit next to each other?"
    return Item("circular_seating_not_adjacent", "near_miss", "logic_change", q, [f"{v:,}"])


GENERATORS: dict[str, Callable[[random.Random, str], Item]] = {
    "perfect_gas_pressure": gas, "buffer_ph_after_strong_acid": buffer,
    "projectile_range_from_height": projectile, "ss_beam_udl_max_deflection": beam,
    "bayes_ppv": bayes, "hardy_weinberg_carriers": hardy, "cockcroft_gault_crcl": crcl,
    "loan_payment_total_interest": loan, "npv_annuity_salvage": npv, "bond_price_semiannual": bond,
    "ddb_book_value": ddb, "count_divisible_or_not": counting, "circular_seating_not_adjacent": seating,
}
