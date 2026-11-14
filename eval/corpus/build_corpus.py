"""Builds the reasoning-cache eval corpus. Every answer is computed here, not typed by hand."""
import json, math, csv, random, itertools
from datetime import date, timedelta
from scipy import stats, optimize

rows = []

def sig(v, n=4):
    if v == 0:
        return "0"
    if abs(v - round(v)) < 1e-6 and abs(v) >= 1:
        v = round(v)
        return f"{int(v):,}"
    d = n - int(math.floor(math.log10(abs(v)))) - 1
    r = round(v, d)
    if d <= 0:
        return f"{int(r):,}"
    return f"{r:,.{d}f}"

def money(v, cur="$"):
    s = f"{abs(v):,.2f}"
    return f"-{cur}{s}" if v < 0 else f"{cur}{s}"

POS = {"original", "paraphrase", "reparam", "reparam_reworded", "hard_positive"}

def add(entry, domain, vtype, q, value=None, unit="", answer=None, category=None,
        either=False, note=None, hit=None):
    if hit is None:
        hit = vtype in POS
    if answer is None:
        answer = f"{sig(value)} {unit}".strip() if value is not None else ""
    rows.append(dict(
        seed_entry=entry, domain=domain, variant_type=vtype,
        near_miss_category=category, question=q, answer=answer,
        answer_value=None if value is None else round(float(value), 6),
        answer_unit=unit, should_hit=hit,
        expected_entry=entry if hit else None,
        acceptable_either=either, notes=note))

# =====================================================================
# 1. Perfect gas pressure
# =====================================================================
E, D = "perfect_gas_pressure", "chemistry"
R = 0.082057366  # L atm / (mol K)
K = lambda c: c + 273.15
gp = lambda n, V, T: n * R * T / V
p0 = gp(10.0, 4.860, K(27))

add(E, D, "original", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure exerted by the ethane from the perfect gas equation.", p0, "atm")
add(E, D, "paraphrase", "What pressure do 10 moles of ethane exert in a 4.86 litre container at 27 °C, treating it as a perfect gas?", p0, "atm")
add(E, D, "paraphrase", "10.0 mol of ethane gas (C2H6) is held in a 4.860 dm3 vessel at 27°C. Assuming ideal behaviour, calculate the pressure.", p0, "atm")
add(E, D, "paraphrase", "Treating ethane as an ideal gas, find the pressure when 10.0 mol of it occupies 4.860 dm3 at a temperature of 27 °C.", p0, "atm")
add(E, D, "reparam", "Suppose that 2.5 mol N2(g) is confined to 12.0 dm3 at 100°C. Predict the pressure exerted by the nitrogen from the perfect gas equation.", gp(2.5, 12.0, K(100)), "atm")
add(E, D, "reparam", "Suppose that 3.20 mol CO2(g) is confined to 15.0 dm3 at 45°C. Predict the pressure exerted by the carbon dioxide from the perfect gas equation.", gp(3.20, 15.0, K(45)), "atm")
add(E, D, "reparam", "Suppose that 0.750 mol O2(g) is confined to 2.00 dm3 at -10°C. Predict the pressure exerted by the oxygen from the perfect gas equation.", gp(0.750, 2.00, K(-10)), "atm")
add(E, D, "reparam_reworded", "A 1.00 L flask holds 250 mmol of CO2 at 300 K. Assuming perfect-gas behaviour, what pressure does the carbon dioxide exert?", gp(0.250, 1.00, 300), "atm", note="unit variant: mmol, L, K")
add(E, D, "reparam_reworded", "A 25.0 L steel cylinder contains 4.00 mol of argon at 350 K. Assuming ideal-gas behaviour, what pressure does the argon exert?", gp(4.00, 25.0, 350), "atm")
add(E, D, "reparam_reworded", "Calculate the pressure of 1.50 mol of methane in a 500 cm3 container at 0 °C, assuming it behaves as a perfect gas.", gp(1.50, 0.500, K(0)), "atm", note="unit variant: cm3")
add(E, D, "hard_positive", "In lab 3B, a technician records that 10.0 mol C2H6(g) sits in a 4.860 dm3 tank at 27°C. The tank was bought in 2019. Assuming perfect-gas behaviour, what pressure does the ethane exert?", p0, "atm", note="irrelevant numbers (3B, 2019): tests over-strict leftover checks")
add(E, D, "hard_positive", "Ten moles of ethane are confined to a 4.86-litre vessel at twenty-seven degrees Celsius. Treating the gas as ideal, what is its pressure?", p0, "atm", note="numbers written as words")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C and is then compressed isothermally to 2.430 dm3. Predict the new pressure from the perfect gas equation.", gp(10, 2.430, K(27)), "atm", category="process_two_states")
a_vdw, b_vdw = 5.507, 0.0651
vdw = lambda n, V, T: n*R*T/(V - n*b_vdw) - a_vdw*n**2/V**2
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure exerted by the ethane from the van der Waals equation, with a = 5.507 atm dm6 mol-2 and b = 0.0651 dm3 mol-1.", vdw(10, 4.86, K(27)), "atm", category="different_method")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure exerted by the ethane from the van der Waals equation.", vdw(10, 4.86, K(27)), "atm", category="different_method", note="constants not given; reference uses a = 5.507 atm dm6/mol2, b = 0.0651 dm3/mol (Atkins)")
add(E, D, "near_miss", "10.0 mol C2H6(g) confined to 4.860 dm3 exerts a pressure of 50.7 atm. Assuming perfect-gas behaviour, what is its temperature?", 50.7*4.86/(10*R), "K", category="different_target")
add(E, D, "near_miss", "10.0 mol of ethane at 27°C exerts a pressure of 50.7 atm. Treating it as a perfect gas, what volume does it occupy?", 10*R*K(27)/50.7, "dm3", category="different_target")
add(E, D, "near_miss", "A 4.860 dm3 vessel at 27°C contains ethane at a pressure of 50.7 atm. Treating it as a perfect gas, how many moles of ethane are present?", 50.7*4.86/(R*K(27)), "mol", category="different_target")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. By how much does the pressure increase when the gas is heated to 127°C at constant volume? Treat it as a perfect gas.", gp(10, 4.86, K(127)) - p0, "atm", category="change_in_quantity")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) and 5.00 mol CH4(g) are confined together to 4.860 dm3 at 27°C. Predict the total pressure from the perfect gas equation.", gp(15, 4.86, K(27)), "atm", category="mixture")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. What is the density of the gas in g/dm3?", 10*30.069/4.86, "g/dm3", category="different_target")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C, where its compressibility factor is Z = 0.85. Predict the pressure exerted by the ethane.", 0.85*p0, "atm", category="extra_condition")
add(E, D, "near_miss", "10.0 mol C2H6(g) and 5.00 mol CH4(g) are confined together to 4.860 dm3 at 27°C. Treating both as perfect gases, what is the partial pressure of the ethane?", p0, "atm", category="benign_extra_info", either=True, note="cached answer happens to be right; rejecting is also fine")
add(E, D, "near_miss", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure in kPa exerted by the ethane from the perfect gas equation.", p0*101.325, "kPa", category="output_unit_change", either=True, note="hit is fine only if the entry supports output units")
add(E, D, "underspecified", "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3. Predict the pressure exerted by the ethane from the perfect gas equation.", answer="Cannot be determined: temperature is not given.", category="missing_given")

# =====================================================================
# 2. Buffer pH after strong acid
# =====================================================================
E, D = "buffer_ph_after_strong_acid", "chemistry"
lg = math.log10
def buf(ha, a, ka, acid=0.0, base=0.0):
    return -lg(ka) + lg((a - acid + base) / (ha + acid - base))
b0 = buf(0.150, 0.100, 1.8e-5, acid=0.020)
add(E, D, "original", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH after 0.020 mol HCl is added?", b0, "pH")
add(E, D, "paraphrase", "You prepare 500 mL of buffer containing 0.150 mol CH3COOH and 0.100 mol CH3COONa (Ka = 1.8e-5). Find the pH once 0.020 mol of hydrochloric acid has been added.", b0, "pH")
add(E, D, "paraphrase", "After adding 0.020 mol HCl to a 500 mL buffer made from 0.150 mol acetic acid and 0.100 mol sodium acetate, what is the pH? Use Ka = 1.8 × 10^-5.", b0, "pH")
add(E, D, "paraphrase", "An acetic acid/acetate buffer (0.150 mol acid, 0.100 mol acetate, total volume 500 mL, Ka 1.8 × 10⁻⁵) receives 0.020 mol of HCl. Calculate the resulting pH.", b0, "pH")
add(E, D, "reparam", "A buffer is made by dissolving 0.200 mol acetic acid and 0.200 mol sodium acetate in water to make 1.00 L of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH after 0.050 mol HCl is added?", buf(0.2, 0.2, 1.8e-5, acid=0.05), "pH")
add(E, D, "reparam", "A buffer is made by dissolving 0.300 mol acetic acid and 0.250 mol sodium acetate in water to make 750 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH after 0.040 mol HCl is added?", buf(0.3, 0.25, 1.8e-5, acid=0.04), "pH")
add(E, D, "reparam", "A buffer is made by dissolving 0.080 mol formic acid and 0.120 mol sodium formate in water to make 250 mL of solution. Ka for formic acid is 1.8 × 10⁻⁴. What is the pH after 0.015 mol HCl is added?", buf(0.08, 0.12, 1.8e-4, acid=0.015), "pH")
add(E, D, "reparam_reworded", "A solution contains 0.250 mol NH3 and 0.180 mol NH4Cl in 1.00 L. The Ka of NH4+ is 5.6 × 10⁻¹⁰. Find the pH after 0.030 mol of HCl is added.", buf(0.180, 0.250, 5.6e-10, acid=0.030), "pH", note="weak acid is NH4+, conjugate base is NH3")
add(E, D, "reparam_reworded", "75.0 mmol of HF (Ka = 6.8 × 10⁻⁴) and 50.0 mmol of NaF are dissolved to make a buffer. What is its pH after 10.0 mmol of HCl is added?", buf(0.075, 0.050, 6.8e-4, acid=0.010), "pH", note="unit variant: mmol, volume not stated (cancels)")
add(E, D, "reparam_reworded", "Starting with a buffer of 150 mmol acetic acid and 100 mmol acetate ion (Ka 1.8×10^-5), how does the pH come out after you add 20 mmol of strong acid (HCl)?", b0, "pH", note="unit variant: mmol")
add(E, D, "hard_positive", "At 25 °C, a buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH after 0.020 mol HCl is added?", b0, "pH", note="irrelevant number (25 °C)")
add(E, D, "hard_positive", "A buffer contains 0.150 mol acetic acid and 0.100 mol sodium acetate in 500 mL (Ka = 1.8 × 10⁻⁵). A student adds 2.0 mL of concentrated HCl containing 0.020 mol HCl. What is the new pH?", b0, "pH", note="irrelevant number (2.0 mL)")
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH after 0.020 mol NaOH is added?", buf(0.15, 0.10, 1.8e-5, base=0.02), "pH", category="sign_change")
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH of the buffer?", buf(0.15, 0.10, 1.8e-5), "pH", category="missing_step")
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH after 0.120 mol HCl is added?", -lg(0.020/0.5), "pH", category="regime_change", note="HCl exceeds acetate; excess strong acid sets pH (volume change and HA dissociation ignored)")
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. By how much does the pH change when 0.020 mol HCl is added?", b0 - buf(0.15, 0.10, 1.8e-5), "pH units", category="change_in_quantity")
ratio = 10**(4.00 - (-lg(1.8e-5)))
xcap = (0.100 - ratio*0.150)/(1 + ratio)
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. How many moles of HCl can be added before the pH falls below 4.00?", xcap, "mol", category="inverse_problem")
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the acetate ion concentration after 0.020 mol HCl is added?", 0.080/0.5, "M", category="different_target")
x = optimize.brentq(lambda h: h*h/(0.3 - h) - 1.8e-5, 1e-9, 0.29)
add(E, D, "near_miss", "0.150 mol acetic acid is dissolved in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pH?", -lg(x), "pH", category="regime_change", note="no conjugate base: weak acid equilibrium, not a buffer")
add(E, D, "near_miss", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. Ka for acetic acid is 1.8 × 10⁻⁵. What is the pOH after 0.020 mol HCl is added?", 14 - b0, "pOH", category="different_target")
add(E, D, "underspecified", "A buffer is made by dissolving 0.150 mol acetic acid and 0.100 mol sodium acetate in water to make 500 mL of solution. What is the pH after 0.020 mol HCl is added?", b0, "pH", answer=f"{sig(b0)} pH, but only by assuming Ka = 1.8 × 10⁻⁵, which the question does not give", category="missing_given", note="tests whether the extractor invents a known constant")

# =====================================================================
# 3. Projectile from a cliff
# =====================================================================
E, D = "projectile_range_from_height", "physics"
def proj(h, v, deg, g=9.81):
    th = math.radians(deg); vx, vy = v*math.cos(th), v*math.sin(th)
    t = (vy + math.sqrt(vy*vy + 2*g*h))/g
    return vx*t
def tflight(h, v, deg, g=9.81):
    vy = v*math.sin(math.radians(deg)); return (vy + math.sqrt(vy*vy + 2*g*h))/g
x0 = proj(20, 25, 35)
add(E, D, "original", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", x0, "m")
add(E, D, "paraphrase", "From the edge of a cliff 20.0 m high, a ball is thrown at 25.0 m/s at an angle of 35° above horizontal. With g = 9.81 m/s², what horizontal distance does it cover before hitting the ground below?", x0, "m")
add(E, D, "paraphrase", "What is the horizontal range of a ball fired at 25.0 m/s and 35° elevation from a 20.0 m tall cliff top, landing on level ground at the cliff's base? Use g = 9.81 m/s².", x0, "m")
add(E, D, "paraphrase", "Ignoring air resistance (g = 9.81 m/s²), a projectile leaves a 20.0 m cliff with speed 25.0 m/s directed 35° upward from the horizontal. How far from the cliff does it strike the ground?", x0, "m")
add(E, D, "reparam", "A ball is launched from the top of a 45.0 m cliff at 18.0 m/s, 20° above the horizontal. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", proj(45, 18, 20), "m")
add(E, D, "reparam", "A ball is launched from the top of a 10.0 m cliff at 30.0 m/s, 50° above the horizontal. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", proj(10, 30, 50), "m")
add(E, D, "reparam", "A ball is launched from the top of a 20.0 m cliff on the Moon at 25.0 m/s, 35° above the horizontal. Taking g = 1.62 m/s², how far from the base of the cliff does it land?", proj(20, 25, 35, 1.62), "m")
add(E, D, "reparam_reworded", "A stone is thrown at 15 m/s at an angle of 40° above the horizontal from a bridge 12 m above a river. How far horizontally does it travel before hitting the water? Use g = 9.8 m/s².", proj(12, 15, 40, 9.8), "m")
add(E, D, "reparam_reworded", "From the roof of a 30.0 m building, a ball is kicked at 12.0 m/s, 25° above the horizontal. If g = 9.81 m/s², how far from the building does it hit the street?", proj(30, 12, 25), "m")
add(E, D, "reparam_reworded", "A golf ball leaves a 20-metre-high tee platform at 90 km/h, angled 35° upward. Neglecting drag and using g = 9.81 m/s², how far away (horizontally) does it land on the ground below?", x0, "m", note="unit variant: km/h")
add(E, D, "hard_positive", "A ball is thrown horizontally at 25.0 m/s from the top of a 20.0 m cliff. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", proj(20, 25, 0), "m", note="angle implied (0°), not written: grounding check must allow this or it false-misses")
add(E, D, "hard_positive", "A 0.45 kg ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s² and ignoring air resistance, how far from the base of the cliff does it land?", x0, "m", note="irrelevant mass")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², what is its maximum height above the ground below?", 20 + (25*math.sin(math.radians(35)))**2/(2*9.81), "m", category="different_target")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², how long is it in the air?", tflight(20, 25, 35), "s", category="different_target")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², with what speed does it hit the ground?", math.sqrt(25**2 + 2*9.81*20), "m/s", category="different_target")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° below the horizontal. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", proj(20, 25, -35), "m", category="sign_change", either=True, note="right only if the extractor passes -35")
add(E, D, "near_miss", "A ball is launched from level ground at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², how far away does it land?", 25**2*math.sin(math.radians(70))/9.81, "m", category="missing_step", either=True, note="h = 0 is implied; correct only if extractor writes 0")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Air resistance is significant, with a quadratic drag coefficient of 0.002 kg/m and a ball mass of 0.15 kg. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", answer="Requires numerical integration with the drag model; the drag-free answer (81.0 m) is an overestimate", category="different_method")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², how far has it travelled horizontally when it reaches the highest point of its path?", 25*math.cos(math.radians(35))*25*math.sin(math.radians(35))/9.81, "m", category="different_target")
best = optimize.minimize_scalar(lambda d: -proj(20, 25, d), bounds=(0, 89), method="bounded")
add(E, D, "near_miss", "A ball is launched from the top of a 20.0 m cliff at 25.0 m/s. Taking g = 9.81 m/s², at what launch angle does it land farthest from the base of the cliff?", best.x, "degrees", category="inverse_problem")
add(E, D, "underspecified", "A ball is launched from the top of a cliff at 25.0 m/s, 35° above the horizontal. Taking g = 9.81 m/s², how far from the base of the cliff does it land?", answer="Cannot be determined: cliff height is not given.", category="missing_given")

# =====================================================================
# 4. Beam deflection
# =====================================================================
E, D = "ss_beam_udl_max_deflection", "engineering"
def beam(w, L, E_gpa, I):  # w kN/m, L m, E GPa, I m^4 -> mm
    return 5*w*1e3*L**4/(384*E_gpa*1e9*I)*1000
d0 = beam(12, 6, 200, 8.0e-5)
add(E, D, "original", "A simply supported steel beam spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum deflection?", d0, "mm")
add(E, D, "paraphrase", "Find the midspan deflection of a 6.0 m simply supported steel beam under a uniformly distributed load of 12 kN/m, given E = 200 GPa and I = 8.0e-5 m^4.", d0, "mm")
add(E, D, "paraphrase", "A steel beam (E = 200 GPa, I = 8.0 × 10⁻⁵ m⁴) is pinned at both ends over a 6.0 m span and loaded with 12 kN/m along its whole length. How much does it sag at most?", d0, "mm")
add(E, D, "paraphrase", "Calculate the maximum vertical deflection for a simply supported beam: span 6.0 m, UDL 12 kN/m, Young's modulus 200 GPa, second moment of area 8.0 × 10⁻⁵ m⁴.", d0, "mm")
add(E, D, "reparam", "A simply supported steel beam spans 5.0 m and carries a uniform load of 8 kN/m. With E = 200 GPa and I = 5.0 × 10⁻⁵ m⁴, what is the maximum deflection?", beam(8, 5, 200, 5e-5), "mm")
add(E, D, "reparam", "A simply supported steel beam spans 8.0 m and carries a uniform load of 20 kN/m. With E = 200 GPa and I = 2.5 × 10⁻⁴ m⁴, what is the maximum deflection?", beam(20, 8, 200, 2.5e-4), "mm")
add(E, D, "reparam", "A simply supported aluminium beam spans 4.0 m and carries a uniform load of 3 kN/m. With E = 70 GPa and I = 1.2 × 10⁻⁵ m⁴, what is the maximum deflection?", beam(3, 4, 70, 1.2e-5), "mm")
add(E, D, "reparam_reworded", "A timber joist spans 4.2 m between simple supports and carries 2.5 kN/m uniformly. Take E = 11 GPa and I = 1.5 × 10⁸ mm⁴. What is the peak deflection?", beam(2.5, 4.2, 11, 1.5e-4), "mm", note="unit variant: mm^4")
add(E, D, "reparam_reworded", "How far does a 7.5 m glulam beam, simply supported, deflect at midspan under a uniform 6 kN/m load if E = 12 GPa and I = 9.0 × 10⁻⁴ m⁴?", beam(6, 7.5, 12, 9e-4), "mm")
add(E, D, "reparam_reworded", "A steel I-section with I = 8000 cm⁴ (E = 200 GPa) is simply supported over 6 m and carries a uniformly distributed 12 kN/m. Compute its maximum deflection.", d0, "mm", note="unit variant: cm^4")
add(E, D, "hard_positive", "A 300 mm deep simply supported steel beam spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum deflection?", d0, "mm", note="irrelevant depth")
add(E, D, "hard_positive", "A simply supported steel beam spans 6000 mm and carries a uniform load of 12 N/mm. With E = 200,000 MPa and I = 8.0 × 10⁷ mm⁴, what is the maximum deflection?", d0, "mm", note="all-mm unit system")
add(E, D, "near_miss", "A simply supported steel beam spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum bending moment?", 12*6**2/8, "kN·m", category="different_target")
add(E, D, "near_miss", "A steel cantilever beam 6.0 m long carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum deflection?", 12e3*6**4/(8*200e9*8e-5)*1000, "mm", category="different_setup")
add(E, D, "near_miss", "A simply supported steel beam spans 6.0 m and carries a single point load of 72 kN at midspan. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum deflection?", 72e3*6**3/(48*200e9*8e-5)*1000, "mm", category="different_setup")
add(E, D, "near_miss", "A simply supported steel beam spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum shear force?", 12*6/2, "kN", category="different_target")
xq = 1.5
add(E, D, "near_miss", "A simply supported steel beam spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the deflection 1.5 m from the left support?", 12e3*xq*(6**3 - 2*6*xq**2 + xq**3)/(24*200e9*8e-5)*1000, "mm", category="different_target")
add(E, D, "near_miss", "A steel beam fixed at both ends spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum deflection?", 12e3*6**4/(384*200e9*8e-5)*1000, "mm", category="different_setup")
add(E, D, "near_miss", "A simply supported steel beam 300 mm deep spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum bending stress?", (12*36/8)*1e3*0.15/8e-5/1e6, "MPa", category="different_target")
add(E, D, "near_miss", "A simply supported steel beam spans 6.0 m and carries a uniform imposed load of 12 kN/m plus its own self-weight of 0.6 kN/m. With E = 200 GPa and I = 8.0 × 10⁻⁵ m⁴, what is the maximum deflection?", beam(12.6, 6, 200, 8e-5), "mm", category="extra_condition", either=True, note="right only if the extractor sums the loads")
add(E, D, "underspecified", "A simply supported steel beam spans 6.0 m and carries a uniform load of 12 kN/m. With E = 200 GPa, what is the maximum deflection?", answer="Cannot be determined: second moment of area I is not given.", category="missing_given")

# =====================================================================
# 5. Inclusion-exclusion count
# =====================================================================
E, D = "count_divisible_or_not", "math"
cnt = lambda N, a, b, c: sum(1 for k in range(1, N) if (k % a == 0 or k % b == 0) and k % c != 0)
c0 = cnt(1000, 3, 5, 7)
add(E, D, "original", "How many positive integers less than 1000 are divisible by 3 or 5 but not by 7?", c0, "")
add(E, D, "paraphrase", "Count the positive integers below 1000 that are multiples of 3 or of 5, excluding any that are multiples of 7.", c0, "")
add(E, D, "paraphrase", "Among the integers from 1 to 999, how many are divisible by at least one of 3 and 5 while not being divisible by 7?", c0, "")
add(E, D, "paraphrase", "Find the number of positive whole numbers smaller than 1,000 that are divisible by 3 or 5 and not divisible by 7.", c0, "")
add(E, D, "reparam", "How many positive integers less than 500 are divisible by 2 or 7 but not by 5?", cnt(500, 2, 7, 5), "")
add(E, D, "reparam", "How many positive integers less than 2000 are divisible by 4 or 6 but not by 9?", cnt(2000, 4, 6, 9), "")
add(E, D, "reparam", "How many positive integers less than 100 are divisible by 3 or 5 but not by 7?", cnt(100, 3, 5, 7), "")
add(E, D, "reparam_reworded", "Among the integers 1 through 1,999, how many are multiples of 6 or 10 but not multiples of 15?", cnt(2000, 6, 10, 15), "", note="overlapping divisors (15 shares factors with 6 and 10)")
add(E, D, "reparam_reworded", "Count the positive whole numbers under 750 that are multiples of 4 or 9 yet not multiples of 11.", cnt(750, 4, 9, 11), "")
add(E, D, "reparam_reworded", "How many natural numbers less than ten thousand are divisible by either 7 or 11 but not by 13?", cnt(10000, 7, 11, 13), "", note="number written as words")
add(E, D, "hard_positive", "A school has 999 lockers numbered 1 to 999. Every locker whose number is a multiple of 3 or 5 is painted red, except that lockers whose number is a multiple of 7 are left unpainted. How many lockers are painted red?", c0, "", note="story framing")
add(E, D, "hard_positive", "How many positive integers less than 10³ are divisible by 3 or 5 but not by 7?", c0, "", note="10^3 notation")
add(E, D, "near_miss", "How many positive integers less than 1000 are divisible by 3 and 5 but not by 7?", sum(1 for k in range(1, 1000) if k % 15 == 0 and k % 7 != 0), "", category="logic_change")
add(E, D, "near_miss", "How many positive integers less than 1000 are divisible by 3 or 5?", sum(1 for k in range(1, 1000) if k % 3 == 0 or k % 5 == 0), "", category="missing_step")
add(E, D, "near_miss", "How many positive integers less than or equal to 1000 are divisible by 3 or 5 but not by 7?", cnt(1001, 3, 5, 7), "", category="boundary_change", note="1000 itself counts (divisible by 5)")
add(E, D, "near_miss", "What is the sum of all positive integers less than 1000 that are divisible by 3 or 5 but not by 7?", sum(k for k in range(1, 1000) if (k % 3 == 0 or k % 5 == 0) and k % 7 != 0), "", category="different_target")
add(E, D, "near_miss", "How many positive integers less than 1000 are divisible by exactly one of 3 and 5, and not by 7?", sum(1 for k in range(1, 1000) if ((k % 3 == 0) != (k % 5 == 0)) and k % 7 != 0), "", category="logic_change")
add(E, D, "near_miss", "How many positive integers less than 1000 are divisible by 3, 5, or 7?", sum(1 for k in range(1, 1000) if k % 3 == 0 or k % 5 == 0 or k % 7 == 0), "", category="logic_change")
add(E, D, "near_miss", "How many positive integers less than 1000 are divisible by none of 3, 5, and 7?", sum(1 for k in range(1, 1000) if k % 3 and k % 5 and k % 7), "", category="logic_change")
add(E, D, "near_miss", "How many three-digit positive integers are divisible by 3 or 5 but not by 7?", sum(1 for k in range(100, 1000) if (k % 3 == 0 or k % 5 == 0) and k % 7 != 0), "", category="extra_condition")
add(E, D, "near_miss", "How many positive integers less than 1000 are divisible by 3 or 5 and also divisible by 7?", sum(1 for k in range(1, 1000) if (k % 3 == 0 or k % 5 == 0) and k % 7 == 0), "", category="sign_change")
add(E, D, "underspecified", "How many positive integers are divisible by 3 or 5 but not by 7?", answer="Infinitely many: no upper bound is given.", category="missing_given")

# =====================================================================
# 6. Circular seating, two people not adjacent
# =====================================================================
E, D = "circular_seating_not_adjacent", "math"
F = math.factorial
circ = lambda n: F(n-1) - 2*F(n-2)
def brute_circ(n, ok):
    total = 0
    for perm in itertools.permutations(range(1, n)):
        seats = (0,) + perm
        if ok(seats, n):
            total += 1
    return total
adj = lambda s, n, a, b: abs(s.index(a) - s.index(b)) in (1, n-1)
add(E, D, "original", "In how many ways can 8 people be seated around a round table if two particular people, A and B, refuse to sit next to each other?", circ(8), "")
add(E, D, "paraphrase", "Eight people sit at a circular table. Two of them, A and B, will not sit side by side. How many seating arrangements are possible?", circ(8), "")
add(E, D, "paraphrase", "Count the circular arrangements of 8 people in which persons A and B are not adjacent.", circ(8), "")
add(E, D, "paraphrase", "How many distinct ways are there to seat 8 guests around a round table so that guests A and B are never neighbours? Rotations count as the same arrangement.", circ(8), "")
add(E, D, "reparam", "In how many ways can 6 people be seated around a round table if two particular people, A and B, refuse to sit next to each other?", circ(6), "")
add(E, D, "reparam", "In how many ways can 10 people be seated around a round table if two particular people, A and B, refuse to sit next to each other?", circ(10), "")
add(E, D, "reparam", "In how many ways can 12 people be seated around a round table if two particular people, A and B, refuse to sit next to each other?", circ(12), "")
add(E, D, "reparam_reworded", "Seven friends, including Priya and Tom who just broke up, sit around a circular campfire. How many seating arrangements keep Priya and Tom apart?", circ(7), "")
add(E, D, "reparam_reworded", "A round table seats 9 delegates. Two delegates from rival nations may not be placed side by side. How many seating plans are possible, treating rotations as identical?", circ(9), "")
add(E, D, "reparam_reworded", "Five guests sit around a round dinner table. In how many ways can they be seated so the two siblings among them are not neighbours?", circ(5), "")
add(E, D, "hard_positive", "A round table 2 metres across seats 8 people. In how many ways can they be seated if A and B refuse to sit next to each other?", circ(8), "", note="irrelevant number (2 metres)")
add(E, D, "hard_positive", "Eight knights sit at a round table. Sir A and Sir B have quarrelled and must not be adjacent. Up to rotation, how many arrangements are there?", circ(8), "", note="story framing")
add(E, D, "near_miss", "In how many ways can 8 people be seated in a row if two particular people, A and B, refuse to sit next to each other?", F(8) - 2*F(7), "", category="different_setup")
add(E, D, "near_miss", "In how many ways can 8 people be seated around a round table if two particular people, A and B, must sit next to each other?", 2*F(6), "", category="sign_change")
add(E, D, "near_miss", "In how many ways can 8 people be seated around a round table if two particular people, A and B, must sit directly opposite each other?", F(6), "", category="logic_change")
add(E, D, "near_miss", "In how many ways can 8 people be seated around a round table if three particular people, A, B and C, must not have any two of them sitting next to each other?", brute_circ(8, lambda s, n: not (adj(s, n, 0, 1) or adj(s, n, 0, 2) or adj(s, n, 1, 2))), "", category="extra_condition")
add(E, D, "near_miss", "In how many ways can 8 people be seated around a round table if A and B refuse to sit next to each other, and arrangements that are mirror images of each other count as the same?", circ(8)//2, "", category="extra_condition")
add(E, D, "near_miss", "In how many ways can 8 people be seated around a round table if A and B refuse to sit next to each other, and C and D also refuse to sit next to each other?", brute_circ(8, lambda s, n: not adj(s, n, 0, 1) and not adj(s, n, 2, 3)), "", category="extra_condition")
add(E, D, "near_miss", "A round table has 8 numbered seats. In how many ways can 8 people be assigned to the seats if A and B refuse to sit next to each other?", circ(8)*8, "", category="different_setup", note="numbered seats: rotations are distinct")
add(E, D, "near_miss", "In how many ways can 8 people be seated around a round table if exactly one person must sit between A and B?", 2*F(6), "", category="logic_change")
add(E, D, "underspecified", "In how many ways can people be seated around a round table if two particular people, A and B, refuse to sit next to each other?", answer="Cannot be determined: the number of people is not given.", category="missing_given")

# =====================================================================
# 7. Bayes positive predictive value
# =====================================================================
E, D = "bayes_ppv", "probability"
ppv = lambda pr, se, sp: se*pr/(se*pr + (1-sp)*(1-pr))
v0 = ppv(0.015, 0.92, 0.96)
add(E, D, "original", "A disease affects 1.5% of a population. A test for it has 92% sensitivity and 96% specificity. If a randomly chosen person tests positive, what is the probability they actually have the disease?", v0, "")
add(E, D, "paraphrase", "With a prevalence of 1.5%, a sensitivity of 92% and a specificity of 96%, what is the chance that someone who tests positive truly has the disease?", v0, "")
add(E, D, "paraphrase", "1.5% of people have a certain illness. The diagnostic test catches 92% of true cases and correctly clears 96% of healthy people. Given a positive result, what is P(disease)?", v0, "")
add(E, D, "paraphrase", "Compute the positive predictive value of a test with 92% sensitivity and 96% specificity in a population where 1.5% have the condition.", v0, "")
add(E, D, "reparam", "A disease affects 0.5% of a population. A test for it has 99% sensitivity and 95% specificity. If a randomly chosen person tests positive, what is the probability they actually have the disease?", ppv(0.005, 0.99, 0.95), "")
add(E, D, "reparam", "A disease affects 10% of a population. A test for it has 85% sensitivity and 90% specificity. If a randomly chosen person tests positive, what is the probability they actually have the disease?", ppv(0.10, 0.85, 0.90), "")
add(E, D, "reparam", "A disease affects 2% of a population. A test for it has 95% sensitivity and 98% specificity. If a randomly chosen person tests positive, what is the probability they actually have the disease?", ppv(0.02, 0.95, 0.98), "")
add(E, D, "reparam_reworded", "1 in 200 people carry a gene variant. A screening test detects it 97% of the time when present and gives a false positive 3% of the time when absent. Given a positive screen, what is the chance the person carries the variant?", ppv(0.005, 0.97, 0.97), "", note="'1 in 200' and false-positive rate need conversion")
add(E, D, "reparam_reworded", "In a factory, 4% of widgets are defective. An inspection machine flags 90% of defective widgets and wrongly flags 5% of good ones. If a widget is flagged, what is the probability it is actually defective?", ppv(0.04, 0.90, 0.95), "", note="domain shift, same reasoning")
add(E, D, "reparam_reworded", "30% of incoming emails are spam. The filter catches 95% of spam and misflags 2% of legitimate emails. If an email is flagged, what is the probability it is spam?", ppv(0.30, 0.95, 0.98), "", note="domain shift, same reasoning")
add(E, D, "hard_positive", "Out of 10,000 people, 150 have a disease. A test correctly identifies 92% of those who have it and correctly clears 96% of those who don't. If someone tests positive, what is the probability they have the disease?", v0, "", note="frequency format; 10,000 and 150 must be turned into a rate")
add(E, D, "hard_positive", "A disease affects 1.5% of a population. A test for it costs $45 and has 92% sensitivity and 96% specificity. If a randomly chosen person tests positive, what is the probability they actually have the disease?", v0, "", note="irrelevant number ($45)")
npv_ = lambda pr, se, sp: sp*(1-pr)/(sp*(1-pr) + (1-se)*pr)
add(E, D, "near_miss", "A disease affects 1.5% of a population. A test for it has 92% sensitivity and 96% specificity. If a randomly chosen person tests negative, what is the probability they are truly disease-free?", npv_(0.015, 0.92, 0.96), "", category="different_target")
add(E, D, "near_miss", "A disease affects 1.5% of a population. A test for it has 92% sensitivity and 96% specificity. What is the probability that a randomly chosen person tests positive?", 0.92*0.015 + 0.04*0.985, "", category="different_target")
add(E, D, "near_miss", "A disease affects 1.5% of a population. A test for it has 92% sensitivity and 96% specificity. If a person tests positive twice on independent tests, what is the probability they have the disease?", ppv(v0, 0.92, 0.96), "", category="process_two_states")
add(E, D, "near_miss", "A disease affects 1.5% of a population of 10,000. A test for it has 92% sensitivity and 96% specificity. How many false positives would you expect if everyone is tested?", 0.04*0.985*10000, "people", category="different_target")
add(E, D, "near_miss", "A disease affects 1.5% of a population. A test for it has 92% sensitivity and 96% specificity. If a randomly chosen person tests negative, what is the probability they actually have the disease?", 1 - npv_(0.015, 0.92, 0.96), "", category="different_target")
add(E, D, "near_miss", "A disease affects 1.5% of a population. A test for it has 92% sensitivity and 96% specificity. If a randomly chosen person tests positive, what is the probability they do not have the disease?", 1 - v0, "", category="sign_change")
add(E, D, "near_miss", "A test for a disease has 92% sensitivity and 96% specificity. What is the positive likelihood ratio of the test?", 0.92/0.04, "", category="different_target")
add(E, D, "near_miss", "A disease affects 1.5% of a population. A test has 92% sensitivity and 96% specificity, and a person tests positive. What are the posterior odds that they have the disease?", v0/(1 - v0), "", category="different_target")
add(E, D, "underspecified", "A disease affects 1.5% of a population. A test for it has 92% sensitivity. If a randomly chosen person tests positive, what is the probability they actually have the disease?", answer="Cannot be determined: specificity (or false-positive rate) is not given.", category="missing_given")

# =====================================================================
# 8. One-sample t-test
# =====================================================================
E, D = "one_sample_t_test", "statistics"
def tt(xbar, mu, s, n, alpha=0.05, tails=2):
    t = (xbar - mu)/(s/math.sqrt(n)); df = n - 1
    p = 2*stats.t.sf(abs(t), df) if tails == 2 else stats.t.sf(t, df)
    dec = "reject H0" if p < alpha else "fail to reject H0"
    return t, f"t = {t:.3f} (df = {df}), p = {p:.4f}; {dec} at α = {alpha}"
def addt(vt, q, *a, **k):
    t, txt = tt(*a, **k); add(E, D, vt, q, t, "t", answer=txt)
addt("original", "A sample of 36 bolts has a mean length of 50.4 mm and a sample standard deviation of 1.2 mm. At α = 0.05, test whether the true mean length differs from 50 mm. Give the test statistic and your conclusion.", 50.4, 50, 1.2, 36)
addt("paraphrase", "36 bolts were measured: mean 50.4 mm, sample SD 1.2 mm. Using a two-sided test at the 5% level, is there evidence the population mean is not 50 mm? Report t and the decision.", 50.4, 50, 1.2, 36)
addt("paraphrase", "Test H0: μ = 50 mm against H1: μ ≠ 50 mm for bolt lengths, given n = 36, sample mean 50.4 mm and s = 1.2 mm, at significance level 0.05.", 50.4, 50, 1.2, 36)
addt("paraphrase", "Does a sample of 36 bolts with average length 50.4 mm (standard deviation 1.2 mm) show that the true mean differs from the 50 mm target? Use α = 0.05 and give the t statistic.", 50.4, 50, 1.2, 36)
addt("reparam", "A sample of 25 bolts has a mean length of 12.6 mm and a sample standard deviation of 0.9 mm. At α = 0.05, test whether the true mean length differs from 12 mm. Give the test statistic and your conclusion.", 12.6, 12, 0.9, 25)
addt("reparam", "A sample of 50 bolts has a mean length of 101.2 mm and a sample standard deviation of 4.5 mm. At α = 0.05, test whether the true mean length differs from 100 mm. Give the test statistic and your conclusion.", 101.2, 100, 4.5, 50)
addt("reparam", "A sample of 16 bolts has a mean length of 7.85 mm and a sample standard deviation of 0.4 mm. At α = 0.05, test whether the true mean length differs from 8 mm. Give the test statistic and your conclusion.", 7.85, 8, 0.4, 16)
addt("reparam_reworded", "A coffee machine is supposed to dispense 250 mL. In 20 test cups the mean was 246.8 mL with a standard deviation of 6.1 mL. At the 5% level, is there evidence that the average differs from 250 mL?", 246.8, 250, 6.1, 20)
addt("reparam_reworded", "Researchers measured the resting heart rates of 40 athletes: mean 58.3 bpm, standard deviation 7.9 bpm. Test at α = 0.01 whether the mean differs from 60 bpm (two-sided).", 58.3, 60, 7.9, 40, alpha=0.01)
addt("reparam_reworded", "A sample of 64 batteries lasted an average of 19.6 hours (s = 2.4 h). Is the mean lifetime different from 20 hours at a 5% significance level?", 19.6, 20, 2.4, 64)
t, txt = tt(50.4, 50, 1.2, 36)
add(E, D, "hard_positive", "Machine 3 produced a sample of 36 bolts with a mean length of 50.4 mm and a sample standard deviation of 1.2 mm. At α = 0.05, test whether the true mean length differs from 50 mm.", t, "t", answer=txt, note="irrelevant number (Machine 3)")
add(E, D, "hard_positive", "n = 36, x̄ = 50.4 mm, s = 1.2 mm, H0: μ = 50 mm vs H1: μ ≠ 50 mm, α = 0.05. Perform the test.", t, "t", answer=txt, note="terse notation")
t1, txt1 = tt(50.4, 50, 1.2, 36, tails=1)
add(E, D, "near_miss", "A sample of 36 bolts has a mean length of 50.4 mm and a sample standard deviation of 1.2 mm. At α = 0.05, test whether the true mean length is greater than 50 mm. Give the test statistic and your conclusion.", t1, "t", answer=txt1, category="logic_change", note="conclusion flips to reject")
pz = 2*stats.norm.sf(2.0)
add(E, D, "near_miss", "A sample of 36 bolts has a mean length of 50.4 mm. The population standard deviation is known to be 1.2 mm. At α = 0.05, test whether the true mean length differs from 50 mm. Give the test statistic and your conclusion.", 2.0, "z", answer=f"z = 2.000, p = {pz:.4f}; reject H0 at α = 0.05", category="different_method", note="z-test: conclusion flips")
tc = stats.t.ppf(0.975, 35)
add(E, D, "near_miss", "A sample of 36 bolts has a mean length of 50.4 mm and a sample standard deviation of 1.2 mm. Construct a 95% confidence interval for the true mean length.", None, "mm", answer=f"({50.4 - tc*0.2:.3f}, {50.4 + tc*0.2:.3f}) mm", category="different_target")
w = stats.ttest_ind_from_stats(50.4, 1.2, 36, 50.1, 1.0, 40, equal_var=False)
add(E, D, "near_miss", "A sample of 36 bolts from Machine A has a mean length of 50.4 mm (s = 1.2 mm), and a sample of 40 bolts from Machine B has a mean of 50.1 mm (s = 1.0 mm). At α = 0.05, test whether the two machines' mean lengths differ.", w.statistic, "t", answer=f"Welch t = {w.statistic:.3f}, p = {w.pvalue:.4f}; fail to reject H0", category="different_method")
chi = 35*1.44/1.0; pchi = 2*min(stats.chi2.sf(chi, 35), stats.chi2.cdf(chi, 35))
add(E, D, "near_miss", "A sample of 36 bolts has a mean length of 50.4 mm and a sample standard deviation of 1.2 mm. At α = 0.05, test whether the population variance differs from 1.0 mm².", chi, "chi-square", answer=f"χ² = {chi:.2f} (df = 35), p = {pchi:.4f}; fail to reject H0", category="different_method")
add(E, D, "near_miss", "Bolt lengths have a standard deviation of about 1.2 mm. How many bolts must be sampled so that a 95% confidence interval for the mean has a margin of error of at most 0.2 mm?", math.ceil((1.96*1.2/0.2)**2), "bolts", category="inverse_problem")
zp = (8/36 - 0.15)/math.sqrt(0.15*0.85/36); pp = 2*stats.norm.sf(abs(zp))
add(E, D, "near_miss", "In a sample of 36 bolts, 8 were defective. At α = 0.05, test whether the true defect rate differs from 15%.", zp, "z", answer=f"z = {zp:.3f}, p = {pp:.4f}; fail to reject H0", category="different_method")
add(E, D, "near_miss", "A sample of 36 bolts has a mean length of 50.4 mm and a sample standard deviation of 1.2 mm. What is the standard error of the mean?", 0.2, "mm", category="different_target")
add(E, D, "underspecified", "A sample of 36 bolts has a mean length of 50.4 mm. At α = 0.05, test whether the true mean length differs from 50 mm.", answer="Cannot be completed: no standard deviation is given.", category="missing_given")

# =====================================================================
# 9. Mortgage payment and total interest
# =====================================================================
E, D = "loan_payment_total_interest", "finance"
def pmt(P, rate, years, per=12):
    r = rate/per; n = years*per
    return P*r/(1 - (1+r)**-n)
def addm(vt, q, P, rate, yrs, cur="$", note=None):
    m = pmt(P, rate, yrs)
    add(E, D, vt, q, m, cur, answer=f"Payment {money(m, cur)}/month; total interest {money(m*yrs*12 - P, cur)}", note=note)
addm("original", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly, over 30 years. What is the monthly payment, and how much total interest is paid over the life of the loan?", 350000, 0.0625, 30)
addm("paraphrase", "What are the monthly payments on a 30-year, $350,000 home loan at 6.25% per year (monthly compounding), and what is the total interest over the full term?", 350000, 0.0625, 30)
addm("paraphrase", "Calculate the monthly installment and lifetime interest for a $350k mortgage at 6.25% annual interest compounded monthly, repaid over 30 years.", 350000, 0.0625, 30)
addm("paraphrase", "I'm taking out $350,000 at 6.25% for 30 years, paid monthly. How much will I pay each month and how much of that ends up being interest overall?", 350000, 0.0625, 30)
addm("reparam", "A $250,000 mortgage has a 5.5% annual interest rate, compounded monthly, over 15 years. What is the monthly payment, and how much total interest is paid over the life of the loan?", 250000, 0.055, 15)
addm("reparam", "A $500,000 mortgage has a 7.0% annual interest rate, compounded monthly, over 30 years. What is the monthly payment, and how much total interest is paid over the life of the loan?", 500000, 0.07, 30)
addm("reparam", "A $180,000 mortgage has a 4.75% annual interest rate, compounded monthly, over 20 years. What is the monthly payment, and how much total interest is paid over the life of the loan?", 180000, 0.0475, 20)
addm("reparam_reworded", "I'm borrowing $425k for a house at 6.875% APR for 30 years with monthly payments. What will my payment be, and how much interest will I pay in total?", 425000, 0.06875, 30)
addm("reparam_reworded", "A car loan of $32,000 at 8.9% annual interest, compounded monthly, is repaid over 5 years. Find the monthly payment and the total interest.", 32000, 0.089, 5, note="domain shift (car loan)")
addm("reparam_reworded", "What is the monthly instalment on a 25-year, £275,000 loan at 5.1% per year compounded monthly, and what is the total interest?", 275000, 0.051, 25, cur="£", note="currency variant")
m0 = pmt(350000, 0.0625, 30)
ans0 = f"Payment {money(m0)}/month; total interest {money(m0*360 - 350000)}"
add(E, D, "hard_positive", "A house costs $437,500 and the buyer puts 20% down, borrowing $350,000 at a 6.25% annual rate compounded monthly over 30 years. What is the monthly payment, and how much total interest is paid?", m0, "$", answer=ans0, note="irrelevant numbers (437,500, 20%)")
add(E, D, "hard_positive", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly, over 30 years. Property taxes of $4,200 per year are paid separately. What is the monthly loan payment, and how much total interest is paid?", m0, "$", answer=ans0, note="irrelevant number ($4,200)")
r = 0.0625/12
bal5 = 350000*(1+r)**60 - m0*((1+r)**60 - 1)/r
add(E, D, "near_miss", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly, over 30 years. What is the remaining balance after 5 years of payments?", bal5, "$", answer=money(bal5), category="different_target")
add(E, D, "near_miss", "A $350,000 interest-only mortgage has a 6.25% annual interest rate, compounded monthly. What is the monthly payment during the interest-only period?", 350000*r, "$", answer=money(350000*r), category="different_setup")
bal, months, tot = 350000.0, 0, 0.0
while bal > 0:
    i = bal*r; pay = min(m0 + 500, bal + i); bal = bal + i - pay; tot += i; months += 1
add(E, D, "near_miss", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly, over 30 years. If the borrower pays an extra $500 every month, how much total interest is paid and how long does payoff take?", tot, "$", answer=f"Total interest {money(tot)}; paid off in {months} months ({months/12:.1f} years)", category="extra_condition")
Pmax = 2500*(1 - (1+r)**-360)/r
add(E, D, "near_miss", "At a 6.25% annual interest rate, compounded monthly, over 30 years, what is the largest mortgage you can take on if you can afford $2,500 per month?", Pmax, "$", answer=money(Pmax), category="inverse_problem")
bal, int1 = 350000.0, 0.0
for _ in range(12):
    i = bal*r; int1 += i; bal = bal + i - m0
add(E, D, "near_miss", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly, over 30 years. How much interest is paid in the first year?", int1, "$", answer=money(int1), category="different_target")
re = 1.0625**(1/12) - 1; ma = 350000*re/(1 - (1+re)**-360)
add(E, D, "near_miss", "A $350,000 mortgage has a 6.25% annual interest rate, compounded annually, with monthly payments over 30 years. What is the monthly payment, and how much total interest is paid?", ma, "$", answer=f"Payment {money(ma)}/month; total interest {money(ma*360 - 350000)}", category="extra_condition", note="compounding frequency differs from payment frequency")
mb = pmt(350000, 0.0625, 30, per=26)
add(E, D, "near_miss", "A $350,000 mortgage has a 6.25% annual interest rate over 30 years, with payments made every two weeks (26 per year) and interest compounded at each payment. What is each biweekly payment?", mb, "$", answer=money(mb), category="different_setup")
apr = optimize.brentq(lambda a: pmt(343000, a, 30) - m0, 0.01, 0.2)
add(E, D, "near_miss", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly, over 30 years, but the lender charges 2 points ($7,000) upfront. What is the effective APR?", apr*100, "%", category="different_target")
add(E, D, "underspecified", "A $350,000 mortgage has a 6.25% annual interest rate, compounded monthly. What is the monthly payment, and how much total interest is paid over the life of the loan?", answer="Cannot be determined: loan term is not given.", category="missing_given")

# =====================================================================
# 10. NPV with salvage
# =====================================================================
E, D = "npv_annuity_salvage", "finance"
def npv(c0, cf, n, salv, r):
    return -c0 + sum(cf/(1+r)**t for t in range(1, n+1)) + salv/(1+r)**n
def addn(vt, q, c0, cf, n, salv, r, cur="$", note=None):
    v = npv(c0, cf, n, salv, r)
    add(E, D, vt, q, v, cur, answer=f"NPV = {money(v, cur)}; {'accept' if v > 0 else 'reject'}", note=note)
addn("original", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is the NPV, and should the project be accepted?", 120000, 35000, 5, 10000, 0.08)
addn("paraphrase", "With an 8% discount rate, compute the net present value of a project that needs $120,000 today and generates $35,000 per year for five years, with $10,000 salvage at the end. Accept or reject?", 120000, 35000, 5, 10000, 0.08)
addn("paraphrase", "Initial outlay $120,000; annual year-end inflows of $35,000 for 5 years; salvage $10,000 in year 5; cost of capital 8%. Find the NPV and give the investment decision.", 120000, 35000, 5, 10000, 0.08)
addn("paraphrase", "Should a firm invest $120,000 now in a project paying $35,000 at each year-end for 5 years and $10,000 salvage at the end of year 5, if its required return is 8%? Show the NPV.", 120000, 35000, 5, 10000, 0.08)
addn("reparam", "A project costs $250,000 upfront and returns $60,000 at the end of each year for 6 years, plus a $20,000 salvage value at the end of year 6. At a discount rate of 10%, what is the NPV, and should the project be accepted?", 250000, 60000, 6, 20000, 0.10)
addn("reparam", "A project costs $50,000 upfront and returns $12,000 at the end of each year for 4 years, plus a $0 salvage value at the end of year 4. At a discount rate of 9%, what is the NPV, and should the project be accepted?", 50000, 12000, 4, 0, 0.09)
addn("reparam", "A project costs $80,000 upfront and returns $22,000 at the end of each year for 5 years, plus a $5,000 salvage value at the end of year 5. At a discount rate of 12%, what is the NPV, and should the project be accepted?", 80000, 22000, 5, 5000, 0.12)
addn("reparam_reworded", "Our firm is considering a machine costing $95,000. It saves $28,000 per year for 4 years and can be sold for $15,000 at the end. With a 10% cost of capital, what's the NPV? Go or no-go?", 95000, 28000, 4, 15000, 0.10)
addn("reparam_reworded", "Invest €2.0 million today to receive €450,000 at the end of each year for 6 years, plus €300,000 residual value at the end of year 6. The required return is 9%. What is the NPV, and is it worth doing?", 2000000, 450000, 6, 300000, 0.09, cur="€", note="currency variant, millions")
addn("reparam_reworded", "A food truck costs $60k. Expected net cash flow is $18k per year for 5 years, after which the truck is sold for $8k. If the owner's hurdle rate is 11%, does the investment add value? Compute the NPV.", 60000, 18000, 5, 8000, 0.11)
v = npv(120000, 35000, 5, 10000, 0.08); a0 = f"NPV = {money(v)}; accept"
add(E, D, "hard_positive", "Proposed by the marketing team in 2025, a project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is the NPV, and should the project be accepted?", v, "$", answer=a0, note="irrelevant number (2025)")
add(E, D, "hard_positive", "A project costs $120,000 upfront and produces after-tax cash flows of $35,000 at the end of each year for 5 years (the firm's tax rate is 25%), plus a $10,000 after-tax salvage value in year 5. At 8%, what is the NPV and the decision?", v, "$", answer=a0, note="irrelevant number (25% tax): flows are already after-tax")
cfs = [-120000] + [35000]*4 + [45000]
irr = optimize.brentq(lambda k: sum(c/(1+k)**t for t, c in enumerate(cfs)), 0.0, 1.0)
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. What is the project's IRR?", irr*100, "%", category="different_target")
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. What is the simple payback period?", 120000/35000, "years", category="different_target")
pi = (v + 120000)/120000
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is the profitability index?", pi, "", category="different_target")
vd = -120000 + sum(35000/1.08**t for t in range(0, 5)) + 10000/1.08**5
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the beginning of each year for 5 years, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is the NPV?", vd, "$", answer=money(vd), category="timing_change")
vg = -120000 + sum(35000*1.03**(t-1)/1.08**t for t in range(1, 6)) + 10000/1.08**5
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of year 1, growing by 3% per year for 5 years in total, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is the NPV?", vg, "$", answer=money(vg), category="extra_condition")
cum, dp = -120000.0, None
for t in range(1, 6):
    dcf = (35000 + (10000 if t == 5 else 0))/1.08**t
    if cum + dcf >= 0 and dp is None:
        dp = t - 1 + (-cum)/dcf
    cum += dcf
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is the discounted payback period?", dp, "years", category="different_target")
af = (1 - 1.08**-5)/0.08
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. At a discount rate of 8%, what is its equivalent annual annuity?", v/af, "$", answer=money(v/af), category="different_target")
disc = [1/1.08, 1/1.08**2, 1/1.08**3, 1/(1.08**3*1.10), 1/(1.08**3*1.10**2)]
v2 = -120000 + sum(35000*d for d in disc) + 10000*disc[-1]
add(E, D, "near_miss", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. The discount rate is 8% for years 1-3 and 10% for years 4-5. What is the NPV?", v2, "$", answer=money(v2), category="extra_condition")
add(E, D, "underspecified", "A project costs $120,000 upfront and returns $35,000 at the end of each year for 5 years, plus a $10,000 salvage value at the end of year 5. What is the NPV, and should the project be accepted?", answer="Cannot be determined: no discount rate is given.", category="missing_given")

# =====================================================================
# 11. Bond price (semiannual)
# =====================================================================
E, D = "bond_price_semiannual", "finance"
def bp(F, c, yrs, y, f=2):
    n = int(round(yrs*f)); cp = F*c/f; yy = y/f
    return cp*(1 - (1+yy)**-n)/yy + F*(1+yy)**-n
b0 = bp(1000, 0.05, 10, 0.062)
add(E, D, "original", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments. If the market yield is 6.2% (annual, compounded semiannually), what is the bond's price?", b0, "$", answer=money(b0))
add(E, D, "paraphrase", "Price a $1,000-par bond maturing in 10 years with a 5% coupon paid twice a year, when the yield to maturity is 6.2% (semiannual compounding).", b0, "$", answer=money(b0))
add(E, D, "paraphrase", "What is the fair value of a 10-year, 5% semiannual-pay bond with face value $1,000 if investors require a 6.2% annual yield compounded semiannually?", b0, "$", answer=money(b0))
add(E, D, "paraphrase", "A bond has 10 years to maturity, a $1,000 par value and a 5% coupon rate with payments every six months. Given a 6.2% yield (bond-equivalent), calculate its price.", b0, "$", answer=money(b0))
for F_, c_, y_, yld in [(1000, 0.04, 5, 0.035), (1000, 0.07, 20, 0.07), (5000, 0.03, 8, 0.045)]:
    v = bp(F_, c_, y_, yld)
    add(E, D, "reparam", f"A {y_}-year bond with a ${F_:,} face value pays a {c_*100:g}% annual coupon in semiannual installments. If the market yield is {yld*100:g}% (annual, compounded semiannually), what is the bond's price?", v, "$", answer=money(v))
v = bp(1000, 0.0425, 7, 0.046)
add(E, D, "reparam_reworded", "What should you pay for a $1,000 Treasury note maturing in 7 years with a 4.25% coupon paid twice yearly, if comparable yields are 4.60%?", v, "$", answer=money(v))
v = bp(100, 0.065, 12, 0.058)
add(E, D, "reparam_reworded", "A corporate bond has par $100, a 6.5% coupon paid semi-annually and 12 years to maturity. Its yield to maturity is 5.8% (bond-equivalent basis). Price it.", v, "$", answer=money(v))
v = bp(1000, 0.045, 15, 0.0525)
add(E, D, "reparam_reworded", "Price a 15-year, $1,000 par bond with semiannual coupons totalling $45 per year, discounted at a YTM of 5.25%.", v, "$", answer=money(v), note="coupon given in dollars, not percent")
add(E, D, "hard_positive", "A BBB-rated bond issued in 2021 has 10 years left to maturity, a $1,000 face value and a 5% annual coupon paid semiannually. If the market yield is 6.2% (annual, compounded semiannually), what is its price?", b0, "$", answer=money(b0), note="irrelevant number (2021)")
add(E, D, "hard_positive", "A $1,000 face-value bond has 20 semiannual periods remaining and pays a 5% annual coupon in semiannual installments. If the market yield is 6.2% (annual, compounded semiannually), what is the bond's price?", b0, "$", answer=money(b0), note="maturity given as periods")
ytm = optimize.brentq(lambda y: bp(1000, 0.05, 10, y) - 920, 0.001, 0.5)
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments and is priced at $920. What is its yield to maturity?", ytm*100, "%", category="inverse_problem")
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments. If the market yield is 6.2% (annual, compounded semiannually), what is the bond's current yield?", 50/b0*100, "%", category="different_target")
yy = 0.031; cfl = [(t, 25 + (1000 if t == 20 else 0)) for t in range(1, 21)]
mac = sum(t*c/(1+yy)**t for t, c in cfl)/b0/2
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments. If the market yield is 6.2% (annual, compounded semiannually), what is its Macaulay duration?", mac, "years", category="different_target")
va = bp(1000, 0.05, 10, 0.062, f=1)
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% coupon once a year. If the market yield is 6.2% (annual compounding), what is the bond's price?", va, "$", answer=money(va), category="extra_condition", either=True, note="hit is fine only if the entry has a frequency slot")
vz = 1000/1.031**20
add(E, D, "near_miss", "A 10-year zero-coupon bond has a $1,000 face value. If the market yield is 6.2% (annual, compounded semiannually), what is the bond's price?", vz, "$", answer=money(vz), category="different_setup", either=True, note="right only if extractor sets coupon = 0")
dirty = b0*1.031**(1/3); clean = dirty - 25/3
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments. Two months after a coupon payment, with 9 years and 10 months to maturity and a yield of 6.2% (compounded semiannually), what is its clean price?", clean, "$", answer=f"Clean {money(clean)} (dirty {money(dirty)}, accrued {money(25/3)})", category="timing_change")
ytc = optimize.brentq(lambda y: 25*(1 - (1+y/2)**-10)/(y/2) + 1025*(1+y/2)**-10 - 1020, 0.001, 0.5)
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments, trades at $1,020, and is callable in 5 years at $1,025. What is its yield to call?", ytc*100, "%", category="different_target")
dp_ = bp(1000, 0.05, 10, 0.067) - b0
add(E, D, "near_miss", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments. The market yield is 6.2% (annual, compounded semiannually). By how much does its price change if the yield rises by 50 basis points?", dp_, "$", answer=money(dp_), category="change_in_quantity")
add(E, D, "underspecified", "A 10-year bond with a $1,000 face value pays a 5% annual coupon in semiannual installments. What is the bond's price?", answer="Cannot be determined: no market yield is given.", category="missing_given")

# =====================================================================
# 12. UPC intestate spouse share
# =====================================================================
E, D = "upc_intestate_spouse_share", "law"
s5 = lambda est: est if est <= 150000 else 150000 + 0.5*(est - 150000)
add(E, D, "original", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $900,000. He is survived by his wife and two children from a prior marriage who are not her children. How much does the wife inherit?", s5(900000), "$", answer=money(s5(900000)) + " (UPC §2-102(4): first $150,000 plus half the balance)")
add(E, D, "paraphrase", "A husband dies intestate in a UPC state with a $900,000 estate. His wife survives him, along with his two children from an earlier marriage (not hers). Using the UPC base figures, what is the wife's share?", s5(900000), "$", answer=money(s5(900000)))
add(E, D, "paraphrase", "Applying UPC §2-102 base amounts: decedent (male) leaves $900,000, no will, a surviving spouse, and two kids from his first marriage who are not the spouse's descendants. Compute the surviving spouse's intestate share.", s5(900000), "$", answer=money(s5(900000)))
add(E, D, "paraphrase", "How much of a $900,000 intestate estate goes to the widow under the Uniform Probate Code if the deceased husband also left two children from a previous marriage who are not hers? Use the Code's unadjusted dollar amounts.", s5(900000), "$", answer=money(s5(900000)))
for est in (600000, 2400000):
    add(E, D, "reparam", f"Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of ${est:,}. He is survived by his wife and two children from a prior marriage who are not her children. How much does the wife inherit?", s5(est), "$", answer=money(s5(est)))
add(E, D, "reparam", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $1,500,000. He is survived by his wife and one son from a prior marriage who is not her child. How much does the wife inherit?", s5(1500000), "$", answer=money(s5(1500000)))
add(E, D, "reparam_reworded", "Under the UPC's intestacy provisions (base amounts), Maria dies intestate leaving $750,000. Her husband survives her, as does her daughter from an earlier relationship. How much does the husband take?", s5(750000), "$", answer=money(s5(750000)))
add(E, D, "reparam_reworded", "A woman dies without a will in a UPC state with $1.2M in probate assets. Survivors: her husband and her son from a previous marriage. Using the UPC base figures, what is the husband's intestate share?", s5(1200000), "$", answer=money(s5(1200000)))
add(E, D, "reparam_reworded", "Intestate decedent, UPC jurisdiction (base amounts), net probate estate $400,000. Survivors: a spouse and three children, one of whom is from the decedent's previous marriage. What is the spouse's share?", s5(400000), "$", answer=money(s5(400000)), note="only one child is not the spouse's; §2-102(4) still applies")
add(E, D, "hard_positive", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $120,000. He is survived by his wife and two children from a prior marriage who are not her children. How much does the wife inherit?", 120000, "$", answer=money(120000) + " (estate is below the $150,000 threshold, so she takes all of it)", note="exercises the small-estate branch")
add(E, D, "hard_positive", "Married for 12 years, a man dies without a will in a Uniform Probate Code state (base dollar amounts), leaving an estate of $900,000. He is survived by his wife and two children from a prior marriage who are not her children. How much does the wife inherit?", s5(900000), "$", answer=money(s5(900000)), note="irrelevant number (12 years)")
add(E, D, "near_miss", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $900,000. He is survived by his wife and their two children together; she has no other descendants. How much does the wife inherit?", 900000, "$", answer=money(900000) + " (entire estate, §2-102(1)(B))", category="different_rule_branch")
add(E, D, "near_miss", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $900,000. He is survived by his wife, their two children together, and her son from an earlier relationship. How much does the wife inherit?", 225000 + 0.5*675000, "$", answer=money(225000 + 0.5*675000) + " (§2-102(3))", category="different_rule_branch")
add(E, D, "near_miss", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $900,000. He has no descendants and is survived by his wife and his father. How much does the wife inherit?", 300000 + 0.75*600000, "$", answer=money(300000 + 0.75*600000) + " (§2-102(2))", category="different_rule_branch")
add(E, D, "near_miss", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $900,000. He is survived by his wife and two children from a prior marriage who are not her children. How much does each child inherit?", (900000 - s5(900000))/2, "$", answer=money((900000 - s5(900000))/2) + " each", category="different_target")
add(E, D, "near_miss", "In a Uniform Probate Code state, a man dies with a will leaving his entire $900,000 estate to his two children from a prior marriage, leaving nothing to his wife of 15 years. What can the wife claim?", answer="Not intestacy: she may claim an elective share under UPC §2-202, which for a 15+ year marriage is 50% of the augmented estate (about $450,000 if the augmented estate is $900,000)", category="different_doctrine")
add(E, D, "near_miss", "Under the Uniform Probate Code, a man dies without a will, leaving an estate of $900,000. He and his wife divorced the year before he died. He is survived by his ex-wife and two children from a prior marriage. How much does the ex-wife inherit?", 0, "$", answer="$0 (a divorced spouse is not a surviving spouse, §2-802); each child takes $450,000", category="extra_condition")
add(E, D, "near_miss", "Under California law, a man dies without a will, leaving an estate of $900,000. He is survived by his wife and two children from a prior marriage who are not her children. How much does the wife inherit?", answer="Not the UPC: California's community/separate property rules apply, so the share depends on how the assets are characterized", category="different_jurisdiction")
add(E, D, "near_miss", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will. His assets total $900,000, including a $200,000 life insurance policy naming his children as beneficiaries. He is survived by his wife and two children from a prior marriage who are not her children. How much does the wife inherit by intestacy?", s5(700000), "$", answer=money(s5(700000)) + " (insurance is non-probate, so the intestate estate is $700,000)", category="extra_condition", either=True, note="right only if extractor subtracts the non-probate asset")
add(E, D, "underspecified", "Under the Uniform Probate Code (using its base dollar amounts), a man dies without a will, leaving an estate of $900,000. He is survived by his wife and two children. How much does the wife inherit?", answer="Cannot be determined: it depends on whether the children are also hers and whether she has other descendants (anywhere from $525,000 to the full $900,000).", category="missing_given")

# =====================================================================
# 13. FRCP 12(a) answer deadline
# =====================================================================
E, D = "frcp_answer_deadline", "law"
H = {date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,5,25), date(2026,6,19), date(2026,7,3),
     date(2026,9,7), date(2026,10,12), date(2026,11,11), date(2026,11,26), date(2026,12,25),
     date(2027,1,1), date(2027,1,18), date(2027,2,15), date(2027,5,31), date(2027,6,18), date(2027,7,5)}
def dl(start, days):
    d = start + timedelta(days=days)
    while d.weekday() >= 5 or d in H:
        d += timedelta(days=1)
    return d
fd = lambda d: d.strftime("%A, %B %-d, %Y")
def addd(vt, q, start, days, note=None, category=None, either=False, extra=""):
    d = dl(start, days)
    add(E, D, vt, q, None, "date", answer=fd(d) + extra, category=category, either=either, note=note)
s0 = date(2026, 3, 6)
addd("original", "Under Federal Rule of Civil Procedure 12(a), a defendant who did not waive service is served with the summons and complaint on Friday, March 6, 2026. What is the last day to serve an answer?", s0, 21)
addd("paraphrase", "A defendant in federal court was served on March 6, 2026 and did not waive service. Under FRCP 12(a)(1)(A)(i), by what date must the answer be served?", s0, 21)
addd("paraphrase", "When is a federal answer due if the defendant was personally served on Friday, March 6, 2026 (no waiver of service)? Apply Rule 12(a).", s0, 21)
addd("paraphrase", "Service of the summons and complaint happened on 3/6/2026, and the defendant never waived service. Under the Federal Rules of Civil Procedure, what is the deadline to answer?", s0, 21)
for s in (date(2026, 11, 2), date(2026, 1, 21), date(2026, 5, 7)):
    addd("reparam", f"Under Federal Rule of Civil Procedure 12(a), a defendant who did not waive service is served with the summons and complaint on {fd(s)}. What is the last day to serve an answer?", s, 21)
addd("reparam_reworded", "Our client was personally served with a federal complaint on Tuesday, September 15, 2026. She hasn't waived service. When is her answer due under FRCP 12?", date(2026, 9, 15), 21)
addd("reparam_reworded", "Defendant was served on 2026-07-10 in a federal civil action (no waiver). Compute the answer deadline under Rule 12(a)(1)(A)(i).", date(2026, 7, 10), 21, note="ISO date format")
addd("reparam_reworded", "A corporation is served with process on Saturday, August 1, 2026 in a U.S. district court case and did not waive service. By what date must it answer?", date(2026, 8, 1), 21, note="day 21 is a Saturday: Rule 6(a)(1)(C) rolls to Monday")
addd("hard_positive", f"Under Federal Rule of Civil Procedure 12(a), a defendant who did not waive service is served with the summons and complaint on {fd(date(2026,12,4))}. What is the last day to serve an answer?", date(2026, 12, 4), 21, note="day 21 is Christmas: rolls past the holiday and weekend (unexercised branch)")
addd("hard_positive", "Under Federal Rule of Civil Procedure 12(a), a defendant who did not waive service is served on Friday, March 6, 2026 with a 47-page complaint alleging breach of contract. What is the last day to serve an answer?", s0, 21, note="irrelevant number (47 pages)")
addd("near_miss", "Under Federal Rule of Civil Procedure 12(a), a defendant timely waived service after the plaintiff sent the waiver request on Friday, March 6, 2026. What is the last day to serve an answer?", s0, 60, category="different_rule_branch", extra=" (60 days after the request was sent)")
addd("near_miss", "Under Federal Rule of Civil Procedure 12(a), the United States is a defendant, and the U.S. attorney was served on Friday, March 6, 2026. What is the last day for the United States to serve an answer?", s0, 60, category="different_rule_branch", extra=" (60 days, Rule 12(a)(2))")
addd("near_miss", "Under Federal Rule of Civil Procedure 12(a), a defendant located in France waived service after the request was sent on Friday, March 6, 2026. What is the last day to serve an answer?", s0, 90, category="different_rule_branch", extra=" (90 days for a defendant outside any U.S. judicial district)")
addd("near_miss", "Under Federal Rule of Civil Procedure 12(a)(4), the court denied the defendant's Rule 12(b)(6) motion to dismiss on Friday, March 6, 2026. What is the last day to serve the answer?", s0, 14, category="different_rule_branch", extra=" (14 days after notice of the denial)")
addd("near_miss", "A defendant was served with a state-court complaint on Friday, March 6, 2026. Under 28 U.S.C. § 1446(b), what is the last day to file a notice of removal to federal court?", s0, 30, category="different_doctrine", extra=" (30 days; day 30 is a Sunday, so it rolls to Monday)")
addd("near_miss", "Under the Indiana Rules of Trial Procedure, a defendant is personally served with the summons and complaint in Indiana state court on Friday, March 6, 2026. What is the last day to answer?", s0, 20, category="different_jurisdiction", extra=" (20 days under Ind. Trial Rule 6(C))")
addd("near_miss", "Under Federal Rule of Civil Procedure 12(a), a plaintiff is served with an answer containing a counterclaim on Friday, March 6, 2026. What is the last day to serve a reply to the counterclaim?", s0, 21, category="benign_extra_info", either=True, note="different rule (12(a)(1)(B)), same 21 days")
add(E, D, "near_miss", "Under Federal Rule of Civil Procedure 12(a), a defendant who did not waive service is served with the summons and complaint on Friday, March 6, 2026. How many days does the defendant have to serve an answer?", 21, "days", answer="21 days", category="different_target")
add(E, D, "underspecified", "Under Federal Rule of Civil Procedure 12(a), a defendant who did not waive service is served with the summons and complaint. What is the last day to serve an answer?", answer="Cannot be determined: the service date is not given (the period is 21 days after service).", category="missing_given")

# =====================================================================
# 14. DDB book value
# =====================================================================
E, D = "ddb_book_value", "accounting"
def ddb(cost, salv, life, year, factor=2.0):
    bv = cost
    for _ in range(year):
        bv -= max(min(bv*factor/life, bv - salv), 0)
    return bv
def addb(vt, q, cost, salv, life, year, cur="$", note=None):
    v = ddb(cost, salv, life, year); add(E, D, vt, q, v, cur, answer=money(v, cur), note=note)
addb("original", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using double-declining-balance depreciation, what is its book value at the end of year 3?", 80000, 8000, 6, 3)
addb("paraphrase", "Using the double-declining-balance method, find the book value after three years of an asset that cost $80,000, has a 6-year life and an $8,000 salvage value.", 80000, 8000, 6, 3)
addb("paraphrase", "An $80,000 machine with $8,000 residual value is depreciated over 6 years by double-declining balance. What is its carrying amount at the end of year 3?", 80000, 8000, 6, 3)
addb("paraphrase", "Cost $80,000; salvage $8,000; life 6 years; method DDB. What is the net book value at the close of the third year?", 80000, 8000, 6, 3)
addb("reparam", "Equipment costs $50,000, has a salvage value of $5,000, and a useful life of 5 years. Using double-declining-balance depreciation, what is its book value at the end of year 3?", 50000, 5000, 5, 3)
addb("reparam", "Equipment costs $120,000, has a salvage value of $10,000, and a useful life of 8 years. Using double-declining-balance depreciation, what is its book value at the end of year 4?", 120000, 10000, 8, 4)
addb("reparam", "Equipment costs $30,000, has a salvage value of $2,000, and a useful life of 10 years. Using double-declining-balance depreciation, what is its book value at the end of year 2?", 30000, 2000, 10, 2)
addb("reparam_reworded", "A delivery van bought for $45,000 is expected to last 5 years and be worth $5,000 at the end. Under the double-declining-balance method, what is its carrying amount after the second year?", 45000, 5000, 5, 2)
addb("reparam_reworded", "Compute the net book value at the end of year 4 for a machine with cost $200,000, residual value $20,000 and a 10-year life, using 200% declining balance.", 200000, 20000, 10, 4, note="'200% declining balance' = DDB")
addb("reparam_reworded", "A laptop fleet costing €24,000 has a €2,400 residual value and a 4-year life. Using DDB, what is the book value at the end of year 2?", 24000, 2400, 4, 2, cur="€", note="currency variant")
addb("hard_positive", "Equipment costs $10,000, has a salvage value of $3,000, and a useful life of 4 years. Using double-declining-balance depreciation, what is its book value at the end of year 3?", 10000, 3000, 4, 3, note="salvage floor kicks in during year 2 (unexercised branch)")
addb("hard_positive", "Equipment installed by a crew of 4 technicians costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using double-declining-balance depreciation, what is its book value at the end of year 3?", 80000, 8000, 6, 3, note="irrelevant number (4 technicians)")
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using straight-line depreciation, what is its book value at the end of year 3?", 80000 - 3*12000, "$", answer=money(44000), category="different_method")
dep3 = ddb(80000, 8000, 6, 2) - ddb(80000, 8000, 6, 3)
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using double-declining-balance depreciation, what is the depreciation expense in year 3?", dep3, "$", answer=money(dep3), category="different_target")
syd = 80000 - 72000*(6 + 5 + 4)/21
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using sum-of-the-years'-digits depreciation, what is its book value at the end of year 3?", syd, "$", answer=money(syd), category="different_method")
d150 = ddb(80000, 8000, 6, 3, factor=1.5)
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using 150% declining-balance depreciation, what is its book value at the end of year 3?", d150, "$", answer=money(d150), category="different_method", either=True, note="right only if entry has a factor slot")
mac = 80000*(1 - 0.20 - 0.32 - 0.192)
add(E, D, "near_miss", "Equipment costs $80,000 and is 5-year MACRS property (half-year convention). What is its tax basis at the end of year 3?", mac, "$", answer=money(mac), category="different_method")
acc = 80000 - ddb(80000, 8000, 6, 3)
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using double-declining-balance depreciation, what is the accumulated depreciation at the end of year 3?", acc, "$", answer=money(acc), category="different_target")
uop = 80000 - (72000/120000)*(25000 + 30000 + 22000)
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and is expected to produce 120,000 units. It produces 25,000, 30,000 and 22,000 units in years 1-3. Using units-of-production depreciation, what is its book value at the end of year 3?", uop, "$", answer=money(uop), category="different_method")
add(E, D, "near_miss", "Equipment costs $80,000, has a salvage value of $8,000, and a useful life of 6 years. Using double-declining-balance depreciation with a switch to straight-line when that gives a larger charge, what is its book value at the end of year 3?", ddb(80000, 8000, 6, 3), "$", answer=money(ddb(80000, 8000, 6, 3)) + " (no switch happens by year 3)", category="benign_extra_info", either=True)
add(E, D, "underspecified", "Equipment costs $80,000 and has a salvage value of $8,000. Using double-declining-balance depreciation, what is its book value at the end of year 3?", answer="Cannot be determined: useful life is not given.", category="missing_given")

# =====================================================================
# 15. Hardy-Weinberg carriers
# =====================================================================
E, D = "hardy_weinberg_carriers", "biology"
def hw(N, aff):
    q = math.sqrt(aff/N); return 2*(1-q)*q*N
add(E, D, "original", "In a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be carriers?", hw(2500, 196), "people")
add(E, D, "paraphrase", "Cystic fibrosis is autosomal recessive. If 196 of 2,500 people in a population are affected, how many heterozygous carriers would Hardy-Weinberg predict?", hw(2500, 196), "people")
add(E, D, "paraphrase", "Assuming HWE, estimate the number of carriers of the recessive CF allele in a group of 2,500 where 196 individuals have the disease.", hw(2500, 196), "people")
add(E, D, "paraphrase", "Out of 2,500 individuals, 196 show the recessive cystic fibrosis phenotype. Under Hardy-Weinberg assumptions, how many are expected to be unaffected carriers?", hw(2500, 196), "people")
for N_, a_ in [(10000, 1), (5000, 45), (1200, 12)]:
    add(E, D, "reparam", f"In a population of {N_:,} people, {a_} {'has' if a_ == 1 else 'have'} cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be carriers?", hw(N_, a_), "people")
add(E, D, "reparam_reworded", "In an isolated village of 4,000, 160 people have albinism, an autosomal recessive trait. If the population is in Hardy-Weinberg equilibrium, about how many heterozygous carriers are there?", hw(4000, 160), "people")
add(E, D, "reparam_reworded", "A recessive coat-colour allele is present in a herd of 900 cattle, and 81 animals show the recessive phenotype. Under HWE, estimate the number of heterozygotes.", hw(900, 81), "animals")
add(E, D, "reparam_reworded", "Phenylketonuria affects 1 in 10,000 newborns. Under Hardy-Weinberg assumptions, how many carriers would you expect in a city of 2,000,000?", hw(2000000, 200), "people", note="incidence given as a rate, not a count")
add(E, D, "hard_positive", "A survey conducted over 3 years found that in a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be carriers?", hw(2500, 196), "people", note="irrelevant number (3 years)")
add(E, D, "hard_positive", "In a population of 2,500 people, 7.84% have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be carriers?", hw(2500, 196), "people", note="affected given as a percentage")
add(E, D, "near_miss", "In a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, what is the frequency of the dominant allele?", 1 - 0.28, "", category="different_target")
add(E, D, "near_miss", "In a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be homozygous dominant?", 0.72**2*2500, "people", category="different_target")
pc = 2*0.72*0.28/(1 - 0.0784)
add(E, D, "near_miss", "In a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, if two unaffected people from this population have a child, what is the probability the child has cystic fibrosis?", pc**2*0.25, "", category="different_target")
qx = 196/2500
add(E, D, "near_miss", "Among 2,500 males in a population, 196 have an X-linked recessive condition. Assuming Hardy-Weinberg equilibrium, how many carrier females are expected among 2,500 females?", 2*qx*(1-qx)*2500, "people", category="different_setup")
qd = math.sqrt(1 - 196/2500)
add(E, D, "near_miss", "In a population of 2,500 people, 196 have an autosomal dominant condition. Assuming Hardy-Weinberg equilibrium, how many affected people are heterozygous?", 2*(1-qd)*qd*2500, "people", category="different_setup")
chi = (4**2/1296) + (8**2/1008) + (4**2/196)
add(E, D, "near_miss", "In a population of 2,500 people, genotype counts at the CF locus are 1,300 AA, 1,000 Aa and 200 aa. Is the population in Hardy-Weinberg equilibrium at α = 0.05?", chi, "chi-square", answer=f"χ² = {chi:.3f} (df = 1), p = {stats.chi2.sf(chi, 1):.3f}; consistent with HWE", category="different_method")
add(E, D, "near_miss", "In a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, what fraction of the unaffected people are carriers?", pc, "", category="different_target")
add(E, D, "near_miss", "In a population of 2,500 people, 196 have cystic fibrosis, a recessive condition. If affected individuals do not reproduce, what will the frequency of the recessive allele be after one generation?", 0.28/1.28, "", category="process_two_states")
add(E, D, "underspecified", "196 people in a population have cystic fibrosis, a recessive condition. Assuming Hardy-Weinberg equilibrium, how many people are expected to be carriers?", answer="Cannot be determined: population size is not given.", category="missing_given")

# =====================================================================
# 16. Cockcroft-Gault
# =====================================================================
E, D = "cockcroft_gault_crcl", "medicine"
cg = lambda age, wt, scr, female=True: (140 - age)*wt/(72*scr)*(0.85 if female else 1)
add(E, D, "original", "Using the Cockcroft-Gault equation, estimate the creatinine clearance of a 68-year-old woman who weighs 62 kg and has a serum creatinine of 1.4 mg/dL.", cg(68, 62, 1.4), "mL/min")
add(E, D, "paraphrase", "Estimate CrCl by Cockcroft-Gault for a female patient aged 68, weight 62 kg, serum creatinine 1.4 mg/dL.", cg(68, 62, 1.4), "mL/min")
add(E, D, "paraphrase", "A 68-year-old woman (62 kg) has an SCr of 1.4 mg/dL. What is her creatinine clearance according to the Cockcroft-Gault formula?", cg(68, 62, 1.4), "mL/min")
add(E, D, "paraphrase", "Apply the Cockcroft–Gault equation to a 62 kg, 68-year-old female with serum creatinine 1.4 mg/dL. What estimated creatinine clearance do you get?", cg(68, 62, 1.4), "mL/min")
for a_, w_, s_ in [(75, 58, 1.0), (45, 70, 0.8), (82, 50, 1.8)]:
    add(E, D, "reparam", f"Using the Cockcroft-Gault equation, estimate the creatinine clearance of a {a_}-year-old woman who weighs {w_} kg and has a serum creatinine of {s_} mg/dL.", cg(a_, w_, s_), "mL/min")
add(E, D, "reparam_reworded", "Estimate CrCl (Cockcroft-Gault) for a woman aged 59 with a body weight of 143 lb and serum creatinine 0.9 mg/dL.", cg(59, 143*0.45359237, 0.9), "mL/min", note="unit variant: lb")
add(E, D, "reparam_reworded", "Female patient, 71 y, 55 kg, SCr 2.1 mg/dL. Calculate her estimated creatinine clearance using Cockcroft–Gault.", cg(71, 55, 2.1), "mL/min")
add(E, D, "reparam_reworded", "What is the Cockcroft-Gault creatinine clearance for a 64-year-old woman weighing 70 kg whose serum creatinine is 106 µmol/L?", cg(64, 70, 106/88.42), "mL/min", note="unit variant: µmol/L")
add(E, D, "hard_positive", "Using the Cockcroft-Gault equation, estimate the creatinine clearance of a 78-year-old man who weighs 70 kg and has a serum creatinine of 1.3 mg/dL.", cg(78, 70, 1.3, female=False), "mL/min", note="male branch (no 0.85 factor), never exercised by the seed")
add(E, D, "hard_positive", "A 68-year-old woman with a blood pressure of 128/82 weighs 62 kg and has a serum creatinine of 1.4 mg/dL. Using the Cockcroft-Gault equation, estimate her creatinine clearance.", cg(68, 62, 1.4), "mL/min", note="irrelevant numbers (128/82)")
ibw = 45.5 + 2.3*4
add(E, D, "near_miss", "Using the Cockcroft-Gault equation with ideal body weight, estimate the creatinine clearance of a 68-year-old woman who is 5 ft 4 in tall, weighs 62 kg, and has a serum creatinine of 1.4 mg/dL.", cg(68, ibw, 1.4), "mL/min", category="extra_condition", note="IBW (Devine) = 54.7 kg")
k, al = 0.7, -0.241; ratio = 1.4/k
ckd = 142*min(ratio, 1)**al*max(ratio, 1)**-1.200*0.9938**68*1.012
add(E, D, "near_miss", "Using the 2021 CKD-EPI creatinine equation, estimate the eGFR of a 68-year-old woman with a serum creatinine of 1.4 mg/dL.", ckd, "mL/min/1.73m²", category="different_method")
add(E, D, "near_miss", "Using the bedside Schwartz equation, estimate the eGFR of a 10-year-old who is 140 cm tall with a serum creatinine of 0.6 mg/dL.", 0.413*140/0.6, "mL/min/1.73m²", category="different_method")
ibw2 = 45.5 + 2.3*(160/2.54 - 60); adj = ibw2 + 0.4*(95 - ibw2)
add(E, D, "near_miss", "A 68-year-old woman is 160 cm tall, weighs 95 kg, and has a serum creatinine of 1.4 mg/dL. Using the Cockcroft-Gault equation with adjusted body weight, estimate her creatinine clearance.", cg(68, adj, 1.4), "mL/min", category="extra_condition")
add(E, D, "near_miss", "A 68-year-old woman has a 24-hour urine collection of 1,440 mL with a urine creatinine of 80 mg/dL, and a serum creatinine of 1.4 mg/dL. What is her measured creatinine clearance?", 80*1440/(1.4*1440), "mL/min", category="different_method")
mdrd = 175*1.4**-1.154*68**-0.203*0.742
add(E, D, "near_miss", "Using the 4-variable MDRD equation, estimate the eGFR of a 68-year-old woman with a serum creatinine of 1.4 mg/dL.", mdrd, "mL/min/1.73m²", category="different_method")
add(E, D, "near_miss", "Using the Cockcroft-Gault equation, a 68-year-old woman who weighs 62 kg has an estimated creatinine clearance of 37.6 mL/min. Which kidney function range (in mL/min) does this fall into: ≥90, 60-89, 45-59, 30-44, 15-29, or <15?", None, "", answer="30-44 mL/min", category="different_target")
add(E, D, "near_miss", "A 68-year-old woman weighs 62 kg. What serum creatinine would give her a Cockcroft-Gault creatinine clearance of 30 mL/min?", (140 - 68)*62*0.85/(72*30), "mg/dL", category="inverse_problem")
add(E, D, "underspecified", "Using the Cockcroft-Gault equation, estimate the creatinine clearance of a 68-year-old woman who has a serum creatinine of 1.4 mg/dL.", answer="Cannot be determined: body weight is not given.", category="missing_given")

# =====================================================================
# 17. Per-unit tax: buyer price and DWL
# =====================================================================
E, D = "linear_market_tax_dwl", "economics"
def tax(a, b, c, d, t):  # Qd = a - bP, Qs = c + dP, tax t on sellers
    P0 = (a - c)/(b + d); Q0 = a - b*P0
    Pb = (a - c + d*t)/(b + d); Q = a - b*Pb
    return Pb, 0.5*t*(Q0 - Q), Q, P0, Q0
def adde(vt, q, a, b, c, d, t, note=None):
    Pb, dwl, *_ = tax(a, b, c, d, t)
    add(E, D, vt, q, Pb, "$", answer=f"Buyers pay {money(Pb)}; DWL = {money(dwl)}", note=note)
adde("original", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit tax is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", 120, 2, -30, 3, 5)
adde("paraphrase", "Given Qd = 120 - 2P and Qs = 3P - 30, the government levies a $5 tax per unit on producers. Find the price consumers pay after the tax and the resulting deadweight loss.", 120, 2, -30, 3, 5)
adde("paraphrase", "A market has demand Q = 120 − 2P and supply Q = −30 + 3P. If sellers must pay a $5 per-unit excise tax, what price do buyers face, and how large is the deadweight loss?", 120, 2, -30, 3, 5)
adde("paraphrase", "With demand Q = 120 – 2P, supply Q = 3P – 30 and a $5 per-unit tax collected from sellers, compute the consumer price and the welfare loss (DWL).", 120, 2, -30, 3, 5)
adde("reparam", "Demand is Q = 200 − 4P and supply is Q = 6P − 50. A $10 per-unit tax is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", 200, 4, -50, 6, 10)
adde("reparam", "Demand is Q = 90 − P and supply is Q = 2P. A $6 per-unit tax is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", 90, 1, 0, 2, 6)
adde("reparam", "Demand is Q = 500 − 5P and supply is Q = 5P − 100. A $4 per-unit tax is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", 500, 5, -100, 5, 4)
adde("reparam_reworded", "Inverse demand is P = 80 − Q and inverse supply is P = 20 + 0.5Q. A $3 per-unit tax is placed on producers. What do consumers now pay, and what is the deadweight loss?", 80, 1, -40, 2, 3, note="inverse-form curves")
adde("reparam_reworded", "The government levies a $2 excise on producers in a market where Qd = 50 − 2P and Qs = −10 + 4P. Find the consumer price and the deadweight loss.", 50, 2, -10, 4, 2)
adde("reparam_reworded", "In the market for concert tickets, demand is Q = 300 − 3P and supply is Q = 2P − 20. A $10 per-ticket tax is charged to sellers. What price do fans pay, and what is the deadweight loss of the tax?", 300, 3, -20, 2, 10)
adde("hard_positive", "A market with 40 identical firms has demand Q = 120 − 2P and supply Q = 3P − 30. A $5 per-unit tax is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", 120, 2, -30, 3, 5, note="irrelevant number (40 firms)")
adde("hard_positive", "Demand: Q = 120 − 2P. Supply: Q = 3P − 30. Policy: $5/unit excise levied on producers. Find P(buyers) and DWL.", 120, 2, -30, 3, 5, note="terse notation")
Pb, dwl, Qt, P0, Q0 = tax(120, 2, -30, 3, 5)
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit tax is imposed. What price do sellers receive after the tax?", Pb - 5, "$", answer=money(Pb - 5), category="different_target")
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit tax is imposed on sellers. How much tax revenue does the government collect?", 5*Qt, "$", answer=money(5*Qt), category="different_target")
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit tax is imposed on sellers. What is consumer surplus after the tax?", 0.5*(60 - Pb)*Qt, "$", answer=money(0.5*(60 - Pb)*Qt), category="different_target")
Pbs = (150 - 15)/5
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit subsidy is paid to sellers. What is the new price buyers pay, and what is the deadweight loss?", Pbs, "$", answer=f"Buyers pay {money(Pbs)}; DWL = {money(0.5*5*(120 - 2*Pbs - Q0))}", category="sign_change")
Qs_c, Qd_c = 3*25 - 30, 120 - 2*25
dwl_c = 0.5*((120 - Qs_c)/2 - (Qs_c + 30)/3)*(Q0 - Qs_c)
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A binding price ceiling of $25 is imposed. What is the resulting shortage, and what is the deadweight loss?", Qd_c - Qs_c, "units", answer=f"Shortage = {Qd_c - Qs_c} units; DWL = {money(dwl_c)}", category="different_setup")
Pa = 150/(2 + 3/1.1); Qa = 120 - 2*Pa
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A 10% ad valorem tax (on the price buyers pay) is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", Pa, "$", answer=f"Buyers pay {money(Pa)}; DWL = {money(0.5*(Pa - Pa/1.1)*(Q0 - Qa))}", category="different_method")
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit tax is imposed on sellers. What share of the tax burden falls on buyers?", (Pb - P0)/5*100, "%", category="different_target")
add(E, D, "near_miss", "Demand is Q = 120 − 2P and supply is Q = 3P − 30. A $5 per-unit tax is imposed on buyers instead of sellers. What total price (including tax) do buyers pay, and what is the deadweight loss?", Pb, "$", answer=f"Buyers pay {money(Pb)} in total; DWL = {money(dwl)}", category="benign_extra_info", either=True, note="incidence is the same, so the cached answer is right")
add(E, D, "underspecified", "Demand is Q = 120 − 2P. A $5 per-unit tax is imposed on sellers. What is the new price buyers pay, and what is the deadweight loss?", answer="Cannot be determined: the supply curve is not given.", category="missing_given")

# =====================================================================
# Out-of-domain and topical-but-conceptual negatives
# =====================================================================
ood = [
    ("math", "How many divisors does 196 have?", 9, "", None, "unrelated_compute"),
    ("math", "What is 17% of 340?", 57.8, "", None, "unrelated_compute"),
    ("general", "What is the capital of Australia?", None, "", "Canberra", "unrelated_fact"),
    ("general", "Translate 'good morning' into Spanish.", None, "", "Buenos días", "unrelated_task"),
    ("general", "Write a haiku about autumn.", None, "", "(open-ended creative task)", "unrelated_task"),
    ("chemistry", "Explain why real gases deviate from the perfect gas law at high pressure.", None, "", "Molecules have finite volume and attract each other, which matter more when they are crowded together", "topical_conceptual"),
    ("chemistry", "What is the Henderson-Hasselbalch equation, and when does it break down?", None, "", "pH = pKa + log([A-]/[HA]); it fails for very dilute buffers, extreme ratios, or strong acids/bases", "topical_conceptual"),
    ("physics", "Why does a projectile follow a parabolic path when air resistance is ignored?", None, "", "Constant horizontal velocity plus constant vertical acceleration gives a quadratic y(x)", "topical_conceptual"),
    ("engineering", "What is the difference between a simply supported beam and a cantilever?", None, "", "Supported at both ends (free to rotate) vs fixed at one end and free at the other", "topical_conceptual"),
    ("math", "Prove that there are infinitely many primes.", None, "", "Euclid: multiply any finite list of primes, add 1, and the result has a prime factor not on the list", "topical_conceptual"),
    ("probability", "What is the difference between sensitivity and specificity?", None, "", "Sensitivity = P(positive | disease); specificity = P(negative | no disease)", "topical_conceptual"),
    ("statistics", "When should you use a t-test instead of a z-test?", None, "", "When the population standard deviation is unknown and estimated from the sample", "topical_conceptual"),
    ("finance", "Should I choose a 15-year or a 30-year mortgage?", None, "", "(open-ended trade-off: higher payment and less interest vs lower payment and flexibility)", "topical_conceptual"),
    ("finance", "What are the weaknesses of NPV compared with IRR?", None, "", "(conceptual) NPV depends on choosing a discount rate and is not a percentage return; IRR has its own reinvestment and multiple-root problems", "topical_conceptual"),
    ("finance", "Why do bond prices fall when interest rates rise?", None, "", "Fixed coupons become less attractive than new bonds, so the price drops until yields match", "topical_conceptual"),
    ("law", "What is the purpose of intestacy laws?", None, "", "To distribute a decedent's estate according to presumed intent when there is no valid will", "topical_conceptual"),
    ("law", "What happens if a defendant misses the deadline to answer a federal complaint?", None, "", "The plaintiff can seek entry of default and then a default judgment under FRCP 55", "topical_conceptual"),
    ("accounting", "What is the difference between depreciation and amortization?", None, "", "Depreciation applies to tangible assets; amortization applies to intangibles", "topical_conceptual"),
    ("biology", "What are the assumptions of Hardy-Weinberg equilibrium?", None, "", "Large population, random mating, no mutation, no migration, no selection", "topical_conceptual"),
    ("medicine", "Why is the Cockcroft-Gault equation less accurate in obese patients?", None, "", "Total body weight overstates muscle mass, so it overestimates creatinine clearance", "topical_conceptual"),
    ("economics", "What is tax incidence?", None, "", "How the burden of a tax is split between buyers and sellers, set by relative elasticities", "topical_conceptual"),
]
for dom, q, v, u, ans, cat in ood:
    add(None, dom, "out_of_domain", q, v, u, answer=ans, category=cat)

# =====================================================================
# Splits and ids
# =====================================================================
random.seed(42)
groups = {}
for r in rows:
    if r["variant_type"] == "original":
        r["split"] = "seed"
    else:
        groups.setdefault((r["seed_entry"], r["should_hit"]), []).append(r)
for grp in groups.values():
    random.shuffle(grp)
    cut = round(len(grp)*0.6)
    for i, r in enumerate(grp):
        r["split"] = "dev" if i < cut else "test"
for i, r in enumerate(rows, 1):
    r["id"] = f"q{i:03d}"

cols = ["id", "split", "seed_entry", "domain", "variant_type", "near_miss_category", "should_hit",
        "expected_entry", "acceptable_either", "question", "answer", "answer_value", "answer_unit", "notes"]
with open("cache_eval_corpus.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({c: r[c] for c in cols}, ensure_ascii=False) + "\n")
with open("cache_eval_corpus.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in cols})

from collections import Counter
print("total", len(rows))
print(Counter(r["variant_type"] for r in rows))
print("should_hit", Counter(r["should_hit"] for r in rows))
print("either", sum(r["acceptable_either"] for r in rows))
print(Counter(r["split"] for r in rows))
print(Counter(r["near_miss_category"] for r in rows if r["near_miss_category"]))
