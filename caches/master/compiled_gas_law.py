"""Compiled agentic reasoning cache: gas_law.

Produced by arc.compiler through static resolution: every do() is a direct call, every
load() is a parameter or a dictionary key. lm()/extract()/convert() remain because they
need the target model at invocation time. No embedding model or vector store is used.
"""

from arc.runtime import Context, convert, extract, invoke_tool, lm, narrate, smart_convert


def cross_check_perfect_gas_pressure_using(ctx, amount_in_moles, temperature_in_kelvin, volume_in_cubic_metres, pressure_in_atmospheres):
    """Cross-check the perfect gas pressure using R = 0.08206 dm3 atm K-1 mol-1 with volume in dm3 and compare it with the SI-derived pressure in atmospheres."""
    n = float(convert(ctx, amount_in_moles, float))
    T = float(convert(ctx, temperature_in_kelvin, float))
    V_m3 = float(convert(ctx, volume_in_cubic_metres, float))
    p_atm = float(convert(ctx, pressure_in_atmospheres, float))
    narrate(ctx, f'Loaded n = {n} mol, T = {T} K, V = {V_m3} m3, SI-derived p = {p_atm:.6g} atm')
    if V_m3 <= 0:
        raise ValueError('Volume must be positive')
    V_dm3 = V_m3 * 1000.0
    narrate(ctx, f'Volume in cubic decimetres: {V_m3} m3 x 1000 = {V_dm3:.6g} dm3')
    R = 0.08206
    nRT = n * R * T
    narrate(ctx, f'nRT = {n} x {R} x {T} = {nRT:.6g} dm3 atm')
    p_check = nRT / V_dm3
    narrate(ctx, f'Check pressure p = nRT / V = {nRT:.6g} / {V_dm3:.6g} = {p_check:.6g} atm')
    rel = abs(p_check - p_atm) / abs(p_atm) if p_atm != 0 else abs(p_check)
    agree = rel < 0.001
    narrate(ctx, f"Relative difference {rel:.3g}; results {('agree' if agree else 'do not agree')}")
    return {'the check pressure in atmospheres': p_check, 'whether the two results agree': agree}


def split_physical_quantity_string_such_as(ctx, quantity_string_with_value_and_unit):
    """Split a physical quantity string into its numeric value and its unit string."""
    import re
    text = convert(ctx, quantity_string_with_value_and_unit, str)
    narrate(ctx, f'Parsing the quantity string: {text!r}')
    s = str(text).strip()
    s_norm = s.replace('−', '-').replace('×', 'x').replace('\u2009', ' ').replace('\xa0', ' ')
    m = re.match('^\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)(?:\\s*(?:x|\\*)\\s*10\\s*(?:\\^|\\*\\*)?\\s*([-+]?\\d+))?\\s*(.*)$', s_norm)
    if m:
        value = float(m.group(1))
        if m.group(2) is not None:
            value *= 10 ** int(m.group(2))
        unit = m.group(3).strip()
    else:
        num_txt = extract(ctx, s, 'the numeric value of this physical quantity', d_type=str)
        value = smart_convert(ctx, num_txt, float)
        unit = extract(ctx, s, 'the unit of this physical quantity (without the number)', d_type=str)
        unit = str(unit).strip()
    narrate(ctx, f'Numeric value = {value}, unit = {unit!r}')
    return {'the numeric value': value, 'the unit string': unit}


def convert_pressure_in_pascals_to_atmospheres(ctx, pressure_in_pascals):
    """Convert a pressure in pascals to atmospheres, bar, kilopascals and megapascals."""
    p = convert(ctx, pressure_in_pascals, float)
    p = float(p)
    narrate(ctx, f'Loaded pressure p = {p:.6g} Pa')
    atm = p / 101325.0
    narrate(ctx, f'In atmospheres: {p:.6g} Pa / 101325 Pa atm-1 = {atm:.6g} atm')
    bar = p / 100000.0
    narrate(ctx, f'In bar: {p:.6g} Pa / 1e5 Pa bar-1 = {bar:.6g} bar')
    kpa = p / 1000.0
    narrate(ctx, f'In kilopascals: {p:.6g} Pa / 1000 = {kpa:.6g} kPa')
    mpa = p / 1000000.0
    narrate(ctx, f'In megapascals: {p:.6g} Pa / 1e6 = {mpa:.6g} MPa')
    return {'the pressure in atmospheres': atm, 'the pressure in bar': bar, 'the pressure in kilopascals': kpa, 'the pressure in megapascals': mpa}


def apply_perfect_gas_law_p_nrt(ctx, amount_in_moles, temperature_in_kelvin, volume_in_cubic_metres):
    """Apply the perfect gas law p = nRT/V in SI units (R = 8.314 J K-1 mol-1) to obtain the pressure in pascals."""
    n = convert(ctx, amount_in_moles, float)
    T = convert(ctx, temperature_in_kelvin, float)
    V = convert(ctx, volume_in_cubic_metres, float)
    narrate(ctx, f'Loaded n = {n} mol, T = {T} K, V = {V} m3')
    if V is None or float(V) <= 0:
        raise ValueError('Volume must be positive')
    if T is None or float(T) <= 0:
        raise ValueError('Temperature must be above absolute zero')
    R = 8.314
    nRT = float(n) * R * float(T)
    narrate(ctx, f'nRT = {n} x {R} x {T} = {nRT} J')
    p = nRT / float(V)
    narrate(ctx, f'p = nRT / V = {nRT} J / {V} m3 = {p:.6g} Pa')
    return {'the pressure in pascals': p}


def convert_temperature_given_in_celsius_fahrenheit(ctx, temperature_value, temperature_unit):
    """Convert a temperature given in Celsius, Fahrenheit, Rankine or kelvin to kelvin."""
    value = convert(ctx, temperature_value, float)
    unit = convert(ctx, temperature_unit, str)
    narrate(ctx, f'Converting temperature {value} {unit!r} to kelvin')
    u = str(unit).strip().lower()
    u = u.replace('°', '').replace('º', '').replace('deg', '').replace('degrees', '').replace(' ', '').rstrip('.')
    u = u.replace('℃', 'c').replace('℉', 'f').replace('K', 'k')
    v = float(value)
    if u in ('c', 'celsius', 'centigrade', 'reesc', 'eesc', 'eescelsius', 'reescelsius'):
        kind = 'celsius'
    elif u in ('f', 'fahrenheit'):
        kind = 'fahrenheit'
    elif u in ('k', 'kelvin', 'kelvins'):
        kind = 'kelvin'
    elif u in ('r', 'rankine', 'ra'):
        kind = 'rankine'
    elif 'celsius' in u or u.endswith('c'):
        kind = 'celsius'
    elif 'fahrenheit' in u or u.endswith('f'):
        kind = 'fahrenheit'
    elif 'kelvin' in u or u.endswith('k'):
        kind = 'kelvin'
    else:
        ans = lm(ctx, f"Which temperature scale is the unit '{unit}'? Answer with exactly one word: celsius, fahrenheit, kelvin or rankine.")
        kind = str(ans).strip().lower()
        narrate(ctx, f'Language model identifies the scale as {kind!r}')
    if 'fahr' in kind:
        T = (v - 32.0) * 5.0 / 9.0 + 273.15
        narrate(ctx, f'Fahrenheit to kelvin: ({v} - 32) x 5/9 + 273.15 = {T} K')
    elif 'rank' in kind:
        T = v * 5.0 / 9.0
        narrate(ctx, f'Rankine to kelvin: {v} x 5/9 = {T} K')
    elif 'kelv' in kind:
        T = v
        narrate(ctx, f'Temperature already in kelvin: {T} K')
    else:
        T = v + 273.15
        narrate(ctx, f'Celsius to kelvin: {v} + 273.15 = {T} K')
    if T <= 0:
        narrate(ctx, f'Warning: temperature {T} K is not above absolute zero')
    return {'the temperature in kelvin': T}


def convert_volume_given_in_common_unit(ctx, volume_value, volume_unit):
    """Convert a volume given in a common unit (dm3, L, cm3, mL, m3, ft3, gal) to cubic metres."""
    value = convert(ctx, volume_value, float)
    unit = convert(ctx, volume_unit, str)
    narrate(ctx, f'Converting volume {value} {unit!r} to cubic metres')
    u = str(unit).strip().lower()
    u = u.replace('³', '3').replace('^', '').replace('**', '').replace(' ', '').rstrip('.')
    u = u.replace('µ', 'u').replace('μ', 'u')
    factors = {'m3': 1.0, 'cubicmetre': 1.0, 'cubicmeter': 1.0, 'cubicmetres': 1.0, 'cubicmeters': 1.0, 'dm3': 0.001, 'l': 0.001, 'litre': 0.001, 'liter': 0.001, 'litres': 0.001, 'liters': 0.001, 'cubicdecimetre': 0.001, 'cubicdecimeter': 0.001, 'cubicdecimetres': 0.001, 'cubicdecimeters': 0.001, 'cm3': 1e-06, 'cc': 1e-06, 'ml': 1e-06, 'millilitre': 1e-06, 'milliliter': 1e-06, 'millilitres': 1e-06, 'milliliters': 1e-06, 'cubiccentimetre': 1e-06, 'cubiccentimeter': 1e-06, 'cubiccentimetres': 1e-06, 'cubiccentimeters': 1e-06, 'mm3': 1e-09, 'ul': 1e-09, 'microlitre': 1e-09, 'microliter': 1e-09, 'km3': 1000000000.0, 'ft3': 0.028316846592, 'in3': 1.6387064e-05, 'gal': 0.003785411784, 'gallon': 0.003785411784, 'gallons': 0.003785411784}
    if u in factors:
        factor = factors[u]
        narrate(ctx, f'Recognised unit {unit!r}; conversion factor to m3 is {factor}')
    else:
        narrate(ctx, f'Unit {unit!r} not directly recognised; asking the language model for its cubic-metre equivalent')
        ans = lm(ctx, f"How many cubic metres is 1 '{unit}' (a unit of volume)? Answer with only a number.")
        factor = smart_convert(ctx, ans, float)
        narrate(ctx, f'Language model gives conversion factor {factor}')
    v_m3 = float(value) * factor
    narrate(ctx, f'Volume in cubic metres: {value} x {factor} = {v_m3} m3')
    return {'the volume in cubic metres': v_m3}


def convert_amount_of_substance_given_with(ctx, amount_value, amount_unit):
    """Convert an amount of substance given with a unit (mol, mmol, kmol, umol, nmol) to moles."""
    value = convert(ctx, amount_value, float)
    unit = convert(ctx, amount_unit, str)
    narrate(ctx, f'Converting amount {value} {unit!r} to moles')
    u = str(unit).strip().lower().replace('µ', 'u').replace('μ', 'u').replace(' ', '').rstrip('.')
    factors = {'mol': 1.0, 'mole': 1.0, 'moles': 1.0, 'mols': 1.0, 'mmol': 0.001, 'millimol': 0.001, 'millimole': 0.001, 'millimoles': 0.001, 'kmol': 1000.0, 'kilomol': 1000.0, 'kilomole': 1000.0, 'kilomoles': 1000.0, 'umol': 1e-06, 'micromol': 1e-06, 'micromole': 1e-06, 'micromoles': 1e-06, 'nmol': 1e-09, 'nanomol': 1e-09, 'nanomole': 1e-09, 'nanomoles': 1e-09}
    if u in factors:
        factor = factors[u]
        narrate(ctx, f'Recognised unit {unit!r} directly; conversion factor to mol is {factor}')
    else:
        narrate(ctx, f"Unit {unit!r} not directly recognised; checking whether it begins with a known amount unit (e.g. 'mol C2H6')")
        factor = None
        for key in sorted(factors, key=len, reverse=True):
            if u.startswith(key):
                factor = factors[key]
                narrate(ctx, f'Unit starts with {key!r}; conversion factor to mol is {factor}')
                break
        if factor is None:
            narrate(ctx, f'Asking the language model for the mole equivalent of 1 {unit!r}')
            ans = lm(ctx, f"How many moles is 1 '{unit}' (a unit of amount of substance)? Answer with only a number.")
            factor = smart_convert(ctx, ans, float)
            narrate(ctx, f'Language model gives conversion factor {factor}')
    moles = float(value) * factor
    narrate(ctx, f'Amount in moles: {value} x {factor} = {moles} mol')
    return {'the amount in moles': moles}


def predict_pressure_exerted_by_gas_confined(ctx, amount_of_gas_with_unit, chemical_formula_of_gas_with_its, volume_of_gas_with_unit, temperature_of_gas_with_unit, common_name_of_gas):
    """Predict the pressure exerted by a gas confined to a given volume at a given temperature using the perfect gas law, reporting it in pascals, megapascals and atmospheres."""
    amount_str = convert(ctx, amount_of_gas_with_unit, str)
    formula = convert(ctx, chemical_formula_of_gas_with_its, str)
    volume_str = convert(ctx, volume_of_gas_with_unit, str)
    temp_str = convert(ctx, temperature_of_gas_with_unit, str)
    name = convert(ctx, common_name_of_gas, str)
    narrate(ctx, f'Gas: {name} ({formula}); amount = {amount_str}, volume = {volume_str}, temperature = {temp_str}')

    def parse(q):
        r = split_physical_quantity_string_such_as(ctx, quantity_string_with_value_and_unit=q)
        return (float(convert(ctx, r['the numeric value'], float)), convert(ctx, r['the unit string'], str))
    n_val, n_unit = parse(amount_str)
    v_val, v_unit = parse(volume_str)
    t_val, t_unit = parse(temp_str)
    narrate(ctx, f'Parsed: n = {n_val} {n_unit}, V = {v_val} {v_unit}, T = {t_val} {t_unit}')
    r = convert_amount_of_substance_given_with(ctx, amount_value=n_val, amount_unit=n_unit)
    n = float(convert(ctx, r['the amount in moles'], float))
    r = convert_volume_given_in_common_unit(ctx, volume_value=v_val, volume_unit=v_unit)
    V = float(convert(ctx, r['the volume in cubic metres'], float))
    r = convert_temperature_given_in_celsius_fahrenheit(ctx, temperature_value=t_val, temperature_unit=t_unit)
    T = float(convert(ctx, r['the temperature in kelvin'], float))
    narrate(ctx, f'SI values: n = {n} mol, V = {V:.6g} m3, T = {T} K')
    r = apply_perfect_gas_law_p_nrt(ctx, amount_in_moles=n, temperature_in_kelvin=T, volume_in_cubic_metres=V)
    p_pa = float(convert(ctx, r['the pressure in pascals'], float))
    narrate(ctx, f'Perfect gas pressure p = {p_pa:.6g} Pa')
    r = convert_pressure_in_pascals_to_atmospheres(ctx, pressure_in_pascals=p_pa)
    p_atm = float(convert(ctx, r['the pressure in atmospheres'], float))
    p_mpa = float(convert(ctx, r['the pressure in megapascals'], float))
    p_bar = float(convert(ctx, r['the pressure in bar'], float))
    narrate(ctx, f'p = {p_mpa:.4g} MPa = {p_atm:.4g} atm = {p_bar:.4g} bar')
    try:
        r = cross_check_perfect_gas_pressure_using(ctx, amount_in_moles=n, temperature_in_kelvin=T, volume_in_cubic_metres=V, pressure_in_atmospheres=p_atm)
        chk = convert(ctx, r['the check pressure in atmospheres'], float)
        narrate(ctx, f'Cross-check with R = 0.08206 dm3 atm K-1 mol-1 gives {float(chk):.4g} atm')
    except Exception as e:
        narrate(ctx, f'Cross-check skipped: {e}')
    answer = f'p ≈ {p_pa:.3e} Pa = {p_mpa:.3g} MPa ≈ {p_atm:.3g} atm'
    narrate(ctx, f'Final answer: {answer}')
    return {'the final pressure answer': answer}

# Entry-point parameter description -> root argument
ENTRY_PARAMETERS = {'the amount of gas with unit': 'amount_of_gas_with_unit', 'the chemical formula of the gas with its phase': 'chemical_formula_of_gas_with_its', 'the volume of the gas with unit': 'volume_of_gas_with_unit', 'the temperature of the gas with unit': 'temperature_of_gas_with_unit', 'the common name of the gas': 'common_name_of_gas'}


def run(values: dict) -> tuple[dict, Context]:
    """Run the compiled root on extracted parameter values (keyed by description)."""
    ctx = Context()
    output = predict_pressure_exerted_by_gas_confined(ctx, **{ENTRY_PARAMETERS[k]: v for k, v in values.items() if k in ENTRY_PARAMETERS})
    return output, ctx
