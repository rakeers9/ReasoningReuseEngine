"""Compiled agentic reasoning cache: gas_law.

Produced by arc.compiler through static resolution: every do() is a direct call, every
load() is a parameter or a dictionary key. lm()/extract()/convert() remain because they
need the target model at invocation time. No embedding model or vector store is used.
"""

from arc.runtime import Context, convert, extract, invoke_tool, lm, narrate, smart_convert


def cross_check_perfect_gas_pressure_in(ctx, amount_of_substance_of_gas_n, temperature_of_gas_in_kelvin_k, volume_of_gas_sample_with_unit, perfect_gas_pressure_in_atmospheres_atm):
    """Cross-check a perfect gas pressure in atmospheres by recomputing p = nRT/V with R in L atm K-1 mol-1 and volume in litres, then compare it with the primary result."""
    import re
    narrate(ctx, 'Cross-checking the perfect gas pressure using R = 0.082057 L atm K-1 mol-1 with the volume in litres')

    def parse(x):
        text = str(x).strip()
        m = re.search('[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?', text)
        if m is None:
            raise ValueError('No number found in: ' + text)
        unit = text[m.end():].strip().lower().replace('^', '').replace(' ', '').replace('³', '3')
        return (float(m.group(0)), unit)
    n, _ = parse(convert(ctx, amount_of_substance_of_gas_n, str))
    narrate(ctx, 'Amount n = ' + str(n) + ' mol')
    T, _ = parse(convert(ctx, temperature_of_gas_in_kelvin_k, str))
    narrate(ctx, 'Temperature T = ' + str(T) + ' K')
    v_value, v_unit = parse(convert(ctx, volume_of_gas_sample_with_unit, str))
    factors = {'l': 1.0, 'dm3': 1.0, 'cm3': 0.001, 'ml': 0.001, 'm3': 1000.0}
    if v_unit not in factors:
        raise ValueError('Unrecognised volume unit: ' + v_unit)
    V_L = v_value * factors[v_unit]
    narrate(ctx, 'Volume V = ' + str(V_L) + ' L')
    p_primary, _ = parse(convert(ctx, perfect_gas_pressure_in_atmospheres_atm, str))
    narrate(ctx, 'Primary pressure = ' + str(p_primary) + ' atm')
    if n <= 0 or T <= 0 or V_L <= 0:
        raise ValueError('Amount, temperature and volume must be positive')
    R_L = 0.082057
    numerator = n * R_L * T
    narrate(ctx, 'Numerator nRT = ' + str(numerator) + ' L atm')
    p_check = numerator / V_L
    narrate(ctx, 'Cross-check pressure p = nRT/V = ' + str(p_check) + ' atm')
    rel_diff = abs(p_check - p_primary) / abs(p_primary)
    agree = rel_diff <= 0.01
    narrate(ctx, 'Relative difference ' + str(rel_diff) + '; agreement with primary result: ' + str(agree))
    return {'the cross-check pressure in atmospheres (atm)': p_check, 'agreement of the cross-check with the primary pressure': agree}


def convert_pressure_in_pascals_into_atmospheres(ctx, perfect_gas_pressure_in_pascals_pa):
    """Convert a pressure in pascals into atmospheres."""
    import re
    narrate(ctx, 'Converting the perfect gas pressure from pascals into atmospheres')
    raw = convert(ctx, perfect_gas_pressure_in_pascals_pa, str)
    narrate(ctx, 'Loaded pressure: ' + str(raw))
    text = str(raw).strip()
    pattern = '[-+]?(?:[0-9]+[.]?[0-9]*|[.][0-9]+)(?:[eE][-+]?[0-9]+)?'
    match = re.search(pattern, text)
    if match is None:
        raise ValueError('No numeric pressure found in: ' + text)
    pascals = float(match.group(0))
    unit = text[match.end():].strip().lower()
    if unit.startswith('kpa'):
        pascals = pascals * 1000.0
    elif unit.startswith('mpa'):
        pascals = pascals * 1000000.0
    narrate(ctx, 'Pressure in pascals: ' + str(pascals) + ' Pa')
    atm_in_pa = 101325.0
    atmospheres = pascals / atm_in_pa
    narrate(ctx, 'Pressure ' + str(pascals) + ' Pa divided by 101325 Pa per atm gives ' + str(atmospheres) + ' atm')
    return {'the perfect gas pressure in atmospheres (atm)': atmospheres}


def compute_pressure_in_pascals_of_perfect(ctx, amount_of_substance_of_gas_n, molar_gas_constant_r_in_j, temperature_of_gas_in_kelvin_k, volume_of_gas_sample_in_cubic):
    """Compute the pressure in pascals of a perfect gas from pV = nRT given amount, gas constant, kelvin temperature and volume in cubic metres."""
    narrate(ctx, 'Computing the perfect gas pressure in pascals from pV = nRT')

    def to_number(x):
        if isinstance(x, str):
            return float(x.strip().split()[0])
        return float(x)
    n_raw = convert(ctx, amount_of_substance_of_gas_n, float)
    n = to_number(n_raw)
    narrate(ctx, 'Amount of substance n = ' + str(n) + ' mol')
    R_raw = convert(ctx, molar_gas_constant_r_in_j, float)
    R = to_number(R_raw)
    narrate(ctx, 'Molar gas constant R = ' + str(R) + ' J K-1 mol-1')
    T_raw = convert(ctx, temperature_of_gas_in_kelvin_k, float)
    T = to_number(T_raw)
    narrate(ctx, 'Temperature T = ' + str(T) + ' K')
    V_raw = convert(ctx, volume_of_gas_sample_in_cubic, float)
    V = to_number(V_raw)
    narrate(ctx, 'Volume V = ' + str(V) + ' m3')
    if n <= 0 or V <= 0 or T <= 0:
        raise ValueError('Amount, volume and temperature must be positive')
    numerator = n * R * T
    narrate(ctx, 'Numerator nRT = ' + str(numerator) + ' J')
    pressure = numerator / V
    narrate(ctx, 'Pressure p = nRT / V = ' + str(pressure) + ' Pa')
    return {'the perfect gas pressure in pascals (Pa)': pressure}


def provide_molar_gas_constant_r_in(ctx):
    """Provide the molar gas constant R in SI units of J K-1 mol-1."""
    narrate(ctx, 'Providing the molar gas constant R in SI units')
    R = 8.314
    narrate(ctx, 'Molar gas constant R = 8.314 J K-1 mol-1')
    return {'the molar gas constant R in J K-1 mol-1': R}


def convert_temperature_given_in_degrees_celsius(ctx, temperature_of_gas_with_unit_c):
    """Convert a temperature given in degrees Celsius or kelvin into kelvin."""
    narrate(ctx, 'Converting the temperature of the gas into kelvin')
    raw = convert(ctx, temperature_of_gas_with_unit_c, str)
    narrate(ctx, 'Loaded temperature: ' + str(raw))
    text = str(raw).strip()
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] in '.+-'):
        idx += 1
    number = float(text[:idx])
    unit = text[idx:].strip().lower().replace('°', '').replace(' ', '')
    if unit in ('c', 'degc', 'celsius'):
        kelvin = number + 273.15
    elif unit in ('k', 'kelvin'):
        kelvin = number
    else:
        answer = lm(ctx, 'Which unit is ' + unit + ' : Celsius or kelvin? Reply with only C or K.').strip().lower()
        if answer.startswith('c'):
            kelvin = number + 273.15
        elif answer.startswith('k'):
            kelvin = number
        else:
            raise ValueError('Unrecognised temperature unit')
    narrate(ctx, 'Temperature ' + str(number) + ' ' + unit + ' converted to ' + str(kelvin) + ' K')
    return {'the temperature of the gas in kelvin (K)': kelvin}


def convert_gas_sample_volume_given_in(ctx, volume_of_gas_sample_with_unit):
    """Convert a gas sample volume given in dm3, L, cm3 or m3 into cubic metres."""
    narrate(ctx, 'Converting the gas sample volume into cubic metres')
    raw = convert(ctx, volume_of_gas_sample_with_unit, str)
    narrate(ctx, 'Loaded gas sample volume: ' + str(raw))
    text = str(raw).strip()
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] in '.+-eE'):
        idx += 1
    number = float(text[:idx])
    unit = text[idx:].strip().lower().replace('^', '').replace(' ', '').replace('³', '3')
    factors = {'m3': 1.0, 'dm3': 0.001, 'l': 0.001, 'cm3': 1e-06, 'ml': 1e-06}
    if unit not in factors:
        unit = lm(ctx, 'Which of m3, dm3, L, cm3 is the same unit as ' + unit + '? Reply with only the unit.').strip().lower()
    if unit not in factors:
        raise ValueError('Unrecognised volume unit')
    volume_m3 = number * factors[unit]
    narrate(ctx, 'Volume ' + str(number) + ' ' + unit + ' converted to ' + str(volume_m3) + ' m3')
    return {'the volume of the gas sample in cubic metres (m3)': volume_m3}


def predict_pressure_exerted_by_perfect_gas(ctx, amount_of_substance_of_gas_n, chemical_identity_of_gas_as_name, volume_of_gas_sample_with_unit, temperature_of_gas_with_unit_c):
    """Predict the pressure exerted by a perfect gas from its amount, identity, volume and temperature."""
    narrate(ctx, 'Predicting the pressure of a perfect gas from pV = nRT')
    n_raw = convert(ctx, amount_of_substance_of_gas_n, str)
    narrate(ctx, 'Loaded amount of substance: ' + str(n_raw))
    gas_raw = convert(ctx, chemical_identity_of_gas_as_name, str)
    narrate(ctx, 'Gas identity: ' + str(gas_raw) + ' (treated as a perfect gas; identity does not change the ideal gas calculation)')
    volume_raw = convert(ctx, volume_of_gas_sample_with_unit, str)
    narrate(ctx, 'Loaded volume: ' + str(volume_raw))
    temp_raw = convert(ctx, temperature_of_gas_with_unit_c, str)
    narrate(ctx, 'Loaded temperature: ' + str(temp_raw))
    vol_result = convert_gas_sample_volume_given_in(ctx, volume_of_gas_sample_with_unit=volume_raw)
    V_m3 = convert(ctx, vol_result['the volume of the gas sample in cubic metres (m3)'], str)
    narrate(ctx, 'Volume in cubic metres: ' + str(V_m3))
    temp_result = convert_temperature_given_in_degrees_celsius(ctx, temperature_of_gas_with_unit_c=temp_raw)
    T_K = convert(ctx, temp_result['the temperature of the gas in kelvin (K)'], str)
    narrate(ctx, 'Temperature in kelvin: ' + str(T_K))
    R_result = provide_molar_gas_constant_r_in(ctx)
    R = convert(ctx, R_result['the molar gas constant R in J K-1 mol-1'], str)
    narrate(ctx, 'Molar gas constant R: ' + str(R) + ' J K-1 mol-1')
    pa_result = compute_pressure_in_pascals_of_perfect(ctx, amount_of_substance_of_gas_n=n_raw, molar_gas_constant_r_in_j=R, temperature_of_gas_in_kelvin_k=T_K, volume_of_gas_sample_in_cubic=V_m3)
    p_pa = convert(ctx, pa_result['the perfect gas pressure in pascals (Pa)'], str)
    narrate(ctx, 'Perfect gas pressure: ' + str(p_pa) + ' Pa')
    atm_result = convert_pressure_in_pascals_into_atmospheres(ctx, perfect_gas_pressure_in_pascals_pa=p_pa)
    p_atm = convert(ctx, atm_result['the perfect gas pressure in atmospheres (atm)'], str)
    narrate(ctx, 'Perfect gas pressure: ' + str(p_atm) + ' atm')
    check_result = cross_check_perfect_gas_pressure_in(ctx, amount_of_substance_of_gas_n=n_raw, temperature_of_gas_in_kelvin_k=T_K, volume_of_gas_sample_with_unit=volume_raw, perfect_gas_pressure_in_atmospheres_atm=p_atm)
    agree = convert(ctx, check_result['agreement of the cross-check with the primary pressure'], str)
    narrate(ctx, 'Cross-check agreement with primary pressure: ' + str(agree))
    return {'the predicted pressure of the perfect gas in atmospheres (atm)': p_atm, 'the predicted pressure of the perfect gas in pascals (Pa)': p_pa}

# Entry-point parameter description -> root argument
ENTRY_PARAMETERS = {'the amount of substance of the gas n with unit (mol)': 'amount_of_substance_of_gas_n', 'the chemical identity of the gas as a name or formula with state (gas phase)': 'chemical_identity_of_gas_as_name', 'the volume of the gas sample with unit (dm3, L, cm3 or m3)': 'volume_of_gas_sample_with_unit', 'the temperature of the gas with unit (°C or K)': 'temperature_of_gas_with_unit_c'}


def run(values: dict) -> tuple[dict, Context]:
    """Run the compiled root on extracted parameter values (keyed by description)."""
    ctx = Context()
    output = predict_pressure_exerted_by_perfect_gas(ctx, **{ENTRY_PARAMETERS[k]: v for k, v in values.items() if k in ENTRY_PARAMETERS})
    return output, ctx
