def semantic_function(ctx):
    '''Predict the pressure exerted by a perfect gas confined to a given volume at a given temperature from the perfect gas equation.'''
    import re
    narrate(ctx, 'Predicting the pressure of a perfect gas from the perfect gas equation pV = nRT')
    amount_text = str(load(ctx, 'the amount of gas n with unit and chemical formula of the gas species and its physical state', d_type=str))
    volume_text = str(load(ctx, 'the volume V occupied by the gas with its unit', d_type=str))
    temperature_text = str(load(ctx, 'the temperature T of the gas with its unit', d_type=str))
    gas_name = str(load(ctx, 'the name of the gas species whose pressure is to be predicted', d_type=str))
    narrate(ctx, 'Loaded amount: ' + amount_text + '; volume: ' + volume_text + '; temperature: ' + temperature_text + '; gas: ' + gas_name)
    pattern = r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?|[+-]?\.\d+)\s*(.*)'
    def parse_quantity(text):
        m = re.search(pattern, text)
        if m is None:
            raise ValueError('No numerical value found in: ' + text)
        return float(m.group(1)), m.group(2).strip()
    n_value, n_rest = parse_quantity(amount_text)
    n_words = n_rest.split()
    n_unit = n_words[0].lower() if n_words else 'mol'
    amount_factors = {'mol': 1.0, 'mole': 1.0, 'moles': 1.0, 'mmol': 1e-3, 'kmol': 1e3}
    if n_unit not in amount_factors:
        raise ValueError('Unsupported amount unit: ' + n_unit)
    n = n_value * amount_factors[n_unit]
    narrate(ctx, f'Amount of gas: n = {n} mol of {gas_name}')
    v_value, v_rest = parse_quantity(volume_text)
    v_unit = v_rest.lower().replace('^', '').replace(' ', '')
    volume_factors_m3 = {'dm3': 1e-3, 'l': 1e-3, 'litre': 1e-3, 'litres': 1e-3, 'liter': 1e-3, 'liters': 1e-3, 'cm3': 1e-6, 'ml': 1e-6, 'm3': 1.0}
    if v_unit not in volume_factors_m3:
        raise ValueError('Unsupported volume unit: ' + v_rest)
    V = v_value * volume_factors_m3[v_unit]
    V_dm3 = V * 1000.0
    narrate(ctx, f'Volume: V = {V} m3 = {V_dm3} dm3')
    t_value, t_rest = parse_quantity(temperature_text)
    t_unit = t_rest.lower().replace('°', '').replace(' ', '').replace('^', '')
    if t_unit in ('c', 'celsius', 'degc'):
        T = t_value + 273.15
    elif t_unit in ('k', 'kelvin'):
        T = t_value
    else:
        raise ValueError('Unsupported temperature unit: ' + t_rest)
    if T <= 0:
        raise ValueError('Temperature must be above absolute zero')
    narrate(ctx, f'Temperature: T = {T} K')
    si_pressure = do(ctx, 'Compute the pressure of a perfect gas from p = nRT/V using SI units, giving the result in pascals.', payloads={
        'amount of gas n in moles': n,
        'absolute temperature in kelvin': T,
        'volume in cubic metres': V,
    })
    p_pa = load(si_pressure, 'perfect gas pressure in pascals', d_type=float, ctx=ctx)
    narrate(ctx, f'Perfect gas pressure p = nRT/V = {p_pa} Pa')
    units = do(ctx, 'Convert a pressure in pascals to atmospheres, bar and megapascals.', payloads={
        'perfect gas pressure in pascals': p_pa,
    })
    p_atm = load(units, 'pressure in atmospheres', d_type=float, ctx=ctx)
    p_bar = load(units, 'pressure in bar', d_type=float, ctx=ctx)
    p_mpa = load(units, 'pressure in megapascals', d_type=float, ctx=ctx)
    narrate(ctx, f'Converted pressure: {p_atm} atm, {p_bar} bar, {p_mpa} MPa')
    R_dm3 = 0.082057
    p_check = n * R_dm3 * T / V_dm3
    agree = abs(p_check - p_atm) <= 1e-3 * abs(p_atm)
    narrate(ctx, f'Cross-check with R = {R_dm3} dm3 atm K-1 mol-1: p = {p_check} atm, agreement: {agree}')
    narrate(ctx, f'Final predicted pressure of {gas_name}: {p_pa} Pa, approximately {p_atm} atm, {p_bar} bar, {p_mpa} MPa')
    return {
        'predicted pressure of the gas in pascals': p_pa,
        'predicted pressure of the gas in atmospheres': p_atm,
        'predicted pressure of the gas in bar': p_bar,
        'predicted pressure of the gas in megapascals': p_mpa,
        'pressures agree flag': agree,
    }
