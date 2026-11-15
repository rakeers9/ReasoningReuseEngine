def semantic_function(ctx):
    '''Cross-check a perfect gas pressure by recomputing p = nRT/V in atm with R = 0.082057 dm3 atm K-1 mol-1 and the volume in cubic decimetres.'''
    narrate(ctx, 'Cross-checking the pressure with R = 0.082057 dm3 atm K-1 mol-1 and volume in cubic decimetres')
    n = float(load(ctx, 'amount of gas n in moles', d_type=float))
    T = float(load(ctx, 'absolute temperature in kelvin', d_type=float))
    volume_text = str(load(ctx, 'volume with its unit', d_type=str)).strip()
    narrate(ctx, f'Loaded n = {n} mol, T = {T} K, volume = {volume_text}')
    i = 0
    while i < len(volume_text) and (volume_text[i].isdigit() or volume_text[i] in '+-.eE'):
        i += 1
    if i == 0:
        raise ValueError('Volume has no numerical value: ' + volume_text)
    value = float(volume_text[:i])
    unit = volume_text[i:].strip().lower().replace('^', '').replace(' ', '')
    factors_to_dm3 = {'dm3': 1.0, 'l': 1.0, 'litre': 1.0, 'litres': 1.0, 'liter': 1.0, 'liters': 1.0, 'cm3': 1e-3, 'ml': 1e-3, 'm3': 1e3}
    if unit not in factors_to_dm3:
        raise ValueError('Unsupported volume unit: ' + unit)
    V_dm3 = value * factors_to_dm3[unit]
    narrate(ctx, f'Volume in cubic decimetres: {V_dm3} dm3')
    R = 0.082057
    p_atm = n * R * T / V_dm3
    narrate(ctx, f'p = nRT/V = {n} x {R} x {T} / {V_dm3} = {p_atm} atm, which cross-checks the SI route')
    return {'cross-checked pressure in atmospheres': p_atm}
