def semantic_function(ctx):
    '''Convert a volume given with its unit such as dm3, L or cm3 into cubic metres.'''
    narrate(ctx, 'Converting the volume with its unit into cubic metres (SI)')
    raw = str(load(ctx, 'volume with its unit', d_type=str)).strip()
    narrate(ctx, f'Loaded volume text: {raw}')
    i = 0
    while i < len(raw) and (raw[i].isdigit() or raw[i] in '.+-eE'):
        i += 1
    value = float(raw[:i])
    unit = raw[i:].strip().lower().replace('³', '3').replace('^', '').replace(' ', '')
    factors = {'m3': 1.0, 'dm3': 1e-3, 'l': 1e-3, 'litre': 1e-3, 'litres': 1e-3, 'liter': 1e-3, 'liters': 1e-3, 'cm3': 1e-6, 'cc': 1e-6, 'ml': 1e-6, 'mm3': 1e-9}
    if unit not in factors:
        unit = str(lm(ctx, f'Which one of m3, dm3, L, cm3, mL, mm3 is the same unit as {unit}? Answer with the unit only.')).strip().lower().replace(' ', '')
        narrate(ctx, f'Recognised unit as: {unit}')
    if unit not in factors:
        raise ValueError(f'Unsupported volume unit: {unit}')
    volume_m3 = value * factors[unit]
    narrate(ctx, f'Volume: {value} {unit} = {value} x {factors[unit]} m3 = {volume_m3} m3')
    return {'volume in cubic metres': volume_m3}
