def semantic_function(ctx):
    '''Convert a gas sample volume given in dm3, L, cm3 or m3 into cubic metres.'''
    narrate(ctx, 'Converting the gas sample volume into cubic metres')
    raw = load(ctx, 'the volume of the gas sample with unit (dm3, L, cm3 or m3)', d_type=str)
    narrate(ctx, 'Loaded gas sample volume: ' + str(raw))
    text = str(raw).strip()
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] in '.+-eE'):
        idx += 1
    number = float(text[:idx])
    unit = text[idx:].strip().lower().replace('^', '').replace(' ', '').replace('³', '3')
    factors = {'m3': 1.0, 'dm3': 1e-3, 'l': 1e-3, 'cm3': 1e-6, 'ml': 1e-6}
    if unit not in factors:
        unit = lm(ctx, 'Which of m3, dm3, L, cm3 is the same unit as ' + unit + '? Reply with only the unit.').strip().lower()
    if unit not in factors:
        raise ValueError('Unrecognised volume unit')
    volume_m3 = number * factors[unit]
    narrate(ctx, 'Volume ' + str(number) + ' ' + unit + ' converted to ' + str(volume_m3) + ' m3')
    return {'the volume of the gas sample in cubic metres (m3)': volume_m3}
