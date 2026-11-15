def semantic_function(ctx):
    """Convert a volume with unit to SI volume in cubic metres."""
    import re
    narrate(ctx, 'Converting the volume of the container to SI volume in cubic metres')
    text = load(ctx, 'the volume of the container holding the gas with unit', d_type=str)
    narrate(ctx, f'Loaded volume: {text}')
    number = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if number is None:
        raise ValueError('no numeric value found in volume: ' + text)
    value = float(number.group())
    unit = text[number.end():].strip().lower()
    unit = unit.replace('³', '3').replace('^', '').replace(' ', '').replace('.', '')
    factors = {'m3': 1.0, 'dm3': 1e-3, 'l': 1e-3, 'cm3': 1e-6, 'ml': 1e-6}
    if unit not in factors:
        raise ValueError('unrecognised volume unit: ' + unit)
    m3 = value * factors[unit]
    litres = m3 * 1000.0
    narrate(ctx, f'Volume {value} {unit} is {m3:.6g} m3, which is {litres:.6g} L')
    return {'volume in m3': m3, 'volume in L': litres}
