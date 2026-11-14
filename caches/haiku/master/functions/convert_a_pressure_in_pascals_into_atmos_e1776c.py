def semantic_function(ctx):
    '''Convert a pressure in pascals into atmospheres.'''
    import re
    narrate(ctx, 'Converting the perfect gas pressure from pascals into atmospheres')
    raw = load(ctx, 'the perfect gas pressure in pascals (Pa)', d_type=str)
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
