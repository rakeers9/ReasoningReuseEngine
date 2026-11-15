def semantic_function(ctx):
    '''Convert a container volume given with its unit to cubic metres.'''
    import re
    raw = load(ctx, 'the volume of the container holding the gas with unit', d_type=str)
    text = str(raw).strip()
    narrate(ctx, 'Loaded container volume text: ' + repr(text))
    match = re.match(r'^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$', text)
    if match:
        number = float(match.group(1))
        unit_text = match.group(2)
        narrate(ctx, 'Read numeric magnitude ' + str(number) + ' with unit text ' + repr(unit_text))
    else:
        unit_text = text
        number = float(extract(ctx, text, 'the numerical magnitude of the volume', d_type=float))
        narrate(ctx, 'Extracted numeric magnitude ' + str(number) + ' from text ' + repr(text))
    factors = {'m3': 1.0, 'dm3': 1e-3, 'l': 1e-3, 'litre': 1e-3, 'litres': 1e-3, 'liter': 1e-3, 'liters': 1e-3, 'cm3': 1e-6, 'ml': 1e-6}
    key = re.sub(r'[\s\*\^\.]', '', unit_text.replace(chr(179), '3')).lower()
    if key not in factors:
        answer = re.sub(r'[\s\*\^\.]', '', str(lm(ctx, 'Which volume unit (m3, dm3, L, cm3 or mL) appears in the text ' + repr(text) + '? Answer with the unit only.'))).lower()
        candidates = [k for k in factors if k in answer]
        if candidates:
            key = max(candidates, key=len)
            narrate(ctx, 'Recognised volume unit ' + key + ' from model answer')
    if key not in factors:
        narrate(ctx, 'Could not recognise volume unit in ' + repr(text))
        raise ValueError('Unrecognised volume unit in ' + repr(text))
    volume_m3 = number * factors[key]
    narrate(ctx, 'Converted volume ' + str(number) + ' ' + key + ' to ' + str(volume_m3) + ' m3')
    return {'volume in cubic metres': volume_m3}