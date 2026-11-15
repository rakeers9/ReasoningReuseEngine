def semantic_function(ctx):
    '''Convert a temperature given with its unit to kelvin.'''
    import re
    narrate(ctx, 'Converting a temperature given with its unit to kelvin')
    raw = load(ctx, 'the temperature of the gas with unit', d_type=str)
    text = str(raw).strip()
    narrate(ctx, 'Loaded temperature text: ' + repr(text))
    match = re.match(r'^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$', text)
    if match:
        number = float(match.group(1))
        unit_text = match.group(2)
        narrate(ctx, 'Read numeric magnitude ' + str(number) + ' with unit text ' + repr(unit_text))
    else:
        number = float(extract(ctx, text, 'the numerical magnitude of the temperature', d_type=float))
        unit_text = text
        narrate(ctx, 'Extracted numeric magnitude ' + str(number) + ' from text ' + repr(text))
    key = unit_text.replace(chr(176), '').lower()
    key = re.sub(r'deg(rees?)?|[\s\*\^\.\(\)]', '', key)
    if key in ('c', 'celsius'):
        kelvin = number + 273.15
        narrate(ctx, 'Unit recognised as degrees Celsius; adding 273.15')
    elif key in ('k', 'kelvin'):
        kelvin = number
        narrate(ctx, 'Unit recognised as kelvin; no conversion needed')
    else:
        answer = str(lm(ctx, 'Is the temperature unit in the text ' + repr(text) + ' Celsius or kelvin? Answer with C or K only.')).strip().lower()
        if answer.startswith('k'):
            kelvin = number
            narrate(ctx, 'Unit identified as kelvin by model')
        elif answer.startswith('c'):
            kelvin = number + 273.15
            narrate(ctx, 'Unit identified as degrees Celsius by model; adding 273.15')
        else:
            narrate(ctx, 'Could not recognise temperature unit in ' + repr(text))
            raise ValueError('Unrecognised temperature unit in ' + repr(text))
    narrate(ctx, 'Converted temperature ' + str(number) + ' to ' + str(kelvin) + ' K')
    return {'absolute temperature in kelvin': kelvin}