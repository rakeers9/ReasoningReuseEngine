def semantic_function(ctx):
    '''Convert a temperature given in degrees Celsius or kelvin into kelvin.'''
    narrate(ctx, 'Converting the temperature of the gas into kelvin')
    raw = load(ctx, 'the temperature of the gas with unit (°C or K)', d_type=str)
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
