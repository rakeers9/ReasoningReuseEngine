def semantic_function(ctx):
    '''Convert a temperature to kelvin and a volume to cubic metres so that both are in SI units.'''
    narrate(ctx, 'Converting the temperature to kelvin and the volume to cubic metres to obtain SI units')
    temp_text = str(load(ctx, 'temperature with its unit', d_type=str)).strip()
    vol_text = str(load(ctx, 'volume with its unit', d_type=str)).strip()
    narrate(ctx, f'Loaded temperature: {temp_text}')
    narrate(ctx, f'Loaded volume: {vol_text}')
    raw = temp_text
    i = 0
    while i < len(raw) and (raw[i].isdigit() or raw[i] in '.+-'):
        i += 1
    t_value = float(raw[:i])
    t_unit = raw[i:].strip().lower().replace('°', '').replace('º', '').replace(' ', '')
    if t_unit not in ('c', 'k', 'kelvin', 'degc', 'celsius'):
        t_unit = str(lm(ctx, f'Is the temperature unit {t_unit} degrees Celsius or kelvin? Answer with C or K only.')).strip().upper()
        t_unit = 'k' if t_unit.startswith('K') else 'c'
    narrate(ctx, f'Temperature unit recognised as: {t_unit}')
    if t_unit in ('k', 'kelvin'):
        t_k = t_value
    else:
        t_result = do(ctx, 'Convert a temperature in degrees Celsius to kelvin by adding 273.15', payloads={'temperature in degrees Celsius': t_value})
        t_k = load(t_result, 'absolute temperature in kelvin', d_type=float, ctx=ctx)
    narrate(ctx, f'Absolute temperature: {t_k} K')
    v_result = do(ctx, 'Convert a volume given with its unit, such as dm3, L or cm3, into cubic metres.', payloads={'volume with its unit': vol_text})
    v_m3 = load(v_result, 'volume in cubic metres', d_type=float, ctx=ctx)
    narrate(ctx, f'Volume in SI: {v_m3} m3')
    return {'absolute temperature in kelvin': t_k, 'volume in cubic metres': v_m3}
