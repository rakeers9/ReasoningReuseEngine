def semantic_function(ctx):
    """Convert a temperature given in Celsius, Fahrenheit, Rankine or kelvin to kelvin."""
    value = load(ctx, "the temperature value", d_type=float)
    unit = load(ctx, "the temperature unit", d_type=str)
    narrate(ctx, f"Converting temperature {value} {unit!r} to kelvin")
    u = str(unit).strip().lower()
    u = u.replace('\u00b0', '').replace('\u00ba', '').replace('deg', '').replace('degrees', '').replace(' ', '').rstrip('.')
    u = u.replace('\u2103', 'c').replace('\u2109', 'f').replace('\u212a', 'k')
    v = float(value)
    if u in ('c', 'celsius', 'centigrade', 'reesc', 'eesc', 'eescelsius', 'reescelsius'):
        kind = 'celsius'
    elif u in ('f', 'fahrenheit'):
        kind = 'fahrenheit'
    elif u in ('k', 'kelvin', 'kelvins'):
        kind = 'kelvin'
    elif u in ('r', 'rankine', 'ra'):
        kind = 'rankine'
    elif 'celsius' in u or u.endswith('c'):
        kind = 'celsius'
    elif 'fahrenheit' in u or u.endswith('f'):
        kind = 'fahrenheit'
    elif 'kelvin' in u or u.endswith('k'):
        kind = 'kelvin'
    else:
        ans = lm(ctx, f"Which temperature scale is the unit '{unit}'? Answer with exactly one word: celsius, fahrenheit, kelvin or rankine.")
        kind = str(ans).strip().lower()
        narrate(ctx, f"Language model identifies the scale as {kind!r}")
    if 'fahr' in kind:
        T = (v - 32.0) * 5.0 / 9.0 + 273.15
        narrate(ctx, f"Fahrenheit to kelvin: ({v} - 32) x 5/9 + 273.15 = {T} K")
    elif 'rank' in kind:
        T = v * 5.0 / 9.0
        narrate(ctx, f"Rankine to kelvin: {v} x 5/9 = {T} K")
    elif 'kelv' in kind:
        T = v
        narrate(ctx, f"Temperature already in kelvin: {T} K")
    else:
        T = v + 273.15
        narrate(ctx, f"Celsius to kelvin: {v} + 273.15 = {T} K")
    if T <= 0:
        narrate(ctx, f"Warning: temperature {T} K is not above absolute zero")
    return {"the temperature in kelvin": T}
