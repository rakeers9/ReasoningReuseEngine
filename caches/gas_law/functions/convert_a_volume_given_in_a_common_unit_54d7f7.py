def semantic_function(ctx):
    """Convert a volume given in a common unit (dm3, L, cm3, mL, m3, ft3, gal) to cubic metres."""
    value = load(ctx, "the volume value", d_type=float)
    unit = load(ctx, "the volume unit", d_type=str)
    narrate(ctx, f"Converting volume {value} {unit!r} to cubic metres")
    u = str(unit).strip().lower()
    u = u.replace('\u00b3', '3').replace('^', '').replace('**', '').replace(' ', '').rstrip('.')
    u = u.replace('\u00b5', 'u').replace('\u03bc', 'u')
    factors = {
        'm3': 1.0, 'cubicmetre': 1.0, 'cubicmeter': 1.0, 'cubicmetres': 1.0, 'cubicmeters': 1.0,
        'dm3': 1e-3, 'l': 1e-3, 'litre': 1e-3, 'liter': 1e-3, 'litres': 1e-3, 'liters': 1e-3,
        'cubicdecimetre': 1e-3, 'cubicdecimeter': 1e-3, 'cubicdecimetres': 1e-3, 'cubicdecimeters': 1e-3,
        'cm3': 1e-6, 'cc': 1e-6, 'ml': 1e-6, 'millilitre': 1e-6, 'milliliter': 1e-6, 'millilitres': 1e-6, 'milliliters': 1e-6,
        'cubiccentimetre': 1e-6, 'cubiccentimeter': 1e-6, 'cubiccentimetres': 1e-6, 'cubiccentimeters': 1e-6,
        'mm3': 1e-9, 'ul': 1e-9, 'microlitre': 1e-9, 'microliter': 1e-9,
        'km3': 1e9,
        'ft3': 0.028316846592, 'in3': 1.6387064e-5,
        'gal': 3.785411784e-3, 'gallon': 3.785411784e-3, 'gallons': 3.785411784e-3,
    }
    if u in factors:
        factor = factors[u]
        narrate(ctx, f"Recognised unit {unit!r}; conversion factor to m3 is {factor}")
    else:
        narrate(ctx, f"Unit {unit!r} not directly recognised; asking the language model for its cubic-metre equivalent")
        ans = lm(ctx, f"How many cubic metres is 1 '{unit}' (a unit of volume)? Answer with only a number.")
        factor = smart_convert(ctx, ans, float)
        narrate(ctx, f"Language model gives conversion factor {factor}")
    v_m3 = float(value) * factor
    narrate(ctx, f"Volume in cubic metres: {value} x {factor} = {v_m3} m3")
    return {"the volume in cubic metres": v_m3}
