def semantic_function(ctx):
    """Parse a specific heat given as text with a unit into J/(g·K) as a Decimal."""
    import re
    from decimal import Decimal
    text = load(ctx, "specific heat text with unit", d_type=str)
    narrate(ctx, f"Parsing specific heat text: {text!r}")
    s = str(text).strip()
    m = re.match(r'^\s*([+-]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$', s)
    if not m:
        narrate(ctx, f"Could not find a number in {text!r}")
        raise ValueError(f"Cannot parse specific heat: {text!r}")
    value = Decimal(m.group(1).replace(',', ''))
    rest = m.group(2).strip()
    narrate(ctx, f"Read numeric value {value} with unit text {rest!r}")
    low = rest.lower()
    low = low.replace('degrees', '').replace('degree', '').replace('deg', '')
    for ch in ['(', ')', '·', '⋅', '*', ' ', '°', 'º', '˚', '-', '\u00b7']:
        low = low.replace(ch, '')
    parts = low.split('/')
    if len(parts) < 2:
        narrate(ctx, f"Unit {rest!r} is not a specific heat unit")
        raise ValueError(f"Unhandled specific heat unit: {rest!r}")
    energy = parts[0]
    tail = ''.join(parts[1:])
    energy_factors = {
        'j': Decimal(1),
        'kj': Decimal(1000),
        'cal': Decimal('4.184'),
        'kcal': Decimal('4184'),
    }
    mass_factors = {
        'g': Decimal(1),
        'kg': Decimal(1000),
        'mg': Decimal('0.001'),
    }
    if tail.endswith('c') or tail.endswith('k'):
        mass = tail[:-1]
    else:
        narrate(ctx, f"Temperature part of unit {rest!r} is not handled")
        raise ValueError(f"Unhandled specific heat unit: {rest!r}")
    if energy not in energy_factors or mass not in mass_factors:
        narrate(ctx, f"Unit {rest!r} is not handled")
        raise ValueError(f"Unhandled specific heat unit: {rest!r}")
    narrate(ctx, f"Energy unit {energy!r} is {energy_factors[energy]} J; mass unit {mass!r} is {mass_factors[mass]} g; a temperature difference of 1 degC equals 1 K")
    result = value * energy_factors[energy] / mass_factors[mass]
    narrate(ctx, f"Specific heat is {result} J/(g·K)")
    return {"specific heat in J/(g·K)": result}
