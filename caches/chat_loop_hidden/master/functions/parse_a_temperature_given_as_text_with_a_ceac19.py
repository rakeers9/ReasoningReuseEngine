def semantic_function(ctx):
    """Parse a temperature given as text with a unit (K, °C, °F, R) into kelvin as a Decimal, applying the proper offset."""
    import re
    from decimal import Decimal
    text = load(ctx, "temperature text with unit", d_type=str)
    narrate(ctx, f"Parsing temperature text: {text!r}")
    s = str(text).strip()
    words = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    m = re.match(r'^\s*([+-]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$', s)
    if m:
        value = Decimal(m.group(1).replace(',', ''))
        rest = m.group(2).strip()
        narrate(ctx, f"Read numeric value {value} with remaining text {rest!r}")
    else:
        tokens = s.split(None, 1)
        first = tokens[0].lower() if tokens else ''
        if first in words:
            value = Decimal(words[first])
            rest = tokens[1].strip() if len(tokens) > 1 else ''
            narrate(ctx, f"Read number word {first!r} as {value}; remaining text {rest!r}")
        else:
            narrate(ctx, f"Could not find a number in {text!r}")
            raise ValueError(f"Cannot parse temperature: {text!r}")
    scale = Decimal(1)
    low = rest.lower().strip()
    for word, f in (('thousand', 10**3), ('million', 10**6), ('billion', 10**9)):
        if low.startswith(word):
            scale = Decimal(f)
            low = low[len(word):].strip()
            break
    value = value * scale
    low = low.replace('º', '°').replace('˚', '°').replace('degrees', '').replace('degree', 'deg').replace('deg', '').replace('°', '').replace(' ', '').strip()
    if low in ('k', 'kelvin', 'kelvins'):
        kelvin = value
        narrate(ctx, f"Value is already in kelvin: {kelvin} K")
    elif low in ('c', 'celsius', 'centigrade'):
        kelvin = value + Decimal('273.15')
        narrate(ctx, f"Celsius {value} plus 273.15 gives {kelvin} K")
    elif low in ('f', 'fahrenheit'):
        kelvin = (value - Decimal(32)) * Decimal(5) / Decimal(9) + Decimal('273.15')
        narrate(ctx, f"Fahrenheit {value} converts to {kelvin} K")
    elif low in ('r', 'ra', 'rankine'):
        kelvin = value * Decimal(5) / Decimal(9)
        narrate(ctx, f"Rankine {value} converts to {kelvin} K")
    else:
        narrate(ctx, f"Unit {low!r} is not a handled temperature unit")
        raise ValueError(f"Unhandled temperature unit: {rest!r}")
    return {"temperature in K": kelvin}