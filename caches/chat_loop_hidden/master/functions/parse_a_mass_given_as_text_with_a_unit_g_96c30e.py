def semantic_function(ctx):
    """Parse a mass given as text with a unit into grams as a Decimal."""
    import re
    from decimal import Decimal
    text = load(ctx, "mass text with unit", d_type=str)
    narrate(ctx, f"Parsing mass text: {text!r}")
    s = str(text).strip()
    words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
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
            raise ValueError(f"Cannot parse mass: {text!r}")
    scale = Decimal(1)
    low = rest.lower().strip()
    for word, f in (('thousand', 10**3), ('million', 10**6), ('billion', 10**9)):
        if low.startswith(word):
            scale = Decimal(f)
            low = low[len(word):].strip()
            narrate(ctx, f"Applied scale word {word!r}, multiplier {scale}")
            break
    low = low.replace('.', '').strip()
    factors = {
        'g': Decimal(1), 'gram': Decimal(1), 'grams': Decimal(1), 'gm': Decimal(1),
        'kg': Decimal(1000), 'kilogram': Decimal(1000), 'kilograms': Decimal(1000),
        'mg': Decimal('0.001'), 'milligram': Decimal('0.001'), 'milligrams': Decimal('0.001'),
        'lb': Decimal('453.59237'), 'lbs': Decimal('453.59237'), 'pound': Decimal('453.59237'), 'pounds': Decimal('453.59237'),
    }
    if low not in factors:
        narrate(ctx, f"Unit {low!r} is not a handled mass unit")
        raise ValueError(f"Unhandled mass unit: {rest!r}")
    narrate(ctx, f"Unit {low!r} is worth {factors[low]} g")
    grams = value * scale * factors[low]
    narrate(ctx, f"Value {value} with unit {low!r} converts to {grams} g")
    return {"mass in grams": grams}
