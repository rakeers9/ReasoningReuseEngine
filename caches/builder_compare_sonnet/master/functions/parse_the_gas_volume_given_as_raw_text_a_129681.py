def semantic_function(ctx):
    """Parse the gas volume given as raw text and convert it to cubic metres, handling m3, dm3, L, mL and cm3."""
    import re
    from decimal import Decimal
    raw = load(ctx, "the gas volume with unit as raw text", d_type=str)
    narrate(ctx, f"Loaded the raw gas volume text: {raw!r}")
    text = str(raw).strip().lower().replace('\u00b5', 'u').replace('\u03bc', 'u')
    text = text.replace('\u00b3', '3').replace('^', '').replace('\u2113', 'l')
    narrate(ctx, f"Normalised text to {text!r}")
    scales = {'k': 10**3, 'thousand': 10**3, 'million': 10**6, 'bn': 10**9, 'billion': 10**9}
    small = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
             'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
             'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
             'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
             'seventy': 70, 'eighty': 80, 'ninety': 90}
    unit_factors = {
        'm3': Decimal(1), 'cubicmetre': Decimal(1), 'cubicmetres': Decimal(1), 'cubicmeter': Decimal(1), 'cubicmeters': Decimal(1),
        'dm3': Decimal('0.001'), 'cubicdecimetre': Decimal('0.001'), 'cubicdecimetres': Decimal('0.001'),
        'l': Decimal('0.001'), 'litre': Decimal('0.001'), 'litres': Decimal('0.001'), 'liter': Decimal('0.001'), 'liters': Decimal('0.001'),
        'ml': Decimal('0.000001'), 'millilitre': Decimal('0.000001'), 'millilitres': Decimal('0.000001'),
        'milliliter': Decimal('0.000001'), 'milliliters': Decimal('0.000001'),
        'cm3': Decimal('0.000001'), 'cc': Decimal('0.000001'), 'cubiccentimetre': Decimal('0.000001'), 'cubiccentimetres': Decimal('0.000001'),
        'mm3': Decimal('0.000000001'),
    }
    m = re.match(r'^\s*([-+]?\d[\d,]*(?:\.\d*)?(?:e[-+]?\d+)?|[-+]?\.\d+(?:e[-+]?\d+)?)\s*(.*)$', text)
    if m:
        number = Decimal(m.group(1).replace(',', ''))
        rest = m.group(2).strip()
        narrate(ctx, f"Text starts with digits; number is {number}")
    else:
        narrate(ctx, "Text does not start with digits; parsing number words")
        tokens = re.split(r'[\s\-]+', text)
        total = 0
        current = 0
        idx = 0
        found = False
        while idx < len(tokens):
            t = tokens[idx]
            if t in small:
                current += small[t]
                found = True
            elif t == 'hundred':
                current = (current or 1) * 100
                found = True
            elif t == 'and':
                pass
            elif t in ('thousand', 'million', 'billion') and found:
                total += (current or 1) * scales[t]
                current = 0
            else:
                break
            idx += 1
        if not found:
            raise ValueError(f"Cannot parse gas volume from {raw!r}")
        number = Decimal(total + current)
        rest = ' '.join(tokens[idx:]).strip()
        narrate(ctx, f"Number words give {number}")
    narrate(ctx, f"Remaining unit text is {rest!r}")
    compact = rest.replace(' ', '')
    if compact in unit_factors:
        factor = unit_factors[compact]
    else:
        mm = re.match(r'^(thousand|million|billion|bn|k)\s*(.+)$', rest)
        if mm and mm.group(2).replace(' ', '') in unit_factors:
            number = number * scales[mm.group(1)]
            factor = unit_factors[mm.group(2).replace(' ', '')]
            narrate(ctx, f"Applied scale word {mm.group(1)}; number is now {number}")
        else:
            raise ValueError(f"Unrecognised volume unit in {raw!r}")
    narrate(ctx, f"Unit {compact!r} converts to m3 with factor {factor}")
    volume = number * factor
    narrate(ctx, f"Volume in m3: {volume}")
    return {"volume in m3": volume}
