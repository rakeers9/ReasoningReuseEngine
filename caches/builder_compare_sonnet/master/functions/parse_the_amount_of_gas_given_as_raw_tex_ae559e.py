def semantic_function(ctx):
    """Parse the amount of gas given as raw text (number, scale word, unit) and convert it to moles."""
    import re
    from decimal import Decimal
    raw = load(ctx, "the amount of gas with unit as raw text", d_type=str)
    narrate(ctx, f"Loaded the raw amount of gas text: {raw!r}")
    text = str(raw).strip().lower().replace('\u00b5', 'u').replace('\u03bc', 'u')
    narrate(ctx, f"Normalised text to {text!r}")
    unit_factors = {
        'mol': Decimal(1), 'mole': Decimal(1), 'moles': Decimal(1), 'mols': Decimal(1),
        'mmol': Decimal('0.001'), 'millimol': Decimal('0.001'), 'millimole': Decimal('0.001'), 'millimoles': Decimal('0.001'),
        'kmol': Decimal(1000), 'kilomol': Decimal(1000), 'kilomole': Decimal(1000), 'kilomoles': Decimal(1000),
        'umol': Decimal('0.000001'), 'micromol': Decimal('0.000001'), 'micromole': Decimal('0.000001'), 'micromoles': Decimal('0.000001'),
        'nmol': Decimal('0.000000001'),
    }
    scales = {'k': 10**3, 'thousand': 10**3, 'million': 10**6, 'bn': 10**9, 'billion': 10**9}
    small = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
             'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
             'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
             'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
             'seventy': 70, 'eighty': 80, 'ninety': 90}
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
                narrate(ctx, f"Number word {t!r} adds {small[t]}")
            elif t == 'hundred':
                current = (current or 1) * 100
                found = True
                narrate(ctx, "Hundred multiplier applied")
            elif t == 'and':
                narrate(ctx, "Skipping the word 'and'")
            elif t in ('thousand', 'million', 'billion') and found:
                total += (current or 1) * scales[t]
                current = 0
                narrate(ctx, f"Scale word {t!r} applied")
            else:
                narrate(ctx, f"Token {t!r} is not part of the number; number words end")
                break
            idx += 1
        if not found:
            narrate(ctx, "No number could be found in the text")
            raise ValueError(f"Cannot parse amount of gas from {raw!r}")
        number = Decimal(total + current)
        rest = ' '.join(tokens[idx:]).strip()
        narrate(ctx, f"Number words give {number}")
    narrate(ctx, f"Remaining unit text is {rest!r}")
    compact = rest.replace(' ', '')
    if compact in unit_factors:
        factor = unit_factors[compact]
        narrate(ctx, f"Unit {compact!r} converts to mol with factor {factor}")
    else:
        mm = re.match(r'^(thousand|million|billion|bn|k)\s*(.+)$', rest)
        if mm and mm.group(2).strip() in unit_factors:
            number = number * scales[mm.group(1)]
            factor = unit_factors[mm.group(2).strip()]
            narrate(ctx, f"Applied scale word {mm.group(1)}; number is now {number}, unit factor {factor}")
        else:
            narrate(ctx, f"Unit text {rest!r} is not recognised")
            raise ValueError(f"Unrecognised amount unit in {raw!r}")
    moles = number * factor
    narrate(ctx, f"Amount of gas in mol: {moles}")
    return {"amount of gas in mol": moles}
