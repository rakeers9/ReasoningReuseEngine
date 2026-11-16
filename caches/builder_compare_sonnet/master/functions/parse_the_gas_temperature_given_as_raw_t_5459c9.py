def semantic_function(ctx):
    """Parse the gas temperature given as raw text and convert it to kelvin, handling K, °C and °F."""
    import re
    from decimal import Decimal
    raw = load(ctx, "the gas temperature with unit as raw text", d_type=str)
    narrate(ctx, f"Loaded the raw gas temperature text: {raw!r}")
    text = str(raw).strip().lower().replace('\u00b5', 'u')
    text = text.replace('\u2103', '\u00b0c').replace('\u2109', '\u00b0f').replace('\u212a', 'k')
    text = text.replace('\u00ba', '\u00b0')
    narrate(ctx, f"Normalised text to {text!r}")
    small = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
             'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
             'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
             'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
             'seventy': 70, 'eighty': 80, 'ninety': 90}
    scales = {'thousand': 1000, 'million': 10**6, 'billion': 10**9}
    m = re.match(r'^\s*([-+\u2212]?\s*\d[\d,]*(?:\.\d*)?(?:e[-+]?\d+)?|[-+\u2212]?\s*\.\d+(?:e[-+]?\d+)?)\s*(.*)$', text)
    if m:
        num_text = m.group(1).replace(',', '').replace('\u2212', '-').replace(' ', '')
        number = Decimal(num_text)
        rest = m.group(2).strip()
        narrate(ctx, f"Text starts with digits; number is {number}")
    else:
        narrate(ctx, "Text does not start with digits; parsing number words")
        tokens = re.split(r'[\s\-]+', text)
        sign = 1
        if tokens and tokens[0] in ('minus', 'negative'):
            sign = -1
            tokens = tokens[1:]
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
            elif t in scales and found:
                total += (current or 1) * scales[t]
                current = 0
            else:
                break
            idx += 1
        if not found:
            raise ValueError(f"Cannot parse gas temperature from {raw!r}")
        number = Decimal(sign * (total + current))
        rest = ' '.join(tokens[idx:]).strip()
        narrate(ctx, f"Number words give {number}")
    narrate(ctx, f"Remaining unit text is {rest!r}")
    unit = rest.replace(' ', '').replace('degrees', '').replace('degree', '').replace('\u00b0', '')
    narrate(ctx, f"Compact unit is {unit!r}")
    if unit in ('k', 'kelvin', 'kelvins'):
        kelvin = number
        narrate(ctx, "Already in kelvin; no conversion needed")
    elif unit in ('c', 'celsius', 'centigrade'):
        kelvin = number + Decimal('273.15')
        narrate(ctx, f"Celsius to kelvin: add 273.15 giving {kelvin}")
    elif unit in ('f', 'fahrenheit'):
        kelvin = (number - Decimal(32)) * Decimal(5) / Decimal(9) + Decimal('273.15')
        narrate(ctx, f"Fahrenheit to kelvin: (F - 32) * 5/9 + 273.15 giving {kelvin}")
    else:
        raise ValueError(f"Unrecognised temperature unit in {raw!r}")
    if kelvin <= 0:
        raise ValueError(f"Temperature {raw!r} is not above absolute zero")
    narrate(ctx, f"Temperature in K: {kelvin}")
    return {"temperature in K": kelvin}
