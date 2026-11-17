def semantic_function(ctx):
    """Parse a money amount from raw text with currency symbols, separators and scale suffixes into an exact dollar amount."""
    import re
    from decimal import Decimal
    raw = load(ctx, "the money amount as raw text", d_type=str)
    narrate(ctx, f"Loaded raw money text: {raw!r}")
    text = str(raw).strip().lower()
    negative = False
    if text.startswith('(') and text.endswith(')'):
        negative = True
        text = text[1:-1].strip()
        narrate(ctx, "Parentheses mean the amount is negative")
    if text.startswith('-'):
        negative = True
        text = text[1:].strip()
        narrate(ctx, "Leading minus sign means the amount is negative")
    elif text.startswith('+'):
        text = text[1:].strip()
        narrate(ctx, "Leading plus sign removed")
    text = text.replace('us$', '').replace('usd', '').replace('dollars', '').replace('dollar', '')
    text = text.replace('$', '').replace('\u20ac', '').replace('\u00a3', '').strip()
    narrate(ctx, f"After removing currency symbols the text is {text!r}")
    if text.startswith('-'):
        negative = True
        text = text[1:].strip()
        narrate(ctx, "Minus sign found after the currency symbol, amount is negative")
    m = re.match(r'^([0-9][0-9,]*\.?[0-9]*|\.[0-9]+)\s*([a-z]*)\.?$', text)
    if not m:
        narrate(ctx, "The text does not look like a money amount")
        raise ValueError(f"Cannot parse money amount: {raw!r}")
    num_str = m.group(1).replace(',', '')
    suffix = m.group(2)
    narrate(ctx, f"Digits after removing thousands separators: {num_str}; scale suffix: {suffix!r}")
    scales = {
        '': 0,
        'k': 3, 'thousand': 3,
        'm': 6, 'mm': 6, 'mn': 6, 'mil': 6, 'million': 6,
        'b': 9, 'bn': 9, 'billion': 9,
    }
    if suffix not in scales:
        narrate(ctx, "The scale suffix is not recognised")
        raise ValueError(f"Unknown scale suffix {suffix!r} in {raw!r}")
    value = Decimal(num_str) * (Decimal(10) ** scales[suffix])
    if negative:
        value = -value
        narrate(ctx, "Applying the negative sign to the amount")
    narrate(ctx, f"Number {num_str} with scale suffix {suffix!r} gives {value} dollars")
    return {"amount in dollars": value}
