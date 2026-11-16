def semantic_function(ctx):
    """Parse a percentage rate from raw text such as '7.5%' or '7.5 percent' into an exact Decimal annual rate as a fraction."""
    import re
    from decimal import Decimal
    raw = load(ctx, "the percentage rate as raw text", d_type=str)
    narrate(ctx, f"Loaded raw rate text: {raw!r}")
    text = str(raw).strip().lower()
    negative = False
    if text.startswith('-'):
        negative = True
        text = text[1:].strip()
        narrate(ctx, "Leading minus sign means the rate is negative")
    elif text.startswith('+'):
        text = text[1:].strip()
    m = re.match(r'^([0-9][0-9,]*\.?[0-9]*|\.[0-9]+)\s*(%|percent|per cent|pct|percentage|bps|basis points|basis point|bp)?(?:\s*(?:per\s+year|per\s+annum|annually|a year|p\.a\.|pa|apr|annual))?\.?$', text)
    if not m:
        narrate(ctx, "The text does not look like a percentage rate")
        raise ValueError(f"Cannot parse rate: {raw!r}")
    num_str = m.group(1).replace(',', '')
    unit = m.group(2) or '%'
    number = Decimal(num_str)
    narrate(ctx, f"Number {num_str} with unit {unit!r}")
    if unit in ('bps', 'basis points', 'basis point', 'bp'):
        fraction = number / Decimal(10000)
        narrate(ctx, "Basis points divided by 10000 give a fraction")
    else:
        fraction = number / Decimal(100)
        narrate(ctx, "Percent divided by 100 gives a fraction")
    if negative:
        fraction = -fraction
    narrate(ctx, f"Annual rate as a decimal fraction is {fraction}")
    return {"annual rate as a decimal fraction": fraction}
