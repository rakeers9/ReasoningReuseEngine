def semantic_function(ctx):
    """Round the level periodic payment to the cent (half up) and compute total interest as rounded payment times number of payments minus principal."""
    import re
    from decimal import Decimal, getcontext, ROUND_HALF_UP
    getcontext().prec = 60

    def to_dec(raw, what):
        if isinstance(raw, Decimal):
            return raw
        if isinstance(raw, bool):
            raise ValueError(f"Cannot parse {what}: {raw!r}")
        if isinstance(raw, int):
            return Decimal(raw)
        if isinstance(raw, float):
            return Decimal(str(raw))
        text = str(raw).strip().lower().replace('$', '')
        m = re.search(r'-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?|-?\.\d+', text)
        if not m:
            raise ValueError(f"Cannot parse {what}: {raw!r}")
        val = Decimal(m.group(0).replace(',', ''))
        rest = text[m.end():].strip()
        scale = Decimal(1)
        if rest.startswith(('k', 'thousand')):
            scale = Decimal(10) ** 3
        elif rest.startswith(('million', 'mm')) or rest == 'm':
            scale = Decimal(10) ** 6
        elif rest.startswith(('billion', 'bn')) or rest == 'b':
            scale = Decimal(10) ** 9
        return val * scale

    pmt_raw = load(ctx, "unrounded periodic payment in dollars", d_type=str)
    pmt = to_dec(pmt_raw, "unrounded periodic payment")
    narrate(ctx, f"Loaded unrounded periodic payment in dollars: {pmt}")
    n_raw = load(ctx, "number of payments", d_type=str)
    n_dec = to_dec(n_raw, "number of payments")
    if n_dec != n_dec.to_integral_value() or n_dec <= 0:
        raise ValueError(f"Number of payments must be a positive whole number: {n_dec}")
    n = int(n_dec)
    narrate(ctx, f"Loaded number of payments: {n}")
    p_raw = load(ctx, "principal in dollars", d_type=str)
    principal = to_dec(p_raw, "principal")
    narrate(ctx, f"Loaded principal in dollars: {principal}")
    mp = pmt.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    narrate(ctx, f"Payment rounded to the cent (half up): {mp}")
    interest = mp * Decimal(n) - principal
    narrate(ctx, f"Total interest = rounded payment * {n} - principal = {interest}")
    interest = interest.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return {"monthly payment in dollars": mp, "total interest in dollars": interest}
