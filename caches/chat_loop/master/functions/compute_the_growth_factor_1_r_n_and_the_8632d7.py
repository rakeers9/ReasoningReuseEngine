def semantic_function(ctx):
    """Compute the growth factor and the unrounded level periodic annuity payment from principal, periodic rate and number of payments."""
    import re
    from decimal import Decimal, getcontext
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
        return Decimal(m.group(0).replace(',', ''))

    principal_raw = load(ctx, "principal in dollars", d_type=str)
    principal = to_dec(principal_raw, "principal")
    narrate(ctx, f"Loaded principal in dollars: {principal}")
    rate_raw = load(ctx, "periodic rate", d_type=str)
    r = to_dec(rate_raw, "periodic rate")
    narrate(ctx, f"Loaded periodic rate: {r}")
    n_raw = load(ctx, "number of payments", d_type=str)
    n_dec = to_dec(n_raw, "number of payments")
    if n_dec != n_dec.to_integral_value() or n_dec <= 0:
        raise ValueError(f"Number of payments must be a positive whole number: {n_dec}")
    n = int(n_dec)
    narrate(ctx, f"Loaded number of payments: {n}")
    G = (1 + r) ** n
    narrate(ctx, f"Growth factor G = (1 + r)^n = {G}")
    if r == 0:
        pmt = principal / Decimal(n)
        narrate(ctx, f"Zero rate, so payment = principal / n = {pmt}")
    else:
        pmt = principal * r * G / (G - 1)
        narrate(ctx, f"Level payment M = P*r*G/(G-1) = {pmt}")
    return {"growth factor": G, "unrounded periodic payment in dollars": pmt}
