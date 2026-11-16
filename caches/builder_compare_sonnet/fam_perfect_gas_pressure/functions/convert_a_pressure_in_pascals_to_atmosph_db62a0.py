def semantic_function(ctx):
    """Convert a pressure in pascals to atmospheres by dividing by 101325 Pa per atm."""
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    p_raw = load(ctx, "pressure in Pa", d_type=str)
    k_raw = load(ctx, "pascals per atmosphere", d_type=str)
    narrate(ctx, f"Loaded pressure {p_raw!r} Pa and conversion factor {k_raw!r} Pa per atm")
    p = Decimal(str(p_raw).strip())
    k = Decimal(str(k_raw).strip())
    if k <= 0:
        raise ValueError("Pascals per atmosphere must be positive")
    p_atm = p / k
    narrate(ctx, f"Pressure in atm = {p} / {k} = {p_atm}")
    return {"pressure in atm": p_atm}