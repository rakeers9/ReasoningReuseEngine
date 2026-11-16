def semantic_function(ctx):
    """Compute pressure in pascals as nRT/V from moles, temperature in kelvin, volume in cubic metres and the gas constant R = 8.314462618 J/(mol K), using Decimal arithmetic."""
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    n_raw = load(ctx, "amount of gas in mol", d_type=str)
    T_raw = load(ctx, "temperature in K", d_type=str)
    V_raw = load(ctx, "volume in m3", d_type=str)
    R_raw = load(ctx, "gas constant R in J/(mol K)", d_type=str)
    narrate(ctx, f"Loaded n={n_raw!r} mol, T={T_raw!r} K, V={V_raw!r} m3, R={R_raw!r} J/(mol K)")
    n = Decimal(str(n_raw).strip())
    T = Decimal(str(T_raw).strip())
    V = Decimal(str(V_raw).strip())
    R = Decimal(str(R_raw).strip())
    if n <= 0:
        raise ValueError("Amount of gas must be positive")
    if T <= 0:
        raise ValueError("Temperature must be above absolute zero")
    if V <= 0:
        raise ValueError("Volume must be positive")
    nRT = n * R * T
    narrate(ctx, f"nRT = {nRT} J")
    P = nRT / V
    narrate(ctx, f"Pressure p = nRT/V = {P} Pa")
    return {"pressure in Pa": P}