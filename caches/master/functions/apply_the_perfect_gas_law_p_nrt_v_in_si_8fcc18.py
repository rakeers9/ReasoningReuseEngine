def semantic_function(ctx):
    """Apply the perfect gas law p = nRT/V in SI units (R = 8.314 J K-1 mol-1) to obtain the pressure in pascals."""
    n = load(ctx, "the amount in moles", d_type=float)
    T = load(ctx, "the temperature in kelvin", d_type=float)
    V = load(ctx, "the volume in cubic metres", d_type=float)
    narrate(ctx, f"Loaded n = {n} mol, T = {T} K, V = {V} m3")
    if V is None or float(V) <= 0:
        raise ValueError("Volume must be positive")
    if T is None or float(T) <= 0:
        raise ValueError("Temperature must be above absolute zero")
    R = 8.314
    nRT = float(n) * R * float(T)
    narrate(ctx, f"nRT = {n} x {R} x {T} = {nRT} J")
    p = nRT / float(V)
    narrate(ctx, f"p = nRT / V = {nRT} J / {V} m3 = {p:.6g} Pa")
    return {"the pressure in pascals": p}
