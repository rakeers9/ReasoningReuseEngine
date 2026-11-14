def semantic_function(ctx):
    """Cross-check the perfect gas pressure using R = 0.08206 dm3 atm K-1 mol-1 with volume in dm3 and compare it with the SI-derived pressure in atmospheres."""
    n = float(load(ctx, "the amount in moles", d_type=float))
    T = float(load(ctx, "the temperature in kelvin", d_type=float))
    V_m3 = float(load(ctx, "the volume in cubic metres", d_type=float))
    p_atm = float(load(ctx, "the pressure in atmospheres", d_type=float))
    narrate(ctx, f"Loaded n = {n} mol, T = {T} K, V = {V_m3} m3, SI-derived p = {p_atm:.6g} atm")
    if V_m3 <= 0:
        raise ValueError("Volume must be positive")
    V_dm3 = V_m3 * 1000.0
    narrate(ctx, f"Volume in cubic decimetres: {V_m3} m3 x 1000 = {V_dm3:.6g} dm3")
    R = 0.08206
    nRT = n * R * T
    narrate(ctx, f"nRT = {n} x {R} x {T} = {nRT:.6g} dm3 atm")
    p_check = nRT / V_dm3
    narrate(ctx, f"Check pressure p = nRT / V = {nRT:.6g} / {V_dm3:.6g} = {p_check:.6g} atm")
    rel = abs(p_check - p_atm) / abs(p_atm) if p_atm != 0 else abs(p_check)
    agree = rel < 1e-3
    narrate(ctx, f"Relative difference {rel:.3g}; results {'agree' if agree else 'do not agree'}")
    return {
        "the check pressure in atmospheres": p_check,
        "whether the two results agree": agree,
    }
