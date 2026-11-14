def semantic_function(ctx):
    """Predict the pressure exerted by a gas confined to a given volume at a given temperature using the perfect gas law, reporting it in pascals, megapascals and atmospheres."""
    amount_str = load(ctx, "the amount of gas with unit", d_type=str)
    formula = load(ctx, "the chemical formula of the gas with its phase", d_type=str)
    volume_str = load(ctx, "the volume of the gas with unit", d_type=str)
    temp_str = load(ctx, "the temperature of the gas with unit", d_type=str)
    name = load(ctx, "the common name of the gas", d_type=str)
    narrate(ctx, f"Gas: {name} ({formula}); amount = {amount_str}, volume = {volume_str}, temperature = {temp_str}")

    def parse(q):
        r = do(ctx, "Split a physical quantity string such as '10.0 mol' or '27°C' into its numeric value and its unit.",
               payloads={"the quantity string with value and unit": q})
        return float(load(r, "the numeric value", d_type=float, ctx=ctx)), load(r, "the unit string", d_type=str, ctx=ctx)

    n_val, n_unit = parse(amount_str)
    v_val, v_unit = parse(volume_str)
    t_val, t_unit = parse(temp_str)
    narrate(ctx, f"Parsed: n = {n_val} {n_unit}, V = {v_val} {v_unit}, T = {t_val} {t_unit}")

    r = do(ctx, "Convert an amount of substance given with a unit (mol, mmol, kmol, umol, nmol) to moles.",
           payloads={"the amount value": n_val, "the amount unit": n_unit})
    n = float(load(r, "the amount in moles", d_type=float, ctx=ctx))
    r = do(ctx, "Convert a volume given in a common unit (dm3, L, cm3, mL, m3) to cubic metres.",
           payloads={"the volume value": v_val, "the volume unit": v_unit})
    V = float(load(r, "the volume in cubic metres", d_type=float, ctx=ctx))
    r = do(ctx, "Convert a temperature given in Celsius, Fahrenheit, Rankine or kelvin to kelvin.",
           payloads={"the temperature value": t_val, "the temperature unit": t_unit})
    T = float(load(r, "the temperature in kelvin", d_type=float, ctx=ctx))
    narrate(ctx, f"SI values: n = {n} mol, V = {V:.6g} m3, T = {T} K")

    r = do(ctx, "Apply the perfect gas law p = nRT/V in SI units with R = 8.314 J K-1 mol-1 to obtain the pressure in pascals.",
           payloads={"the amount in moles": n, "the temperature in kelvin": T, "the volume in cubic metres": V})
    p_pa = float(load(r, "the pressure in pascals", d_type=float, ctx=ctx))
    narrate(ctx, f"Perfect gas pressure p = {p_pa:.6g} Pa")

    r = do(ctx, "Convert a pressure in pascals to atmospheres, bar, kilopascals and megapascals.",
           payloads={"the pressure in pascals": p_pa})
    p_atm = float(load(r, "the pressure in atmospheres", d_type=float, ctx=ctx))
    p_mpa = float(load(r, "the pressure in megapascals", d_type=float, ctx=ctx))
    p_bar = float(load(r, "the pressure in bar", d_type=float, ctx=ctx))
    narrate(ctx, f"p = {p_mpa:.4g} MPa = {p_atm:.4g} atm = {p_bar:.4g} bar")

    try:
        r = do(ctx, "Cross-check the perfect gas pressure using R = 0.08206 dm3 atm K-1 mol-1 with volume in dm3 and compare it with the SI-derived pressure in atmospheres.",
               payloads={"the amount in moles": n, "the temperature in kelvin": T, "the volume in cubic metres": V, "the pressure in atmospheres": p_atm})
        chk = load(r, "the check pressure in atmospheres", d_type=float, ctx=ctx)
        narrate(ctx, f"Cross-check with R = 0.08206 dm3 atm K-1 mol-1 gives {float(chk):.4g} atm")
    except Exception as e:
        narrate(ctx, f"Cross-check skipped: {e}")

    answer = f"p ≈ {p_pa:.3e} Pa = {p_mpa:.3g} MPa ≈ {p_atm:.3g} atm"
    narrate(ctx, f"Final answer: {answer}")
    return {"the final pressure answer": answer}
