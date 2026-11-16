def semantic_function(ctx):
    """Compute the pressure of an ideal gas in pascals and atmospheres from the amount of gas, its volume and its temperature using pV = nRT."""
    amount_raw = load(ctx, "the amount of gas with unit", d_type=str)
    gas_raw = load(ctx, "the chemical formula or name of the gas", d_type=str)
    volume_raw = load(ctx, "the volume of the gas with unit", d_type=str)
    temp_raw = load(ctx, "the temperature of the gas with unit", d_type=str)
    narrate(ctx, f"Loaded amount {amount_raw!r}, gas {gas_raw!r}, volume {volume_raw!r}, temperature {temp_raw!r}")
    narrate(ctx, "The identity of the gas does not affect the ideal gas calculation")
    n_res = do(ctx, "Parse the amount of gas given as raw text (number, scale word, unit) and convert it to moles.",
               payloads={"the amount of gas with unit as raw text": amount_raw})
    n = load(n_res, "amount of gas in mol", d_type=str, ctx=ctx)
    v_res = do(ctx, "Parse the gas volume given as raw text and convert it to cubic metres, handling m3, dm3, L, mL and cm3.",
               payloads={"the gas volume with unit as raw text": volume_raw})
    V = load(v_res, "volume in m3", d_type=str, ctx=ctx)
    t_res = do(ctx, "Parse the gas temperature given as raw text and convert it to kelvin, handling K, \u00b0C and \u00b0F.",
               payloads={"the gas temperature with unit as raw text": temp_raw})
    T = load(t_res, "temperature in K", d_type=str, ctx=ctx)
    narrate(ctx, f"Parsed n={n} mol, V={V} m3, T={T} K")
    p_res = do(ctx, "Compute pressure in pascals as nRT/V from moles, temperature in kelvin, volume in cubic metres and the gas constant R using Decimal arithmetic.",
               payloads={"amount of gas in mol": str(n),
                         "temperature in K": str(T),
                         "volume in m3": str(V),
                         "gas constant R in J/(mol K)": "8.314462618"})
    p_pa = load(p_res, "pressure in Pa", d_type=str, ctx=ctx)
    narrate(ctx, f"Pressure in Pa: {p_pa}")
    a_res = do(ctx, "Convert a pressure in pascals to atmospheres by dividing by 101325 Pa per atm.",
               payloads={"pressure in Pa": str(p_pa),
                         "pascals per atmosphere": "101325"})
    p_atm = load(a_res, "pressure in atm", d_type=str, ctx=ctx)
    narrate(ctx, f"Pressure in atm: {p_atm}")
    return {"pressure in Pa": p_pa, "pressure in atm": p_atm}