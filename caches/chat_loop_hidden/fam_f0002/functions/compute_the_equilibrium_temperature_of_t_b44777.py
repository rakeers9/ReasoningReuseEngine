def semantic_function(ctx):
    """Compute the equilibrium temperature of two bodies mixed in an isolated system from their masses, specific heats and initial temperatures."""
    m_w_txt = load(ctx, "the mass of the water (the colder body) with unit", d_type=str)
    t_w_txt = load(ctx, "the initial temperature of the water with unit", d_type=str)
    m_a_txt = load(ctx, "the mass of the aluminum object (the hot body) with unit", d_type=str)
    t_a_txt = load(ctx, "the initial temperature of the aluminum object with unit", d_type=str)
    c_a_txt = load(ctx, "the specific heat of aluminum with unit", d_type=str)
    c_w_txt = load(ctx, "the specific heat of water with unit", d_type=str)
    narrate(ctx, f"Loaded water mass {m_w_txt!r}, water temperature {t_w_txt!r}, aluminum mass {m_a_txt!r}, aluminum temperature {t_a_txt!r}, aluminum specific heat {c_a_txt!r}, water specific heat {c_w_txt!r}")

    def get_mass(txt):
        r = do(ctx, "Parse a mass given as text with a unit (g, kg, mg, lb) into grams as a Decimal", payloads={"mass text with unit": txt})
        return load(r, "mass in grams", d_type=str, ctx=ctx)

    def get_temp(txt):
        r = do(ctx, "Parse a temperature given as text with a unit (K, °C, °F) into kelvin as a Decimal, applying the proper offset", payloads={"temperature text with unit": txt})
        return load(r, "temperature in K", d_type=str, ctx=ctx)

    def get_cp(txt):
        r = do(ctx, "Parse a specific heat given as text with a unit (J/(g·°C), J/(g·K), J/(kg·K), kJ/(kg·K), cal/(g·°C)) into J/(g·K) as a Decimal", payloads={"specific heat text with unit": txt})
        return load(r, "specific heat in J/(g·K)", d_type=str, ctx=ctx)

    m_w = get_mass(m_w_txt)
    m_a = get_mass(m_a_txt)
    t_w = get_temp(t_w_txt)
    t_a = get_temp(t_a_txt)
    c_a = get_cp(c_a_txt)
    c_w = get_cp(c_w_txt)
    narrate(ctx, f"Parsed: water {m_w} g at {t_w} K, c={c_w} J/(g·K); aluminum {m_a} g at {t_a} K, c={c_a} J/(g·K)")

    def get_hc(m, c):
        r = do(ctx, "Compute the heat capacity of a body as mass in grams times specific heat in J/(g·K)", payloads={"mass in grams": m, "specific heat in J/(g·K)": c})
        return load(r, "heat capacity in J/K", d_type=str, ctx=ctx)

    hc_w = get_hc(m_w, c_w)
    hc_a = get_hc(m_a, c_a)
    narrate(ctx, f"Heat capacities: water {hc_w} J/K, aluminum {hc_a} J/K")

    r = do(ctx, "Solve the energy balance for the final temperature as the heat-capacity-weighted average of the initial temperatures", payloads={
        "heat capacity of the first body in J/K": hc_w,
        "initial temperature of the first body in K": t_w,
        "heat capacity of the second body in J/K": hc_a,
        "initial temperature of the second body in K": t_a,
    })
    t_eq = load(r, "equilibrium temperature in K", d_type=str, ctx=ctx)
    narrate(ctx, f"Equilibrium temperature = {t_eq} K")

    chk = do(ctx, "Verify that the heat gained by the colder body equals the heat lost by the hotter body at the equilibrium temperature", payloads={
        "heat capacity of the first body in J/K": hc_w,
        "initial temperature of the first body in K": t_w,
        "heat capacity of the second body in J/K": hc_a,
        "initial temperature of the second body in K": t_a,
        "equilibrium temperature in K": t_eq,
    })
    holds = load(chk, "balance holds", d_type=bool, ctx=ctx)
    narrate(ctx, f"Energy balance check holds: {holds}")
    return {"equilibrium temperature in K": t_eq}