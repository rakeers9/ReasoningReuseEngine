
def semantic_function(ctx):
    """Calculate the pressure exerted by a gas
    using PV = nRT, rearranged to P = nRT/V."""

    molar_amount_str = load(ctx,
        "the molar amount of gas with unit",
        d_type=str)
    volume_str = load(ctx,
        "the volume of the container with unit",
        d_type=str)
    temperature_str = load(ctx,
        "the temperature with unit", d_type=str)
    gas_name = load(ctx,
        "the common name of the gas", d_type=str)
    gas_model = load(ctx,
        "the gas model or assumption to use",
        d_type=str)

    narrate(ctx, f"Calculating pressure for "
        f"{gas_name} using the {gas_model} law")
    narrate(ctx, f"Given: n = {molar_amount_str}, "
        f"V = {volume_str}, T = {temperature_str}")

    n = extract(ctx, molar_amount_str,
        "the numeric molar amount", d_type=float)
    narrate(ctx, f"Extracted amount of substance: n = {n} mol")
    volume_value = extract(ctx, volume_str,
        "the numeric volume value", d_type=float)
    narrate(ctx, f"Extracted volume: V = {volume_value} dm^3")

    # Convert temperature via utility function
    narrate(ctx, "Converting temperature...")
    temp_result = do(ctx,
        "Convert a temperature value from Celsius "
        "or Kelvin to Kelvin given a temperature "
        "string with unit",
        payloads={
            "the temperature value with unit":
                temperature_str})
    T = load(temp_result,
        "the temperature in Kelvin",
        d_type=float, ctx=ctx)
    narrate(ctx, f"Temperature in Kelvin: T = {T} K")

    R = 8.314 # kPa.dm^3.mol^-1.K^-1
    narrate(ctx, "Using R = 8.314 kPa·dm^3·mol^-1·K^-1")
    narrate(ctx, f"P = nRT/V = {n} × {R} × {T}"
        f" / {volume_value}")

    pressure_kpa = n * R * T / volume_value
    narrate(ctx, f"P = {pressure_kpa:.2f} kPa")

    # Convert to other units via utility function
    narrate(ctx, "Converting pressure to other units...")
    pressure_result = do(ctx,
        "Convert a pressure value from kPa to "
        "multiple common units including MPa, "
        "bar, and atm",
        payloads={
            "the pressure value in kilopascals "
            "(kPa)": pressure_kpa})

    pressure_mpa = load(pressure_result,
        "the pressure in MPa",
        d_type=float, ctx=ctx)
    pressure_bar = load(pressure_result,
        "the pressure in bar", d_type=float, ctx=ctx)
    pressure_atm = load(pressure_result,
        "the pressure in atm", d_type=float, ctx=ctx)
    narrate(ctx, f"Final: {pressure_kpa:.2f} kPa = {pressure_mpa:.2f} MPa"
        f" = {pressure_bar:.2f} bar = {pressure_atm:.2f} atm")

    return {
        "the calculated pressure in kPa":
            pressure_kpa,
        "the calculated pressure in MPa":
            pressure_mpa,
        "the calculated pressure in bar": pressure_bar,
        "the calculated pressure in atm": pressure_atm,
        "summary of the calculation":
            f"The pressure exerted by {n} mol of {gas_name} in {volume_value} dm^3 "
            f"at {T} K is {pressure_mpa:.2f} MPa (or {pressure_kpa / 1000:.2f} x 10^3 kPa)",
    }
