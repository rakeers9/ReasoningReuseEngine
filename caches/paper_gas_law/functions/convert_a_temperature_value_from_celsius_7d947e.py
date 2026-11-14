
def semantic_function(ctx):
    """Convert a temperature value from Celsius
    or Kelvin to Kelvin."""

    temp_with_unit = load(ctx,
        "the temperature value with unit",
        d_type=str)
    narrate(ctx,
        f"Received temperature specification: {temp_with_unit}")

    temp_value = extract(ctx, temp_with_unit,
        "the numeric temperature value",
        d_type=float)
    narrate(ctx,
        f"Extracted numeric temperature value: {temp_value}")

    temp_unit = lm(ctx,
        f"What is the temperature unit in "
        f"'{temp_with_unit}'? "
        f"Reply with exactly one of: C, K")
    temp_unit = temp_unit.strip().upper()
    narrate(ctx,
        f"Detected temperature unit: {temp_unit}")

    if "C" in temp_unit:
        temp_kelvin = temp_value + 273
        narrate(ctx,
            f"{temp_value} °C + 273 = {temp_kelvin} K")
    else:
        temp_kelvin = temp_value
        narrate(ctx,
            f"Already in Kelvin: {temp_kelvin} K")

    return {
        "the temperature in Kelvin": temp_kelvin
    }
