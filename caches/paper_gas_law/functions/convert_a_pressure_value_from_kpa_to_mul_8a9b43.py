
def semantic_function(ctx):
    """Convert a pressure value from kPa to multiple
    common units including MPa, bar, and atm."""

    pressure_kpa = load(ctx,
        "the pressure value in kilopascals (kPa)",
        d_type=float)

    pressure_mpa = pressure_kpa / 1000
    narrate(ctx, f"{pressure_kpa:.2f} kPa / 1000 = {pressure_mpa:.2f} MPa")
    pressure_bar = pressure_kpa / 100
    narrate(ctx, f"{pressure_kpa:.2f} kPa / 100 = {pressure_bar:.2f} bar")
    pressure_atm = pressure_kpa / 101.325
    narrate(ctx, f"{pressure_kpa:.2f} kPa / 101.325 = {pressure_atm:.2f} atm")

    return {
        "the pressure in kPa": pressure_kpa,
        "the pressure in MPa": pressure_mpa,
        "the pressure in bar": pressure_bar,
        "the pressure in atm": pressure_atm,
    }
