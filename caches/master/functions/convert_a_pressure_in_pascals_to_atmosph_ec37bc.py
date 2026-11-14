def semantic_function(ctx):
    """Convert a pressure in pascals to atmospheres, bar, kilopascals and megapascals."""
    p = load(ctx, "the pressure in pascals", d_type=float)
    p = float(p)
    narrate(ctx, f"Loaded pressure p = {p:.6g} Pa")
    atm = p / 101325.0
    narrate(ctx, f"In atmospheres: {p:.6g} Pa / 101325 Pa atm-1 = {atm:.6g} atm")
    bar = p / 1.0e5
    narrate(ctx, f"In bar: {p:.6g} Pa / 1e5 Pa bar-1 = {bar:.6g} bar")
    kpa = p / 1.0e3
    narrate(ctx, f"In kilopascals: {p:.6g} Pa / 1000 = {kpa:.6g} kPa")
    mpa = p / 1.0e6
    narrate(ctx, f"In megapascals: {p:.6g} Pa / 1e6 = {mpa:.6g} MPa")
    return {
        "the pressure in atmospheres": atm,
        "the pressure in bar": bar,
        "the pressure in kilopascals": kpa,
        "the pressure in megapascals": mpa,
    }
