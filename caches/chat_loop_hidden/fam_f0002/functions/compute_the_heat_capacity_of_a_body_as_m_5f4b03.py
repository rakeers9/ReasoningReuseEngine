def semantic_function(ctx):
    """Compute the heat capacity of a body in J/K as its mass in grams times its specific heat in J/(g·K)."""
    from decimal import Decimal, InvalidOperation
    mass_raw = load(ctx, "mass in grams", d_type=str)
    cp_raw = load(ctx, "specific heat in J/(g·K)", d_type=str)
    narrate(ctx, f"Loaded mass in grams: {mass_raw!r}")
    narrate(ctx, f"Loaded specific heat in J/(g·K): {cp_raw!r}")
    def to_dec(x, name):
        s = str(x).strip().replace(',', '')
        try:
            return Decimal(s)
        except InvalidOperation:
            narrate(ctx, f"Could not parse {name} from {x!r}")
            raise ValueError(f"Cannot parse {name}: {x!r}")
    mass = to_dec(mass_raw, "mass")
    cp = to_dec(cp_raw, "specific heat")
    if mass <= 0:
        raise ValueError("Mass must be positive")
    if cp <= 0:
        raise ValueError("Specific heat must be positive")
    hc = mass * cp
    narrate(ctx, f"Heat capacity = mass {mass} g * specific heat {cp} J/(g·K) = {hc} J/K")
    return {"heat capacity in J/K": hc}
