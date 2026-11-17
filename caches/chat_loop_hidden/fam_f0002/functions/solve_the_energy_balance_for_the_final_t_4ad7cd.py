def semantic_function(ctx):
    """Solve the energy balance for the equilibrium temperature as the heat-capacity-weighted average of the two initial temperatures."""
    from decimal import Decimal, InvalidOperation
    def to_dec(x, name):
        s = str(x).strip().replace(',', '')
        try:
            return Decimal(s)
        except InvalidOperation:
            narrate(ctx, f"Could not parse {name} from {x!r}")
            raise ValueError(f"Cannot parse {name}: {x!r}")
    c1_raw = load(ctx, "heat capacity of the first body in J/K", d_type=str)
    t1_raw = load(ctx, "initial temperature of the first body in K", d_type=str)
    c2_raw = load(ctx, "heat capacity of the second body in J/K", d_type=str)
    t2_raw = load(ctx, "initial temperature of the second body in K", d_type=str)
    narrate(ctx, f"Loaded heat capacity of first body: {c1_raw!r} J/K")
    narrate(ctx, f"Loaded initial temperature of first body: {t1_raw!r} K")
    narrate(ctx, f"Loaded heat capacity of second body: {c2_raw!r} J/K")
    narrate(ctx, f"Loaded initial temperature of second body: {t2_raw!r} K")
    c1 = to_dec(c1_raw, "first heat capacity")
    t1 = to_dec(t1_raw, "first temperature")
    c2 = to_dec(c2_raw, "second heat capacity")
    t2 = to_dec(t2_raw, "second temperature")
    if c1 <= 0 or c2 <= 0:
        raise ValueError("Heat capacities must be positive")
    total = c1 + c2
    narrate(ctx, f"Total heat capacity = {c1} + {c2} = {total} J/K")
    numerator = c1 * t1 + c2 * t2
    narrate(ctx, f"Weighted sum of temperatures = {numerator}")
    t_eq = numerator / total
    narrate(ctx, f"Energy balance C1(Teq-T1)+C2(Teq-T2)=0 gives Teq = {numerator}/{total} = {t_eq} K")
    return {"equilibrium temperature in K": t_eq, "total heat capacity in J/K": total}
