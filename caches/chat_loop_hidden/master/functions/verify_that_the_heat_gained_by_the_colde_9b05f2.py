def semantic_function(ctx):
    """Verify that the heat gained by the colder body equals the heat lost by the hotter body at the equilibrium temperature."""
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
    teq_raw = load(ctx, "equilibrium temperature in K", d_type=str)
    narrate(ctx, f"Loaded heat capacity of first body: {c1_raw!r} J/K")
    narrate(ctx, f"Loaded initial temperature of first body: {t1_raw!r} K")
    narrate(ctx, f"Loaded heat capacity of second body: {c2_raw!r} J/K")
    narrate(ctx, f"Loaded initial temperature of second body: {t2_raw!r} K")
    narrate(ctx, f"Loaded equilibrium temperature: {teq_raw!r} K")
    c1 = to_dec(c1_raw, "first heat capacity")
    t1 = to_dec(t1_raw, "first temperature")
    c2 = to_dec(c2_raw, "second heat capacity")
    t2 = to_dec(t2_raw, "second temperature")
    teq = to_dec(teq_raw, "equilibrium temperature")
    # identify the colder and hotter bodies
    if t1 <= t2:
        cc, tc, ch, th = c1, t1, c2, t2
    else:
        cc, tc, ch, th = c2, t2, c1, t1
    gained = cc * (teq - tc)
    lost = ch * (th - teq)
    narrate(ctx, f"Heat gained by colder body = {cc} * ({teq} - {tc}) = {gained} J")
    narrate(ctx, f"Heat lost by hotter body = {ch} * ({th} - {teq}) = {lost} J")
    diff = abs(gained - lost)
    tol = Decimal('1e-20') * max(abs(gained), abs(lost), Decimal(1))
    holds = diff <= tol
    narrate(ctx, f"Difference = {diff} J; balance holds: {holds}")
    return {"heat gained in J": gained, "heat lost in J": lost, "balance holds": holds}
