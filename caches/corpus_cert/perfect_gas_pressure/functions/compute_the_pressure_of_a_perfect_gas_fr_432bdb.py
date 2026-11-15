def semantic_function(ctx):
    '''Compute the pressure of a perfect gas from p = nRT/V using SI units, giving the result in pascals.'''
    narrate(ctx, 'Computing the perfect gas pressure p = nRT/V in SI units')
    n = load(ctx, 'amount of gas n in moles', d_type=float)
    T = load(ctx, 'absolute temperature in kelvin', d_type=float)
    V = load(ctx, 'volume in cubic metres', d_type=float)
    narrate(ctx, f'Loaded n = {n} mol, T = {T} K, V = {V} m3')
    R = 8.314
    nRT = n * R * T
    narrate(ctx, f'nRT = {n} x {R} x {T} = {nRT} J')
    p = nRT / V
    narrate(ctx, f'p = nRT/V = {nRT} / {V} = {p} Pa')
    return {'perfect gas pressure in pascals': p}