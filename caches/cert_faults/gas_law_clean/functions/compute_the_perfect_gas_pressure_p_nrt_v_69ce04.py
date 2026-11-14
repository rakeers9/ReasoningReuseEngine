def semantic_function(ctx):
    '''Compute the perfect gas pressure p = nRT/V in SI units from the amount, temperature and volume.'''
    narrate(ctx, 'Computing the perfect gas pressure p = nRT/V in SI units')
    R = 8.314
    n = load(ctx, 'the amount of substance of the gas in mol', d_type=float)
    narrate(ctx, f'Loaded amount of substance: {n} mol')
    T = load(ctx, 'temperature in K', d_type=float)
    narrate(ctx, f'Loaded temperature: {T} K')
    V = load(ctx, 'volume in m3', d_type=float)
    narrate(ctx, f'Loaded volume: {V} m3')
    if not (n > 0 and T > 0 and V > 0):
        raise ValueError('amount, temperature and volume must all be positive')
    nRT = n * R * T
    narrate(ctx, f'nRT = {n} mol x {R} J K-1 mol-1 x {T} K = {nRT:.5g} J')
    p = nRT / V
    narrate(ctx, f'p = nRT/V = {nRT:.5g} J / {V:.4g} m3 = {p:.4g} Pa')
    return {'pressure in Pa': p}