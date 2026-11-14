def semantic_function(ctx):
    '''Cross-check a perfect gas pressure by recomputing it with the gas constant in L atm K-1 mol-1.'''
    import math
    narrate(ctx, 'Cross-checking the perfect gas pressure using R = 0.082057 L atm K-1 mol-1')
    n = load(ctx, 'the amount of substance of the gas in mol', d_type=float)
    narrate(ctx, f'Loaded amount of substance: {n} mol')
    T = load(ctx, 'temperature in K', d_type=float)
    narrate(ctx, f'Loaded temperature: {T} K')
    volume_text = load(ctx, 'the volume of the container holding the gas with unit', d_type=str)
    narrate(ctx, f'Loaded volume: {volume_text}')
    p_pa = load(ctx, 'pressure in Pa', d_type=float)
    narrate(ctx, f'Loaded pressure: {p_pa} Pa')
    conv = do(ctx, 'Convert a volume with unit to SI volume in cubic metres', payloads={'the volume of the container holding the gas with unit': volume_text})
    V_L = load(conv, 'volume in L', d_type=float, ctx=ctx)
    narrate(ctx, f'Volume in litres: {V_L} L')
    if not (n > 0 and T > 0 and V_L > 0):
        raise ValueError('amount, temperature and volume must all be positive')
    R = 0.082057
    p_atm = n * R * T / V_L
    narrate(ctx, f'p = nRT/V = {n} x {R} x {T} / {V_L} = {p_atm:.4g} atm')
    p_atm_from_pa = p_pa / 101325.0
    narrate(ctx, f'Same pressure from the SI result: {p_atm_from_pa:.4g} atm')
    agree = math.isclose(p_atm, p_atm_from_pa, rel_tol=1e-3)
    narrate(ctx, 'The two pressure computations agree' if agree else 'The two pressure computations disagree')
    return {'cross-checked pressure in atm': p_atm, 'agreement of the two pressure computations': agree}
