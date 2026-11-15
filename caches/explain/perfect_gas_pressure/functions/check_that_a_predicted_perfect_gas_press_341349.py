def semantic_function(ctx):
    '''Check that a predicted perfect gas pressure is insensitive to rounding the temperature to 300 K.'''
    narrate(ctx, 'Checking that the predicted pressure is insensitive to rounding the temperature to 300 K')
    n = float(load(ctx, 'amount of substance in mol', d_type=float))
    narrate(ctx, f'Loaded amount n = {n} mol')
    V = float(load(ctx, 'volume in cubic metres', d_type=float))
    narrate(ctx, f'Loaded volume V = {V} m3')
    T = float(load(ctx, 'absolute temperature in kelvin', d_type=float))
    narrate(ctx, f'Loaded absolute temperature T = {T} K')
    p_original = float(load(ctx, 'pressure in pascals', d_type=float))
    narrate(ctx, f'Loaded predicted pressure p = {p_original} Pa')
    T_rounded = 300.0
    narrate(ctx, f'Rounding the temperature to {T_rounded} K for the check')
    result = do(ctx, 'Compute the pressure of a perfect gas from its amount, absolute temperature and volume using p = nRT/V', payloads={'amount of substance in mol': n, 'absolute temperature in kelvin': T_rounded, 'volume in cubic metres': V})
    p_rounded = float(load(result, 'pressure in pascals', d_type=float, ctx=ctx))
    narrate(ctx, f'Pressure at the rounded temperature p = {p_rounded} Pa')
    sig = 3
    a = float(f'{p_original:.{sig}g}')
    b = float(f'{p_rounded:.{sig}g}')
    agree = a == b
    narrate(ctx, f'Rounded to {sig} significant figures: {a} Pa versus {b} Pa')
    narrate(ctx, 'Pressures agree' if agree else 'Pressures do not agree')
    return {'pressure agrees at the rounded temperature to the stated significant figures': agree, 'pressure at the rounded temperature in pascals': p_rounded}
