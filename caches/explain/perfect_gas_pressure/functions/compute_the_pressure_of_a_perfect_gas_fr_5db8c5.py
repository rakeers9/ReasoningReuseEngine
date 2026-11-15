def semantic_function(ctx):
    '''Compute the pressure of a perfect gas from its amount, absolute temperature and volume using p = nRT/V.'''
    narrate(ctx, 'Computing the pressure of a perfect gas from p = nRT/V')
    n = float(load(ctx, 'the amount of substance in mol', d_type=float))
    narrate(ctx, 'Loaded amount of substance n = ' + str(n) + ' mol')
    T = float(load(ctx, 'the absolute temperature in kelvin', d_type=float))
    narrate(ctx, 'Loaded absolute temperature T = ' + str(T) + ' K')
    V = float(load(ctx, 'the volume in cubic metres', d_type=float))
    narrate(ctx, 'Loaded volume V = ' + str(V) + ' m3')
    if n <= 0 or T <= 0 or V <= 0:
        narrate(ctx, 'Amount, temperature and volume must all be positive')
        raise ValueError('Non-physical input to perfect gas equation')
    R = 8.314
    narrate(ctx, 'Using gas constant R = 8.314 J K^-1 mol^-1, equivalent to Pa m3 K^-1 mol^-1')
    nRT = n * R * T
    narrate(ctx, 'Product nRT = ' + str(nRT) + ' J')
    p = nRT / V
    narrate(ctx, 'Pressure p = nRT/V = ' + str(p) + ' Pa, since 1 J = 1 Pa m3')
    return {'pressure in pascals': p}