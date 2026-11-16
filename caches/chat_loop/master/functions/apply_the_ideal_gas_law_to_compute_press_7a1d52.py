def semantic_function(ctx):
    '''Apply the ideal gas law to compute pressure in pascals from amount in moles, temperature in kelvin and volume in cubic metres.'''
    from decimal import Decimal
    narrate(ctx, 'Applying the ideal gas law p = nRT/V to get the pressure in pascals')
    n = Decimal(str(load(ctx, 'amount of substance in moles', d_type=str)).strip())
    T = Decimal(str(load(ctx, 'temperature in kelvin', d_type=str)).strip())
    V = Decimal(str(load(ctx, 'volume in cubic metres', d_type=str)).strip())
    narrate(ctx, 'Amount of substance n = ' + str(n) + ' mol')
    narrate(ctx, 'Temperature T = ' + str(T) + ' K')
    narrate(ctx, 'Volume V = ' + str(V) + ' m3')
    if n < 0:
        raise ValueError('amount of substance must be non-negative, got ' + str(n))
    if T <= 0:
        raise ValueError('temperature in kelvin must be strictly positive, got ' + str(T))
    if V <= 0:
        raise ValueError('volume in cubic metres must be strictly positive, got ' + str(V))
    R = Decimal('8.314462618')
    narrate(ctx, 'Molar gas constant R = 8.314462618 J/(K mol)')
    nRT = n * R * T
    narrate(ctx, 'nRT = ' + str(nRT) + ' J')
    p = nRT / V
    narrate(ctx, 'Pressure p = nRT/V = ' + str(p) + ' Pa')
    return {'pressure in pascals': p}
