def semantic_function(ctx):
    '''Compute the pressure in pascals of a perfect gas from pV = nRT given amount, gas constant, kelvin temperature and volume in cubic metres.'''
    narrate(ctx, 'Computing the perfect gas pressure in pascals from pV = nRT')
    def to_number(x):
        if isinstance(x, str):
            return float(x.strip().split()[0])
        return float(x)
    n_raw = load(ctx, 'the amount of substance of the gas n with unit (mol)', d_type=float)
    n = to_number(n_raw)
    narrate(ctx, 'Amount of substance n = ' + str(n) + ' mol')
    R_raw = load(ctx, 'the molar gas constant R in J K-1 mol-1', d_type=float)
    R = to_number(R_raw)
    narrate(ctx, 'Molar gas constant R = ' + str(R) + ' J K-1 mol-1')
    T_raw = load(ctx, 'the temperature of the gas in kelvin (K)', d_type=float)
    T = to_number(T_raw)
    narrate(ctx, 'Temperature T = ' + str(T) + ' K')
    V_raw = load(ctx, 'the volume of the gas sample in cubic metres (m3)', d_type=float)
    V = to_number(V_raw)
    narrate(ctx, 'Volume V = ' + str(V) + ' m3')
    if n <= 0 or V <= 0 or T <= 0:
        raise ValueError('Amount, volume and temperature must be positive')
    numerator = n * R * T
    narrate(ctx, 'Numerator nRT = ' + str(numerator) + ' J')
    pressure = numerator / V
    narrate(ctx, 'Pressure p = nRT / V = ' + str(pressure) + ' Pa')
    return {'the perfect gas pressure in pascals (Pa)': pressure}