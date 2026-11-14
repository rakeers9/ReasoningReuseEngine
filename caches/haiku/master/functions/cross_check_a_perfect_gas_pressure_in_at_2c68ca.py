def semantic_function(ctx):
    '''Cross-check a perfect gas pressure in atmospheres by recomputing p = nRT/V with R in L atm K-1 mol-1 and volume in litres, then compare it with the primary result.'''
    import re
    narrate(ctx, 'Cross-checking the perfect gas pressure using R = 0.082057 L atm K-1 mol-1 with the volume in litres')
    def parse(x):
        text = str(x).strip()
        m = re.search('[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?', text)
        if m is None:
            raise ValueError('No number found in: ' + text)
        unit = text[m.end():].strip().lower().replace('^', '').replace(' ', '').replace('³', '3')
        return float(m.group(0)), unit
    n, _ = parse(load(ctx, 'the amount of substance of the gas n with unit (mol)', d_type=str))
    narrate(ctx, 'Amount n = ' + str(n) + ' mol')
    T, _ = parse(load(ctx, 'the temperature of the gas in kelvin (K)', d_type=str))
    narrate(ctx, 'Temperature T = ' + str(T) + ' K')
    v_value, v_unit = parse(load(ctx, 'the volume of the gas sample with unit (dm3, L, cm3 or m3)', d_type=str))
    factors = {'l': 1.0, 'dm3': 1.0, 'cm3': 0.001, 'ml': 0.001, 'm3': 1000.0}
    if v_unit not in factors:
        raise ValueError('Unrecognised volume unit: ' + v_unit)
    V_L = v_value * factors[v_unit]
    narrate(ctx, 'Volume V = ' + str(V_L) + ' L')
    p_primary, _ = parse(load(ctx, 'the perfect gas pressure in atmospheres (atm)', d_type=str))
    narrate(ctx, 'Primary pressure = ' + str(p_primary) + ' atm')
    if n <= 0 or T <= 0 or V_L <= 0:
        raise ValueError('Amount, temperature and volume must be positive')
    R_L = 0.082057
    numerator = n * R_L * T
    narrate(ctx, 'Numerator nRT = ' + str(numerator) + ' L atm')
    p_check = numerator / V_L
    narrate(ctx, 'Cross-check pressure p = nRT/V = ' + str(p_check) + ' atm')
    rel_diff = abs(p_check - p_primary) / abs(p_primary)
    agree = rel_diff <= 0.01
    narrate(ctx, 'Relative difference ' + str(rel_diff) + '; agreement with primary result: ' + str(agree))
    return {'the cross-check pressure in atmospheres (atm)': p_check, 'agreement of the cross-check with the primary pressure': agree}
