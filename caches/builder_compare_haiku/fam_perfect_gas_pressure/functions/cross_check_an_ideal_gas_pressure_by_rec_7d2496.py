from decimal import Decimal, localcontext

def semantic_function(ctx):
    '''Cross-check an ideal gas pressure by recomputing nRT/V in SI units and comparing it with the primary pressure.'''
    narrate(ctx, 'Cross-checking the ideal gas pressure by recomputing nRT/V in SI units with R = 8.314 J/(mol K) and 101325 Pa per atm')
    n = Decimal(str(load(ctx, 'amount of substance n in mol', d_type=str)).strip())
    T = Decimal(str(load(ctx, 'temperature in kelvin', d_type=str)).strip())
    V_L = Decimal(str(load(ctx, 'volume in litres', d_type=str)).strip())
    p_primary = Decimal(str(load(ctx, 'pressure in atm', d_type=str)).strip())
    narrate(ctx, 'Inputs: n = ' + str(n) + ' mol, T = ' + str(T) + ' K, V = ' + str(V_L) + ' L, primary pressure = ' + str(p_primary) + ' atm')
    if n <= 0 or T <= 0 or V_L <= 0 or p_primary <= 0:
        narrate(ctx, 'Amount, temperature, volume and pressure must all be positive for the cross-check')
        raise ValueError('cross-check inputs must be positive')
    R_SI = Decimal('8.314')
    PA_PER_ATM = Decimal('101325')
    L_PER_M3 = Decimal('1000')
    with localcontext() as c:
        c.prec = 60
        V_m3 = V_L / L_PER_M3
        nRT_SI = n * R_SI * T
        p_pa = nRT_SI / V_m3
        p_atm_si = p_pa / PA_PER_ATM
        rel_diff = (p_atm_si - p_primary) / p_primary
    narrate(ctx, 'Volume in SI = ' + str(V_m3) + ' m3')
    narrate(ctx, 'nRT = ' + str(nRT_SI) + ' Pa m3')
    narrate(ctx, 'SI pressure = ' + str(p_pa) + ' Pa')
    narrate(ctx, 'SI cross-check pressure = ' + str(p_atm_si) + ' atm, relative difference from primary = ' + str(rel_diff))
    return {'SI cross-check pressure in atm': p_atm_si, 'relative difference between SI cross-check and primary pressure': rel_diff}
