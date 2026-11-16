from decimal import Decimal, localcontext

def semantic_function(ctx):
    '''Compute the ideal gas pressure p = nRT/V from amount, gas constant, temperature and volume using exact decimal arithmetic.'''
    narrate(ctx, 'Computing the ideal gas pressure p = nRT/V with exact decimal arithmetic')
    n = Decimal(str(load(ctx, 'amount of substance n in mol', d_type=str)).strip())
    R = Decimal(str(load(ctx, 'gas constant R in L·atm/(mol·K)', d_type=str)).strip())
    T = Decimal(str(load(ctx, 'temperature in kelvin', d_type=str)).strip())
    V = Decimal(str(load(ctx, 'volume in litres', d_type=str)).strip())
    narrate(ctx, 'Amount n = ' + str(n) + ' mol')
    narrate(ctx, 'Gas constant R = ' + str(R) + ' L·atm/(mol·K)')
    narrate(ctx, 'Temperature T = ' + str(T) + ' K')
    narrate(ctx, 'Volume V = ' + str(V) + ' L')
    if n <= 0 or R <= 0 or T <= 0 or V <= 0:
        narrate(ctx, 'Amount, gas constant, temperature and volume must all be positive')
        raise ValueError('ideal gas inputs must be positive')
    with localcontext() as c:
        c.prec = 60
        nRT = n * R * T
        p = nRT / V
    narrate(ctx, 'nRT = ' + str(nRT) + ' L·atm')
    narrate(ctx, 'Pressure p = nRT/V = ' + str(p) + ' atm')
    return {'pressure in atm': p}
