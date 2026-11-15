def semantic_function(ctx):
    '''Express a pressure given in pascals in megapascals, bar and atmospheres for reference.'''
    narrate(ctx, 'Expressing the pressure in megapascals, bar and atmospheres for reference')
    p_pa = float(load(ctx, 'pressure in pascals', d_type=float))
    narrate(ctx, 'Loaded pressure p = ' + str(p_pa) + ' Pa')
    p_mpa = p_pa / 1e6
    p_bar = p_pa / 1e5
    p_atm = p_pa / 101325.0
    narrate(ctx, 'Since 1 MPa = 1e6 Pa, p = ' + str(p_mpa) + ' MPa')
    narrate(ctx, 'Since 1 bar = 1e5 Pa, p = ' + str(p_bar) + ' bar')
    narrate(ctx, 'Since 1 atm = 101325 Pa, p = ' + str(p_atm) + ' atm')
    return {'pressure in megapascals': p_mpa, 'pressure in bar': p_bar, 'pressure in atmospheres': p_atm}