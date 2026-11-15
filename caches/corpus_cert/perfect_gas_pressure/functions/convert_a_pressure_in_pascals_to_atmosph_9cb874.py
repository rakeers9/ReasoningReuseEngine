def semantic_function(ctx):
    '''Convert a pressure in pascals to atmospheres, bar and megapascals.'''
    narrate(ctx, 'Converting the pressure in pascals to atmospheres, bar and megapascals')
    p_pa = float(load(ctx, 'perfect gas pressure in pascals', d_type=float))
    narrate(ctx, f'Loaded pressure: {p_pa} Pa')
    p_atm = p_pa / 101325.0
    p_bar = p_pa / 1e5
    p_mpa = p_pa / 1e6
    narrate(ctx, f'Pressure in atm: {p_pa} / 101325 = {p_atm} atm')
    narrate(ctx, f'Pressure in bar: {p_pa} / 1e5 = {p_bar} bar')
    narrate(ctx, f'Pressure in MPa: {p_pa} / 1e6 = {p_mpa} MPa')
    return {'pressure in atmospheres': p_atm, 'pressure in bar': p_bar, 'pressure in megapascals': p_mpa}