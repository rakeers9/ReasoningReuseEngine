def semantic_function(ctx):
    '''Convert a pressure in pascals to atmospheres and megapascals for reporting.'''
    narrate(ctx, 'Converting the pressure from pascals to atmospheres and megapascals for reporting')
    pressure_pa = load(ctx, 'pressure in Pa', d_type=float)
    narrate(ctx, f'Loaded pressure: {pressure_pa} Pa')
    pressure_atm = pressure_pa / 101325.0
    pressure_mpa = pressure_pa / 1e6
    narrate(ctx, f'Using 1 atm = 1.01325e5 Pa: {pressure_pa} Pa = {pressure_atm:.4g} atm')
    narrate(ctx, f'In megapascals: {pressure_pa} Pa = {pressure_mpa:.4g} MPa')
    return {'pressure in atm': pressure_atm, 'pressure in MPa': pressure_mpa}
