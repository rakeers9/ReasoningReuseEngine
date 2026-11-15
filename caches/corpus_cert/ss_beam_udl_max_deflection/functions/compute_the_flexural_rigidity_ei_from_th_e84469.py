def semantic_function(ctx):
    '''Compute the flexural rigidity EI from the modulus of elasticity in pascals and the second moment of area in metres to the fourth.'''
    narrate(ctx, 'Computing the flexural rigidity EI as the product of E in pascals and I in metres to the fourth')
    E_pa = load(ctx, 'the modulus of elasticity in pascals', d_type=float)
    I_m4 = load(ctx, 'the second moment of area in metres to the fourth', d_type=float)
    narrate(ctx, 'Loaded E = ' + str(E_pa) + ' Pa and I = ' + str(I_m4) + ' m^4')
    EI = E_pa * I_m4
    narrate(ctx, 'Flexural rigidity EI = ' + str(EI) + ' N m^2')
    return {'the flexural rigidity EI in newton square metres': EI}