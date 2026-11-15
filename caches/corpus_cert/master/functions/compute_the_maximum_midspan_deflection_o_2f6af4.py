def semantic_function(ctx):
    '''Compute the maximum midspan deflection of a simply supported beam under a uniform load over its full span as 5wL^4 divided by 384EI.'''
    narrate(ctx, 'Computing the maximum midspan deflection of a simply supported beam under a full-span uniform load using 5wL^4 divided by 384EI')
    L_m = load(ctx, 'the span length in metres', d_type=float)
    w_npm = load(ctx, 'the uniform load intensity in newtons per metre', d_type=float)
    E_pa = load(ctx, 'the modulus of elasticity in pascals', d_type=float)
    I_m4 = load(ctx, 'the second moment of area in metres to the fourth', d_type=float)
    narrate(ctx, 'Loaded span L = ' + str(L_m) + ' m, uniform load w = ' + str(w_npm) + ' N/m, E = ' + str(E_pa) + ' Pa, I = ' + str(I_m4) + ' m^4')
    rigidity_result = do(ctx, 'Compute the flexural rigidity EI from the modulus of elasticity in pascals and the second moment of area in metres to the fourth', payloads={'the modulus of elasticity in pascals': E_pa, 'the second moment of area in metres to the fourth': I_m4})
    EI = load(rigidity_result, 'the flexural rigidity EI in newton square metres', d_type=float, ctx=ctx)
    narrate(ctx, 'Flexural rigidity EI = ' + str(EI) + ' N m^2')
    numerator = 5.0 * w_npm * L_m ** 4
    narrate(ctx, 'Numerator 5wL^4 = ' + str(numerator) + ' N m^3')
    denominator = 384.0 * EI
    narrate(ctx, 'Denominator 384EI = ' + str(denominator) + ' N m^2')
    delta_m = numerator / denominator
    narrate(ctx, 'Maximum midspan deflection delta_max = ' + str(delta_m) + ' m, occurring at midspan')
    return {'the maximum midspan deflection in metres': delta_m}
