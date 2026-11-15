def semantic_function(ctx):
    '''Compute the maximum deflection in millimetres of a simply supported beam under a full-span uniform load from its span, load intensity, modulus of elasticity and second moment of area, reporting the location at midspan.'''
    narrate(ctx, 'Computing the maximum deflection of a simply supported beam under a uniform load from the given span, load, modulus and second moment of area')
    span_text = load(ctx, 'the span length L of the simply supported beam with unit', d_type=str)
    load_text = load(ctx, 'the uniform load intensity w acting over the full span of the beam with unit', d_type=str)
    e_text = load(ctx, 'the modulus of elasticity E of the beam material with unit', d_type=str)
    i_text = load(ctx, 'the second moment of area I of the beam cross-section with unit', d_type=str)
    units_result = do(ctx, 'Convert the beam span, uniform load intensity, modulus of elasticity and second moment of area to SI base units', payloads={'the span length L of the simply supported beam with unit': span_text, 'the uniform load intensity w acting over the full span of the beam with unit': load_text, 'the modulus of elasticity E of the beam material with unit': e_text, 'the second moment of area I of the beam cross-section with unit': i_text})
    L_m = load(units_result, 'the span length in metres', d_type=float, ctx=ctx)
    w_npm = load(units_result, 'the uniform load intensity in newtons per metre', d_type=float, ctx=ctx)
    E_pa = load(units_result, 'the modulus of elasticity in pascals', d_type=float, ctx=ctx)
    I_m4 = load(units_result, 'the second moment of area in metres to the fourth', d_type=float, ctx=ctx)
    narrate(ctx, 'SI inputs: span ' + str(L_m) + ' m, load ' + str(w_npm) + ' N/m, E ' + str(E_pa) + ' Pa, I ' + str(I_m4) + ' m^4')
    midspan_result = do(ctx, 'Compute the maximum midspan deflection of a simply supported beam under a uniform load over its full span as 5wL^4 divided by 384EI', payloads={'the span length in metres': L_m, 'the uniform load intensity in newtons per metre': w_npm, 'the modulus of elasticity in pascals': E_pa, 'the second moment of area in metres to the fourth': I_m4})
    delta_m = load(midspan_result, 'the maximum midspan deflection in metres', d_type=float, ctx=ctx)
    narrate(ctx, 'Maximum midspan deflection = ' + str(delta_m) + ' m, occurring at midspan')
    plaus_result = do(ctx, 'Check the plausibility of a beam deflection by computing the span-to-deflection ratio and comparing it with the typical L/360 serviceability limit', payloads={'the span length in metres': L_m, 'the maximum midspan deflection in metres': delta_m})
    ratio = load(plaus_result, 'the span-to-deflection ratio', d_type=float, ctx=ctx)
    judgement = load(plaus_result, 'the plausibility judgement of the deflection', d_type=str, ctx=ctx)
    narrate(ctx, 'Span-to-deflection ratio L/delta = ' + str(ratio) + '; ' + judgement)
    delta_mm = delta_m * 1000.0
    narrate(ctx, 'Final answer: maximum deflection = ' + str(round(delta_mm, 2)) + ' mm at midspan')
    return {'the maximum deflection of the beam in millimetres with location at midspan': delta_mm}
