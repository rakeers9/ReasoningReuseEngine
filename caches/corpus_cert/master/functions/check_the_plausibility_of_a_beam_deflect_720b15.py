def semantic_function(ctx):
    '''Check the plausibility of a beam deflection by computing the span-to-deflection ratio and comparing it with the typical L/360 serviceability limit.'''
    narrate(ctx, 'Checking plausibility of the midspan deflection by computing the span-to-deflection ratio')
    span_m = load(ctx, 'the span length in metres', d_type=float)
    delta_m = load(ctx, 'the maximum midspan deflection in metres', d_type=float)
    narrate(ctx, 'Loaded span L = ' + str(span_m) + ' m and maximum deflection delta = ' + str(delta_m) + ' m')
    limit_ratio = 360.0
    if delta_m > 0:
        ratio = span_m / delta_m
    else:
        ratio = float('inf')
    narrate(ctx, 'Span-to-deflection ratio L/delta = ' + str(ratio))
    if ratio >= limit_ratio:
        judgement = 'within the typical serviceability limit of L/360, so the deflection is plausible'
    else:
        judgement = 'exceeds the typical serviceability limit of L/360, so the deflection may be excessive'
    narrate(ctx, 'Comparing with the typical L/360 serviceability limit: ' + judgement)
    return {
        'the span-to-deflection ratio': ratio,
        'the plausibility judgement of the deflection': judgement,
    }