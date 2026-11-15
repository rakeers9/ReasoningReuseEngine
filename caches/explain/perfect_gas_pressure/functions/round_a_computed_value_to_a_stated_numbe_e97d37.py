def semantic_function(ctx):
    '''Round a computed value to a stated number of significant figures and attach its unit.'''
    narrate(ctx, 'Rounding a computed value to a stated number of significant figures and attaching its unit')
    value = float(load(ctx, 'the computed value', d_type=float))
    narrate(ctx, 'Loaded computed value ' + repr(value))
    unit = str(load(ctx, 'the unit of the computed value', d_type=str)).strip()
    narrate(ctx, 'Loaded unit ' + repr(unit))
    sig = int(load(ctx, 'the number of significant figures to report', d_type=int))
    narrate(ctx, 'Loaded number of significant figures to report: ' + str(sig))
    if sig < 1:
        narrate(ctx, 'Significant figures must be at least 1')
        raise ValueError('Number of significant figures must be at least 1')
    sci = '{:.{}e}'.format(value, sig - 1)
    narrate(ctx, 'Rounded ' + repr(value) + ' to ' + str(sig) + ' significant figures: ' + sci)
    text = sci + ' ' + unit
    narrate(ctx, 'Rounded value with unit: ' + text)
    return {'rounded value with unit': text}
