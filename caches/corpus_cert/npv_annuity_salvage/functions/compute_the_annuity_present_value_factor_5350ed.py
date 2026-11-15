def semantic_function(ctx):
    '''Compute the annuity present value factor for a decimal discount rate and a number of periods.'''
    narrate(ctx, 'Computing the annuity present value factor PVIFA(r,n) = (1 - (1+r)^-n) / r')
    rate = load(ctx, 'the discount rate as a decimal', d_type=float)
    narrate(ctx, f'Discount rate loaded as decimal {rate}')
    periods = load(ctx, 'the number of periods until the cash is received', d_type=int)
    narrate(ctx, f'Number of periods loaded as {periods}')
    discount_result = do(ctx,
        'Compute the discount factor (1+r)^-n for a decimal discount rate and a number of periods',
        payloads={
            'the discount rate as a decimal': rate,
            'the number of periods until the cash is received': periods,
        })
    discount_factor = load(discount_result, 'the discount factor for the given periods', d_type=float, ctx=ctx)
    narrate(ctx, f'Discount factor (1+r)^-n = {discount_factor}')
    if rate == 0:
        factor = float(periods)
        narrate(ctx, 'Rate is zero, so the annuity factor equals the number of periods')
    else:
        factor = (1.0 - discount_factor) / rate
        narrate(ctx, f'Annuity factor = (1 - {discount_factor}) / {rate} = {factor}')
    return {'the annuity present value factor': factor}
