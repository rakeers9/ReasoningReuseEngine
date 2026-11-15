def semantic_function(ctx):
    '''Compute the present value of a single future cash amount received after a given number of periods.'''
    narrate(ctx, 'Computing the present value of a single future cash amount received after a number of periods')
    amount = load(ctx, 'the future cash amount as a number', d_type=float)
    rate = load(ctx, 'the discount rate as a decimal', d_type=float)
    periods = load(ctx, 'the number of periods until the cash is received as an integer', d_type=int)
    narrate(ctx, f'Future cash amount {amount}, discount rate {rate}, number of periods {periods}')
    discount_result = do(ctx,
        'Compute the discount factor (1+r)^-n for a decimal discount rate and a number of periods',
        payloads={
            'the discount rate as a decimal': rate,
            'the number of periods until the cash is received': periods,
        })
    discount_factor = load(discount_result, 'the discount factor for the given periods', d_type=float, ctx=ctx)
    narrate(ctx, f'Discount factor (1+r)^-n = {discount_factor}')
    pv = amount * discount_factor
    narrate(ctx, f'Present value = {amount} x {discount_factor} = {pv:.2f}')
    return {'the present value of the future cash amount': pv}