def semantic_function(ctx):
    '''Compute the present value of a level annuity of equal end-of-period cash payments over a given number of periods.'''
    narrate(ctx, 'Computing the present value of a level annuity of equal end-of-period payments')
    payment = load(ctx, 'the annual cash inflow as a number', d_type=float)
    rate = load(ctx, 'the discount rate as a decimal', d_type=float)
    periods = load(ctx, 'the project life as an integer number of years', d_type=int)
    narrate(ctx, f'Level payment {payment}, discount rate {rate}, number of periods {periods}')
    factor_result = do(ctx,
        'Compute the annuity present value factor for a decimal discount rate and a number of periods',
        payloads={
            'the discount rate as a decimal': rate,
            'the number of periods until the cash is received': periods,
        })
    factor = load(factor_result, 'the annuity present value factor', d_type=float, ctx=ctx)
    narrate(ctx, f'Annuity present value factor = {factor}')
    pv = payment * factor
    narrate(ctx, f'PV of annual inflows = {payment} x {factor} = {pv:.2f}')
    return {'the present value of the annual inflows': pv}
