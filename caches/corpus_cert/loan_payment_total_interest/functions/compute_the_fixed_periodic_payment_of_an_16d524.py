def semantic_function(ctx):
    '''Compute the fixed periodic payment of an amortizing loan from the principal, the periodic interest rate and the compound growth factor.'''
    narrate(ctx, 'Computing the fixed monthly payment M = P*r*(1+r)^n / ((1+r)^n - 1)')
    principal = load(ctx, 'the principal amount of the loan with currency unit', d_type=float)
    r = load(ctx, 'the periodic interest rate per compounding period', d_type=float)
    growth = load(ctx, 'the compound growth factor (1+r)^n', d_type=float)
    narrate(ctx, f'Loaded principal P = {principal}, periodic rate r = {r}, growth factor (1+r)^n = {growth}')
    denominator = growth - 1.0
    if denominator == 0:
        raise ValueError('Growth factor equals 1 so the amortization formula is undefined; a zero-rate loan needs a different formula')
    payment_factor = growth / denominator
    narrate(ctx, f'Payment factor (1+r)^n / ((1+r)^n - 1) = {payment_factor}')
    interest_part = principal * r
    narrate(ctx, f'Periodic interest on principal P x r = {interest_part}')
    payment = interest_part * payment_factor
    narrate(ctx, f'Fixed monthly payment M = {payment}')
    return {'the fixed monthly payment amount': payment}