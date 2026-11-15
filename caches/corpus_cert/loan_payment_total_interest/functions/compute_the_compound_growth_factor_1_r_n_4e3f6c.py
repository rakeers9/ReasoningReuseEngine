import math

def semantic_function(ctx):
    '''Compute the compound growth factor (1+r)^n for a periodic rate r and a number of periods n.'''
    narrate(ctx, 'Computing the compound growth factor (1+r)^n from the periodic rate and the period count')
    r = load(ctx, 'the periodic interest rate per compounding period', d_type=float)
    n_raw = load(ctx, 'the total number of payment periods', d_type=float)
    n = int(round(n_raw))
    narrate(ctx, f'Loaded periodic rate r = {r} and number of periods n = {n}')
    log_result = do(ctx, 'Evaluate the natural logarithm of one plus a small periodic interest rate by series expansion', payloads={'the periodic interest rate per compounding period': r})
    ln_base = load(log_result, 'the natural logarithm of one plus the periodic rate', d_type=float, ctx=ctx)
    narrate(ctx, f'ln(1+r) = {ln_base}')
    growth = math.exp(n * ln_base)
    narrate(ctx, f'(1+r)^n = exp(n x ln(1+r)) = exp({n} x {ln_base}) = {growth}')
    return {'the compound growth factor (1+r)^n': growth}
