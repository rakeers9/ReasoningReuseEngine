import math

def semantic_function(ctx):
    '''Evaluate the natural logarithm of one plus a small periodic interest rate by series expansion.'''
    narrate(ctx, 'Evaluating ln(1+r) by power series for the periodic interest rate')
    r = load(ctx, 'the periodic interest rate per compounding period', d_type=float)
    narrate(ctx, f'Loaded periodic rate r = {r}')
    if abs(r) >= 1:
        narrate(ctx, 'Rate is not small enough for the series; using the direct logarithm instead')
        result = math.log(1 + r)
        narrate(ctx, f'ln(1+r) = {result}')
        return {'the natural logarithm of one plus the periodic rate': result}
    series_sum = 0.0
    power = 1.0
    terms_used = 0
    for k in range(1, 201):
        power *= r
        term = power / k
        if k % 2 == 1:
            series_sum += term
        else:
            series_sum -= term
        terms_used = k
        if abs(term) < 1e-18:
            break
    narrate(ctx, f'Summed {terms_used} series terms: ln(1+r) = {series_sum}')
    return {'the natural logarithm of one plus the periodic rate': series_sum}
