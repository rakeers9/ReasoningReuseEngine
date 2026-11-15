import math
def semantic_function(ctx):
    '''Look up the two-tailed critical t value for a significance level and degrees of freedom.'''
    narrate(ctx, 'Looking up the two-tailed critical t value for the significance level and degrees of freedom')
    alpha = load(ctx, 'the significance level alpha for the two-tailed test', d_type=float)
    nu = load(ctx, 'the degrees of freedom n-1', d_type=int)
    narrate(ctx, f'Loaded alpha = {alpha} and degrees of freedom = {nu}')
    if not (0 < alpha < 1) or nu < 1:
        narrate(ctx, 'Alpha must lie in (0,1) and degrees of freedom must be at least 1; no critical value can be found')
        raise ValueError('invalid alpha or degrees of freedom')
    narrate(ctx, 'Inputs are valid; building the Student t density and integrating it numerically')
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    def pdf(x):
        return c * (1.0 + x * x / nu) ** (-(nu + 1) / 2.0)
    def half_mass(t):
        n_int = 4000
        h = t / n_int
        odd = math.fsum(pdf(h * (2 * k + 1)) for k in range(n_int // 2))
        even = math.fsum(pdf(h * 2 * k) for k in range(1, n_int // 2))
        return (pdf(0.0) + pdf(t) + 4.0 * odd + 2.0 * even) * h / 3.0
    lo, hi = 0.0, 1e4
    for _ in range(100):
        mid = (lo + hi) / 2.0
        tail = 1.0 - 2.0 * half_mass(mid)
        if tail > alpha:
            lo = mid
            narrate(ctx, f'Two-sided tail {tail:.5f} at t = {mid:.5f} exceeds alpha, so the critical value lies higher')
        else:
            hi = mid
            narrate(ctx, f'Two-sided tail {tail:.5f} at t = {mid:.5f} does not exceed alpha, so the critical value lies lower')
    crit = (lo + hi) / 2.0
    narrate(ctx, f'Two-tailed critical t value for alpha = {alpha} with df = {nu}: {crit:.4f}')
    return {'the two-tailed critical t value': crit}
