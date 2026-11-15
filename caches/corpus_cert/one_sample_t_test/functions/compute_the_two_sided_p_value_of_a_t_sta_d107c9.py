import math
def semantic_function(ctx):
    '''Compute the two-sided p-value of a t statistic with given degrees of freedom.'''
    narrate(ctx, 'Computing the two-sided p-value of the t statistic')
    t = load(ctx, 'the test statistic t', d_type=float)
    nu = load(ctx, 'the degrees of freedom n-1', d_type=int)
    narrate(ctx, f'Loaded t = {t} and degrees of freedom = {nu}')
    if nu < 1:
        raise ValueError('degrees of freedom must be at least 1')
    a = abs(t)
    c = math.exp(math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0)) / math.sqrt(nu * math.pi)
    def pdf(x):
        return c * (1.0 + x * x / nu) ** (-(nu + 1) / 2.0)
    n_int = 4000
    h = a / n_int
    odd = math.fsum(pdf(h * (2 * k + 1)) for k in range(n_int // 2))
    even = math.fsum(pdf(h * 2 * k) for k in range(1, n_int // 2))
    area = (pdf(0.0) + pdf(a) + 4.0 * odd + 2.0 * even) * h / 3.0
    p = 1.0 - 2.0 * area
    p = min(1.0, max(0.0, p))
    narrate(ctx, f'Area between 0 and |t| under the t density: {area:.6f}')
    narrate(ctx, f'Two-sided p-value = 1 - 2 * area = {p:.4f}')
    return {'the two-sided p-value': p}
