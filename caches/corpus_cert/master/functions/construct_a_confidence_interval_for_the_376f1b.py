import math
def semantic_function(ctx):
    '''Construct a confidence interval for the population mean from the sample statistics and the two-tailed critical t value, and check whether the hypothesized mean lies inside it.'''
    narrate(ctx, 'Constructing the confidence interval for the mean and checking the hypothesized mean against it')
    n = load(ctx, 'the sample size n, the number of observations in the sample', d_type=int)
    xbar = load(ctx, 'the sample mean of the measured quantity with its unit', d_type=float)
    s = load(ctx, 'the sample standard deviation of the measured quantity with its unit', d_type=float)
    mu0 = load(ctx, 'the hypothesized population mean under the null hypothesis, with its unit', d_type=float)
    t_crit = load(ctx, 'the two-tailed critical t value', d_type=float)
    narrate(ctx, f'Loaded n = {n}, sample mean = {xbar}, sample standard deviation = {s}, hypothesized mean = {mu0}, two-tailed critical t value = {t_crit}')
    if n < 2:
        raise ValueError('sample size must be at least 2')
    if s <= 0:
        raise ValueError('sample standard deviation must be positive')
    se = s / math.sqrt(n)
    narrate(ctx, f'Standard error of the mean = s / sqrt(n) = {se:.4f}')
    margin = t_crit * se
    narrate(ctx, f'Margin of error = critical value * standard error = {margin:.4f}')
    lower = xbar - margin
    upper = xbar + margin
    narrate(ctx, f'Confidence interval = ({lower:.3f}, {upper:.3f})')
    inside = lower <= mu0 <= upper
    if inside:
        narrate(ctx, f'The hypothesized mean {mu0} lies inside the interval, so the interval agrees with failing to reject H0')
    else:
        narrate(ctx, f'The hypothesized mean {mu0} lies outside the interval, so the interval agrees with rejecting H0')
    return {
        'the confidence interval lower and upper bounds': [lower, upper],
        'the agreement between the interval and the test decision': inside,
    }
