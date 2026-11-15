import math
def semantic_function(ctx):
    """Compute the one-sample t statistic, its degrees of freedom and the standard error of the mean from the sample size, sample mean, sample standard deviation and hypothesized mean."""
    narrate(ctx, "Computing the one-sample t statistic and the standard error of the mean")
    n = load(ctx, "the sample size n, the number of observations in the sample", d_type=int)
    xbar = load(ctx, "the sample mean of the measured quantity with its unit", d_type=float)
    s = load(ctx, "the sample standard deviation of the measured quantity with its unit", d_type=float)
    mu0 = load(ctx, "the hypothesized population mean under the null hypothesis, with its unit", d_type=float)
    narrate(ctx, f"Loaded n = {n}, sample mean = {xbar}, sample standard deviation = {s}, hypothesized mean = {mu0}")
    se = s / math.sqrt(n)
    narrate(ctx, f"Standard error of the mean = s / sqrt(n) = {se}")
    df = n - 1
    narrate(ctx, f"Degrees of freedom = n - 1 = {df}")
    t = (xbar - mu0) / se
    narrate(ctx, f"Test statistic t = (sample mean - hypothesized mean) / standard error = {t}")
    return {
        "the test statistic t": t,
        "the degrees of freedom n-1": df,
        "the standard error of the mean": se,
    }
