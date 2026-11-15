def semantic_function(ctx):
    '''Test whether a population mean differs from a hypothesized value at a given significance level and report the test statistic and conclusion.'''
    narrate(ctx, 'Testing whether the population mean differs from the hypothesized value at the given significance level')
    n = load(ctx, 'the sample size n, the number of observations in the sample', d_type=int)
    xbar = load(ctx, 'the sample mean of the measured quantity with its unit', d_type=float)
    s = load(ctx, 'the sample standard deviation of the measured quantity with its unit', d_type=float)
    mu0 = load(ctx, 'the hypothesized population mean under the null hypothesis, with its unit', d_type=float)
    alpha = load(ctx, 'the significance level alpha for the two-tailed test', d_type=float)
    narrate(ctx, f'Loaded n = {n}, sample mean = {xbar}, sample standard deviation = {s}, hypothesized mean = {mu0}, alpha = {alpha}')
    narrate(ctx, f'Hypotheses: H0: mu = {mu0} versus H1: mu differs from {mu0} (two-sided)')
    tr = do(ctx, 'Compute the one-sample t statistic, degrees of freedom and standard error of the mean from the sample size, sample mean, sample standard deviation and hypothesized mean.', payloads={'the sample size n, the number of observations in the sample': n, 'the sample mean of the measured quantity with its unit': xbar, 'the sample standard deviation of the measured quantity with its unit': s, 'the hypothesized population mean under the null hypothesis, with its unit': mu0})
    t_stat = load(tr, 'the test statistic t', d_type=float, ctx=ctx)
    df = load(tr, 'the degrees of freedom n-1', d_type=int, ctx=ctx)
    se = load(tr, 'the standard error of the mean', d_type=float, ctx=ctx)
    narrate(ctx, f'Standard error = {se}, degrees of freedom = {df}, test statistic t = {t_stat}')
    cr = do(ctx, 'Look up the two-tailed critical t value for a significance level and degrees of freedom.', payloads={'the significance level alpha for the two-tailed test': alpha, 'the degrees of freedom n-1': df})
    crit = load(cr, 'the two-tailed critical t value', d_type=float, ctx=ctx)
    narrate(ctx, f'Two-tailed critical value = +/-{abs(crit)}')
    pr = do(ctx, 'Compute the two-sided p-value of a t statistic with given degrees of freedom.', payloads={'the test statistic t': t_stat, 'the degrees of freedom n-1': df})
    p_value = load(pr, 'the two-sided p-value', d_type=float, ctx=ctx)
    narrate(ctx, f'Two-sided p-value = {p_value}')
    dr = do(ctx, 'Decide whether to reject the null hypothesis by comparing the absolute test statistic with the two-tailed critical value.', payloads={'the test statistic t': t_stat, 'the two-tailed critical t value': crit})
    decision = load(dr, 'the reject or fail to reject decision', d_type=str, ctx=ctx)
    cir = do(ctx, 'Construct a confidence interval for the population mean from the sample statistics and the two-tailed critical t value, and check whether the hypothesized mean lies inside it.', payloads={'the sample size n, the number of observations in the sample': n, 'the sample mean of the measured quantity with its unit': xbar, 'the sample standard deviation of the measured quantity with its unit': s, 'the hypothesized population mean under the null hypothesis, with its unit': mu0, 'the two-tailed critical t value': crit})
    ci = load(cir, 'the confidence interval lower and upper bounds', ctx=ctx)
    ci_agree = load(cir, 'the agreement between the interval and the test decision', ctx=ctx)
    narrate(ctx, f'Confidence interval = {ci}, hypothesized mean inside interval = {ci_agree}')
    rejected = not decision.lower().startswith('fail')
    if rejected:
        conclusion = f'Reject H0 at alpha = {alpha}: there is sufficient evidence that the true mean differs from {mu0}'
    else:
        conclusion = f'Fail to reject H0 at alpha = {alpha}: there is not sufficient evidence that the true mean differs from {mu0}'
    narrate(ctx, conclusion)
    return {
        'the test statistic t with degrees of freedom': {'t': t_stat, 'df': df},
        'the two-tailed critical values': [-abs(crit), abs(crit)],
        'the two-sided p-value': p_value,
        'the reject or fail to reject decision': decision,
        'the confidence interval agreement check': ci_agree,
        'the final reported conclusion about the mean': conclusion
    }
