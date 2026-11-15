def semantic_function(ctx):
    '''Decide whether to reject the null hypothesis by comparing the absolute test statistic with the two-tailed critical value.'''
    narrate(ctx, 'Deciding whether to reject H0 by comparing |t| with the two-tailed critical value')
    t = load(ctx, 'the test statistic t', d_type=float)
    crit = load(ctx, 'the two-tailed critical t value', d_type=float)
    narrate(ctx, f'Loaded test statistic t = {t} and two-tailed critical value = {crit}')
    crit = abs(crit)
    if abs(t) > crit:
        decision = 'reject H0'
        narrate(ctx, f'|t| = {abs(t)} exceeds the critical value {crit}, so H0 is rejected')
    else:
        decision = 'fail to reject H0'
        narrate(ctx, f'|t| = {abs(t)} does not exceed the critical value {crit}, so H0 is not rejected')
    return {'the reject or fail to reject decision': decision}