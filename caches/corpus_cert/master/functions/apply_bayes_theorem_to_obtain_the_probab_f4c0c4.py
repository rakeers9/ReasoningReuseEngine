def semantic_function(ctx):
    '''Apply Bayes theorem to obtain the probability of disease given a positive test from sensitivity, prevalence, and total positive probability.'''
    narrate(ctx, 'Applying Bayes theorem: P(D|+) = P(+|D) * P(D) / P(+)')
    sensitivity = load(ctx, 'sensitivity P(+|D)', d_type=float)
    p_disease = load(ctx, 'prevalence probability P(D)', d_type=float)
    p_positive = load(ctx, 'total probability of a positive test P(+)', d_type=float)
    narrate(ctx, 'Loaded sensitivity P(+|D) = ' + str(sensitivity))
    narrate(ctx, 'Loaded prevalence P(D) = ' + str(p_disease))
    narrate(ctx, 'Loaded total probability of a positive test P(+) = ' + str(p_positive))
    checks = [('sensitivity', sensitivity), ('prevalence', p_disease), ('total positive probability', p_positive)]
    for label, value in checks:
        if not (0.0 <= value <= 1.0):
            raise ValueError(label + ' must be a probability between 0 and 1')
    if p_positive <= 0.0:
        raise ValueError('total probability of a positive test must be greater than zero')
    numerator = sensitivity * p_disease
    posterior = numerator / p_positive
    narrate(ctx, 'Numerator P(+|D) P(D) = ' + str(numerator))
    narrate(ctx, 'Posterior P(D|+) = ' + str(numerator) + ' / ' + str(p_positive) + ' = ' + str(posterior))
    return {'posterior probability of disease given a positive test P(D|+)': posterior}
