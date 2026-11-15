def semantic_function(ctx):
    '''Compute the total probability of a positive test by the law of total probability over disease and non-disease groups.'''
    narrate(ctx, 'Computing the total probability of a positive test P(+) by the law of total probability')
    sensitivity = load(ctx, "sensitivity P(+|D)", d_type=float)
    p_disease = load(ctx, "prevalence probability P(D)", d_type=float)
    p_not_disease = load(ctx, "probability of not having the disease P(D')", d_type=float)
    false_positive_rate = load(ctx, "false-positive rate P(+|D')", d_type=float)
    narrate(ctx, 'Loaded sensitivity P(+|D) = ' + str(sensitivity))
    narrate(ctx, 'Loaded prevalence P(D) = ' + str(p_disease))
    narrate(ctx, 'Loaded probability of not having the disease, P(not D) = ' + str(p_not_disease))
    narrate(ctx, 'Loaded false-positive rate P(+|not D) = ' + str(false_positive_rate))
    checks = [('sensitivity', sensitivity), ('prevalence', p_disease), ('probability of not having the disease', p_not_disease), ('false-positive rate', false_positive_rate)]
    for label, value in checks:
        if not (0.0 <= value <= 1.0):
            raise ValueError(label + ' must be a probability between 0 and 1')
    true_positive_part = sensitivity * p_disease
    false_positive_part = false_positive_rate * p_not_disease
    narrate(ctx, 'True-positive share P(+|D) P(D) = ' + str(true_positive_part))
    narrate(ctx, 'False-positive share P(+|not D) P(not D) = ' + str(false_positive_part))
    p_positive = true_positive_part + false_positive_part
    narrate(ctx, 'Total probability of a positive test P(+) = ' + str(p_positive))
    return {'total probability of a positive test P(+)': p_positive}