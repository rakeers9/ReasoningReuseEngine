def semantic_function(ctx):
    '''Apply Bayes theorem to divide the true positive joint probability by the total probability of a positive test to get the posterior probability of disease given a positive test.'''
    narrate(ctx, 'Applying Bayes theorem: posterior = true positive joint probability / total probability of a positive test')
    true_positive = load(ctx, 'true positive joint probability P(positive and disease)', d_type=float)
    total_positive = load(ctx, 'total probability of a positive test P(positive)', d_type=float)
    narrate(ctx, f'Loaded P(positive and disease) = {true_positive} and P(positive) = {total_positive}')
    if total_positive <= 0:
        raise ValueError('Total probability of a positive test must be positive for Bayes theorem')
    posterior = round(true_positive / total_positive, 12)
    narrate(ctx, f'P(disease | positive) = {true_positive} / {total_positive} = {posterior}')
    return {'posterior probability of disease given a positive test': posterior}