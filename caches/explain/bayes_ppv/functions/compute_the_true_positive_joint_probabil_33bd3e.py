def semantic_function(ctx):
    '''Compute the true positive joint probability, the false positive joint probability and the total probability of a positive test by the law of total probability.'''
    narrate(ctx, 'Computing the joint probabilities of true and false positives and the total probability of a positive test')
    prevalence = load(ctx, 'prevalence probability P(disease)', d_type=float)
    sensitivity = load(ctx, 'sensitivity probability P(positive | disease)', d_type=float)
    false_positive_rate = load(ctx, 'false positive probability P(positive | no disease) as one minus specificity', d_type=float)
    no_disease = load(ctx, 'probability of no disease P(not disease)', d_type=float)
    narrate(ctx, f'Loaded P(disease) = {prevalence}, P(positive | disease) = {sensitivity}, P(positive | no disease) = {false_positive_rate}, P(not disease) = {no_disease}')
    true_positive = round(sensitivity * prevalence, 12)
    narrate(ctx, f'True positive joint probability P(positive and disease) = {sensitivity} x {prevalence} = {true_positive}')
    false_positive = round(false_positive_rate * no_disease, 12)
    narrate(ctx, f'False positive joint probability P(positive and no disease) = {false_positive_rate} x {no_disease} = {false_positive}')
    total_positive = round(true_positive + false_positive, 12)
    narrate(ctx, f'Total probability of a positive test P(positive) = {true_positive} + {false_positive} = {total_positive}')
    return {
        'true positive joint probability P(positive and disease)': true_positive,
        'false positive joint probability P(positive and no disease)': false_positive,
        'total probability of a positive test P(positive)': total_positive,
    }
