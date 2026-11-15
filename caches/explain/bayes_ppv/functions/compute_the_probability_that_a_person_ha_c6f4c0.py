def semantic_function(ctx):
    '''Compute the probability that a person has the disease given a positive test from the prevalence, sensitivity and specificity of the test.'''
    narrate(ctx, 'Computing the probability of disease given a positive test from prevalence, sensitivity and specificity')
    prevalence_raw = load(ctx, 'the prevalence of the disease in the population with unit (base rate, percentage or probability)', d_type=str)
    sensitivity_raw = load(ctx, 'the sensitivity of the test, i.e. the true positive rate P(positive | disease), with unit (percentage or probability)', d_type=str)
    specificity_raw = load(ctx, 'the specificity of the test, i.e. the true negative rate P(negative | no disease), with unit (percentage or probability)', d_type=str)
    narrate(ctx, f'Loaded prevalence {prevalence_raw}, sensitivity {sensitivity_raw} and specificity {specificity_raw}')
    rates = do(ctx, 'Convert the percentage prevalence, sensitivity and specificity into probabilities in the unit interval', payloads={
        'the prevalence of the disease in the population with unit (base rate, percentage or probability)': prevalence_raw,
        'the sensitivity of the test, i.e. the true positive rate P(positive | disease), with unit (percentage or probability)': sensitivity_raw,
        'the specificity of the test, i.e. the true negative rate P(negative | no disease), with unit (percentage or probability)': specificity_raw,
    })
    prevalence = load(rates, 'prevalence probability P(disease)', d_type=float, ctx=ctx)
    sensitivity = load(rates, 'sensitivity probability P(positive | disease)', d_type=float, ctx=ctx)
    false_positive = load(rates, 'false positive probability P(positive | no disease) as one minus specificity', d_type=float, ctx=ctx)
    no_disease = load(rates, 'probability of no disease P(not disease)', d_type=float, ctx=ctx)
    narrate(ctx, f'Probabilities: P(disease) = {prevalence}, P(positive | disease) = {sensitivity}, P(positive | no disease) = {false_positive}, P(not disease) = {no_disease}')
    joint = do(ctx, 'Compute the true positive joint probability, the false positive joint probability and the total probability of a positive test from prevalence, sensitivity and false positive rate by the law of total probability.', payloads={
        'prevalence probability P(disease)': prevalence,
        'sensitivity probability P(positive | disease)': sensitivity,
        'false positive probability P(positive | no disease) as one minus specificity': false_positive,
        'probability of no disease P(not disease)': no_disease,
    })
    true_positive = load(joint, 'true positive joint probability P(positive and disease)', d_type=float, ctx=ctx)
    total_positive = load(joint, 'total probability of a positive test P(positive)', d_type=float, ctx=ctx)
    narrate(ctx, f'Joint true positive probability = {true_positive}, total positive probability = {total_positive}')
    bayes = do(ctx, 'Apply Bayes theorem to divide the true positive joint probability by the total probability of a positive test to get the posterior probability of disease given a positive test', payloads={
        'true positive joint probability P(positive and disease)': true_positive,
        'total probability of a positive test P(positive)': total_positive,
    })
    posterior = load(bayes, 'posterior probability of disease given a positive test P(disease | positive)', d_type=float, ctx=ctx)
    narrate(ctx, f'Posterior probability P(disease | positive) = {posterior}')
    pct = do(ctx, 'Express a probability as a percentage by multiplying it by one hundred.', payloads={
        'posterior probability of disease given a positive test P(disease | positive)': posterior,
    })
    percentage = load(pct, 'posterior probability as a percentage', d_type=float, ctx=ctx)
    narrate(ctx, f'Expressed as a percentage: {percentage}%')
    return {
        'probability of disease given a positive test as a probability': posterior,
        'probability of disease given a positive test as a percentage': percentage,
    }
