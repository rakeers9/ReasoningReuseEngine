def semantic_function(ctx):
    '''Convert the prevalence, sensitivity and specificity of a diagnostic test into probabilities in the unit interval and derive their complements.'''
    import re
    narrate(ctx, 'Converting prevalence, sensitivity and specificity into probabilities in the unit interval')
    def to_probability(raw, label):
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = float(raw)
            percent_marked = False
        else:
            text = str(raw).strip()
            match = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)', text)
            if match is None:
                raise ValueError('No number found for ' + label + ': ' + text)
            value = float(match.group(0))
            percent_marked = ('%' in text) or ('percent' in text.lower())
        if percent_marked or value > 1.0:
            value = value / 100.0
        if value < 0.0 or value > 1.0:
            raise ValueError('Value out of range for ' + label + ': ' + str(raw))
        narrate(ctx, 'Read ' + label + ' as ' + str(raw) + ', probability ' + str(round(value, 12)))
        return value
    prevalence = to_probability(load(ctx, 'the prevalence of the disease in the population with unit (base rate, percentage or probability)', d_type=str), 'prevalence')
    sensitivity = to_probability(load(ctx, 'the sensitivity of the test, i.e. the true positive rate P(positive | disease), with unit (percentage or probability)', d_type=str), 'sensitivity')
    specificity = to_probability(load(ctx, 'the specificity of the test, i.e. the true negative rate P(negative | no disease), with unit (percentage or probability)', d_type=str), 'specificity')
    false_positive = round(1.0 - specificity, 12)
    no_disease = round(1.0 - prevalence, 12)
    narrate(ctx, 'False positive probability P(positive | no disease) = 1 - specificity = ' + str(false_positive))
    narrate(ctx, 'Probability of no disease P(not disease) = 1 - prevalence = ' + str(no_disease))
    return {
        'prevalence probability P(disease)': prevalence,
        'sensitivity probability P(positive | disease)': sensitivity,
        'specificity probability P(negative | no disease)': specificity,
        'false positive probability P(positive | no disease) as one minus specificity': false_positive,
        'probability of no disease P(not disease)': no_disease,
    }
