def semantic_function(ctx):
    """Verify a posterior probability by counting expected true and false positives in a hypothetical population of fixed size."""
    narrate(ctx, "Verifying the posterior probability by counting expected positives in a hypothetical population of 10000 people")
    p_disease = load(ctx, "prevalence probability P(D)", d_type=float)
    sensitivity = load(ctx, "sensitivity P(+|D)", d_type=float)
    false_positive_rate = load(ctx, "false-positive rate P(+|not D)", d_type=float)
    posterior = load(ctx, "posterior probability of disease given a positive test P(D|+)", d_type=float)
    narrate(ctx, "Loaded prevalence P(D) = " + str(p_disease))
    narrate(ctx, "Loaded sensitivity P(+|D) = " + str(sensitivity))
    narrate(ctx, "Loaded false-positive rate P(+|not D) = " + str(false_positive_rate))
    narrate(ctx, "Loaded posterior P(D|+) = " + str(posterior))
    checks = [("prevalence", p_disease), ("sensitivity", sensitivity), ("false-positive rate", false_positive_rate)]
    for label, value in checks:
        if not (0.0 <= value <= 1.0):
            raise ValueError(label + " must be a probability between 0 and 1")
    population = 10000
    diseased = population * p_disease
    healthy = population - diseased
    true_positives = diseased * sensitivity
    false_positives = healthy * false_positive_rate
    total_positives = true_positives + false_positives
    narrate(ctx, "Expected true positives out of " + str(population) + " people: " + str(true_positives))
    narrate(ctx, "Expected false positives: " + str(false_positives))
    narrate(ctx, "Expected total positives: " + str(total_positives))
    if total_positives <= 0.0:
        raise ValueError("no positive results expected, so the fraction is undefined")
    fraction = true_positives / total_positives
    agrees = abs(fraction - posterior) < 1e-3
    narrate(ctx, "Counted fraction of positives that are true positives = " + str(fraction))
    narrate(ctx, "Counting check agrees with the Bayes posterior: " + str(agrees))
    return {
        "fraction of positive results that are true positives": fraction,
        "counting check agrees with posterior": agrees,
    }
