def semantic_function(ctx):
    """Derive the probability of not having the disease and the false-positive rate from prevalence and specificity."""
    narrate(ctx, "Deriving complements from prevalence P(D) and specificity P(-|D')")
    p_disease = load(ctx, "prevalence probability P(D)", d_type=float)
    specificity = load(ctx, "specificity P(-|D')", d_type=float)
    narrate(ctx, "Loaded prevalence P(D) = " + str(p_disease))
    narrate(ctx, "Loaded specificity P(-|D') = " + str(specificity))
    if not (0.0 <= p_disease <= 1.0) or not (0.0 <= specificity <= 1.0):
        raise ValueError("Prevalence and specificity must be probabilities between 0 and 1")
    p_not_disease = 1.0 - p_disease
    false_positive_rate = 1.0 - specificity
    narrate(ctx, "P(D') = 1 - P(D) = " + str(p_not_disease))
    narrate(ctx, "False-positive rate P(+|D') = 1 - P(-|D') = " + str(false_positive_rate))
    return {
        "probability of not having the disease P(D')": p_not_disease,
        "false-positive rate P(+|D')": false_positive_rate,
    }