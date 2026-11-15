def semantic_function(ctx):
    """Convert the prevalence, sensitivity, and specificity percentages into probabilities between 0 and 1."""
    import re
    narrate(ctx, "Converting prevalence, sensitivity, and specificity percentages into probabilities between 0 and 1")
    def to_probability(raw, label):
        text = str(raw).strip().lower()
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
        if match is None:
            raise ValueError("No number found for " + label + ": " + text)
        number = float(match.group(0))
        if "%" in text or "percent" in text:
            probability = number / 100.0
        elif number <= 1.0:
            probability = number
        else:
            probability = number / 100.0
        if not (0.0 <= probability <= 1.0):
            raise ValueError("Probability out of range for " + label + ": " + str(probability))
        narrate(ctx, "Converted " + label + " from '" + str(raw) + "' to probability " + str(probability))
        return probability
    prevalence_raw = load(ctx, "the prevalence of the disease in the population as a percentage with unit (the fraction of the population that has the disease)", d_type=str)
    sensitivity_raw = load(ctx, "the sensitivity of the test as a percentage with unit (the probability that a person with the disease tests positive)", d_type=str)
    specificity_raw = load(ctx, "the specificity of the test as a percentage with unit (the probability that a person without the disease tests negative)", d_type=str)
    p_disease = to_probability(prevalence_raw, "prevalence")
    sensitivity = to_probability(sensitivity_raw, "sensitivity")
    specificity = to_probability(specificity_raw, "specificity")
    narrate(ctx, "Prevalence P(D) = " + str(p_disease) + ", sensitivity P(+|D) = " + str(sensitivity) + ", specificity P(-|D') = " + str(specificity))
    return {
        "prevalence probability P(D)": p_disease,
        "sensitivity P(+|D)": sensitivity,
        "specificity P(-|D')": specificity,
    }