def semantic_function(ctx):
    '''Compute the posterior probability of disease given a positive test from prevalence, sensitivity, and specificity percentages.'''
    narrate(ctx, 'Computing P(disease | positive test) from prevalence, sensitivity and specificity')
    prev_desc = "the prevalence of the disease in the population as a percentage with unit (the fraction of the population that has the disease)"
    sens_desc = "the sensitivity of the test as a percentage with unit (the probability that a person with the disease tests positive)"
    spec_desc = "the specificity of the test as a percentage with unit (the probability that a person without the disease tests negative)"
    prev_text = load(ctx, prev_desc, d_type=str)
    sens_text = load(ctx, sens_desc, d_type=str)
    spec_text = load(ctx, spec_desc, d_type=str)
    narrate(ctx, 'Loaded prevalence ' + str(prev_text) + ', sensitivity ' + str(sens_text) + ', specificity ' + str(spec_text))
    conv = do(ctx, "Convert the prevalence, sensitivity, and specificity percentages into probabilities between 0 and 1", payloads={prev_desc: prev_text, sens_desc: sens_text, spec_desc: spec_text})
    p_d = load(conv, "prevalence probability P(D)", d_type=float, ctx=ctx)
    sens = load(conv, "sensitivity P(+|D)", d_type=float, ctx=ctx)
    spec = load(conv, "specificity P(-|D')", d_type=float, ctx=ctx)
    narrate(ctx, 'Converted to probabilities: P(D) = ' + str(p_d) + ', sensitivity = ' + str(sens) + ', specificity = ' + str(spec))
    comp = do(ctx, "Derive the probability of not having the disease and the false-positive rate from prevalence and specificity.", payloads={"prevalence probability P(D)": p_d, "specificity P(-|D')": spec})
    p_not_d = load(comp, "probability of not having the disease P(D')", d_type=float, ctx=ctx)
    fpr = load(comp, "false-positive rate P(+|D')", d_type=float, ctx=ctx)
    narrate(ctx, 'Complements: P(not D) = ' + str(p_not_d) + ', false-positive rate = ' + str(fpr))
    tot = do(ctx, "Compute the total probability of a positive test by the law of total probability over disease and non-disease groups", payloads={"sensitivity P(+|D)": sens, "prevalence probability P(D)": p_d, "probability of not having the disease P(D')": p_not_d, "false-positive rate P(+|D')": fpr})
    p_pos = load(tot, "total probability of a positive test P(+)", d_type=float, ctx=ctx)
    narrate(ctx, 'Total probability of a positive test P(+) = ' + str(p_pos))
    bayes = do(ctx, "Apply Bayes' theorem to obtain the probability of disease given a positive test from sensitivity, prevalence, and total positive probability", payloads={"sensitivity P(+|D)": sens, "prevalence probability P(D)": p_d, "total probability of a positive test P(+)": p_pos})
    posterior = load(bayes, "posterior probability of disease given a positive test P(D|+)", d_type=float, ctx=ctx)
    narrate(ctx, 'Posterior P(D|+) = ' + str(posterior) + ' (about ' + str(round(100 * posterior, 1)) + ' percent)')
    ver = do(ctx, "Verify a posterior probability by counting expected true and false positives in a hypothetical population of fixed size.", payloads={"prevalence probability P(D)": p_d, "sensitivity P(+|D)": sens, "false-positive rate P(+|D')": fpr, "posterior probability of disease given a positive test P(D|+)": posterior})
    check = load(ver, "fraction of positive results that are true positives", d_type=float, ctx=ctx)
    narrate(ctx, 'Counting check: fraction of positives that are true positives = ' + str(check))
    narrate(ctx, 'Base-rate interpretation: the disease is rare, so false positives from the healthy majority are comparable to true positives, and a positive result is far from conclusive.')
    return {"posterior probability of disease given a positive test with base-rate interpretation": posterior, "fraction of positive results that are true positives": check}
