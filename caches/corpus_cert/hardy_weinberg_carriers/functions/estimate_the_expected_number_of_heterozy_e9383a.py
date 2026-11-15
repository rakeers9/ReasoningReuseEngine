def semantic_function(ctx):
    '''Estimate the expected number of heterozygous carriers of a recessive condition in a Hardy-Weinberg population and express them as a percentage of the population.'''
    narrate(ctx, 'Estimating expected heterozygous carriers of a recessive condition under Hardy-Weinberg equilibrium')
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=int)
    narrate(ctx, f'Total population size N = {total}')
    affected = load(ctx, 'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition', d_type=int)
    narrate(ctx, f'Individuals showing the recessive phenotype = {affected}')
    condition = load(ctx, 'the name of the genetic condition or trait', d_type=str)
    narrate(ctx, f'Condition = {condition}')
    mode = load(ctx, 'the inheritance mode of the condition (must be recessive for carrier counting)', d_type=str)
    narrate(ctx, f'Inheritance mode = {mode}')
    validation = do(ctx, 'Validate that the recessive Hardy-Weinberg carrier estimate applies to the given population size, affected count, and inheritance mode.', payloads={
        'the total population size N (count of individuals in the population)': total,
        'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition': affected,
        'the inheritance mode of the condition (must be recessive for carrier counting)': mode,
    })
    valid = load(validation, 'inputs are valid for carrier estimation', d_type=bool, ctx=ctx)
    narrate(ctx, f'Input validation result: {valid}')
    if not valid:
        raise ValueError('inputs do not satisfy the recessive Hardy-Weinberg carrier estimate')
    estimate = do(ctx, 'Estimate the expected number of heterozygous carriers and the carrier frequency from the affected count and population size under Hardy-Weinberg equilibrium.', payloads={
        'the total population size N (count of individuals in the population)': total,
        'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition': affected,
    })
    carriers = load(estimate, 'expected number of heterozygous carriers', d_type=float, ctx=ctx)
    narrate(ctx, f'Expected heterozygous carriers = {carriers}')
    percent_result = do(ctx, 'Express a count as a percentage of the total population size.', payloads={
        'expected number of heterozygous carriers': carriers,
        'the total population size N (count of individuals in the population)': total,
    })
    percentage = load(percent_result, 'carrier percentage of population', d_type=float, ctx=ctx)
    narrate(ctx, f'Carriers are {percentage} percent of the population')
    return {
        'expected number of heterozygous carriers': carriers,
        'carrier percentage of population': percentage,
    }
