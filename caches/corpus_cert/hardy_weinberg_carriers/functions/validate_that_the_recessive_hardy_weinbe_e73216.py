def semantic_function(ctx):
    '''Validate that the recessive Hardy-Weinberg carrier estimate applies to the given population size, affected count, and inheritance mode.'''
    narrate(ctx, 'Validating inputs for the recessive Hardy-Weinberg carrier estimate')
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=int)
    affected = load(ctx, 'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition', d_type=int)
    mode = load(ctx, 'the inheritance mode of the condition (must be recessive for carrier counting)', d_type=str)
    narrate(ctx, 'Population size N = ' + str(total))
    narrate(ctx, 'Individuals with the recessive phenotype = ' + str(affected))
    narrate(ctx, 'Inheritance mode = ' + str(mode))
    problems = []
    if total < 1:
        problems.append('population size must be a positive integer')
    if affected < 0 or affected > total:
        problems.append('affected count must be between 0 and the population size')
    if 'recessive' not in str(mode).lower():
        problems.append('inheritance mode must be recessive for carrier counting')
    for problem in problems:
        narrate(ctx, 'Invalid input: ' + problem)
    valid = len(problems) == 0
    narrate(ctx, 'Inputs are valid for carrier estimation: ' + str(valid))
    return {'inputs are valid for carrier estimation': valid}