def semantic_function(ctx):
    '''Express a count as a percentage of the total population size.'''
    narrate(ctx, 'Expressing a count as a percentage of the total population size')
    count = load(ctx, 'the expected number of heterozygous carriers', d_type=float)
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=float)
    if total <= 0:
        raise ValueError('population size must be positive')
    percentage = round(count / total * 100, 6)
    narrate(ctx, f'Percentage = {count} / {total} x 100 = {percentage}%')
    return {'carrier percentage of population': percentage}