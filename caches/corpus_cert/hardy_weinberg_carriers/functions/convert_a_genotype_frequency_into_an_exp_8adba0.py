def semantic_function(ctx):
    '''Convert a genotype frequency into an expected number of individuals by multiplying by the total population size.'''
    narrate(ctx, 'Converting a genotype frequency into an expected number of individuals by multiplying by the population size')
    frequency = load(ctx, 'the genotype frequency to convert', d_type=float)
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=int)
    if total <= 0:
        raise ValueError('population size must be positive')
    if frequency < 0 or frequency > 1:
        raise ValueError('genotype frequency must lie between 0 and 1')
    expected = round(frequency * total, 6)
    narrate(ctx, f'Expected individuals = {frequency} x {total} = {expected}')
    return {'expected number of individuals': expected}