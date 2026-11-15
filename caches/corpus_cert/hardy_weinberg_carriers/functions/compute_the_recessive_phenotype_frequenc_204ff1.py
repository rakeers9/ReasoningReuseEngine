def semantic_function(ctx):
    """Compute the recessive phenotype frequency q squared as the affected count divided by the total population size."""
    narrate(ctx, 'Computing q^2 as the affected count divided by the total population size')
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=int)
    affected = load(ctx, 'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition', d_type=int)
    if total <= 0:
        raise ValueError('population size must be positive')
    if affected < 0 or affected > total:
        raise ValueError('affected count must lie between 0 and the population size')
    q_squared = affected / total
    narrate(ctx, f'Total population N = {total}')
    narrate(ctx, f'Individuals with the recessive phenotype = {affected}')
    narrate(ctx, f'q^2 = {affected} / {total} = {q_squared}')
    return {'recessive phenotype frequency q squared': q_squared}
