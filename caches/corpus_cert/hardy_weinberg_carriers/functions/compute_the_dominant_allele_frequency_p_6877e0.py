def semantic_function(ctx):
    '''Compute the dominant allele frequency p as one minus the recessive allele frequency.'''
    narrate(ctx, 'Computing the dominant allele frequency p as one minus the recessive allele frequency q')
    q = load(ctx, 'recessive allele frequency q', d_type=float)
    if q < 0 or q > 1:
        raise ValueError('recessive allele frequency must lie between 0 and 1')
    p = 1 - q
    narrate(ctx, f'Dominant allele frequency p = 1 - {q} = {p}')
    return {'dominant allele frequency p': p}