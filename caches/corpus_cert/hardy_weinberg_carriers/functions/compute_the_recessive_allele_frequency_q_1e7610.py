def semantic_function(ctx):
    '''Compute the recessive allele frequency q as the square root of the recessive phenotype frequency.'''
    narrate(ctx, 'Computing the recessive allele frequency q as the square root of q squared')
    q_squared = load(ctx, 'recessive phenotype frequency q squared', d_type=float)
    if q_squared < 0 or q_squared > 1:
        raise ValueError('recessive phenotype frequency must lie between 0 and 1')
    q = q_squared ** 0.5
    narrate(ctx, f'Recessive allele frequency q = sqrt({q_squared}) = {q}')
    return {'recessive allele frequency q': q}