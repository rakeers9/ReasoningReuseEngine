def semantic_function(ctx):
    """Compute the heterozygote carrier genotype frequency 2pq from allele frequencies p and q."""
    narrate(ctx, 'Computing the heterozygote carrier frequency 2pq from p and q')
    p = load(ctx, 'dominant allele frequency p', d_type=float)
    q = load(ctx, 'recessive allele frequency q', d_type=float)
    if p < 0 or p > 1 or q < 0 or q > 1:
        raise ValueError('allele frequencies must lie between 0 and 1')
    carrier_frequency = 2 * p * q
    narrate(ctx, f'Carrier frequency 2pq = 2 x {p} x {q} = {carrier_frequency}')
    return {'carrier genotype frequency 2pq': carrier_frequency}
