def semantic_function(ctx):
    '''Estimate the expected number of heterozygous carriers and the carrier frequency from the affected count and population size under Hardy-Weinberg equilibrium.'''
    import math
    narrate(ctx, 'Estimating heterozygous carriers under Hardy-Weinberg equilibrium from the affected count and population size')
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=int)
    affected = load(ctx, 'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition', d_type=int)
    narrate(ctx, f'Population size N = {total}; individuals with the recessive phenotype = {affected}')
    if total <= 0 or affected < 0 or affected > total:
        raise ValueError('population size must be positive and the affected count must lie between 0 and N')
    q_squared = affected / total
    narrate(ctx, f'Recessive phenotype frequency q^2 = {affected} / {total} = {q_squared}')
    q = round(math.sqrt(q_squared), 10)
    narrate(ctx, f'Recessive allele frequency q = sqrt({q_squared}) = {q}')
    p = round(1 - q, 10)
    narrate(ctx, f'Dominant allele frequency p = 1 - {q} = {p}')
    carrier_frequency = 2 * p * q
    narrate(ctx, f'Carrier genotype frequency 2pq = 2 x {p} x {q} = {carrier_frequency}')
    carriers = carrier_frequency * total
    narrate(ctx, f'Expected carriers = {carrier_frequency} x {total} = {carriers}')
    recessive_alleles = 2 * total * q
    accounted = recessive_alleles - 2 * affected
    narrate(ctx, f'Allele accounting: 2Nq = {recessive_alleles}; minus 2 x {affected} affected gives {accounted} carriers')
    if abs(accounted - carriers) > 1e-6:
        narrate(ctx, 'Allele accounting does not match the carrier count')
    else:
        narrate(ctx, 'Allele accounting matches the carrier count')
    return {'expected number of heterozygous carriers': round(carriers, 6), 'carrier genotype frequency 2pq': round(carrier_frequency, 6)}