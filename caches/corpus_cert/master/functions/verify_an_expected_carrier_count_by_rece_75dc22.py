def semantic_function(ctx):
    '''Verify an expected carrier count by recessive allele accounting: total recessive alleles 2Nq minus two per affected individual.'''
    narrate(ctx, 'Checking the expected carrier count by recessive allele accounting')
    total = load(ctx, 'the total population size N (count of individuals in the population)', d_type=int)
    q = load(ctx, 'recessive allele frequency q', d_type=float)
    affected = load(ctx, 'the number of individuals showing the recessive phenotype (homozygous recessive count) with the condition', d_type=int)
    carriers = load(ctx, 'expected number of heterozygous carriers', d_type=float)
    total_recessive_alleles = 2 * total * q
    narrate(ctx, f'Total recessive alleles 2Nq = 2 x {total} x {q} = {total_recessive_alleles}')
    affected_alleles = 2 * affected
    narrate(ctx, f'Recessive alleles carried by affected individuals = 2 x {affected} = {affected_alleles}')
    implied_carriers = total_recessive_alleles - affected_alleles
    narrate(ctx, f'Remaining recessive alleles, one per carrier = {implied_carriers}')
    matches = abs(implied_carriers - carriers) < 0.5
    narrate(ctx, f'Expected carriers {carriers} agree with allele accounting: {matches}')
    return {'allele accounting check result': matches, 'allele-implied carrier count': round(implied_carriers)}