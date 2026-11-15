def semantic_function(ctx):
    '''Count positive integers below a bound that are divisible by the exclusion divisor and by at least one inclusion divisor.'''
    narrate(ctx, 'Counting positive integers below the bound divisible by the exclusion divisor and by at least one inclusion divisor')
    bound = load(ctx, 'the exclusive upper bound N for the positive integers being counted, as a positive integer', d_type=int)
    narrate(ctx, f'Loaded exclusive bound N = {bound}')
    inclusion = [int(d) for d in load(ctx, 'the set of inclusion divisors d1 or d2 (positive integers) such that a counted integer must be divisible by at least one of them', d_type=list)]
    narrate(ctx, f'Loaded inclusion divisors {inclusion}')
    excl = int(load(ctx, 'the exclusion divisor d (a positive integer) such that counted integers must not be divisible by it', d_type=int))
    narrate(ctx, f'Loaded exclusion divisor {excl}')
    limit = bound - 1
    narrate(ctx, f'Largest counted integer is {limit}, since the bound is exclusive')
    lcm_divisors = []
    for d in inclusion:
        r = do(ctx, 'Compute the least common multiple of two positive integers', payloads={'the first positive integer': excl, 'the second positive integer': d})
        combined = load(r, 'the least common multiple of the two integers', d_type=int, ctx=ctx)
        narrate(ctx, f'lcm({excl}, {d}) = {combined}, so multiples of {combined} are divisible by both {excl} and {d}')
        lcm_divisors.append(combined)
    u = do(ctx, 'Count positive integers up to a limit that are divisible by at least one divisor in a given list, using inclusion-exclusion over lcms', payloads={'the largest value allowed for the counted integers': limit, 'the list of divisors of which at least one must divide a counted integer': lcm_divisors})
    count = load(u, 'the count of positive integers up to the limit divisible by at least one of the divisors', d_type=int, ctx=ctx)
    narrate(ctx, f'Count of integers below {bound} divisible by {excl} and by at least one of {inclusion}: {count}')
    return {'the count of positive integers less than N divisible by the exclusion divisor and by at least one inclusion divisor': count}
