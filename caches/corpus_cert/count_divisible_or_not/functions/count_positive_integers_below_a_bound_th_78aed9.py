def semantic_function(ctx):
    '''Count positive integers below a bound that are divisible by some inclusion divisor but not by the exclusion divisor.'''
    narrate(ctx, 'Counting positive integers below the bound divisible by some inclusion divisor but not by the exclusion divisor')
    bound = int(load(ctx, 'the exclusive upper bound N for the positive integers being counted, as a positive integer', d_type=int))
    narrate(ctx, f'Loaded exclusive bound N = {bound}')
    inclusion = [int(d) for d in load(ctx, 'the set of inclusion divisors d1 or d2 (positive integers) such that a counted integer must be divisible by at least one of them', d_type=list)]
    narrate(ctx, f'Loaded inclusion divisors {inclusion}')
    excl = int(load(ctx, 'the exclusion divisor d (a positive integer) such that counted integers must not be divisible by it', d_type=int))
    narrate(ctx, f'Loaded exclusion divisor {excl}')
    limit = bound - 1
    narrate(ctx, f'Largest counted integer is {limit}, since the bound is exclusive')
    a = do(ctx, 'Count positive integers up to a limit that are divisible by at least one divisor in a given list, using inclusion-exclusion over lcms', payloads={'the largest value allowed for the counted integers': limit, 'the list of divisors of which at least one must divide a counted integer': inclusion})
    count_a = int(load(a, 'the count of positive integers up to the limit divisible by at least one of the divisors', d_type=int, ctx=ctx))
    narrate(ctx, f'Count of integers below {bound} divisible by at least one of {inclusion}: {count_a}')
    b = do(ctx, 'Count positive integers below a bound that are divisible by the exclusion divisor and by at least one inclusion divisor, via inclusion-exclusion over lcms', payloads={'the exclusive upper bound N for the positive integers being counted, as a positive integer': bound, 'the set of inclusion divisors d1 or d2 (positive integers) such that a counted integer must be divisible by at least one of them': inclusion, 'the exclusion divisor d (a positive integer) such that counted integers must not be divisible by it': excl})
    count_b = int(load(b, 'the count of positive integers less than N divisible by the exclusion divisor and by at least one inclusion divisor', d_type=int, ctx=ctx))
    narrate(ctx, f'Count of integers below {bound} divisible by {excl} and by at least one of {inclusion}: {count_b}')
    result = count_a - count_b
    narrate(ctx, f'Subtracting the overlap: {count_a} - {count_b} = {result}')
    return {'the count of positive integers less than N divisible by an inclusion divisor but not by the exclusion divisor': result}
