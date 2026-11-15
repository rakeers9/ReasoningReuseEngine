def semantic_function(ctx):
    '''Count positive integers up to a limit that are divisible by at least one divisor in a given list, using inclusion-exclusion over lcms.'''
    narrate(ctx, 'Counting positive integers up to the limit divisible by at least one divisor, via inclusion-exclusion')
    limit = load(ctx, 'the largest value allowed for the counted integers', d_type=int)
    divisors = [int(d) for d in load(ctx, 'the list of divisors of which at least one must divide a counted integer', d_type=list)]
    narrate(ctx, f'Limit {limit}, divisors {divisors}')
    k = len(divisors)
    total = 0
    for mask in range(1, 1 << k):
        subset = [divisors[i] for i in range(k) if (mask >> i) & 1]
        current = subset[0]
        for d in subset[1:]:
            r = do(ctx, 'Compute the least common multiple of two positive integers', payloads={'the first positive integer': current, 'the second positive integer': d})
            current = load(r, 'the least common multiple of the two integers', d_type=int, ctx=ctx)
        m = do(ctx, 'Count the positive multiples of a divisor that do not exceed a limit', payloads={'the divisor whose multiples are counted': current, 'the largest value allowed for the counted multiples': limit})
        count = load(m, 'the number of positive multiples of the divisor not exceeding the limit', d_type=int, ctx=ctx)
        sign = 1 if len(subset) % 2 == 1 else -1
        total += sign * count
        narrate(ctx, f'Subset {subset}: lcm {current}, {count} multiples, sign {sign:+d}')
    narrate(ctx, f'Inclusion-exclusion total: {total}')
    return {'the count of positive integers up to the limit divisible by at least one of the divisors': total}
