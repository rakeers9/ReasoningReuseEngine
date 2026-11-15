def semantic_function(ctx):
    '''Count the positive multiples of a divisor that do not exceed a limit.'''
    narrate(ctx, 'Counting the positive multiples of a divisor that do not exceed a limit')
    divisor = load(ctx, 'the divisor whose multiples are counted', d_type=int)
    limit = load(ctx, 'the largest value allowed for the counted multiples', d_type=int)
    if divisor <= 0:
        narrate(ctx, 'The divisor is not positive, so there are no counted multiples')
        return {'the number of positive multiples of the divisor not exceeding the limit': 0}
    count = max(0, limit // divisor)
    narrate(ctx, f'floor({limit}/{divisor}) = {count} positive multiples of {divisor} do not exceed {limit}')
    return {'the number of positive multiples of the divisor not exceeding the limit': count}
