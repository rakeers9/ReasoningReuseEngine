def semantic_function(ctx):
    '''Count the seatings of n people around a round table in which two specified people do not sit next to each other.'''
    narrate(ctx, 'Counting seatings in which the two specified people are not next to each other')
    n = load(ctx, 'the total number of distinct people to be seated around the round table, as a positive integer count', d_type=int)
    narrate(ctx, 'Number of people to seat: ' + str(n))
    first = load(ctx, 'the name or label of the first of the two particular people who must not sit next to each other', d_type=str)
    second = load(ctx, 'the name or label of the second of the two particular people who must not sit next to each other', d_type=str)
    narrate(ctx, 'Restricted pair: ' + str(first) + ' and ' + str(second))
    total_result = do(ctx, 'Count the total number of distinct circular seating arrangements of n people around a round table.', payloads={'the total number of distinct people to be seated around the round table, as a positive integer count': n})
    total = load(total_result, 'the total number of circular seatings of the people around the table', d_type=int, ctx=ctx)
    narrate(ctx, 'Total circular seatings: ' + str(total))
    adjacent_result = do(ctx, 'Count the circular seatings in which two specified people sit next to each other by merging them into one block and multiplying by their internal orderings.', payloads={'the total number of distinct people to be seated around the round table, as a positive integer count': n, 'the name or label of the first of the two particular people who must not sit next to each other': first, 'the name or label of the second of the two particular people who must not sit next to each other': second})
    adjacent = load(adjacent_result, 'the number of circular seatings in which the two specified people sit next to each other', d_type=int, ctx=ctx)
    narrate(ctx, 'Seatings with the pair adjacent: ' + str(adjacent))
    not_adjacent = total - adjacent
    narrate(ctx, 'Seatings with the pair not adjacent: ' + str(total) + ' - ' + str(adjacent) + ' = ' + str(not_adjacent))
    return {'the number of circular seatings in which the two specified people are not next to each other': not_adjacent}
