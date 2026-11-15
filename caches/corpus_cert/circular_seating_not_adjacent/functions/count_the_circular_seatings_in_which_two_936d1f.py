import math
def semantic_function(ctx):
    '''Count the circular seatings in which two specified people sit next to each other by merging them into one block and multiplying by their internal orderings.'''
    narrate(ctx, 'Counting circular seatings in which the two specified people sit next to each other')
    n = load(ctx, 'the total number of distinct people to be seated around the round table, as a positive integer count', d_type=int)
    first = load(ctx, 'the name or label of the first of the two particular people who must not sit next to each other', d_type=str)
    second = load(ctx, 'the name or label of the second of the two particular people who must not sit next to each other', d_type=str)
    key = 'the number of circular seatings in which the two specified people sit next to each other'
    narrate(ctx, 'Seating ' + str(n) + ' people with ' + str(first) + ' and ' + str(second) + ' forced to sit together')
    if n < 2:
        narrate(ctx, 'Fewer than two people, so no adjacent pair exists')
        return {key: 0}
    if n == 2:
        narrate(ctx, 'Only two people, so they are always adjacent: 1 seating')
        return {key: 1}
    units = n - 1
    narrate(ctx, 'Merging the pair into one block leaves ' + str(n - 2) + ' other people, giving ' + str(units) + ' units around the table')
    units_result = do(ctx, 'Count the distinct circular arrangements of k distinct units around a round table up to rotation', payloads={'the number of distinct units to be arranged around the round table': units})
    unit_arrangements = load(units_result, 'the number of circular arrangements of the units up to rotation', d_type=int, ctx=ctx)
    internal = math.factorial(2)
    narrate(ctx, 'Internal orderings of the block of ' + str(first) + ' and ' + str(second) + ': 2! = ' + str(internal))
    adjacent = unit_arrangements * internal
    narrate(ctx, 'Seatings with the pair adjacent: ' + str(unit_arrangements) + ' x ' + str(internal) + ' = ' + str(adjacent))
    return {key: adjacent}