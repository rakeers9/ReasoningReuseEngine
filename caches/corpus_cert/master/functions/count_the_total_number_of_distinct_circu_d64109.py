def semantic_function(ctx):
    '''Count the total number of distinct circular seating arrangements of n people around a round table.'''
    narrate(ctx, 'Counting the total circular seatings of the people around the table')
    n = load(ctx, 'the total number of distinct people to be seated around the round table, as a positive integer count', d_type=int)
    narrate(ctx, 'Number of people to seat: ' + str(n))
    units_result = do(ctx, 'Count the distinct circular arrangements of k distinct units around a round table up to rotation', payloads={'the number of distinct units to be arranged around the round table': n})
    total = load(units_result, 'the number of circular arrangements of the units up to rotation', d_type=int, ctx=ctx)
    narrate(ctx, 'Total circular seatings of ' + str(n) + ' people: (' + str(n) + '-1)! = ' + str(total))
    return {'the total number of circular seatings of the people around the table': total}