def semantic_function(ctx):
    '''Count the distinct circular arrangements of k distinct units around a round table up to rotation.'''
    import math
    k = load(ctx, 'the number of distinct units to be arranged around the round table', d_type=int)
    narrate(ctx, 'Loaded number of units to seat around the round table: ' + str(k))
    if k < 1:
        narrate(ctx, 'No units to arrange, so there are 0 circular arrangements')
        return {'the number of circular arrangements of the units up to rotation': 0}
    count = math.factorial(k - 1)
    narrate(ctx, 'Circular arrangements of ' + str(k) + ' units up to rotation: (' + str(k) + '-1)! = ' + str(count))
    return {'the number of circular arrangements of the units up to rotation': count}