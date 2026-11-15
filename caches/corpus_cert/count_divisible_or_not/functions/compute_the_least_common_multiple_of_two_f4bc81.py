import math
def semantic_function(ctx):
    '''Compute the least common multiple of two positive integers.'''
    narrate(ctx, 'Computing the least common multiple of two positive integers')
    a = load(ctx, 'the first positive integer', d_type=int)
    b = load(ctx, 'the second positive integer', d_type=int)
    g = math.gcd(a, b)
    result = abs(a * b) // g
    narrate(ctx, f'lcm({a}, {b}) = {result}')
    return {'the least common multiple of the two integers': result}