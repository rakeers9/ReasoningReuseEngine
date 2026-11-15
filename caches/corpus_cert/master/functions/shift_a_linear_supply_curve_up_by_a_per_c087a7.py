def semantic_function(ctx):
    '''Shift a linear supply curve up by a per-unit tax on sellers to express it as a function of the buyer price.'''
    narrate(ctx, 'Expressing the post-tax supply curve in terms of the buyer price')
    a_s = load(ctx, 'intercept a of the supply curve Q = a + bP', d_type=float)
    b_s = load(ctx, 'slope b of the supply curve Q = a + bP', d_type=float)
    t = load(ctx, 'per-unit tax on sellers in dollars', d_type=float)
    narrate(ctx, f'Supply: Q = {a_s} + {b_s}P; per-unit tax on sellers t = {t}')
    narrate(ctx, 'Sellers receive P_s = P_b - t, so substitute P_s = P_b - t into the supply curve')
    new_a = a_s - b_s * t
    new_b = b_s
    narrate(ctx, f'Post-tax supply: Q = {a_s} + {b_s}(P_b - {t}) = {new_a} + {new_b}P_b')
    return {'post-tax supply intercept a in buyer price': new_a, 'post-tax supply slope b in buyer price': new_b}