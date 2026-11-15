import math

def semantic_function(ctx):
    '''Compute the time until a projectile launched from a given height returns to ground level under constant gravity.'''
    narrate(ctx, 'Computing the time of flight until the ball reaches ground level')
    h = load(ctx, 'the height of the cliff above its base', d_type=float)
    vy = load(ctx, 'the initial vertical velocity component', d_type=float)
    g = load(ctx, 'the gravitational acceleration magnitude', d_type=float)
    narrate(ctx, f'Height h = {h} m, initial vertical velocity v0y = {vy} m/s, gravity g = {g} m/s^2')
    if g <= 0:
        narrate(ctx, 'Gravity must be positive; no landing time can be computed')
        return {'time of flight until the ball reaches ground level': None}
    a = g / 2.0
    b = -vy
    c = -h
    narrate(ctx, f'Landing condition y(t) = 0 gives {a:.4f} t^2 + {b:.4f} t + {c:.4f} = 0')
    disc = b * b - 4.0 * a * c
    narrate(ctx, f'Discriminant = {disc:.4f}')
    if disc < 0:
        narrate(ctx, 'No real root: the ball never reaches ground level')
        return {'time of flight until the ball reaches ground level': None}
    root = (-b + math.sqrt(disc)) / (2.0 * a)
    narrate(ctx, f'Positive root t = (v0y + sqrt(v0y^2 + 2gh)) / g = {root:.3f} s')
    return {'time of flight until the ball reaches ground level': root}
