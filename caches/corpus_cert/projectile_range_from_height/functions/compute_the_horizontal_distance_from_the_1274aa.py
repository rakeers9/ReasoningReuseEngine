import math
import re

def semantic_function(ctx):
    '''Compute the horizontal distance from the cliff base to where a ball launched from the top of a cliff lands.'''
    narrate(ctx, 'Computing the horizontal landing distance of a ball launched from the top of a cliff')
    def number_in(text):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(text))
        if m is None:
            return smart_convert(ctx, text, float)
        return float(m.group(0))
    h = number_in(load(ctx, 'the height of the cliff above its base, with unit', d_type=str))
    v0 = number_in(load(ctx, 'the launch speed of the ball, with unit', d_type=str))
    theta_deg = number_in(load(ctx, 'the launch angle of the ball measured above the horizontal, with unit', d_type=str))
    g = number_in(load(ctx, 'the gravitational acceleration magnitude, with unit', d_type=str))
    narrate(ctx, f'Cliff height h = {h} m, launch speed v0 = {v0} m/s, launch angle = {theta_deg} degrees, gravity g = {g} m/s^2')
    split = do(ctx, 'Resolve a launch velocity into horizontal and vertical components using the launch angle above horizontal', payloads={'the launch speed of the ball': v0, 'the launch angle of the ball measured above the horizontal': theta_deg})
    vx = load(split, 'initial horizontal velocity component', d_type=float, ctx=ctx)
    vy = load(split, 'initial vertical velocity component', d_type=float, ctx=ctx)
    narrate(ctx, f'Horizontal velocity component v0x = {vx:.3f} m/s, vertical velocity component v0y = {vy:.3f} m/s')
    flight = do(ctx, 'Compute the time until a projectile launched from a given height returns to ground level under constant gravity', payloads={'the height of the cliff above its base': h, 'the initial vertical velocity component': vy, 'the gravitational acceleration magnitude': g})
    t = load(flight, 'time of flight until the ball reaches ground level', d_type=float, ctx=ctx)
    if t is None:
        narrate(ctx, 'The ball never reaches ground level, so no landing distance exists')
        return {'horizontal distance from the base of the cliff to the landing point': None}
    narrate(ctx, f'Time of flight t = {t:.3f} s')
    x = vx * t
    narrate(ctx, f'Horizontal landing distance x = v0x * t = {vx:.3f} * {t:.3f} = {x:.1f} m')
    return {'horizontal distance from the base of the cliff to the landing point': x}
