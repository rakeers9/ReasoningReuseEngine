import math

def semantic_function(ctx):
    '''Resolve a launch velocity into horizontal and vertical components using the launch angle above horizontal.'''
    narrate(ctx, 'Resolving the launch velocity into horizontal and vertical components')
    speed = load(ctx, 'the launch speed of the ball, with unit', d_type=float)
    angle_deg = load(ctx, 'the launch angle of the ball measured above the horizontal, in degrees', d_type=float)
    narrate(ctx, f'Launch speed v0 = {speed} and launch angle theta = {angle_deg} degrees')
    theta = math.radians(angle_deg)
    vx = speed * math.cos(theta)
    vy = speed * math.sin(theta)
    narrate(ctx, f'Horizontal component v0x = v0 cos(theta) = {vx:.3f}')
    narrate(ctx, f'Vertical component v0y = v0 sin(theta) = {vy:.3f}')
    return {
        'initial horizontal velocity component': vx,
        'initial vertical velocity component': vy,
    }
