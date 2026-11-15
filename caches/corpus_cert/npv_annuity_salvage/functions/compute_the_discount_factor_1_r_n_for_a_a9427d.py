def semantic_function(ctx):
    """Compute the discount factor (1+r)^-n for a decimal discount rate and a number of periods."""
    narrate(ctx, 'Computing the discount factor (1+r)^-n for the given rate and periods')
    rate = load(ctx, 'the discount rate as a decimal', d_type=float)
    periods = load(ctx, 'the number of periods until the cash is received', d_type=int)
    growth = 1.0 + rate
    growth_power = growth ** periods
    factor = 1.0 / growth_power
    narrate(ctx, f'Growth factor (1 + r)^n = {growth_power}, so discount factor (1 + r)^-n = {factor}')
    return {'the discount factor for the given periods': factor}