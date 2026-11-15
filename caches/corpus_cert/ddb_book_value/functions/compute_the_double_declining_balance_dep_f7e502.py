from fractions import Fraction

def semantic_function(ctx):
    """Compute the double-declining-balance depreciation rate from the useful life."""
    narrate(ctx, 'Computing the double-declining-balance rate from the useful life')
    life = load(ctx, 'the useful life of the asset in years', d_type=int)
    narrate(ctx, f'Useful life loaded: {life} years')
    if life <= 0:
        raise ValueError('useful life must be a positive number of years')
    straight_line_rate = Fraction(1, life)
    ddb_rate = 2 * straight_line_rate
    narrate(ctx, f'Straight-line rate = 1 / {life} = {straight_line_rate}')
    narrate(ctx, f'DDB rate = 2 x straight-line rate = {ddb_rate} per year')
    return {'DDB depreciation rate per year': ddb_rate}