from fractions import Fraction

def semantic_function(ctx):
    '''Verify the book value of a double-declining-balance depreciating asset at a requested year using the closed-form expression cost times one minus rate raised to the year count, floored at salvage value.'''
    narrate(ctx, 'Verifying the book value with the closed-form expression cost x (1 - rate)^year, floored at salvage value')
    cost = load(ctx, 'the original cost of the asset (basis) with currency unit', d_type=float)
    rate_raw = load(ctx, 'DDB depreciation rate per year', d_type=str)
    salvage = load(ctx, 'the salvage value of the asset (estimated residual value) with currency unit', d_type=float)
    target_year = load(ctx, 'the year at which the book value is requested (the target year, as a year number)', d_type=int)
    cost_f = Fraction(str(cost))
    salvage_f = Fraction(str(salvage))
    rate = Fraction(str(rate_raw).strip())
    year = int(target_year)
    if year < 0:
        raise ValueError('target year must be non-negative')
    narrate(ctx, f'Cost {float(cost_f):.2f}, rate {rate}, salvage {float(salvage_f):.2f}, year {year}')
    value = cost_f * (1 - rate) ** year
    floor = min(salvage_f, cost_f)
    if value < floor:
        value = floor
        narrate(ctx, 'Closed-form value fell below salvage value, so it is held at the salvage floor')
    narrate(ctx, f'Closed-form book value after {year} years: {float(value):.2f}')
    return {'closed-form book value at the requested year': round(float(value), 2)}
