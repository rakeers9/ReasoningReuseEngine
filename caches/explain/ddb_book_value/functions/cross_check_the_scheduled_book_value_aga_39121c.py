def semantic_function(ctx):
    '''Cross-check the scheduled book value against the closed-form expression cost times (1 - rate) raised to the target year, when the salvage floor is not reached.'''
    from fractions import Fraction
    narrate(ctx, 'Cross-checking the schedule result against the closed-form expression cost x (1 - rate)^year')
    cost = load(ctx, 'the original acquisition cost of the asset with currency unit', d_type=float)
    rate_raw = load(ctx, 'annual depreciation rate as a fraction', d_type=str)
    year = load(ctx, 'the target year number for which the end-of-year book value is requested', d_type=int)
    schedule_bv = load(ctx, 'book value at end of target year from schedule', d_type=float)
    salvage = load(ctx, 'the salvage value (residual value) of the asset at the end of its useful life with currency unit', d_type=float)
    cost_f = Fraction(str(cost))
    rate = Fraction(str(rate_raw).strip())
    schedule_f = Fraction(str(schedule_bv))
    salvage_f = Fraction(str(salvage))
    closed = cost_f * (1 - rate) ** int(year)
    narrate(ctx, f'Closed-form value {float(cost_f):.2f} x (1 - {rate})^{year} = {float(closed):.2f}')
    tol = Fraction(1, 100)
    if closed >= salvage_f:
        agrees = abs(closed - schedule_f) < tol
        narrate(ctx, f'Salvage floor not reached; comparing closed form {float(closed):.2f} with schedule {float(schedule_f):.2f}')
    else:
        agrees = abs(schedule_f - salvage_f) < tol
        narrate(ctx, f'Closed form {float(closed):.2f} falls below salvage {float(salvage_f):.2f}; schedule should sit at salvage {float(salvage_f):.2f}')
    narrate(ctx, f'Schedule value {float(schedule_f):.2f} agrees with closed form: {bool(agrees)}')
    return {'closed-form book value agrees with schedule flag': bool(agrees)}
