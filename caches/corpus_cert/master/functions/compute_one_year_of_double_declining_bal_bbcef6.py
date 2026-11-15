from fractions import Fraction

def semantic_function(ctx):
    '''Compute one year of double-declining-balance depreciation from the beginning book value and rate, limited so the ending book value never falls below salvage value.'''
    narrate(ctx, 'Computing one year of double-declining-balance depreciation')
    beginning_value = load(ctx, 'beginning-of-year book value of the asset with currency unit', d_type=float)
    rate_raw = load(ctx, 'DDB depreciation rate per year', d_type=str)
    salvage = load(ctx, 'the salvage value of the asset (estimated residual value) with currency unit', d_type=float)
    beginning = Fraction(str(beginning_value))
    rate = Fraction(str(rate_raw).strip())
    salvage_floor = Fraction(str(salvage))
    narrate(ctx, f'Beginning book value {float(beginning):.2f}, DDB rate {rate}, salvage value {float(salvage_floor):.2f}')
    charge = beginning * rate
    ending = beginning - charge
    lowest_allowed = min(salvage_floor, beginning)
    if ending < lowest_allowed:
        ending = lowest_allowed
        narrate(ctx, 'Depreciation limited so book value does not fall below salvage value')
    charge = beginning - ending
    narrate(ctx, f'Depreciation charge = {float(charge):.2f}, ending book value = {float(ending):.2f}')
    return {'depreciation charge for the year': round(float(charge), 2), 'ending book value for the year': round(float(ending), 2)}
