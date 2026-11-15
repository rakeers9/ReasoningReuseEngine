def semantic_function(ctx):
    """Compute one year of declining-balance depreciation from the beginning book value and the annual rate, capped so the ending book value does not fall below salvage value."""
    from fractions import Fraction
    narrate(ctx, 'Computing one year of declining-balance depreciation with a salvage value floor')
    begin_raw = load(ctx, 'beginning book value for the year', d_type=float)
    rate_raw = load(ctx, 'annual depreciation rate as a fraction', d_type=str)
    salvage_raw = load(ctx, 'the salvage value (residual value) of the asset at the end of its useful life with currency unit', d_type=float)
    begin = Fraction(str(begin_raw))
    rate = Fraction(str(rate_raw).strip())
    salvage = Fraction(str(salvage_raw))
    narrate(ctx, f'Beginning book value {float(begin):.2f}; annual rate {rate} ({float(rate) * 100:.2f}%); salvage value {float(salvage):.2f}')
    raw_dep = begin * rate
    room = begin - salvage
    if room <= 0:
        dep = Fraction(0)
        narrate(ctx, 'Book value is already at or below salvage value; no depreciation this year')
    elif raw_dep > room:
        dep = room
        narrate(ctx, f'Raw depreciation {float(raw_dep):.2f} would breach salvage; capped at {float(dep):.2f}')
    else:
        dep = raw_dep
        narrate(ctx, f'Depreciation = {float(begin):.2f} x {rate} = {float(dep):.2f}; salvage floor not reached')
    end_bv = begin - dep
    narrate(ctx, f'Ending book value = {float(begin):.2f} - {float(dep):.2f} = {float(end_bv):.2f}')
    return {
        'depreciation expense for the year': round(float(dep), 2),
        'ending book value for the year': round(float(end_bv), 2),
    }
