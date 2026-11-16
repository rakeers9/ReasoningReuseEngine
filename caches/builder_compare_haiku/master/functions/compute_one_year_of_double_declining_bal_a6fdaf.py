def semantic_function(ctx):
    '''Compute one year of double-declining-balance depreciation as the rate times the opening book value, capped so the closing book value does not fall below the salvage value.'''
    from decimal import Decimal, localcontext
    from fractions import Fraction
    narrate(ctx, 'Computing the capped double-declining-balance depreciation for one year')
    raw_book = load(ctx, 'the opening book value in dollars as an exact decimal', d_type=None)
    raw_rate = load(ctx, 'the double-declining-balance rate as an exact fraction', d_type=None)
    raw_salvage = load(ctx, 'the salvage value in dollars as an exact decimal', d_type=None)
    narrate(ctx, 'Loaded opening book value ' + str(raw_book) + ', rate ' + str(raw_rate) + ' and salvage ' + str(raw_salvage))
    with localcontext() as c:
        c.prec = 60
        book = Decimal(str(raw_book))
        salvage = Decimal(str(raw_salvage))
        rate_frac = Fraction(str(raw_rate))
        if book < 0 or salvage < 0:
            raise ValueError('Book value and salvage value must be non-negative')
        if rate_frac <= 0 or rate_frac > 1:
            raise ValueError('DDB rate must be in (0, 1]')
        rate = Decimal(rate_frac.numerator) / Decimal(rate_frac.denominator)
        if book <= salvage:
            dep = Decimal(0)
            narrate(ctx, 'Opening book value is already at or below salvage so no depreciation is taken')
        else:
            dep = book * rate
            room = book - salvage
            if dep > room:
                narrate(ctx, 'Rate times book would breach salvage so depreciation is capped at ' + str(room))
                dep = room
            else:
                narrate(ctx, 'Depreciation is rate times opening book value = ' + str(dep))
    return {'depreciation for the year in dollars': dep}
