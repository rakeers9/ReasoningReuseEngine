from decimal import Decimal, localcontext
from fractions import Fraction

def to_dollars(value, label):
    if isinstance(value, Decimal):
        return value
    if value is None or isinstance(value, bool):
        raise ValueError('No dollar amount for ' + label)
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    s = str(value).strip().lower()
    for junk in ['$', ',', ' ', 'usd', 'dollars', 'dollar']:
        s = s.replace(junk, '')
    scale = Decimal(1)
    for word, factor in [('billion', Decimal(10) ** 9), ('bn', Decimal(10) ** 9), ('million', Decimal(10) ** 6), ('mn', Decimal(10) ** 6), ('thousand', Decimal(10) ** 3), ('k', Decimal(10) ** 3)]:
        if s.endswith(word) and len(s) > len(word):
            s = s[:-len(word)]
            scale = factor
            break
    try:
        return Decimal(s) * scale
    except Exception:
        raise ValueError('Cannot parse dollar amount for ' + label + ': ' + str(value))

def semantic_function(ctx):
    '''Compute one year of double-declining-balance depreciation as the larger of the declining-balance and straight-line charges, capped so book value stays at or above salvage, and return the charge and year-end book value.'''
    narrate(ctx, 'Computing one year of double-declining-balance depreciation with straight-line switch test and salvage cap')
    bv_raw = load(ctx, 'beginning book value for the year in dollars', d_type=None)
    salvage_raw = load(ctx, 'salvage value in dollars', d_type=None)
    rate_raw = load(ctx, 'double-declining-balance rate as an exact fraction', d_type=None)
    years_raw = load(ctx, 'remaining useful life in years including the current year', d_type=None)
    narrate(ctx, 'Loaded beginning book value text: ' + str(bv_raw))
    narrate(ctx, 'Loaded salvage value text: ' + str(salvage_raw))
    narrate(ctx, 'Loaded DDB rate: ' + str(rate_raw))
    narrate(ctx, 'Loaded remaining useful life text: ' + str(years_raw))
    bv = to_dollars(bv_raw, 'beginning book value')
    salvage = to_dollars(salvage_raw, 'salvage value')
    rate = Fraction(str(rate_raw).strip())
    if rate <= 0:
        raise ValueError('DDB rate must be positive')
    years_s = str(years_raw).strip()
    if not years_s.isdigit():
        raise ValueError('Remaining useful life must be a whole number of years: ' + str(years_raw))
    remaining = int(years_s)
    if remaining < 1:
        raise ValueError('Remaining useful life must be at least 1 year')
    with localcontext() as c:
        c.prec = 50
        ddb = bv * Decimal(rate.numerator) / Decimal(rate.denominator)
    narrate(ctx, 'Declining-balance charge = beginning book value x rate = ' + str(ddb) + ' dollars')
    sl_result = do(ctx, 'Compute the straight-line depreciation charge for the year on the remaining depreciable base over the remaining years', payloads={'beginning book value for the year in dollars': str(bv), 'salvage value in dollars': str(salvage), 'remaining useful life in years including the current year': remaining})
    sl = to_dollars(load(sl_result, 'straight-line charge for the year in dollars', d_type=None, ctx=ctx), 'straight-line charge')
    narrate(ctx, 'Straight-line charge on remaining base = ' + str(sl) + ' dollars')
    with localcontext() as c:
        c.prec = 50
        candidate = max(ddb, sl)
    if candidate == ddb:
        narrate(ctx, 'Declining-balance charge is at least the straight-line charge, so keep double-declining-balance for this year')
    else:
        narrate(ctx, 'Straight-line charge is larger, so switch to straight-line for this year')
    cap_result = do(ctx, 'Cap a candidate depreciation charge so the resulting book value does not fall below salvage value, parsing each dollar input robustly first.', payloads={'candidate depreciation charge in dollars': str(candidate), 'beginning book value for the year in dollars': str(bv), 'salvage value in dollars': str(salvage)})
    capped = to_dollars(load(cap_result, 'capped depreciation charge in dollars', d_type=None, ctx=ctx), 'capped charge')
    with localcontext() as c:
        c.prec = 50
        bv_end = bv - capped
    narrate(ctx, 'Depreciation charge for the year: ' + str(capped) + ' dollars')
    narrate(ctx, 'Book value at year end: ' + str(bv_end) + ' dollars')
    return {'depreciation charge for the year in dollars': capped, 'book value at year end in dollars': bv_end}
