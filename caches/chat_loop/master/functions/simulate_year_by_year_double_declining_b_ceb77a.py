from decimal import Decimal
from fractions import Fraction
import re

def to_dollars(value, label):
    if isinstance(value, Decimal):
        return value
    if value is None or isinstance(value, bool):
        raise ValueError('No dollar amount for ' + label)
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    s = re.sub(r'usd|dollars?|[$,\s]', '', str(value).strip().lower())
    m = re.fullmatch(r'([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))(billion|bn|million|mn|thousand|k|m)?', s)
    if m is None:
        raise ValueError('Cannot parse dollar amount for ' + label + ': ' + str(value))
    scales = {None: Decimal(1), 'billion': Decimal(10) ** 9, 'bn': Decimal(10) ** 9, 'million': Decimal(10) ** 6, 'mn': Decimal(10) ** 6, 'm': Decimal(10) ** 6, 'thousand': Decimal(10) ** 3, 'k': Decimal(10) ** 3}
    return Decimal(m.group(1)) * scales[m.group(2)]

def semantic_function(ctx):
    '''Simulate year-by-year double-declining-balance depreciation with salvage floor to find the book value after a number of years.'''
    narrate(ctx, 'Simulating double-declining-balance depreciation year by year with a salvage floor')
    cost = to_dollars(load(ctx, 'asset cost in dollars', d_type=None), 'asset cost')
    salvage = to_dollars(load(ctx, 'salvage value in dollars', d_type=None), 'salvage value')
    life_s = str(load(ctx, 'useful life in years', d_type=None)).strip()
    years_s = str(load(ctx, 'number of elapsed years', d_type=None)).strip()
    rate_s = str(load(ctx, 'double-declining-balance rate as an exact fraction', d_type=None)).strip()
    narrate(ctx, 'Loaded asset cost: ' + str(cost) + ' dollars; salvage value: ' + str(salvage) + ' dollars')
    narrate(ctx, 'Loaded useful life: ' + life_s + ' years; elapsed years: ' + years_s)
    narrate(ctx, 'Loaded DDB rate: ' + rate_s)
    if not life_s.isdigit() or int(life_s) < 1:
        raise ValueError('Useful life must be a whole number of years of at least 1: ' + life_s)
    if not years_s.isdigit():
        raise ValueError('Elapsed years must be a non-negative whole number: ' + years_s)
    if salvage > cost:
        raise ValueError('Salvage value exceeds asset cost')
    life = int(life_s)
    years = int(years_s)
    rate = Fraction(rate_s)
    if rate <= 0:
        raise ValueError('DDB rate must be positive')
    bv = cost
    for y in range(1, min(years, life) + 1):
        remaining = life - y + 1
        res = do(ctx, 'Compute one year of double-declining-balance depreciation as the larger of the declining-balance and straight-line charges, capped so book value stays at or above salvage, and return the charge and year-end book value.', payloads={'beginning book value for the year in dollars': str(bv), 'salvage value in dollars': str(salvage), 'double-declining-balance rate as an exact fraction': str(rate), 'remaining useful life in years including the current year': remaining})
        charge = to_dollars(load(res, 'depreciation charge for the year in dollars', d_type=None, ctx=ctx), 'depreciation charge')
        bv = to_dollars(load(res, 'book value at year end in dollars', d_type=None, ctx=ctx), 'year-end book value')
        narrate(ctx, 'Year ' + str(y) + ': depreciation ' + str(charge) + ' dollars, year-end book value ' + str(bv) + ' dollars')
    if years > life:
        narrate(ctx, 'Elapsed years exceed useful life, so book value is held at salvage value')
        bv = salvage
    narrate(ctx, 'Book value after ' + str(years) + ' years: ' + str(bv) + ' dollars')
    return {'book value after the elapsed years in dollars': bv}
