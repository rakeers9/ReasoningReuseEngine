from decimal import Decimal, localcontext
from fractions import Fraction

def semantic_function(ctx):
    '''Simulate year-by-year double-declining-balance depreciation from cost up to a target year while keeping book value at or above salvage.'''
    narrate(ctx, 'Simulating double-declining-balance depreciation year by year up to the target year, keeping book value at or above salvage')
    raw_cost = load(ctx, 'the cost in dollars as an exact decimal', d_type=None)
    raw_salvage = load(ctx, 'the salvage value in dollars as an exact decimal', d_type=None)
    raw_rate = load(ctx, 'the double-declining-balance rate as an exact fraction', d_type=None)
    raw_year = load(ctx, 'the target year as an integer', d_type=None)
    narrate(ctx, 'Loaded cost ' + str(raw_cost) + ', salvage ' + str(raw_salvage) + ', rate ' + str(raw_rate) + ' and target year ' + str(raw_year))
    cost = Decimal(str(raw_cost).strip())
    salvage = Decimal(str(raw_salvage).strip())
    rate = Fraction(str(raw_rate).strip())
    year = int(str(raw_year).strip())
    if cost <= salvage:
        raise ValueError('Cost must exceed salvage value')
    if year < 0:
        raise ValueError('Target year must be non-negative')
    if rate <= 0 or rate > 1:
        raise ValueError('DDB rate must be in (0, 1]')
    book = cost
    narrate(ctx, 'Starting book value at cost ' + str(book) + ' dollars')
    for y in range(1, year + 1):
        result = do(ctx, 'Compute one year of double-declining-balance depreciation as the rate times the opening book value, capped so the closing book value does not fall below the salvage value', payloads={'the opening book value in dollars as an exact decimal': str(book), 'the double-declining-balance rate as an exact fraction': str(rate), 'the salvage value in dollars as an exact decimal': str(salvage)})
        dep = Decimal(str(load(result, 'depreciation for the year in dollars', d_type=None, ctx=ctx)))
        with localcontext() as c:
            c.prec = 60
            book = book - dep
        narrate(ctx, 'Year ' + str(y) + ': depreciation ' + str(dep) + ' dollars, closing book value ' + str(book) + ' dollars')
    narrate(ctx, 'Book value at the end of year ' + str(year) + ' is ' + str(book) + ' dollars')
    return {'book value at the end of the target year in dollars': book}
