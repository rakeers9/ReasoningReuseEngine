from decimal import Decimal, localcontext
import re

SCALES = {None: Decimal(1), 'billion': Decimal(10) ** 9, 'bn': Decimal(10) ** 9, 'million': Decimal(10) ** 6, 'mn': Decimal(10) ** 6, 'm': Decimal(10) ** 6, 'thousand': Decimal(10) ** 3, 'k': Decimal(10) ** 3}

def semantic_function(ctx):
    '''Compute the straight-line depreciation charge for the year on the remaining depreciable base over the remaining years.'''
    narrate(ctx, 'Computing straight-line charge on the remaining depreciable base over the remaining years')
    bv_raw = load(ctx, 'beginning book value for the year in dollars', d_type=None)
    salvage_raw = load(ctx, 'salvage value in dollars', d_type=None)
    years_raw = load(ctx, 'remaining useful life in years including the current year', d_type=None)
    narrate(ctx, 'Loaded beginning book value text: ' + str(bv_raw))
    narrate(ctx, 'Loaded salvage value text: ' + str(salvage_raw))
    narrate(ctx, 'Loaded remaining useful life text: ' + str(years_raw))
    bv_s = re.sub(r'usd|dollars?|[$,\s]', '', str(bv_raw).strip().lower())
    bv_m = re.fullmatch(r'(?P<num>[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+))(?P<scale>billion|bn|million|mn|thousand|k|m)?', bv_s)
    if bv_m is None:
        narrate(ctx, 'Beginning book value text could not be parsed as a dollar amount')
        raise ValueError('Cannot parse beginning book value: ' + str(bv_raw))
    bv = Decimal(bv_m.group('num')) * SCALES[bv_m.group('scale')]
    narrate(ctx, 'Beginning book value in dollars: ' + str(bv))
    sal_s = re.sub(r'usd|dollars?|[$,\s]', '', str(salvage_raw).strip().lower())
    sal_m = re.fullmatch(r'(?P<num>[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+))(?P<scale>billion|bn|million|mn|thousand|k|m)?', sal_s)
    if sal_m is None:
        narrate(ctx, 'Salvage value text could not be parsed as a dollar amount')
        raise ValueError('Cannot parse salvage value: ' + str(salvage_raw))
    salvage = Decimal(sal_m.group('num')) * SCALES[sal_m.group('scale')]
    narrate(ctx, 'Salvage value in dollars: ' + str(salvage))
    years_s = str(years_raw).strip().lower()
    if re.fullmatch(r'[0-9]+', years_s):
        remaining = int(years_s)
        narrate(ctx, 'Remaining years read directly as digits: ' + str(remaining))
    else:
        narrate(ctx, 'Remaining years are written as words, so the year-count parser is used')
        years_result = do(ctx, 'Parse a count of years written as digits or English number words into an integer.', payloads={'number of years as text copied from the question': years_raw})
        remaining = load(years_result, 'whole number of years as an integer', d_type=int, ctx=ctx)
        narrate(ctx, 'Remaining years parsed from words: ' + str(remaining))
    if remaining < 1:
        narrate(ctx, 'Remaining useful life is below one year, which is invalid')
        raise ValueError('Remaining useful life must be at least 1 year')
    base = bv - salvage
    if base < 0:
        narrate(ctx, 'Book value is below salvage, so the depreciable base is zero')
        base = Decimal(0)
    else:
        narrate(ctx, 'Depreciable base above salvage: ' + str(base) + ' dollars')
    with localcontext() as c:
        c.prec = 50
        sl = base / Decimal(remaining)
    narrate(ctx, 'Straight-line charge = base / remaining years = ' + str(sl) + ' dollars')
    return {'straight-line charge for the year in dollars': sl}
