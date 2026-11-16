from decimal import Decimal, localcontext
import re

def semantic_function(ctx):
    '''Iterate double-declining-balance yearly depreciation with a salvage-value floor from the original cost for the elapsed years and return the unrounded book value.'''
    narrate(ctx, 'Iterating double-declining-balance depreciation with a salvage-value floor from the original cost')
    cost_raw = load(ctx, 'original cost in dollars', d_type=str)
    narrate(ctx, 'Original cost as given: ' + str(cost_raw))
    salvage_raw = load(ctx, 'salvage value in dollars', d_type=str)
    narrate(ctx, 'Salvage value as given: ' + str(salvage_raw))
    life_raw = load(ctx, 'useful life in years', d_type=str)
    narrate(ctx, 'Useful life as given: ' + str(life_raw))
    years_raw = load(ctx, 'number of years elapsed', d_type=str)
    narrate(ctx, 'Years elapsed as given: ' + str(years_raw))

    def _dec(text, label):
        s = str(text).strip().replace(',', '').replace('$', '')
        if re.fullmatch(r'\+?\d+(\.\d+)?', s) is None:
            raise ValueError('Cannot parse ' + label + ' from: ' + str(text))
        return Decimal(s)

    ys = str(years_raw).strip()
    if re.fullmatch(r'\d+', ys) is None:
        raise ValueError('Cannot parse number of years from: ' + str(years_raw))
    years = int(ys)
    with localcontext() as c:
        c.prec = 50
        cost = _dec(cost_raw, 'cost')
        salvage = _dec(salvage_raw, 'salvage value')
        life = _dec(life_raw, 'useful life')
        if life <= 0:
            raise ValueError('Useful life must be positive')
        if salvage < 0 or cost < salvage:
            raise ValueError('Salvage value must be non-negative and not exceed cost')
        rate = Decimal(2) / life
        narrate(ctx, 'Annual double-declining-balance rate is 2 divided by ' + str(life) + ' = ' + str(rate) + ' per year')
        bv = cost
        for y in range(1, years + 1):
            room = bv - salvage
            if room < 0:
                room = Decimal(0)
            dep = bv * rate
            capped = dep > room
            if capped:
                dep = room
            bv = bv - dep
            note = ' (capped so book value stays at salvage)' if capped else ''
            narrate(ctx, 'End of year ' + str(y) + ': depreciation ' + str(dep) + ' dollars, book value ' + str(bv) + ' dollars' + note)
    narrate(ctx, 'Book value after ' + str(years) + ' years, unrounded: ' + str(bv) + ' dollars')
    return {'book value after elapsed years in dollars unrounded': bv}
