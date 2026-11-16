from decimal import Decimal, localcontext
import re

def semantic_function(ctx):
    '''Apply one year of double-declining-balance depreciation to a beginning book value, capping the charge so the ending book value does not fall below salvage value.'''
    narrate(ctx, 'Applying one year of double-declining-balance depreciation with a salvage-value floor')
    raw_bv = load(ctx, 'the beginning book value of the year in dollars', d_type=str)
    narrate(ctx, 'Beginning book value as given: ' + str(raw_bv))
    raw_rate = load(ctx, 'the annual double-declining-balance rate as exact fraction', d_type=str)
    narrate(ctx, 'Annual double-declining-balance rate as given: ' + str(raw_rate))
    raw_salvage = load(ctx, 'the salvage value in dollars', d_type=str)
    narrate(ctx, 'Salvage value as given: ' + str(raw_salvage))
    def _num(text):
        s = str(text).strip().replace(',', '').replace('$', '')
        if re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)', s) is None:
            raise ValueError('Cannot parse amount from: ' + str(text))
        return Decimal(s)
    with localcontext() as c:
        c.prec = 50
        bv = _num(raw_bv)
        salvage = _num(raw_salvage)
        s = str(raw_rate).strip()
        if '/' in s:
            parts = s.split('/')
            if len(parts) != 2:
                raise ValueError('Cannot parse rate from: ' + s)
            rate = _num(parts[0]) / _num(parts[1])
        else:
            rate = _num(s)
        if rate <= 0:
            raise ValueError('DDB rate must be positive')
        room = bv - salvage
        if room < 0:
            room = Decimal(0)
        candidate = bv * rate
        if candidate > room:
            dep = room
            narrate(ctx, 'Depreciation capped so book value does not fall below salvage value')
        else:
            dep = candidate
        end = bv - dep
        narrate(ctx, 'Depreciation this year: ' + str(dep) + ' dollars')
        narrate(ctx, 'Ending book value this year: ' + str(end) + ' dollars')
    return {'depreciation charged this year in dollars': dep, 'ending book value in dollars': end}
