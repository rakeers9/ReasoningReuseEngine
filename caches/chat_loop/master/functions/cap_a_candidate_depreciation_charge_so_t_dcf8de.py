import re
from decimal import Decimal

def semantic_function(ctx):
    '''Limit a depreciation charge so that the resulting book value does not fall below salvage value.'''
    narrate(ctx, 'Capping a depreciation charge so the resulting book value does not fall below salvage value')
    charge_raw = load(ctx, 'candidate depreciation charge in dollars', d_type=None)
    bv_raw = load(ctx, 'beginning book value for the year in dollars', d_type=None)
    salvage_raw = load(ctx, 'salvage value in dollars', d_type=None)
    scales = {None: Decimal(1), 'billion': Decimal(10) ** 9, 'bn': Decimal(10) ** 9, 'million': Decimal(10) ** 6, 'mn': Decimal(10) ** 6, 'm': Decimal(10) ** 6, 'thousand': Decimal(10) ** 3, 'k': Decimal(10) ** 3}
    def to_dollars(value, label):
        if isinstance(value, Decimal):
            return value
        if value is None or isinstance(value, bool):
            raise ValueError('No dollar amount for ' + label)
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        s = re.sub(r'usd|dollars?|[$,\s]', '', str(value).strip().lower())
        m = re.fullmatch(r'([+-]?(?:\d+(?:\.\d+)?|\.\d+))(billion|bn|million|mn|thousand|k|m)?', s)
        if not m:
            raise ValueError('Cannot parse dollar amount for ' + label + ': ' + str(value))
        return Decimal(m.group(1)) * scales[m.group(2)]
    charge = to_dollars(charge_raw, 'candidate depreciation charge')
    bv = to_dollars(bv_raw, 'beginning book value')
    salvage = to_dollars(salvage_raw, 'salvage value')
    narrate(ctx, 'Candidate charge: ' + str(charge) + ', beginning book value: ' + str(bv) + ', salvage: ' + str(salvage))
    if charge < 0:
        raise ValueError('Depreciation charge cannot be negative')
    headroom = bv - salvage
    if headroom < 0:
        headroom = Decimal(0)
    capped = min(charge, headroom)
    if capped < charge:
        narrate(ctx, 'Charge limited to ' + str(capped) + ' so book value stays at salvage value')
    else:
        narrate(ctx, 'Charge not binding: book value after the charge stays at or above salvage value')
    narrate(ctx, 'Capped depreciation charge in dollars: ' + str(capped))
    return {'capped depreciation charge in dollars': capped}
