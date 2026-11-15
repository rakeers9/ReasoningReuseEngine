import decimal
from decimal import Decimal, ROUND_HALF_UP

def semantic_function(ctx):
    '''Round a monetary amount to the nearest cent for reporting.'''
    narrate(ctx, 'Rounding a monetary amount to the nearest cent for reporting')
    amount = load(ctx, 'the monetary amount to be rounded', d_type=float)
    narrate(ctx, f'Loaded monetary amount {amount}')
    cents = Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    rounded = float(cents)
    narrate(ctx, f'Rounded to the nearest cent: {rounded:.2f}')
    return {'the rounded monetary amount in dollars and cents': rounded}