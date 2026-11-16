def semantic_function(ctx):
    '''Round a dollar amount to the nearest cent using half-up rounding.'''
    from decimal import Decimal, ROUND_HALF_UP
    narrate(ctx, 'Rounding the dollar amount to the nearest cent using half-up rounding')
    raw = load(ctx, 'dollar amount at full precision', d_type=None)
    narrate(ctx, 'Loaded dollar amount ' + str(raw))
    text = str(raw).strip().replace(',', '').replace('$', '')
    amount = Decimal(text)
    rounded = amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    narrate(ctx, 'Rounded ' + str(amount) + ' dollars to ' + str(rounded) + ' dollars')
    return {'dollar amount rounded to the cent': rounded}