def semantic_function(ctx):
    '''Convert a pressure given in pascals into bar using the exact factor 1 bar = 100000 Pa.'''
    from decimal import Decimal, localcontext
    narrate(ctx, 'Converting the pressure from pascals into bar')
    raw_pressure = load(ctx, 'pressure in pascals', d_type=str)
    narrate(ctx, 'Loaded pressure in pascals: ' + repr(raw_pressure))
    pascals = Decimal(str(raw_pressure).strip())
    if not pascals.is_finite() or pascals < 0:
        raise ValueError('pressure in pascals must be finite and non-negative, got ' + str(pascals))
    with localcontext() as dec_ctx:
        dec_ctx.prec = 60
        bar = pascals / Decimal('100000')
    narrate(ctx, 'Pressure in bar = ' + str(pascals) + ' Pa / 100000 Pa per bar = ' + str(bar) + ' bar')
    return {'pressure in bar': bar}