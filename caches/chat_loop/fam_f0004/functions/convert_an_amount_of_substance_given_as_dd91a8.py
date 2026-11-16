def semantic_function(ctx):
    '''Convert an amount of substance given as raw text into moles using exact unit factors.'''
    from decimal import Decimal
    narrate(ctx, 'Converting the amount of substance into moles')
    raw_amount = load(ctx, 'the amount of gas in moles with unit', d_type=str)
    narrate(ctx, 'Loaded amount text: ' + repr(raw_amount))
    parsed = do(ctx, 'Parse a raw quantity text into its numeric magnitude with any scale word applied and its unit symbol', payloads={'quantity as raw text copied from the question': raw_amount})
    magnitude = Decimal(str(load(parsed, 'numeric magnitude of the quantity', d_type=Decimal, ctx=ctx)))
    unit_text = str(load(parsed, 'unit symbol of the quantity', d_type=str, ctx=ctx))
    narrate(ctx, 'Parsed magnitude ' + str(magnitude) + ' with unit ' + repr(unit_text))
    if magnitude < 0:
        raise ValueError('amount of substance must be non-negative, got ' + str(magnitude))
    key = ''.join(unit_text.split()).lower()
    factors = {'mol': Decimal('1'), 'mole': Decimal('1'), 'moles': Decimal('1'), 'mmol': Decimal('1E-3'), 'kmol': Decimal('1E3')}
    factor = factors.get(key)
    if factor is None:
        raise ValueError('unsupported amount unit: ' + repr(unit_text))
    narrate(ctx, 'Unit ' + repr(unit_text) + ' has exact factor ' + str(factor) + ' to moles')
    moles = magnitude * factor
    narrate(ctx, 'Amount of substance in moles: ' + str(moles))
    return {'amount of substance in moles': moles}
