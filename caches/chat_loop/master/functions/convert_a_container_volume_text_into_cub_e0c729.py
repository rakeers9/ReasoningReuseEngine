def semantic_function(ctx):
    '''Convert a container volume text into cubic metres using exact unit factors.'''
    import re
    from decimal import Decimal
    narrate(ctx, 'Converting the container volume into cubic metres')
    raw_volume = load(ctx, 'the volume of the flask or container holding the gas with unit', d_type=str)
    narrate(ctx, 'Loaded volume text: ' + repr(raw_volume))
    parsed = do(ctx, 'Parse a raw quantity text into its numeric magnitude with any scale word applied and its unit symbol', payloads={'quantity as raw text copied from the question': raw_volume})
    magnitude = Decimal(str(load(parsed, 'numeric magnitude of the quantity', d_type=Decimal, ctx=ctx)))
    unit_text = str(load(parsed, 'unit symbol of the quantity', d_type=str, ctx=ctx))
    narrate(ctx, 'Parsed magnitude ' + str(magnitude) + ' with unit ' + repr(unit_text))
    if magnitude <= 0:
        raise ValueError('container volume must be strictly positive, got ' + str(magnitude))
    key = re.sub(r'\s+', '', unit_text).lower().replace('^', '').replace('\u00b3', '3')
    factors = {
        'm3': Decimal('1'),
        'dm3': Decimal('1E-3'),
        'l': Decimal('1E-3'),
        'litre': Decimal('1E-3'),
        'litres': Decimal('1E-3'),
        'liter': Decimal('1E-3'),
        'liters': Decimal('1E-3'),
        'cm3': Decimal('1E-6'),
        'cc': Decimal('1E-6'),
        'ml': Decimal('1E-6'),
        'millilitre': Decimal('1E-6'),
        'millilitres': Decimal('1E-6'),
        'milliliter': Decimal('1E-6'),
        'milliliters': Decimal('1E-6'),
        'mm3': Decimal('1E-9'),
    }
    factor = factors.get(key)
    if factor is None:
        raise ValueError('unsupported volume unit: ' + repr(unit_text))
    narrate(ctx, 'Unit ' + repr(unit_text) + ' has exact factor ' + str(factor) + ' to cubic metres')
    volume = magnitude * factor
    narrate(ctx, 'Volume in cubic metres: ' + str(volume))
    return {'volume in cubic metres': volume}
