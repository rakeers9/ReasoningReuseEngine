def semantic_function(ctx):
    '''Convert a temperature text into kelvin, applying the Celsius or Fahrenheit offset when needed.'''
    from decimal import Decimal, localcontext
    narrate(ctx, 'Converting the gas temperature into kelvin')
    raw_temperature = load(ctx, 'the temperature of the gas with unit', d_type=str)
    narrate(ctx, 'Loaded temperature text: ' + repr(raw_temperature))
    parsed = do(ctx, 'Parse a raw quantity text into its numeric magnitude with any scale word applied and its unit symbol', payloads={'quantity as raw text copied from the question': raw_temperature})
    magnitude = Decimal(str(load(parsed, 'numeric magnitude of the quantity', d_type=Decimal, ctx=ctx)))
    unit_text = str(load(parsed, 'unit symbol of the quantity', d_type=str, ctx=ctx))
    narrate(ctx, 'Parsed temperature ' + str(magnitude) + ' with unit ' + repr(unit_text))
    key = ''.join(unit_text.split()).lower().replace('\u00b0', '').replace('deg', '')
    with localcontext() as lc:
        lc.prec = 60
        if key in ('k', 'kelvin', 'kelvins'):
            kelvin = magnitude
            narrate(ctx, 'Temperature is already in kelvin, so no offset is applied')
        elif key in ('c', 'celsius', 'centigrade'):
            kelvin = magnitude + Decimal('273.15')
            narrate(ctx, 'Celsius offset 273.15 applied')
        elif key in ('f', 'fahrenheit'):
            celsius = (magnitude - Decimal(32)) * Decimal(5) / Decimal(9)
            kelvin = celsius + Decimal('273.15')
            narrate(ctx, 'Fahrenheit converted to Celsius, then offset 273.15 applied')
        else:
            raise ValueError('unsupported temperature unit: ' + repr(unit_text))
    if kelvin <= 0:
        raise ValueError('temperature must be above absolute zero, got ' + str(kelvin) + ' K')
    narrate(ctx, 'Temperature in kelvin: ' + str(kelvin))
    return {'temperature in kelvin': kelvin}
