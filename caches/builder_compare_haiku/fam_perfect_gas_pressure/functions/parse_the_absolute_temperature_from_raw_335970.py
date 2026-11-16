import re
from decimal import Decimal, localcontext

def semantic_function(ctx):
    """Parse the absolute temperature from raw text and convert it exactly to kelvin."""
    narrate(ctx, 'Parsing the gas temperature from raw text and converting it to kelvin')
    raw = load(ctx, 'the absolute temperature of the gas with unit', d_type=str)
    text = str(raw).strip()
    narrate(ctx, 'Raw temperature text is: ' + text)
    m = re.match(r'^\s*([+-]?)\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)\s*(.+?)\s*$', text)
    if m is None:
        narrate(ctx, 'Could not find a number followed by a unit in the temperature text')
        raise ValueError('cannot parse a temperature with unit from: ' + text)
    number = Decimal(m.group(1) + m.group(2).replace(',', ''))
    unit_text = m.group(3)
    unit_key = unit_text.lower().replace(' ', '').replace('.', '').replace('°', '').replace('º', '')
    with localcontext() as lc:
        lc.prec = 60
        if unit_key in ('k', 'kelvin', 'kelvins'):
            kelvin = number
        elif unit_key in ('c', 'degc', 'celsius', 'degreesc', 'degreescelsius', 'centigrade'):
            kelvin = number + Decimal('273.15')
        elif unit_key in ('f', 'degf', 'fahrenheit', 'degreesf', 'degreesfahrenheit'):
            kelvin = (number + Decimal('459.67')) * Decimal(5) / Decimal(9)
        elif unit_key in ('r', 'degr', 'rankine'):
            kelvin = number * Decimal(5) / Decimal(9)
        else:
            narrate(ctx, 'Unsupported temperature unit: ' + unit_text)
            raise ValueError('unsupported temperature unit: ' + unit_text)
    narrate(ctx, 'Temperature ' + unit_text + ' converts to ' + str(kelvin) + ' K')
    if kelvin <= 0:
        narrate(ctx, 'Absolute temperature must be above 0 K')
        raise ValueError('absolute temperature must be above 0 K')
    return {'temperature in kelvin': kelvin}
