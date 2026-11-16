import re
from decimal import Decimal

def semantic_function(ctx):
    '''Parse the gas volume from raw text including its unit and convert it exactly to litres.'''
    narrate(ctx, 'Parsing the gas volume from raw text and converting it exactly to litres')
    raw = load(ctx, 'the volume occupied by the gas with unit', d_type=str)
    text = str(raw).strip()
    narrate(ctx, 'Raw volume text is: ' + text)
    m = re.match(r'^\s*([+-]?)\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)\s*(.+?)\s*$', text)
    if m is None:
        narrate(ctx, 'Could not find a number followed by a unit in the volume text')
        raise ValueError('cannot parse a volume with unit from: ' + text)
    number = Decimal(m.group(1) + m.group(2).replace(',', ''))
    unit_text = m.group(3)
    unit_key = unit_text.lower().replace(' ', '').replace('-', '').replace('^', '').replace('³', '3').replace('.', '')
    factors = {'m3': Decimal('1000'), 'cubicmetre': Decimal('1000'), 'cubicmetres': Decimal('1000'), 'cubicmeter': Decimal('1000'), 'cubicmeters': Decimal('1000'), 'dm3': Decimal('1'), 'cubicdecimetre': Decimal('1'), 'cubicdecimetres': Decimal('1'), 'cubicdecimeter': Decimal('1'), 'cubicdecimeters': Decimal('1'), 'l': Decimal('1'), 'litre': Decimal('1'), 'litres': Decimal('1'), 'liter': Decimal('1'), 'liters': Decimal('1'), 'ml': Decimal('0.001'), 'millilitre': Decimal('0.001'), 'millilitres': Decimal('0.001'), 'milliliter': Decimal('0.001'), 'milliliters': Decimal('0.001'), 'cm3': Decimal('0.001'), 'cc': Decimal('0.001'), 'cubiccentimetre': Decimal('0.001'), 'cubiccentimetres': Decimal('0.001'), 'cubiccentimeter': Decimal('0.001'), 'cubiccentimeters': Decimal('0.001'), 'µl': Decimal('0.000001'), 'μl': Decimal('0.000001'), 'ul': Decimal('0.000001'), 'microlitre': Decimal('0.000001'), 'microlitres': Decimal('0.000001'), 'microliter': Decimal('0.000001'), 'microliters': Decimal('0.000001')}
    if unit_key not in factors:
        narrate(ctx, 'Unsupported volume unit: ' + unit_text)
        raise ValueError('unsupported volume unit: ' + unit_text)
    volume_l = number * factors[unit_key]
    if volume_l <= 0:
        narrate(ctx, 'Volume must be positive')
        raise ValueError('volume must be positive')
    narrate(ctx, 'Volume is ' + str(number) + ' ' + unit_text + ' which equals ' + str(volume_l) + ' L')
    return {'volume in litres': volume_l}
