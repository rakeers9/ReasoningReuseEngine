def semantic_function(ctx):
    '''Convert a temperature given in degrees Celsius or kelvin to thermodynamic temperature in kelvin.'''
    import re
    narrate(ctx, 'Converting the temperature of the gas to thermodynamic temperature in kelvin')
    text = load(ctx, 'the temperature of the gas with unit', d_type=str)
    narrate(ctx, f'Loaded temperature: {text}')
    number = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    value = float(number.group())
    is_celsius = re.search(r'(°\s*C|deg\s*C|degC|\bC)\s*$', text.strip()) is not None
    if is_celsius:
        kelvin = value + 273.15
        narrate(ctx, f'Temperature is in degrees Celsius: {value} °C; adding 273.15 gives {kelvin:.2f} K')
    else:
        kelvin = value
        narrate(ctx, f'Temperature is already in kelvin: {kelvin} K')
    return {'temperature in K': kelvin}