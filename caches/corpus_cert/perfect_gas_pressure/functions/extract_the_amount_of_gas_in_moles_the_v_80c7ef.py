def semantic_function(ctx):
    '''Extract the amount of gas in moles, the volume with its unit, the temperature with its unit, and the gas name from the problem parameters.'''
    narrate(ctx, 'Extracting the given quantities for the perfect gas problem from the parameter descriptions')
    amount_text = load(ctx, 'the amount of gas n with unit and chemical formula of the gas species and its physical state', d_type=str)
    volume_text = load(ctx, 'the volume V occupied by the gas with its unit', d_type=str)
    temperature_text = load(ctx, 'the temperature T of the gas with its unit', d_type=str)
    gas_text = load(ctx, 'the name of the gas species whose pressure is to be predicted', d_type=str)
    amount_mol = smart_convert(ctx, extract(ctx, amount_text, 'the numerical value of the amount of substance in moles', d_type=float), float)
    volume_value = smart_convert(ctx, extract(ctx, volume_text, 'the numerical value of the volume', d_type=float), float)
    volume_unit = extract(ctx, volume_text, 'the unit of volume, such as dm3, L or cm3')
    temperature_value = smart_convert(ctx, extract(ctx, temperature_text, 'the numerical value of the temperature', d_type=float), float)
    temperature_unit = extract(ctx, temperature_text, 'the unit of temperature, such as C or K')
    gas_name = extract(ctx, gas_text, 'the name of the gas species')
    narrate(ctx, f'Amount of gas: {amount_mol} mol')
    narrate(ctx, f'Volume: {volume_value} {volume_unit}')
    narrate(ctx, f'Temperature: {temperature_value} {temperature_unit}')
    narrate(ctx, f'Gas species: {gas_name}')
    return {
        'amount of gas n in moles': amount_mol,
        'volume with its unit': f'{volume_value} {volume_unit}',
        'temperature with its unit': f'{temperature_value} {temperature_unit}',
        'name of gas species': gas_name,
    }
