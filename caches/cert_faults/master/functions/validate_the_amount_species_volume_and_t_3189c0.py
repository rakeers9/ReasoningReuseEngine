def semantic_function(ctx):
    '''Validate the amount, species, volume and temperature against the perfect gas validity ranges.'''
    import math
    import re
    narrate(ctx, 'Validating the amount, species, volume and temperature against the perfect gas validity ranges')
    n = load(ctx, 'the amount of substance of the gas in mol', d_type=float)
    narrate(ctx, f'Loaded amount of substance: {n} mol')
    species = load(ctx, 'the chemical species of the gas with its physical state', d_type=str)
    narrate(ctx, f'Loaded species: {species}')
    volume_text = load(ctx, 'the volume of the container holding the gas with unit', d_type=str)
    narrate(ctx, f'Loaded volume: {volume_text}')
    temperature_text = load(ctx, 'the temperature of the gas with unit', d_type=str)
    narrate(ctx, f'Loaded temperature: {temperature_text}')
    number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    problems = []
    if not (math.isfinite(n) and n > 0):
        problems.append('the amount must be a positive finite number of mol')
    is_gas = re.search(r'\(g\)', species) is not None
    if not is_gas:
        answer = lm(ctx, f'Is the substance {species} a gas at the stated conditions? Answer yes or no.')
        is_gas = str(answer).strip().lower().startswith('yes')
    if not is_gas:
        problems.append('the species must be a gas at the stated conditions')
    vol_match = re.search(number_pattern, volume_text)
    if vol_match is None or not float(vol_match.group()) > 0:
        problems.append('the volume must be a positive finite volume')
    temp_match = re.search(number_pattern, temperature_text)
    if temp_match is None:
        problems.append('the temperature must be a number with a unit')
    else:
        t_value = float(temp_match.group())
        is_celsius = re.search(r'(°|deg)?\s*C\s*$', temperature_text.strip()) is not None
        t_kelvin = t_value + 273.15 if is_celsius else t_value
        if not t_kelvin > 0:
            problems.append('the temperature must be above absolute zero')
    valid = len(problems) == 0
    message = 'All inputs lie within the perfect gas validity ranges' if valid else 'Invalid inputs: ' + '; '.join(problems)
    narrate(ctx, message)
    return {'inputs are valid for the perfect gas model': valid, 'validation message': message}