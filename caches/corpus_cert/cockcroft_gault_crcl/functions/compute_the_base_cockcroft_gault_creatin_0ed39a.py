def semantic_function(ctx):
    '''Compute the base Cockcroft-Gault creatinine clearance before sex correction from age, body weight, and serum creatinine.'''
    narrate(ctx, 'Computing the base Cockcroft-Gault creatinine clearance before sex correction')
    age = load(ctx, 'validated age in years', d_type=float)
    weight = load(ctx, 'validated body weight in kg', d_type=float)
    scr = load(ctx, 'validated serum creatinine in mg/dL', d_type=float)
    narrate(ctx, 'Loaded age: ' + str(age) + ' years')
    narrate(ctx, 'Loaded body weight: ' + str(weight) + ' kg')
    narrate(ctx, 'Loaded serum creatinine: ' + str(scr) + ' mg/dL')
    if scr <= 0:
        raise ValueError('Serum creatinine must be strictly positive, got ' + str(scr))
    age_term = 140 - age
    narrate(ctx, 'Age term: 140 - ' + str(age) + ' = ' + str(age_term))
    numerator = age_term * weight
    narrate(ctx, 'Numerator: ' + str(age_term) + ' x ' + str(weight) + ' = ' + str(numerator))
    denominator = 72 * scr
    narrate(ctx, 'Denominator: 72 x ' + str(scr) + ' = ' + str(denominator))
    base = numerator / denominator
    narrate(ctx, 'Base estimate before sex correction: ' + str(numerator) + ' / ' + str(denominator) + ' = ' + str(round(base, 2)) + ' mL/min')
    return {'base creatinine clearance before sex correction in mL/min': base}