def semantic_function(ctx):
    '''Apply the female sex correction factor of 0.85 to a base creatinine clearance, leaving the value unchanged for male patients.'''
    narrate(ctx, 'Applying the sex correction to the base creatinine clearance')
    sex = load(ctx, 'normalised sex as female or male', d_type=str)
    base = load(ctx, 'base creatinine clearance before sex correction in mL/min', d_type=float)
    narrate(ctx, 'Loaded normalised sex: ' + str(sex))
    narrate(ctx, 'Loaded base creatinine clearance before sex correction: ' + str(base) + ' mL/min')
    sex_key = str(sex).strip().lower()
    if sex_key == 'female':
        corrected = base * 0.85
        narrate(ctx, 'Female patient: multiplying by 0.85, ' + str(base) + ' x 0.85 = ' + str(round(corrected, 2)) + ' mL/min')
    elif sex_key == 'male':
        corrected = base
        narrate(ctx, 'Male patient: no sex correction applied, value stays ' + str(round(corrected, 2)) + ' mL/min')
    else:
        raise ValueError('Sex must be female or male, got ' + str(sex))
    narrate(ctx, 'Sex-corrected creatinine clearance: ' + str(round(corrected, 2)) + ' mL/min')
    return {'sex-corrected creatinine clearance in mL/min': corrected}