def semantic_function(ctx):
    '''Estimate the creatinine clearance of a patient using the Cockcroft-Gault equation from age, sex, body weight, and serum creatinine.'''
    narrate(ctx, 'Estimating creatinine clearance with the Cockcroft-Gault equation')
    raw_age = load(ctx, 'the age of the patient in years', d_type=str)
    raw_sex = load(ctx, 'the biological sex of the patient', d_type=str)
    raw_weight = load(ctx, 'the body weight of the patient in kilograms with unit', d_type=str)
    raw_scr = load(ctx, 'the serum creatinine concentration of the patient in mg/dL with unit', d_type=str)
    narrate(ctx, 'Patient inputs as given: age ' + str(raw_age) + ', sex ' + str(raw_sex) + ', weight ' + str(raw_weight) + ', serum creatinine ' + str(raw_scr))
    validated = do(ctx, 'Check that the patient age, sex, body weight, and serum creatinine are within the valid ranges for the Cockcroft-Gault equation and normalise the sex to female or male.', payloads={
        'the age of the patient in years': raw_age,
        'the biological sex of the patient': raw_sex,
        'the body weight of the patient in kilograms with unit': raw_weight,
        'the serum creatinine concentration of the patient in mg/dL with unit': raw_scr,
    })
    age = load(validated, 'validated age in years', d_type=float, ctx=ctx)
    sex = load(validated, 'normalised sex as female or male', d_type=str, ctx=ctx)
    weight = load(validated, 'validated body weight in kg', d_type=float, ctx=ctx)
    scr = load(validated, 'validated serum creatinine in mg/dL', d_type=float, ctx=ctx)
    narrate(ctx, 'Validated inputs: age ' + str(age) + ' years, sex ' + str(sex) + ', weight ' + str(weight) + ' kg, serum creatinine ' + str(scr) + ' mg/dL')
    base_result = do(ctx, 'Compute the base Cockcroft-Gault creatinine clearance before sex correction from age, body weight, and serum creatinine.', payloads={
        'validated age in years': age,
        'validated body weight in kg': weight,
        'validated serum creatinine in mg/dL': scr,
    })
    base = load(base_result, 'base creatinine clearance before sex correction in mL/min', d_type=float, ctx=ctx)
    narrate(ctx, 'Base estimate before sex correction: ' + str(round(base, 2)) + ' mL/min')
    corrected_result = do(ctx, 'Apply the female sex correction factor of 0.85 to a base creatinine clearance for female patients, leaving the value unchanged for male patients.', payloads={
        'normalised sex as female or male': sex,
        'base creatinine clearance before sex correction in mL/min': base,
    })
    crcl = load(corrected_result, 'sex-corrected creatinine clearance in mL/min', d_type=float, ctx=ctx)
    narrate(ctx, 'Sex-corrected creatinine clearance: ' + str(round(crcl, 2)) + ' mL/min')
    narrate(ctx, 'Final estimated creatinine clearance: ' + str(round(crcl, 1)) + ' mL/min')
    return {'estimated creatinine clearance in mL/min': crcl}
