def semantic_function(ctx):
    '''Check that the patient age, sex, body weight, and serum creatinine are within the valid ranges for the Cockcroft-Gault equation and normalise the sex to female or male.'''
    import re
    narrate(ctx, 'Validating the patient inputs for the Cockcroft-Gault equation')
    raw_age = load(ctx, 'the age of the patient in years', d_type=str)
    raw_sex = load(ctx, 'the biological sex of the patient', d_type=str)
    raw_weight = load(ctx, 'the body weight of the patient in kilograms with unit', d_type=str)
    raw_scr = load(ctx, 'the serum creatinine concentration of the patient in mg/dL with unit', d_type=str)
    narrate(ctx, 'Loaded raw age: ' + str(raw_age))
    narrate(ctx, 'Loaded raw sex: ' + str(raw_sex))
    narrate(ctx, 'Loaded raw body weight: ' + str(raw_weight))
    narrate(ctx, 'Loaded raw serum creatinine: ' + str(raw_scr))
    def first_number(text):
        m = re.search(r'[-+]?\d*\.?\d+', str(text))
        if m is None:
            raise ValueError('No numeric value found in ' + str(text))
        return float(m.group(0))
    age = first_number(raw_age)
    weight = first_number(raw_weight)
    scr = first_number(raw_scr)
    sex_text = str(raw_sex).strip().lower()
    if re.search(r'\b(female|woman|women|girl|f)\b', sex_text):
        sex = 'female'
    elif re.search(r'\b(male|man|men|boy|m)\b', sex_text):
        sex = 'male'
    else:
        raise ValueError('Sex must be male or female, got ' + str(raw_sex))
    if age <= 0 or age >= 140:
        raise ValueError('Age must be positive and below 140 for the Cockcroft-Gault term 140 minus age, got ' + str(age))
    if age < 18 or age > 110:
        narrate(ctx, 'Warning: age ' + str(age) + ' is outside the adult range the equation is validated for')
    if weight <= 0:
        raise ValueError('Body weight must be strictly positive, got ' + str(weight))
    if weight < 30 or weight > 250:
        narrate(ctx, 'Warning: body weight ' + str(weight) + ' kg is outside the plausible adult range')
    if scr <= 0:
        raise ValueError('Serum creatinine must be strictly positive because it is the divisor, got ' + str(scr))
    if scr < 0.2 or scr > 15:
        narrate(ctx, 'Warning: serum creatinine ' + str(scr) + ' mg/dL is outside the plausible clinical range')
    narrate(ctx, 'Validated age: ' + str(age) + ' years')
    narrate(ctx, 'Normalised sex: ' + sex)
    narrate(ctx, 'Validated body weight: ' + str(weight) + ' kg')
    narrate(ctx, 'Validated serum creatinine: ' + str(scr) + ' mg/dL')
    return {
        'validated age in years': age,
        'normalised sex as female or male': sex,
        'validated body weight in kg': weight,
        'validated serum creatinine in mg/dL': scr,
    }
