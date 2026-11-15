def semantic_function(ctx):
    '''Convert the beam span, uniform load intensity, modulus of elasticity and second moment of area to SI base units.'''
    import re
    narrate(ctx, 'Converting the span, uniform load intensity, modulus of elasticity and second moment of area to SI base units')
    span_text = load(ctx, 'the span length L of the simply supported beam with unit', d_type=str)
    load_text = load(ctx, 'the uniform load intensity w acting over the full span of the beam with unit', d_type=str)
    e_text = load(ctx, 'the modulus of elasticity E of the beam material with unit', d_type=str)
    i_text = load(ctx, 'the second moment of area I of the beam cross-section with unit', d_type=str)
    superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
    length_scale = {'m': 1.0, 'mm': 0.001, 'cm': 0.01, 'km': 1000.0}
    line_load_scale = {'N/m': 1.0, 'kN/m': 1000.0, 'MN/m': 1000000.0, 'N/mm': 1000.0}
    pressure_scale = {'Pa': 1.0, 'kPa': 1000.0, 'MPa': 1000000.0, 'GPa': 1000000000.0, 'N/mm2': 1000000.0, 'N/m2': 1.0}
    inertia_scale = {'m4': 1.0, 'mm4': 1e-12, 'cm4': 1e-8}
    def to_si(text, table, label):
        s = str(text).translate(superscripts)
        s = s.replace('×', 'x').replace('−', '-').replace('–', '-').strip()
        m = re.match(r'^([-+]?[0-9]+(?:[.][0-9]+)?(?:[eE][-+]?[0-9]+)?)[ ]*(?:[x*][ ]*10\^?[ ]*([-+]?[0-9]+))?[ ]*(.*)$', s)
        if m is None:
            raise ValueError('Cannot read a number from ' + label + ': ' + s)
        value = float(m.group(1))
        if m.group(2) is not None:
            value = value * (10.0 ** int(m.group(2)))
        unit = m.group(3).replace(' ', '').replace('^', '')
        if unit not in table:
            options = ', '.join(table.keys())
            answer = str(lm(ctx, 'Which option is the same unit as ' + repr(unit) + '? Options: ' + options + '. Reply with exactly one option, written exactly as listed, and nothing else.')).strip()
            if answer not in table:
                raise ValueError('Unrecognised unit ' + repr(unit) + ' for ' + label)
            unit = answer
        si_value = value * table[unit]
        narrate(ctx, label + ' given as ' + s + ' converts to ' + str(si_value) + ' in SI base units')
        return si_value
    L_m = to_si(span_text, length_scale, 'span length')
    w_npm = to_si(load_text, line_load_scale, 'uniform load intensity')
    E_pa = to_si(e_text, pressure_scale, 'modulus of elasticity')
    I_m4 = to_si(i_text, inertia_scale, 'second moment of area')
    if L_m <= 0 or E_pa <= 0 or I_m4 <= 0 or w_npm < 0:
        raise ValueError('Span, modulus and second moment of area must be positive and the load non-negative')
    narrate(ctx, 'SI inputs: span ' + str(L_m) + ' m, uniform load ' + str(w_npm) + ' N/m, E ' + str(E_pa) + ' Pa, I ' + str(I_m4) + ' m^4')
    return {
        'the span length in metres': L_m,
        'the uniform load intensity in newtons per metre': w_npm,
        'the modulus of elasticity in pascals': E_pa,
        'the second moment of area in metres to the fourth': I_m4,
    }
