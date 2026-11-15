import re
def semantic_function(ctx):
    '''Determine the pH of a weak acid and conjugate base buffer after a strong acid is added.'''
    def _num(text):
        s = str(text).translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+'))
        s = s.replace('−', '-')
        s = re.sub(r'\s*[×xX*]\s*10\s*\^?\s*([-+]?\d+)', r'e\1', s)
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        if m is None:
            raise ValueError('No number found in ' + str(text))
        return float(m.group(0))
    def _volume_l(text):
        n = _num(text)
        if 'mL' in str(text):
            return n / 1000.0
        return n
    narrate(ctx, 'Determining the buffer pH after a strong acid is added')
    acid_name = load(ctx, 'the name of the weak acid HA that forms the buffer', d_type=str)
    ha_raw = load(ctx, 'the amount in moles of the weak acid HA with unit', d_type=str)
    salt_name = load(ctx, 'the name of the salt that supplies the conjugate base A- of the weak acid', d_type=str)
    a_raw = load(ctx, 'the amount in moles of the conjugate base A- supplied by the salt with unit', d_type=str)
    vol_raw = load(ctx, 'the total volume of the buffer solution with unit', d_type=str)
    ka_raw = load(ctx, 'the acid dissociation constant Ka of the weak acid HA (dimensionless, in scientific notation)', d_type=str)
    acid_raw = load(ctx, 'the amount in moles and identity of the strong acid added to the buffer (HCl)', d_type=str)
    ha_0 = _num(ha_raw)
    a_0 = _num(a_raw)
    volume = _volume_l(vol_raw)
    ka = _num(ka_raw)
    acid_added = _num(acid_raw)
    narrate(ctx, f'Loaded {acid_name} = {ha_0} mol, {salt_name} supplies {a_0} mol A-, volume {volume} L, Ka = {ka}, strong acid added = {acid_added} mol')
    pka_result = do(ctx, 'Compute the pKa of a weak acid from its acid dissociation constant Ka', payloads={'the acid dissociation constant Ka of the weak acid HA (dimensionless, in scientific notation)': ka})
    pka = load(pka_result, 'the pKa of the weak acid', d_type=float, ctx=ctx)
    narrate(ctx, f'pKa = {pka:.4f}')
    neut_result = do(ctx, 'Neutralize an added strong acid against the conjugate base and update the moles of weak acid and conjugate base', payloads={'the moles of the weak acid HA in the buffer': ha_0, 'the moles of the conjugate base A- in the buffer': a_0, 'the moles of the strong acid HCl added': acid_added})
    ha_new = load(neut_result, 'the moles of HA after neutralization', d_type=float, ctx=ctx)
    a_new = load(neut_result, 'the moles of A- after neutralization', d_type=float, ctx=ctx)
    narrate(ctx, f'After neutralization: HA = {ha_new:.3f} mol, A- = {a_new:.3f} mol')
    hh_result = do(ctx, 'Compute buffer pH from pKa and the moles of conjugate base and weak acid using the Henderson-Hasselbalch equation', payloads={'the pKa of the weak acid': pka, 'the moles of the conjugate base A- after neutralization': a_new, 'the moles of HA after neutralization': ha_new})
    ph = load(hh_result, 'the pH of the buffer after strong acid addition', d_type=float, ctx=ctx)
    narrate(ctx, f'Buffer pH after strong acid addition = {ph:.2f}')
    check_result = do(ctx, 'Sanity-check the buffer pH against pKa and the direction of the acid-to-base ratio shift', payloads={'the pH of the buffer after strong acid addition': ph, 'the pKa of the weak acid': pka, 'the moles of HA after neutralization': ha_new, 'the moles of A- after neutralization': a_new})
    check = load(check_result, 'the sanity check result', ctx=ctx)
    narrate(ctx, f'Sanity check: {check}')
    return {'the pH of the buffer after the strong acid is added': ph}
