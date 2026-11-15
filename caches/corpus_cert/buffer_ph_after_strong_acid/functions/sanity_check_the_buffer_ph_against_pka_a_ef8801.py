import math
def semantic_function(ctx):
    '''Sanity-check the buffer pH against pKa and the direction of the acid-to-base ratio shift.'''
    narrate(ctx, 'Sanity-checking the buffer pH against pKa and the base-to-acid ratio')
    ph = load(ctx, 'the pH of the buffer after strong acid addition', d_type=float)
    pka = load(ctx, 'the pKa of the weak acid', d_type=float)
    ha = load(ctx, 'the moles of HA after neutralization', d_type=float)
    a_minus = load(ctx, 'the moles of A- after neutralization', d_type=float)
    narrate(ctx, f'Loaded pH = {ph:.4f}, pKa = {pka:.4f}, HA = {ha:.3f} mol, A- = {a_minus:.3f} mol')
    if ha <= 0 or a_minus <= 0:
        narrate(ctx, 'Both HA and A- must be present for the buffer check to apply; check fails')
        return {'the sanity check result': False, 'the sanity check explanation': 'buffer check not applicable: one species absent'}
    log_ratio = math.log10(a_minus / ha)
    in_range = 0.0 < ph < 14.0
    direction_ok = (ph - pka) * log_ratio >= 0
    passed = in_range and direction_ok
    narrate(ctx, f'log10(A-/HA) = {log_ratio:.4f}; pH - pKa = {ph - pka:.4f}')
    if passed:
        explanation = f'pH {ph:.2f} lies on the expected side of pKa {pka:.2f} for the A-/HA ratio and within 0-14'
    else:
        explanation = 'pH is on the wrong side of pKa for the A-/HA ratio or outside 0-14'
    narrate(ctx, explanation)
    return {'the sanity check result': passed, 'the sanity check explanation': explanation}
