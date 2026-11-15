import math
def semantic_function(ctx):
    '''Compute buffer pH from pKa and the moles of conjugate base and weak acid using the Henderson-Hasselbalch equation.'''
    narrate(ctx, 'Applying the Henderson-Hasselbalch equation pH = pKa + log10(A-/HA)')
    pka = load(ctx, 'the pKa of the weak acid', d_type=float)
    a_minus = load(ctx, 'the moles of the conjugate base A- after neutralization', d_type=float)
    ha = load(ctx, 'the moles of HA after neutralization', d_type=float)
    narrate(ctx, f'Loaded pKa = {pka:.4f}, A- = {a_minus:.3f} mol, HA = {ha:.3f} mol')
    if a_minus <= 0 or ha <= 0:
        raise ValueError('Both A- and HA must be positive for the Henderson-Hasselbalch equation')
    ratio = a_minus / ha
    narrate(ctx, f'Base-to-acid mole ratio A-/HA = {ratio:.4f} (common volume cancels)')
    log_ratio = math.log10(ratio)
    narrate(ctx, f'log10(A-/HA) = {log_ratio:.4f}')
    ph = pka + log_ratio
    narrate(ctx, f'pH = {pka:.4f} + ({log_ratio:.4f}) = {ph:.4f}')
    return {'the pH of the buffer after strong acid addition': ph}
