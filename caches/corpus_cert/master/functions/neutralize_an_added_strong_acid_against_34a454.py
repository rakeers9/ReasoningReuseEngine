def semantic_function(ctx):
    '''Neutralize an added strong acid against the conjugate base and update the moles of weak acid and conjugate base.'''
    narrate(ctx, 'Neutralizing the added strong acid against the conjugate base A- to update the buffer moles')
    ha = load(ctx, 'the moles of the weak acid HA in the buffer', d_type=float)
    a_minus = load(ctx, 'the moles of the conjugate base A- in the buffer', d_type=float)
    acid = load(ctx, 'the moles of the strong acid HCl added', d_type=float)
    narrate(ctx, f'Loaded HA = {ha} mol, A- = {a_minus} mol, strong acid added = {acid} mol')
    reacted = min(acid, a_minus)
    excess = acid - reacted
    ha_new = ha + reacted
    a_new = a_minus - reacted
    narrate(ctx, f'Strong acid reacts completely with A-: {reacted:.3f} mol of A- converted to HA')
    if excess > 0:
        narrate(ctx, f'Excess strong acid of {excess:.3f} mol exceeds the conjugate base; buffer no longer applies')
    narrate(ctx, f'HA after neutralization = {ha_new:.3f} mol; A- after neutralization = {a_new:.3f} mol')
    return {'the moles of HA after neutralization': ha_new, 'the moles of A- after neutralization': a_new}
