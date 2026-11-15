def semantic_function(ctx):
    '''Compute the deadweight loss triangle area from the quantity lost after a per-unit tax and the per-unit tax wedge.'''
    narrate(ctx, 'Computing the deadweight loss triangle from the quantity lost and the per-unit tax wedge')
    q_pre = load(ctx, 'the pre-tax equilibrium quantity', d_type=float)
    q_post = load(ctx, 'the new quantity traded after the tax', d_type=float)
    tax = load(ctx, 'the per-unit tax amount charged on each unit sold, in dollars, levied on sellers', d_type=float)
    narrate(ctx, f'Pre-tax quantity = {q_pre}; post-tax quantity = {q_post}; per-unit tax = {tax}')
    lost = q_pre - q_post
    narrate(ctx, f'Quantity lost = {q_pre} - {q_post} = {lost}')
    dwl = 0.5 * lost * tax
    narrate(ctx, f'Deadweight loss = 1/2 x {lost} x {tax} = {dwl}')
    return {'deadweight loss in dollars': dwl}
