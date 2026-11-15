def semantic_function(ctx):
    '''Apply the accept-if-positive decision rule to a net present value.'''
    narrate(ctx, 'Applying the accept-if-positive decision rule to the net present value')
    npv = load(ctx, 'the net present value of the project', d_type=float)
    narrate(ctx, f'Net present value loaded as {npv:.2f}')
    if npv > 0:
        decision = 'accept'
        narrate(ctx, 'NPV is positive, so the project adds value at the required rate; accept the project')
    else:
        decision = 'reject'
        narrate(ctx, 'NPV is not positive, so the project does not earn the required rate; reject the project')
    narrate(ctx, f'Decision: {decision}')
    return {'the accept or reject decision': decision}