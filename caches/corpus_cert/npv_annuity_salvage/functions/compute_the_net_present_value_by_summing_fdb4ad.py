def semantic_function(ctx):
    '''Compute the net present value by summing the present values of future inflows and subtracting the initial outlay.'''
    narrate(ctx, 'Computing NPV by summing the present values of future inflows and subtracting the initial outlay')
    outlay = load(ctx, 'the initial outlay as a positive amount', d_type=float)
    annuity_pv = load(ctx, 'the present value of the annual inflows', d_type=float)
    salvage_pv = load(ctx, 'the present value of the salvage value', d_type=float)
    narrate(ctx, f'Initial outlay at t=0: {outlay:.2f}')
    narrate(ctx, f'PV of annual inflows: {annuity_pv:.2f}')
    narrate(ctx, f'PV of salvage value: {salvage_pv:.2f}')
    total_pv = annuity_pv + salvage_pv
    narrate(ctx, f'Total PV of future inflows = {annuity_pv:.2f} + {salvage_pv:.2f} = {total_pv:.2f}')
    npv = total_pv - outlay
    narrate(ctx, f'NPV = {total_pv:.2f} - {outlay:.2f} = {npv:.2f}')
    return {'the net present value of the project': npv}
