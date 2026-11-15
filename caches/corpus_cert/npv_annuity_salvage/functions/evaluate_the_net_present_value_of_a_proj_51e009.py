def semantic_function(ctx):
    '''Evaluate the net present value of a project with an upfront outlay, level annual inflows, and a salvage value, then decide whether to accept it.'''
    narrate(ctx, 'Evaluating the NPV of a project with an upfront outlay, level annual inflows, and a salvage value')
    outlay_raw = load(ctx, 'the initial investment outlay at time zero with currency unit', d_type=str)
    inflow_raw = load(ctx, 'the annual cash inflow received at the end of each year with currency unit', d_type=str)
    life_raw = load(ctx, 'the project life in whole years over which annual inflows and salvage occur', d_type=str)
    salvage_raw = load(ctx, 'the salvage value received at the end of the project life with currency unit', d_type=str)
    rate_raw = load(ctx, 'the discount rate as a percentage or decimal for the project cash flows', d_type=str)
    narrate(ctx, f'Loaded raw inputs: outlay {outlay_raw}, annual inflow {inflow_raw}, life {life_raw}, salvage {salvage_raw}, discount rate {rate_raw}')
    norm = do(ctx, 'Normalize raw project cash flow inputs into a positive outlay, numeric inflow, numeric salvage, decimal discount rate, and integer project life.', payloads={
        'the initial investment outlay at time zero with currency unit': outlay_raw,
        'the annual cash inflow received at the end of each year with currency unit': inflow_raw,
        'the project life in whole years over which annual inflows and salvage occur': life_raw,
        'the salvage value received at the end of the project life with currency unit': salvage_raw,
        'the discount rate as a percentage or decimal for the project cash flows': rate_raw,
    })
    outlay = load(norm, 'the initial outlay as a positive amount', d_type=float, ctx=ctx)
    inflow = load(norm, 'the annual cash inflow as a number', d_type=float, ctx=ctx)
    salvage = load(norm, 'the salvage value as a number', d_type=float, ctx=ctx)
    rate = load(norm, 'the discount rate as a decimal', d_type=float, ctx=ctx)
    life = load(norm, 'the project life as an integer number of years', d_type=int, ctx=ctx)
    narrate(ctx, f'Normalized: outlay {outlay}, annual inflow {inflow}, salvage {salvage}, rate {rate}, life {life} years')
    annuity_res = do(ctx, 'Compute the present value of a level annuity of equal end-of-period cash payments over a given number of periods.', payloads={
        'the annual cash inflow as a number': inflow,
        'the discount rate as a decimal': rate,
        'the project life as an integer number of years': life,
    })
    annuity_pv = load(annuity_res, 'the present value of the annual inflows', d_type=float, ctx=ctx)
    narrate(ctx, f'PV of the annual inflows: {annuity_pv:.2f}')
    salvage_pv = None
    try:
        salvage_res = do(ctx, 'Compute the present value of a single future cash amount received after a given number of periods.', payloads={
            'the future cash amount as a number': salvage,
            'the salvage value as a number': salvage,
            'the discount rate as a decimal': rate,
            'the project life as an integer number of years': life,
        })
        salvage_pv = load(salvage_res, 'the present value of the salvage value', d_type=float, ctx=ctx)
    except Exception as exc:
        narrate(ctx, f'Delegated salvage present value step failed ({exc}); computing it directly')
        salvage_pv = salvage / (1 + rate) ** life
    narrate(ctx, f'PV of the salvage value: {salvage_pv:.2f}')
    npv_res = do(ctx, 'Compute the net present value by summing the present values of future inflows and subtracting the initial outlay.', payloads={
        'the initial outlay as a positive amount': outlay,
        'the present value of the annual inflows': annuity_pv,
        'the present value of the salvage value': salvage_pv,
    })
    npv = load(npv_res, 'the net present value of the project', d_type=float, ctx=ctx)
    narrate(ctx, f'Net present value of the project: {npv:.2f}')
    decision_res = do(ctx, 'Apply the accept-if-positive decision rule to a net present value.', payloads={
        'the net present value of the project': npv,
    })
    decision = load(decision_res, 'the accept or reject decision', d_type=str, ctx=ctx)
    narrate(ctx, f'Decision: {decision}')
    return {
        'the net present value of the project': npv,
        'the accept or reject decision': decision,
    }
