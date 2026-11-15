def semantic_function(ctx):
    """Normalize raw project cash flow inputs into a positive outlay, numeric inflow, numeric salvage, decimal discount rate, and integer project life."""
    narrate(ctx, 'Normalizing the project cash flow inputs into consistent units')

    def to_number(raw):
        if isinstance(raw, (int, float)):
            return float(raw)
        cleaned = ''.join(ch for ch in str(raw) if ch.isdigit() or ch in '.-')
        return float(cleaned)

    outlay_raw = load(ctx, 'the initial investment outlay at time zero with currency unit', d_type=str)
    inflow_raw = load(ctx, 'the annual cash inflow received at the end of each year with currency unit', d_type=str)
    life_raw = load(ctx, 'the project life in whole years over which annual inflows and salvage occur', d_type=str)
    salvage_raw = load(ctx, 'the salvage value received at the end of the project life with currency unit', d_type=str)
    rate_raw = load(ctx, 'the discount rate as a percentage or decimal for the project cash flows', d_type=str)

    outlay = abs(to_number(outlay_raw))
    narrate(ctx, f'Initial outlay at time zero normalized to {outlay}')
    inflow = to_number(inflow_raw)
    narrate(ctx, f'Annual inflow normalized to {inflow}')
    salvage = to_number(salvage_raw)
    narrate(ctx, f'Salvage value normalized to {salvage}')
    life = int(round(to_number(life_raw)))
    narrate(ctx, f'Project life normalized to {life} years')
    rate_value = to_number(rate_raw)
    rate_text = str(rate_raw).lower()
    if '%' in rate_text or 'percent' in rate_text or rate_value > 1:
        rate = rate_value / 100.0
    else:
        rate = rate_value
    narrate(ctx, f'Discount rate normalized to decimal {rate}')

    return {
        'the initial outlay as a positive amount': outlay,
        'the annual cash inflow as a number': inflow,
        'the salvage value as a number': salvage,
        'the discount rate as a decimal': rate,
        'the project life as an integer number of years': life,
    }
