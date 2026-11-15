def semantic_function(ctx):
    '''Iterate year-by-year double-declining-balance depreciation on the beginning book value up to the requested year, respecting the salvage floor.'''
    narrate(ctx, 'Iterating year-by-year double-declining-balance depreciation up to the requested year, respecting the salvage floor')
    cost = load(ctx, 'the original cost of the asset (basis) with currency unit', d_type=float)
    rate_raw = load(ctx, 'DDB depreciation rate per year', d_type=str)
    salvage = load(ctx, 'the salvage value of the asset (estimated residual value) with currency unit', d_type=float)
    target_year = load(ctx, 'the year at which the book value is requested (the target year, as a year number)', d_type=int)
    rate_text = str(rate_raw).strip()
    narrate(ctx, f'Original cost {cost:.2f}, DDB rate {rate_text}, salvage value {salvage:.2f}, target year {target_year}')
    if target_year < 0:
        raise ValueError('target year must be non-negative')
    book_value = float(cost)
    for year in range(1, int(target_year) + 1):
        step = do(ctx, 'Compute one year of double-declining-balance depreciation from the beginning book value and rate, limited so the ending book value never falls below salvage value', payloads={
            'beginning-of-year book value': book_value,
            'DDB depreciation rate per year': rate_text,
            'the salvage value of the asset (estimated residual value) with currency unit': float(salvage),
        })
        charge = load(step, 'depreciation charge for the year', d_type=float, ctx=ctx)
        ending = load(step, 'ending book value for the year', d_type=float, ctx=ctx)
        narrate(ctx, f'Year {year}: depreciation {charge:.2f}, ending book value {ending:.2f}')
        book_value = ending
    narrate(ctx, f'Iterative book value at the end of year {target_year}: {book_value:.2f}')
    return {'iterative book value at the end of the requested year': book_value}