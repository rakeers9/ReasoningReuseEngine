import re

def semantic_function(ctx):
    '''Determine the book value of an asset at the end of a requested year under double-declining-balance depreciation with salvage value.'''
    narrate(ctx, 'Determining the book value at the requested year under double-declining-balance depreciation with salvage value')
    cost_text = load(ctx, 'the original cost of the asset (basis) with currency unit', d_type=str)
    narrate(ctx, f'Loaded original cost: {cost_text}')
    salvage_text = load(ctx, 'the salvage value of the asset (estimated residual value) with currency unit', d_type=str)
    narrate(ctx, f'Loaded salvage value: {salvage_text}')
    life_text = load(ctx, 'the useful life of the asset in years', d_type=str)
    narrate(ctx, f'Loaded useful life: {life_text}')
    method_text = load(ctx, 'the depreciation method to apply, which must be double-declining-balance depreciation', d_type=str)
    narrate(ctx, f'Loaded depreciation method: {method_text}')
    year_text = load(ctx, 'the year at which the book value is requested (the target year, as a year number)', d_type=str)
    narrate(ctx, f'Loaded target year: {year_text}')

    def first_number(text):
        match = re.search(r'\d+(?:\.\d+)?', str(text).replace(',', ''))
        if match is None:
            raise ValueError('no number found in ' + str(text))
        return float(match.group(0))

    cost = first_number(cost_text)
    salvage = first_number(salvage_text)
    life = int(first_number(life_text))
    target_year = int(first_number(year_text))
    method_lower = str(method_text).lower()
    if 'declin' not in method_lower and 'ddb' not in method_lower:
        raise ValueError('this function only handles double-declining-balance depreciation')
    if life < 2:
        raise ValueError('useful life must be at least 2 years')
    if target_year < 0:
        raise ValueError('target year must be non-negative')
    if salvage >= cost:
        raise ValueError('salvage value must be below the original cost')
    narrate(ctx, f'Parsed cost {cost:.2f}, salvage {salvage:.2f}, useful life {life} years, target year {target_year}')

    rate_result = do(ctx, 'Compute the double-declining-balance depreciation rate per year from the useful life of an asset', payloads={'the useful life of the asset in years': life})
    rate_text = load(rate_result, 'DDB depreciation rate per year', d_type=str, ctx=ctx)
    narrate(ctx, f'DDB rate for a {life}-year life: {rate_text}')

    iter_result = do(ctx, 'Iterate year-by-year double-declining-balance depreciation on the beginning book value up to the requested year, respecting the salvage floor.', payloads={
        'the original cost of the asset (basis) with currency unit': cost,
        'DDB depreciation rate per year': rate_text,
        'the salvage value of the asset (estimated residual value) with currency unit': salvage,
        'the year at which the book value is requested (the target year, as a year number)': target_year,
    })
    iterative = load(iter_result, 'iterative book value at the end of the requested year', d_type=float, ctx=ctx)
    narrate(ctx, f'Iterative book value after {target_year} years: {iterative:.2f}')

    verify_result = do(ctx, 'Verify the book value of a double-declining-balance depreciating asset at a requested year using the closed-form expression cost times one minus the rate raised to the year count, floored at salvage value.', payloads={
        'the original cost of the asset (basis) with currency unit': cost,
        'DDB depreciation rate per year': rate_text,
        'the salvage value of the asset (estimated residual value) with currency unit': salvage,
        'the year at which the book value is requested (the target year, as a year number)': target_year,
    })
    closed = load(verify_result, 'closed-form book value at the requested year', d_type=float, ctx=ctx)
    narrate(ctx, f'Closed-form book value after {target_year} years: {closed:.2f}')

    if abs(iterative - closed) > 0.01:
        narrate(ctx, 'Iterative and closed-form values disagree; reporting the iterative value')
    else:
        narrate(ctx, 'Iterative and closed-form values agree')

    book_value = round(iterative, 2)
    narrate(ctx, f'Book value at the end of year {target_year}: ${book_value:,.2f}')
    return {'book value at the end of the requested year': book_value}
