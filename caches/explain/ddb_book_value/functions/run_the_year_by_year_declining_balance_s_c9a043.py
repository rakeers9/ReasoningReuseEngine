def semantic_function(ctx):
    '''Run the year-by-year declining-balance schedule from the acquisition cost up to the target year and return the book value at the end of that year.'''
    narrate(ctx, 'Running the year-by-year declining-balance schedule up to the target year')
    cost = load(ctx, 'the original acquisition cost of the asset with currency unit', d_type=float)
    rate = load(ctx, 'annual depreciation rate as a fraction', d_type=str)
    salvage = load(ctx, 'the salvage value (residual value) of the asset at the end of its useful life with currency unit', d_type=float)
    target_year = load(ctx, 'the target year number for which the end-of-year book value is requested', d_type=int)
    rate_text = str(rate).strip()
    narrate(ctx, f'Starting book value {float(cost):.2f}; annual rate {rate_text}; salvage value {float(salvage):.2f}; target year {target_year}')
    book_value = float(cost)
    for year in range(1, int(target_year) + 1):
        step = do(ctx, 'Compute one year of declining-balance depreciation from the beginning book value and annual rate, capped so the ending book value does not fall below the salvage value floor.', payloads={'beginning book value for the year': book_value, 'annual depreciation rate as a fraction': rate_text, 'the salvage value (residual value) of the asset at the end of its useful life with currency unit': float(salvage)})
        dep = load(step, 'depreciation expense for the year', d_type=float, ctx=ctx)
        book_value = load(step, 'ending book value for the year', d_type=float, ctx=ctx)
        narrate(ctx, f'Year {year}: depreciation {float(dep):.2f}, ending book value {float(book_value):.2f}')
    narrate(ctx, f'Book value at end of year {target_year}: {book_value:.2f}')
    return {'book value at end of target year from schedule': book_value}