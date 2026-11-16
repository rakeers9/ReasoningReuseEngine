def semantic_function(ctx):
    '''Compute the book value after a given number of years under double-declining-balance depreciation with a salvage floor.'''
    narrate(ctx, 'Computing book value after elapsed years under double-declining-balance depreciation with a salvage floor')
    cost_text = load(ctx, 'the original cost of the asset in dollars, as text copied from the question including currency symbol and any scale word', d_type=str)
    life_text = load(ctx, 'the useful life of the asset in years, as text copied from the question', d_type=str)
    salvage_text = load(ctx, 'the salvage (residual) value of the asset at the end of its useful life in dollars, as text copied from the question', d_type=str)
    years_text = load(ctx, 'the number of years elapsed since acquisition at which the book value is required, as text copied from the question', d_type=str)
    narrate(ctx, 'Loaded texts: cost ' + str(cost_text) + ', useful life ' + str(life_text) + ', salvage ' + str(salvage_text) + ', elapsed ' + str(years_text))
    money_desc = 'Parse a monetary amount text with currency symbol and scale word into an exact Decimal dollar amount'
    money_key = 'monetary amount as text copied from the question with currency symbol and scale word'
    cost_res = do(ctx, money_desc, payloads={money_key: cost_text})
    cost = load(cost_res, 'amount in dollars as an exact Decimal', d_type=None, ctx=ctx)
    salvage_res = do(ctx, money_desc, payloads={money_key: salvage_text})
    salvage = load(salvage_res, 'amount in dollars as an exact Decimal', d_type=None, ctx=ctx)
    narrate(ctx, 'Parsed cost: ' + str(cost) + ' dollars; salvage: ' + str(salvage) + ' dollars')
    years_desc = 'Parse a count of years written as digits or English number words into an integer.'
    years_key = 'number of years as text copied from the question'
    life_res = do(ctx, years_desc, payloads={years_key: life_text})
    life = int(load(life_res, 'whole number of years as an integer', d_type=None, ctx=ctx))
    elapsed_res = do(ctx, years_desc, payloads={years_key: years_text})
    elapsed = int(load(elapsed_res, 'whole number of years as an integer', d_type=None, ctx=ctx))
    narrate(ctx, 'Parsed useful life: ' + str(life) + ' years; elapsed years: ' + str(elapsed))
    rate_res = do(ctx, 'Compute the double-declining-balance rate as two divided by the useful life in years.', payloads={'useful life of the asset in years': str(life)})
    rate = load(rate_res, 'double-declining-balance rate as an exact fraction', d_type=None, ctx=ctx)
    narrate(ctx, 'DDB rate = 2 / ' + str(life) + ' = ' + str(rate))
    sim_res = do(ctx, 'Simulate year-by-year double-declining-balance depreciation with salvage floor to find the book value after a number of years.', payloads={'asset cost in dollars': str(cost), 'salvage value in dollars': str(salvage), 'useful life in years': str(life), 'number of elapsed years': str(elapsed), 'double-declining-balance rate as an exact fraction': str(rate)})
    book = load(sim_res, 'book value after the elapsed years in dollars', d_type=None, ctx=ctx)
    narrate(ctx, 'Unrounded book value after ' + str(elapsed) + ' years: ' + str(book) + ' dollars')
    fmt_res = do(ctx, 'Format a dollar book value rounded half-up to cents for the final answer.', payloads={'book value after the elapsed years in dollars': str(book)})
    final = load(fmt_res, 'formatted book value in dollars to two decimal places', d_type=None, ctx=ctx)
    narrate(ctx, 'Final book value after ' + str(elapsed) + ' years: ' + str(final) + ' dollars')
    return {'final book value after the elapsed years, formatted in dollars to two decimal places': final}
