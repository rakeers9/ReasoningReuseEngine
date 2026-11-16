def semantic_function(ctx):
    '''Compute the book value at the end of a given year for a machine under double-declining-balance depreciation with a salvage floor.'''
    from decimal import Decimal
    narrate(ctx, 'Computing the end-of-year book value under double-declining-balance depreciation with a salvage floor')
    raw_cost = load(ctx, 'the purchase cost of the machine in dollars, as written with its scale word or suffix (e.g. million or k)', d_type=str)
    raw_salvage = load(ctx, 'the salvage value of the machine in dollars, the residual book value at the end of its useful life', d_type=str)
    raw_life = load(ctx, 'the useful life of the machine in years', d_type=str)
    raw_year = load(ctx, 'the target year at whose end the book value is requested, as a count of years since acquisition', d_type=str)
    narrate(ctx, 'Loaded raw texts: cost ' + repr(raw_cost) + ', salvage ' + repr(raw_salvage) + ', useful life ' + repr(raw_life) + ', target year ' + repr(raw_year))
    cost_result = do(ctx, 'Parse a money amount text with an optional scale word into an exact decimal number of dollars.', payloads={'raw money text with optional scale word such as million or k': raw_cost})
    cost = Decimal(str(load(cost_result, 'money amount in dollars as an exact decimal', d_type=None, ctx=ctx)))
    narrate(ctx, 'Cost in dollars: ' + str(cost))
    salvage_result = do(ctx, 'Parse a money amount text with an optional scale word into an exact decimal number of dollars.', payloads={'raw money text with optional scale word such as million or k': raw_salvage})
    salvage = Decimal(str(load(salvage_result, 'money amount in dollars as an exact decimal', d_type=None, ctx=ctx)))
    narrate(ctx, 'Salvage value in dollars: ' + str(salvage))
    life_result = do(ctx, 'Parse a text giving a count of years into a non-negative whole number of years.', payloads={'raw text giving a count of years': raw_life})
    life = int(str(load(life_result, 'count of years as an integer', d_type=None, ctx=ctx)))
    narrate(ctx, 'Useful life in years: ' + str(life))
    year_result = do(ctx, 'Parse a text giving a count of years into a non-negative whole number of years.', payloads={'raw text giving a count of years': raw_year})
    target = int(str(load(year_result, 'count of years as an integer', d_type=None, ctx=ctx)))
    narrate(ctx, 'Target year: ' + str(target))
    if cost <= salvage:
        raise ValueError('Cost must exceed salvage value')
    if life < 2:
        raise ValueError('Useful life must be at least 2 years for a DDB rate of 100 percent or less')
    if target < 0:
        raise ValueError('Target year must be non-negative')
    rate_result = do(ctx, 'Compute the double-declining-balance depreciation rate as two divided by the useful life in years.', payloads={'useful life in years as an integer': life})
    rate = load(rate_result, 'double-declining-balance rate as an exact fraction', d_type=None, ctx=ctx)
    narrate(ctx, 'DDB rate: ' + str(rate))
    sim_result = do(ctx, 'Simulate year-by-year double-declining-balance depreciation from cost up to a target year while keeping book value at or above salvage.', payloads={'the cost in dollars as an exact decimal': str(cost), 'the salvage value in dollars as an exact decimal': str(salvage), 'the double-declining-balance rate as an exact fraction': str(rate), 'the target year as an integer': target})
    book = Decimal(str(load(sim_result, 'book value at the end of the target year in dollars', d_type=None, ctx=ctx)))
    narrate(ctx, 'Book value at end of year ' + str(target) + ' at full precision: ' + str(book))
    round_result = do(ctx, 'Round a dollar amount to the nearest cent using half-up rounding.', payloads={'dollar amount at full precision': str(book)})
    rounded = load(round_result, 'dollar amount rounded to the cent', d_type=None, ctx=ctx)
    narrate(ctx, 'Reported book value rounded to the cent: ' + str(rounded) + ' USD')
    return {'book value at the end of the target year in dollars rounded to the cent': rounded}