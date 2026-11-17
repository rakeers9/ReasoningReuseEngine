from decimal import Decimal

def semantic_function(ctx):
    '''Compute the book value after a given number of elapsed years under double-declining-balance depreciation that never falls below salvage value.'''
    narrate(ctx, 'Computing the book value after elapsed years under double-declining-balance depreciation with a salvage floor')
    years_raw = load(ctx, 'the number of years elapsed since the asset was acquired, as a whole number of years', d_type=str)
    cost_raw = load(ctx, 'the original cost of the asset in dollars with currency unit', d_type=str)
    life_raw = load(ctx, 'the useful life of the asset in years', d_type=str)
    salvage_raw = load(ctx, 'the estimated salvage value of the asset in dollars with currency unit', d_type=str)
    narrate(ctx, 'Raw inputs: years elapsed ' + str(years_raw) + ', cost ' + str(cost_raw) + ', useful life ' + str(life_raw) + ', salvage value ' + str(salvage_raw))
    money_desc = 'Parse a monetary amount in dollars from raw text, handling currency sign, thousands separators and scale words such as k, thousand, million, bn and billion.'
    money_key = 'raw text of a monetary amount such as $379,000 or $0.42 million'
    whole_desc = 'Parse a whole number of years or a useful life from raw text, handling digits and number words.'
    whole_key = 'raw text of a whole number such as 8 years or 11-year life or eleven years'
    cost_res = do(ctx, money_desc, payloads={money_key: cost_raw})
    cost = str(load(cost_res, 'monetary amount in dollars as exact decimal', d_type=str, ctx=ctx)).strip()
    narrate(ctx, 'Parsed original cost: ' + cost + ' dollars')
    salvage_res = do(ctx, money_desc, payloads={money_key: salvage_raw})
    salvage = str(load(salvage_res, 'monetary amount in dollars as exact decimal', d_type=str, ctx=ctx)).strip()
    narrate(ctx, 'Parsed salvage value: ' + salvage + ' dollars')
    years_res = do(ctx, whole_desc, payloads={whole_key: years_raw})
    years = str(load(years_res, 'whole number as integer', d_type=str, ctx=ctx)).strip()
    narrate(ctx, 'Parsed years elapsed: ' + years)
    life_res = do(ctx, whole_desc, payloads={whole_key: life_raw})
    life = str(load(life_res, 'whole number as integer', d_type=str, ctx=ctx)).strip()
    narrate(ctx, 'Parsed useful life: ' + life + ' years')
    if not years.isdigit() or not life.isdigit():
        raise ValueError('Years elapsed and useful life must be whole numbers')
    years_i = int(years)
    life_i = int(life)
    cost_d = Decimal(cost)
    salvage_d = Decimal(salvage)
    if life_i <= 0:
        raise ValueError('Useful life must be positive')
    if salvage_d < 0 or cost_d < salvage_d:
        raise ValueError('Salvage value must be non-negative and not exceed cost')
    iter_res = do(ctx, 'Iterate double-declining-balance yearly depreciation with a salvage-value floor from the original cost for the elapsed years and return the unrounded book value.', payloads={'original cost in dollars': str(cost_d), 'salvage value in dollars': str(salvage_d), 'useful life in years': str(life_i), 'number of years elapsed': str(years_i)})
    bv = load(iter_res, 'book value after the elapsed years in dollars unrounded', d_type=str, ctx=ctx)
    bv_d = Decimal(str(bv))
    narrate(ctx, 'Book value after ' + str(years_i) + ' years, unrounded: ' + str(bv_d) + ' dollars; rounding is applied only once at the end')
    return {'book value after the elapsed years in dollars': bv_d}
