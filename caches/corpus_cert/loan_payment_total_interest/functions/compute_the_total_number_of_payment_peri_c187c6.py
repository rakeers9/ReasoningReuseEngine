def semantic_function(ctx):
    '''Compute the total number of payment periods from the loan term in years and the compounding frequency per year.'''
    narrate(ctx, 'Computing the total number of payment periods from the loan term and compounding frequency')
    raw_term = load(ctx, 'the loan term in years', d_type=str)
    term_years = smart_convert(ctx, raw_term, float)
    raw_freq = load(ctx, 'the compounding frequency of the interest per year', d_type=str)
    periods_per_year = smart_convert(ctx, raw_freq, int)
    narrate(ctx, f'Loaded term {term_years} years with {periods_per_year} compounding periods per year')
    raw_total = term_years * periods_per_year
    total_periods = int(round(raw_total))
    if abs(raw_total - total_periods) > 1e-9:
        narrate(ctx, f'Warning: term times frequency = {raw_total} is not a whole number of payments')
    narrate(ctx, f'Total number of payment periods n = {term_years} x {periods_per_year} = {total_periods}')
    return {'the total number of payment periods': total_periods}