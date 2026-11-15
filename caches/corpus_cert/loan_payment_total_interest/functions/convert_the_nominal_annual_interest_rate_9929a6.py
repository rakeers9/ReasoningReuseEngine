def semantic_function(ctx):
    '''Convert the nominal annual interest rate percentage into a periodic interest rate per compounding period.'''
    narrate(ctx, 'Converting the nominal annual interest rate into a periodic rate per compounding period')
    raw_rate = load(ctx, 'the nominal annual interest rate as a percentage', d_type=str)
    annual_rate_pct = smart_convert(ctx, raw_rate, float)
    raw_freq = load(ctx, 'the compounding frequency of the interest per year', d_type=str)
    periods_per_year = smart_convert(ctx, raw_freq, int)
    narrate(ctx, f'Loaded annual rate {annual_rate_pct}% compounded {periods_per_year} times per year')
    periodic_rate = (annual_rate_pct / 100.0) / periods_per_year
    narrate(ctx, f'Periodic rate r = {annual_rate_pct}/100/{periods_per_year} = {periodic_rate}')
    return {'the periodic interest rate per compounding period': periodic_rate}