def semantic_function(ctx):
    '''Compute the monthly payment and total interest of a fixed-rate amortizing mortgage from principal, nominal annual rate, compounding frequency and loan term by chaining the registered loan steps.'''
    narrate(ctx, 'Starting mortgage computation of monthly payment and total interest for a fixed-rate amortizing loan')
    raw_principal = load(ctx, 'the principal amount of the loan with currency unit', d_type=str)
    raw_rate = load(ctx, 'the nominal annual interest rate as a percentage', d_type=str)
    raw_freq = load(ctx, 'the compounding frequency of the interest per year', d_type=str)
    raw_term = load(ctx, 'the loan term in years', d_type=str)
    narrate(ctx, f'Loaded raw principal {raw_principal}, raw rate {raw_rate}, raw compounding frequency {raw_freq}, raw term {raw_term}')

    def parse_number(text):
        cleaned = ''.join(ch for ch in str(text) if ch.isdigit() or ch in '.-')
        if cleaned in ('', '.', '-'):
            return float(smart_convert(ctx, str(text), float))
        return float(cleaned)

    principal = parse_number(raw_principal)
    annual_rate_pct = parse_number(raw_rate)
    term_years = parse_number(raw_term)
    word_map = {'annually': 1, 'annual': 1, 'yearly': 1, 'semi-annually': 2, 'semiannually': 2, 'quarterly': 4, 'monthly': 12, 'biweekly': 26, 'weekly': 52, 'daily': 365}
    key = str(raw_freq).strip().lower()
    if key in word_map:
        periods_per_year = word_map[key]
    else:
        periods_per_year = int(round(parse_number(raw_freq)))
    narrate(ctx, f'Parsed principal P = {principal}, annual rate = {annual_rate_pct} percent, term = {term_years} years, periods per year = {periods_per_year}')

    rate_result = do(ctx,
        'Convert the nominal annual interest rate percentage into a periodic interest rate per compounding period.',
        payloads={
            'the nominal annual interest rate as a percentage': annual_rate_pct,
            'the compounding frequency of the interest per year': periods_per_year,
        })
    periodic_rate = load(rate_result, 'the periodic interest rate per compounding period', d_type=float, ctx=ctx)
    narrate(ctx, f'Periodic rate r = {periodic_rate}')

    periods_result = do(ctx,
        'Compute the total number of payment periods from the loan term in years and the compounding frequency per year.',
        payloads={
            'the loan term in years': term_years,
            'the compounding frequency of the interest per year': periods_per_year,
        })
    total_periods = int(round(load(periods_result, 'the total number of payment periods', d_type=float, ctx=ctx)))
    narrate(ctx, f'Total number of payment periods n = {total_periods}')

    growth_result = do(ctx,
        'Compute the compound growth factor (1+r)^n for a periodic rate r and a number of periods n.',
        payloads={
            'the periodic interest rate per compounding period': periodic_rate,
            'the total number of payment periods': total_periods,
        })
    growth = load(growth_result, 'the compound growth factor (1+r)^n', d_type=float, ctx=ctx)
    narrate(ctx, f'Growth factor (1+r)^n = {growth}')

    payment_result = do(ctx,
        'Compute the fixed periodic payment of an amortizing loan from the principal, the periodic interest rate and the compound growth factor.',
        payloads={
            'the principal amount of the loan with currency unit': principal,
            'the periodic interest rate per compounding period': periodic_rate,
            'the compound growth factor (1+r)^n': growth,
        })
    raw_payment = load(payment_result, 'the fixed monthly payment amount', d_type=float, ctx=ctx)
    narrate(ctx, f'Unrounded fixed monthly payment M = {raw_payment}')

    rounded_payment_result = do(ctx,
        'Round a monetary amount to the nearest cent for reporting.',
        payloads={'the monetary amount to be rounded': raw_payment})
    monthly_payment = load(rounded_payment_result, 'the rounded monetary amount in dollars and cents', d_type=float, ctx=ctx)
    narrate(ctx, f'Monthly payment reported to the cent = {monthly_payment}')

    interest_result = do(ctx,
        'Compute the total interest paid over the life of a loan from the periodic payment, the number of payments and the principal.',
        payloads={
            'the fixed monthly payment amount': monthly_payment,
            'the total number of payment periods': total_periods,
            'the principal amount of the loan with currency unit': principal,
        })
    raw_interest = load(interest_result, 'the total interest paid over the life of the loan', d_type=float, ctx=ctx)

    rounded_interest_result = do(ctx,
        'Round a monetary amount to the nearest cent for reporting.',
        payloads={'the monetary amount to be rounded': raw_interest})
    total_interest = load(rounded_interest_result, 'the rounded monetary amount in dollars and cents', d_type=float, ctx=ctx)
    narrate(ctx, f'Total interest over the life of the loan = {total_interest}')

    return {
        'the monthly payment amount': monthly_payment,
        'the total interest paid over the life of the loan': total_interest,
    }
