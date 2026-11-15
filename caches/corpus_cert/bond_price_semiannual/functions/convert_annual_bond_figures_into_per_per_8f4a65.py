import re
def semantic_function(ctx):
    '''Convert annual bond figures into per-period figures: total number of periods, coupon per period, and periodic yield.'''
    narrate(ctx, 'Converting annual bond figures into per-period figures')
    years_text = load(ctx, 'the time to maturity of the bond in years, with unit', d_type=str)
    face_text = load(ctx, 'the face value (par value) of the bond with currency unit', d_type=str)
    coupon_text = load(ctx, 'the annual coupon rate of the bond as a percentage of face value', d_type=str)
    coupon_word = load(ctx, 'the frequency of coupon installments per year, as a payment schedule word', d_type=str)
    yield_text = load(ctx, 'the annual market yield (required return) on the bond as a percentage', d_type=str)
    yield_word = load(ctx, 'the compounding frequency of the market yield as a schedule word', d_type=str)
    def _number(text):
        cleaned = str(text).replace(',', '')
        match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
        if match is None:
            raise ValueError('No number found in ' + str(text))
        return float(match.group())
    def _rate(text):
        value = _number(text)
        if '%' in str(text) or value > 1:
            value = value / 100.0
        return value
    years = _number(years_text)
    face = _number(face_text)
    coupon_rate = _rate(coupon_text)
    annual_yield = _rate(yield_text)
    narrate(ctx, f'Loaded maturity {years} years, face value {face}, coupon rate {coupon_rate}, annual yield {annual_yield}')
    schedule_key = 'the schedule word describing the frequency, such as annual, semiannual, quarterly, or monthly'
    coupon_result = do(ctx,
        'Map a payment or compounding schedule word to its number of periods per year',
        payloads={schedule_key: coupon_word})
    coupon_periods = int(load(coupon_result, 'the number of periods per year as a positive integer', d_type=int, ctx=ctx))
    yield_result = do(ctx,
        'Map a payment or compounding schedule word to its number of periods per year',
        payloads={schedule_key: yield_word})
    yield_periods = int(load(yield_result, 'the number of periods per year as a positive integer', d_type=int, ctx=ctx))
    narrate(ctx, f'Coupon periods per year: {coupon_periods}; yield compounding periods per year: {yield_periods}')
    if coupon_periods <= 0 or yield_periods <= 0:
        raise ValueError('Payment frequencies must be positive integers')
    n_float = years * coupon_periods
    n = int(round(n_float)) if abs(n_float - round(n_float)) < 1e-9 else n_float
    narrate(ctx, f'Total number of coupon periods n = {years} x {coupon_periods} = {n}')
    C = face * coupon_rate / coupon_periods
    narrate(ctx, f'Coupon per period C = {face} x {coupon_rate} / {coupon_periods} = {C}')
    i = annual_yield / yield_periods
    narrate(ctx, f'Periodic yield i = {annual_yield} / {yield_periods} = {i}')
    return {
        'the total number of coupon periods n': n,
        'the coupon payment per period C in currency': C,
        'the periodic yield i as a decimal': i,
    }