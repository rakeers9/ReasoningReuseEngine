import re

def semantic_function(ctx):
    '''Compute the market price of a fixed-coupon bond from its maturity, face value, coupon rate, coupon frequency, market yield and yield compounding frequency.'''
    narrate(ctx, 'Computing the market price of a fixed-coupon bond by discounting its coupons and face value at the periodic market yield')
    maturity_text = load(ctx, 'the time to maturity of the bond in years, with unit', d_type=str)
    narrate(ctx, 'Loaded maturity: ' + maturity_text)
    face_text = load(ctx, 'the face value (par value) of the bond with currency unit', d_type=str)
    narrate(ctx, 'Loaded face value: ' + face_text)
    coupon_text = load(ctx, 'the annual coupon rate of the bond as a percentage of face value', d_type=str)
    narrate(ctx, 'Loaded annual coupon rate: ' + coupon_text)
    coupon_word = load(ctx, 'the frequency of coupon installments per year, as a payment schedule word', d_type=str)
    narrate(ctx, 'Loaded coupon payment frequency: ' + coupon_word)
    yield_text = load(ctx, 'the annual market yield (required return) on the bond as a percentage', d_type=str)
    narrate(ctx, 'Loaded annual market yield: ' + yield_text)
    yield_word = load(ctx, 'the compounding frequency of the market yield as a schedule word', d_type=str)
    narrate(ctx, 'Loaded yield compounding frequency: ' + yield_word)
    face = float(re.search(r'\d+(?:\.\d+)?', face_text.replace(',', '')).group())
    narrate(ctx, 'Parsed face value number: ' + str(face))
    conv = do(ctx,
        'Convert annual bond figures into per-period figures: total number of periods, coupon per period, and periodic yield.',
        payloads={
            'the time to maturity of the bond in years, with unit': maturity_text,
            'the face value (par value) of the bond with currency unit': face_text,
            'the annual coupon rate of the bond as a percentage of face value': coupon_text,
            'the frequency of coupon installments per year, as a payment schedule word': coupon_word,
            'the annual market yield (required return) on the bond as a percentage': yield_text,
            'the compounding frequency of the market yield as a schedule word': yield_word,
        })
    n = load(conv, 'the total number of coupon periods n', d_type=float, ctx=ctx)
    C = load(conv, 'the coupon payment per period C in currency', d_type=float, ctx=ctx)
    i = load(conv, 'the periodic yield i as a decimal', d_type=float, ctx=ctx)
    narrate(ctx, 'Periods n = ' + str(n) + ', coupon per period C = ' + str(C) + ', periodic yield i = ' + str(i))
    pv_result = do(ctx,
        'Compute the present value of a level periodic coupon annuity plus the lump-sum face value repayment at maturity.',
        payloads={
            'the coupon payment per period C in currency': C,
            'the periodic yield i as a decimal': i,
            'the total number of coupon periods n': n,
            'the face value (par value) of the bond with currency unit': face,
        })
    price = load(pv_result, 'the total bond price as the sum of both present values', d_type=float, ctx=ctx)
    narrate(ctx, 'Present value of coupons plus face value gives bond price ' + str(round(price, 2)))
    cls_result = do(ctx,
        'Classify a bond as trading at a premium, at par, or at a discount by comparing its market yield with its coupon rate',
        payloads={
            'the annual coupon rate of the bond as a percentage of face value': coupon_text,
            'the annual market yield (required return) on the bond as a percentage': yield_text,
        })
    classification = load(cls_result, 'the price classification of the bond as premium, par, or discount', d_type=str, ctx=ctx)
    narrate(ctx, 'The bond trades at a ' + classification + ' because the market yield is compared with the coupon rate')
    return {
        'the market price of the bond in currency': round(price, 2),
        'the price classification of the bond as premium, par, or discount': classification,
    }
