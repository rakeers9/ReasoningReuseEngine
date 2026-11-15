import re
def semantic_function(ctx):
    """Classify a bond as trading at a premium, at par, or at a discount by comparing its market yield with its coupon rate."""
    narrate(ctx, 'Classifying the bond as premium, par, or discount by comparing the market yield with the coupon rate')
    coupon_text = load(ctx, 'the annual coupon rate of the bond as a percentage of face value', d_type=str)
    yield_text = load(ctx, 'the annual market yield (required return) on the bond as a percentage', d_type=str)
    def _rate(text):
        match = re.search(r'-?\d+(?:\.\d+)?', str(text).replace(',', ''))
        if match is None:
            raise ValueError('No number found in ' + str(text))
        value = float(match.group())
        if '%' in str(text) or value > 1:
            value = value / 100.0
        return value
    coupon_rate = _rate(coupon_text)
    market_yield = _rate(yield_text)
    narrate(ctx, f'Loaded coupon rate {coupon_rate} and market yield {market_yield}')
    gap = round(market_yield - coupon_rate, 12)
    if gap > 0:
        classification = 'discount'
        reason = 'the market yield exceeds the coupon rate, so the bond trades below face value'
    elif gap < 0:
        classification = 'premium'
        reason = 'the coupon rate exceeds the market yield, so the bond trades above face value'
    else:
        classification = 'par'
        reason = 'the coupon rate equals the market yield, so the bond trades at face value'
    narrate(ctx, f'The bond trades at a {classification}: {reason}')
    return {'the price classification of the bond as premium, par, or discount': classification}
