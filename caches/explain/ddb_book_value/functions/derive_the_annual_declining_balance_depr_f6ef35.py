def semantic_function(ctx):
    """Derive the annual declining-balance depreciation rate from the useful life and the method's acceleration factor."""
    import re
    from fractions import Fraction
    narrate(ctx, 'Deriving the annual declining-balance depreciation rate from the useful life and the method acceleration factor')
    life = load(ctx, 'the useful life of the asset expressed in years', d_type=int)
    method = load(ctx, 'the name of the declining-balance depreciation method to apply with its acceleration factor', d_type=str)
    narrate(ctx, f'Useful life is {life} years; depreciation method is {method}')
    if life <= 0:
        raise ValueError('useful life must be a positive number of years')
    text = str(method).lower()
    pct = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if pct:
        factor = Fraction(pct.group(1)) / 100
    elif 'double' in text or 'ddb' in text:
        factor = Fraction(2)
    else:
        raise ValueError('unrecognised declining-balance method: ' + text)
    rate = factor / life
    narrate(ctx, f'Annual rate = {factor} / {life} = {rate} ({float(rate) * 100:.2f}%)')
    return {'annual depreciation rate as a fraction': rate}
