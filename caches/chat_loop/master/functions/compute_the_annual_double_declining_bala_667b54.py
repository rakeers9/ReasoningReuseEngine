from decimal import Decimal, localcontext
import re

def semantic_function(ctx):
    '''Compute the annual double-declining-balance rate as two divided by the useful life in years.'''
    narrate(ctx, 'Computing the annual double-declining-balance rate as 2 divided by the useful life in years')
    raw = load(ctx, 'the useful life of the asset in years', d_type=str)
    narrate(ctx, 'Useful life as given: ' + str(raw))
    text = str(raw).strip().lower().replace(',', '')
    m = re.search(r'\d+(?:\.\d+)?', text)
    if m is None:
        narrate(ctx, 'Cannot parse a useful life from: ' + str(raw))
        raise ValueError('Cannot parse useful life from: ' + str(raw))
    life = Decimal(m.group(0))
    if life <= 0:
        raise ValueError('Useful life must be positive: ' + str(raw))
    with localcontext() as c:
        c.prec = 50
        rate = Decimal(2) / life
    narrate(ctx, 'Useful life in years: ' + str(life))
    narrate(ctx, 'Annual double-declining-balance rate is 2 divided by ' + str(life) + ' = ' + str(rate) + ' per year')
    return {'annual double-declining-balance rate as exact fraction': rate}
