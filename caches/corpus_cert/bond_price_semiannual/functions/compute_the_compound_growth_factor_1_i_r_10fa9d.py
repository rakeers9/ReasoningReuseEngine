import math

def semantic_function(ctx):
    '''Compute the compound growth factor (1+i) raised to n and the present value discount factor v^n for a periodic yield over n periods, with a logarithm cross-check.'''
    narrate(ctx, 'Computing the compound growth factor and the present value discount factor for the periodic yield over n periods')
    i = float(load(ctx, 'the periodic yield i as a decimal', d_type=float))
    n = float(load(ctx, 'the total number of coupon periods n', d_type=float))
    narrate(ctx, f'Loaded periodic yield i = {i} and total number of periods n = {n}')
    base = 1.0 + i
    if base <= 0:
        raise ValueError('The periodic growth factor 1 + i must be positive')
    growth = base ** n
    growth_log = math.exp(n * math.log(base))
    narrate(ctx, f'Compound growth factor (1+i)^n = (1 + {i})^{n} = {growth:.6f}')
    narrate(ctx, f'Logarithm cross-check: exp({n} x ln(1 + {i})) = {growth_log:.6f}')
    if abs(growth - growth_log) > 1e-9 * max(1.0, abs(growth)):
        narrate(ctx, 'Warning: power and logarithm results disagree; using the logarithm value')
        growth = growth_log
    discount = 1.0 / growth
    narrate(ctx, f'Present value discount factor v^n = 1 / {growth:.6f} = {discount:.6f}')
    return {'the compound growth factor (1+i) raised to n': growth, 'the present value discount factor v^n': discount}
