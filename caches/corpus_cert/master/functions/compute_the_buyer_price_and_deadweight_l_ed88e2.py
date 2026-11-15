def semantic_function(ctx):
    '''Compute the buyer price and deadweight loss after a per-unit tax on sellers in a linear supply and demand market.'''
    import re
    narrate(ctx, 'Computing the new buyer price and deadweight loss after a per-unit tax on sellers in a linear market')
    demand_text = load(ctx, 'The linear demand curve expressing quantity demanded Q as a function of price P (intercept and slope)', d_type=str)
    supply_text = load(ctx, 'The linear supply curve expressing quantity supplied Q as a function of price P (intercept and slope)', d_type=str)
    tax = float(load(ctx, 'The per-unit tax amount charged on each unit sold, in dollars, levied on sellers', d_type=float))
    narrate(ctx, f'Loaded demand {demand_text}, supply {supply_text}, per-unit seller tax {tax}')

    def parse_line(text):
        s = str(text).replace(chr(8722), '-').replace(' ', '')
        rhs = s.split('=')[-1]
        m = re.search(r'([+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)?)P', rhs)
        if m is None:
            raise ValueError('no price term in ' + str(text))
        c = m.group(1)
        if c in ('', '+'):
            slope = 1.0
        elif c == '-':
            slope = -1.0
        else:
            slope = float(c)
        rest = rhs[:m.start()] + rhs[m.end():]
        const = float(rest) if rest not in ('', '+', '-') else 0.0
        return const, slope

    def intersect(a1, b1, a2, b2):
        if b1 == b2:
            raise ValueError('parallel curves have no unique equilibrium')
        p = (a2 - a1) / (b1 - b2)
        return p, a1 + b1 * p

    a_d, b_d = parse_line(demand_text)
    a_s, b_s = parse_line(supply_text)
    narrate(ctx, f'Demand Q = {a_d} + {b_d}P; supply Q = {a_s} + {b_s}P')
    p_star, q_star = intersect(a_d, b_d, a_s, b_s)
    narrate(ctx, f'Pre-tax equilibrium: price {p_star}, quantity {q_star}')
    a_t = a_s - b_s * tax
    narrate(ctx, f'Sellers receive P_s = P_b - {tax}, so post-tax supply is Q = {a_t} + {b_s}P_b')
    p_buyer, q_new = intersect(a_d, b_d, a_t, b_s)
    narrate(ctx, f'New price buyers pay {p_buyer}; new quantity traded {q_new}')
    p_seller = p_buyer - tax
    narrate(ctx, f'Seller net price {p_seller}')
    wedge = p_buyer - p_seller
    consistent = abs(wedge - tax) < 1e-9 and q_new >= 0 and abs((a_s + b_s * p_seller) - q_new) < 1e-9
    narrate(ctx, f'Consistency check (wedge equals tax, supply at seller price matches quantity, quantity nonnegative): {consistent}')
    lost = q_star - q_new
    dwl = 0.5 * lost * tax
    narrate(ctx, f'Quantity lost {lost}; deadweight loss = 1/2 x {lost} x {tax} = {dwl}')
    return {'new price buyers pay': p_buyer, 'seller net price': p_seller, 'new quantity traded': q_new, 'deadweight loss': dwl}