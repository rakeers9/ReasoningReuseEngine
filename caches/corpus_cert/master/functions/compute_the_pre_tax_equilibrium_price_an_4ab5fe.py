def semantic_function(ctx):
    '''Compute the pre-tax equilibrium price and quantity from the demand and supply curves.'''
    narrate(ctx, 'Computing the pre-tax equilibrium from the demand and supply curves')
    a_d = load(ctx, 'intercept a of the demand curve Q = a + bP', d_type=float)
    b_d = load(ctx, 'slope b of the demand curve Q = a + bP', d_type=float)
    a_s = load(ctx, 'intercept a of the supply curve Q = a + bP', d_type=float)
    b_s = load(ctx, 'slope b of the supply curve Q = a + bP', d_type=float)
    narrate(ctx, f'Demand: Q = {a_d} + {b_d}P; supply: Q = {a_s} + {b_s}P')
    solved = do(ctx,
        'Solve two linear quantity-price equations Q = a + bP for their intersection price and quantity',
        payloads={
            'intercept a of the first linear quantity equation Q = a + bP': a_d,
            'slope b of the first linear quantity equation Q = a + bP': b_d,
            'intercept a of the second linear quantity equation Q = a + bP': a_s,
            'slope b of the second linear quantity equation Q = a + bP': b_s,
        })
    price = load(solved, 'intersection price', d_type=float, ctx=ctx)
    quantity = load(solved, 'intersection quantity', d_type=float, ctx=ctx)
    narrate(ctx, f'Pre-tax equilibrium price P* = {price}')
    narrate(ctx, f'Pre-tax equilibrium quantity Q* = {quantity}')
    return {'pre-tax equilibrium price': price, 'pre-tax equilibrium quantity': quantity}