def semantic_function(ctx):
    '''Find the new buyer price and quantity by equating the demand curve with the post-tax supply curve.'''
    narrate(ctx, 'Finding the new equilibrium by equating demand with the post-tax supply curve in buyer price')
    a_d = load(ctx, 'intercept a of the demand curve Q = a + bP', d_type=float)
    narrate(ctx, f'Demand intercept a = {a_d}')
    b_d = load(ctx, 'slope b of the demand curve Q = a + bP', d_type=float)
    narrate(ctx, f'Demand slope b = {b_d}')
    a_t = load(ctx, 'intercept a of the post-tax supply curve Q = a + bP in buyer price', d_type=float)
    narrate(ctx, f'Post-tax supply intercept a in buyer price = {a_t}')
    b_t = load(ctx, 'slope b of the post-tax supply curve Q = a + bP in buyer price', d_type=float)
    narrate(ctx, f'Post-tax supply slope b in buyer price = {b_t}')
    narrate(ctx, f'Demand: Q = {a_d} + {b_d}P; post-tax supply in buyer price: Q = {a_t} + {b_t}P')
    solved = do(ctx,
        'Solve two linear quantity-price equations Q = a + bP for their intersection price and quantity',
        payloads={
            'intercept a of the first linear quantity equation Q = a + bP': a_d,
            'slope b of the first linear quantity equation Q = a + bP': b_d,
            'intercept a of the second linear quantity equation Q = a + bP': a_t,
            'slope b of the second linear quantity equation Q = a + bP': b_t,
        })
    price = load(solved, 'intersection price', d_type=float, ctx=ctx)
    quantity = load(solved, 'intersection quantity', d_type=float, ctx=ctx)
    narrate(ctx, f'New price buyers pay P_b = {price}')
    narrate(ctx, f'New quantity traded Q_1 = {quantity}')
    return {'new price buyers pay': price, 'new quantity traded': quantity}
