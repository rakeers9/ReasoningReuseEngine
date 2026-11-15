def semantic_function(ctx):
    '''Compute the seller net price by subtracting the per-unit tax from the price buyers pay.'''
    narrate(ctx, 'Computing the seller net price as the buyer price minus the per-unit tax')
    buyer_price = load(ctx, 'the new price buyers pay per unit in dollars', d_type=float)
    tax = load(ctx, 'the per-unit tax on sellers in dollars', d_type=float)
    narrate(ctx, f'Buyer price P_b = {buyer_price}; per-unit tax t = {tax}')
    seller_price = buyer_price - tax
    narrate(ctx, f'Seller net price P_s = {buyer_price} - {tax} = {seller_price}')
    return {'seller net price': seller_price}