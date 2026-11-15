def semantic_function(ctx):
    '''Compute the present value of a level periodic coupon annuity plus the lump-sum face value repayment at maturity.'''
    narrate(ctx, 'Computing the present value of the coupon annuity plus the face value repayment')
    C = float(load(ctx, 'the coupon payment per period C in currency', d_type=float))
    i = float(load(ctx, 'the periodic yield i as a decimal', d_type=float))
    n = float(load(ctx, 'the total number of coupon periods n', d_type=float))
    F = float(load(ctx, 'the face value (par value) of the bond with currency unit', d_type=float))
    narrate(ctx, f'Loaded coupon per period C = {C}, periodic yield i = {i}, periods n = {n}, face value F = {F}')
    discount_result = do(ctx,
        'Compute the compound growth factor (1+i) raised to n and the present value discount factor v^n for a periodic yield over n periods, with a logarithm cross-check.',
        payloads={
            'the periodic yield i as a decimal': i,
            'the total number of coupon periods n': n,
        })
    v_n = float(load(discount_result, 'the present value discount factor v^n', d_type=float, ctx=ctx))
    narrate(ctx, f'Present value discount factor v^n = {v_n:.6f}')
    if i == 0:
        annuity_factor = n
    else:
        annuity_factor = (1.0 - v_n) / i
    narrate(ctx, f'Annuity factor (1 - v^n) / i = {annuity_factor:.6f}')
    pv_coupons = C * annuity_factor
    narrate(ctx, f'PV of coupons = {C} x {annuity_factor:.6f} = {pv_coupons:.2f}')
    pv_face = F * v_n
    narrate(ctx, f'PV of face value = {F} x {v_n:.6f} = {pv_face:.2f}')
    price = pv_coupons + pv_face
    narrate(ctx, f'Bond price = {pv_coupons:.2f} + {pv_face:.2f} = {price:.2f}')
    return {
        'the present value of the coupon payments': pv_coupons,
        'the present value of the face value': pv_face,
        'the total bond price as the sum of both present values': price,
    }
