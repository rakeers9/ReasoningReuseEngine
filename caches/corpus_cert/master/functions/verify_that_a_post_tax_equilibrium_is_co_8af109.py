def semantic_function(ctx):
    """Verify that a post-tax equilibrium is consistent: the buyer-seller price wedge equals the per-unit tax and the traded quantity is nonnegative."""
    narrate(ctx, "Verifying the new equilibrium: checking that the buyer-seller price wedge equals the per-unit tax")
    buyer_price = load(ctx, "the new price buyers pay", d_type=float)
    narrate(ctx, f"New price buyers pay = {buyer_price}")
    quantity = load(ctx, "the new quantity traded", d_type=float)
    narrate(ctx, f"New quantity traded = {quantity}")
    seller_price = load(ctx, "the seller net price received per unit", d_type=float)
    narrate(ctx, f"Seller net price = {seller_price}")
    tax = load(ctx, "the per-unit tax amount charged on each unit sold, in dollars, levied on sellers", d_type=float)
    narrate(ctx, f"Per-unit tax = {tax}")
    wedge = buyer_price - seller_price
    wedge_ok = abs(wedge - tax) < 1e-9
    quantity_ok = quantity >= 0
    consistent = wedge_ok and quantity_ok
    narrate(ctx, f"Wedge = {wedge}, matches tax: {wedge_ok}; quantity nonnegative: {quantity_ok}")
    narrate(ctx, f"Equilibrium consistent: {consistent}")
    return {"equilibrium consistency check result": consistent}
