def semantic_function(ctx):
    """Solve two linear quantity-price equations Q = a + bP for their intersection price and quantity."""
    narrate(ctx, "Solving two linear equations of the form Q = a + bP for their intersection point")
    a1 = load(ctx, "intercept a of the first linear quantity equation Q = a + bP", d_type=float)
    b1 = load(ctx, "slope b of the first linear quantity equation Q = a + bP", d_type=float)
    a2 = load(ctx, "intercept a of the second linear quantity equation Q = a + bP", d_type=float)
    b2 = load(ctx, "slope b of the second linear quantity equation Q = a + bP", d_type=float)
    narrate(ctx, f"First line: Q = {a1} + {b1}P; second line: Q = {a2} + {b2}P")
    denom = b1 - b2
    if denom == 0:
        narrate(ctx, "The two lines have equal slopes, so there is no unique intersection")
        return {"intersection price": None, "intersection quantity": None}
    price = (a2 - a1) / denom
    quantity = a1 + b1 * price
    narrate(ctx, f"Setting the two equations equal gives P = {price}")
    narrate(ctx, f"Substituting back gives Q = {quantity}")
    return {"intersection price": price, "intersection quantity": quantity}