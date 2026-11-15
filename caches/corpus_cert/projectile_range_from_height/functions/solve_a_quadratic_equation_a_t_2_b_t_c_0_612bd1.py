import math

def semantic_function(ctx):
    """Solve a quadratic equation a t^2 + b t + c = 0 and return its positive real root."""
    narrate(ctx, "Solving a quadratic equation a t^2 + b t + c = 0 for its positive real root")
    a = load(ctx, "the coefficient a of the squared term in the quadratic", d_type=float)
    b = load(ctx, "the coefficient b of the linear term in the quadratic", d_type=float)
    c = load(ctx, "the coefficient c of the constant term in the quadratic", d_type=float)
    narrate(ctx, f"Coefficients: a = {a}, b = {b}, c = {c}")
    candidates = []
    if a == 0:
        narrate(ctx, "Leading coefficient is zero, so the equation is linear")
        if b != 0:
            candidates = [-c / b]
    else:
        disc = b * b - 4 * a * c
        narrate(ctx, f"Discriminant b^2 - 4ac = {disc}")
        if disc >= 0:
            sq = math.sqrt(disc)
            candidates = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
    positive = [r for r in candidates if r > 0]
    if not positive:
        narrate(ctx, "No positive real root exists")
        return {"a positive real root exists": False, "the positive real root of the quadratic": None}
    root = max(positive)
    narrate(ctx, f"Positive real root: t = {root:.3f}")
    return {"a positive real root exists": True, "the positive real root of the quadratic": root}
