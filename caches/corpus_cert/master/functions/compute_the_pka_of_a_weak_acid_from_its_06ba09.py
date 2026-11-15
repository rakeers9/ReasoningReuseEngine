import math
def semantic_function(ctx):
    """Compute the pKa of a weak acid from its acid dissociation constant Ka."""
    narrate(ctx, "Computing pKa from the acid dissociation constant Ka")
    ka = load(ctx, "the acid dissociation constant Ka of the weak acid HA (dimensionless, in scientific notation)", d_type=float)
    narrate(ctx, f"Loaded Ka = {ka}")
    if ka <= 0:
        raise ValueError("Ka must be positive to compute pKa")
    pka = -math.log10(ka)
    narrate(ctx, f"pKa = -log10(Ka) = {pka:.4f}")
    return {"the pKa of the weak acid": pka}