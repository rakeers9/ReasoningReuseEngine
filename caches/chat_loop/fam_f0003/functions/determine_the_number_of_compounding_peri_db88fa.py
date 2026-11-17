def semantic_function(ctx):
    """Determine the number of compounding periods per year from a frequency description."""
    import re
    raw = load(ctx, "the compounding and payment frequency description as raw text", d_type=str)
    narrate(ctx, f"Loaded frequency text: {raw!r}")
    text = str(raw).strip().lower().replace('-', ' ').replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)
    table = [
        ('semi annually', 2), ('semiannually', 2), ('semi annual', 2), ('semiannual', 2),
        ('twice a year', 2), ('twice per year', 2), ('half yearly', 2), ('every six months', 2),
        ('bi weekly', 26), ('biweekly', 26), ('fortnightly', 26), ('every two weeks', 26),
        ('semi monthly', 24), ('semimonthly', 24),
        ('bimonthly', 6), ('every two months', 6),
        ('quarterly', 4), ('every quarter', 4), ('every three months', 4),
        ('monthly', 12), ('every month', 12), ('each month', 12), ('per month', 12),
        ('weekly', 52), ('every week', 52),
        ('daily', 365), ('every day', 365),
        ('annually', 1), ('annual', 1), ('yearly', 1), ('once a year', 1), ('every year', 1),
    ]
    periods = None
    for key, val in table:
        if key in text:
            periods = val
            narrate(ctx, f"Matched frequency keyword {key!r}, which means {val} periods per year")
            break
    if periods is None:
        narrate(ctx, "The frequency description is not recognised")
        raise ValueError(f"Unknown compounding frequency: {raw!r}")
    narrate(ctx, f"Periods per year: {periods}")
    return {"periods per year": periods}