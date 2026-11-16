import re
from decimal import Decimal, ROUND_HALF_UP

def semantic_function(ctx):
    """Format a dollar book value rounded half-up to cents for the final answer."""
    narrate(ctx, "Formatting the book value in dollars, rounded half-up to cents")
    raw = load(ctx, "book value after the elapsed years in dollars", d_type=None)
    if isinstance(raw, Decimal):
        value = raw
    elif raw is None or isinstance(raw, bool):
        raise ValueError("No dollar amount to format")
    elif isinstance(raw, (int, float)):
        value = Decimal(str(raw))
    else:
        s = re.sub(r"usd|dollars?|[$,\s]", "", str(raw).strip().lower())
        if not re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)", s):
            raise ValueError("Cannot parse dollar amount: " + str(raw))
        value = Decimal(s)
    narrate(ctx, "Book value before rounding: " + str(value) + " dollars")
    cents = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    narrate(ctx, "Book value rounded half-up to cents: " + str(cents) + " dollars")
    return {"formatted book value in dollars to two decimal places": str(cents) + " $"}
