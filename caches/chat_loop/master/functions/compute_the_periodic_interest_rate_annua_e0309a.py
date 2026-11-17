def semantic_function(ctx):
    """Compute the periodic interest rate and the total number of payments from the annual rate, periods per year and term in years."""
    import re
    from decimal import Decimal, getcontext
    getcontext().prec = 60

    def to_dec(raw, what):
        if isinstance(raw, Decimal):
            return raw
        if isinstance(raw, bool):
            raise ValueError(f"Cannot parse {what}: {raw!r}")
        if isinstance(raw, int):
            return Decimal(raw)
        if isinstance(raw, float):
            return Decimal(str(raw))
        text = str(raw).strip().lower()
        percent = '%' in text or 'percent' in text
        m = re.search(r'(-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?|-?\.\d+)(?:\s*/\s*(\d+(?:\.\d+)?))?', text)
        if not m:
            raise ValueError(f"Cannot parse {what}: {raw!r}")
        val = Decimal(m.group(1).replace(',', ''))
        if m.group(2):
            val = val / Decimal(m.group(2))
        if percent:
            val = val / Decimal(100)
        return val

    annual_raw = load(ctx, "annual rate as a decimal fraction", d_type=str)
    annual = to_dec(annual_raw, "annual rate")
    narrate(ctx, f"Loaded annual rate as a decimal fraction: {annual}")
    periods_raw = load(ctx, "periods per year", d_type=str)
    periods = to_dec(periods_raw, "periods per year")
    narrate(ctx, f"Loaded periods per year: {periods}")
    years_raw = load(ctx, "term in years", d_type=str)
    years = to_dec(years_raw, "term in years")
    narrate(ctx, f"Loaded term in years: {years}")
    if periods <= 0:
        raise ValueError("Periods per year must be positive")
    rate = annual / periods
    narrate(ctx, f"Periodic rate = annual rate / periods per year = {annual} / {periods} = {rate}")
    count = periods * years
    narrate(ctx, f"Number of payments = periods per year * term in years = {periods} * {years} = {count}")
    if count != count.to_integral_value():
        raise ValueError(f"Number of payments is not a whole number: {count}")
    count_int = int(count)
    narrate(ctx, f"Number of payments as a whole number: {count_int}")
    return {"periodic rate": rate, "number of payments": count_int}
