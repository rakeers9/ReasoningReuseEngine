def semantic_function(ctx):
    """Solve a fixed-rate amortizing loan by computing the level periodic payment and total interest from principal, annual rate, compounding frequency and term."""
    narrate(ctx, "Solving a fixed-rate amortizing loan")
    principal_text = load(ctx, "the loan principal amount in dollars", d_type=str)
    narrate(ctx, f"Loaded principal text: {principal_text}")
    rate_text = load(ctx, "the annual nominal interest rate as a percentage", d_type=str)
    narrate(ctx, f"Loaded annual rate text: {rate_text}")
    freq_text = load(ctx, "the compounding and payment frequency description", d_type=str)
    narrate(ctx, f"Loaded frequency text: {freq_text}")
    term_text = load(ctx, "the loan term with unit", d_type=str)
    narrate(ctx, f"Loaded term text: {term_text}")

    r1 = do(ctx, "Parse a money amount from raw text with currency symbols, separators and scale suffixes into an exact Decimal dollar amount.",
            payloads={"the money amount as raw text": principal_text})
    principal = load(r1, "amount in dollars", d_type=str, ctx=ctx)
    narrate(ctx, f"Principal in dollars: {principal}")

    r2 = do(ctx, "Parse a percentage rate from raw text such as '7.5%' or '7.5 percent' into an exact Decimal annual rate as a fraction.",
            payloads={"the percentage rate as raw text": rate_text})
    annual = load(r2, "annual rate as a decimal fraction", d_type=str, ctx=ctx)
    narrate(ctx, f"Annual rate as fraction: {annual}")

    r3 = do(ctx, "Parse a loan term from raw text (digits or number words, in years or months) into a term in years and a term in months.",
            payloads={"the loan term with unit as raw text": term_text})
    years = load(r3, "term in years", d_type=str, ctx=ctx)
    narrate(ctx, f"Term in years: {years}")

    r4 = do(ctx, "Determine the number of compounding periods per year from a frequency description such as monthly, quarterly, semiannual, annual, weekly, biweekly or daily.",
            payloads={"the compounding and payment frequency description as raw text": freq_text})
    periods = load(r4, "periods per year", d_type=str, ctx=ctx)
    narrate(ctx, f"Periods per year: {periods}")

    r5 = do(ctx, "Compute the periodic interest rate (annual rate divided by periods per year) and the total number of payments (periods per year times term in years).",
            payloads={"annual rate as a decimal fraction": annual,
                      "periods per year": periods,
                      "term in years": years})
    periodic = load(r5, "periodic rate", d_type=str, ctx=ctx)
    count = load(r5, "number of payments", d_type=str, ctx=ctx)
    narrate(ctx, f"Periodic rate {periodic}, number of payments {count}")

    r6 = do(ctx, "Compute the growth factor (1+r)^n and the unrounded level periodic annuity payment from principal, periodic rate and number of payments, treating a zero rate as principal divided by number of payments.",
            payloads={"principal in dollars": principal,
                      "periodic rate": periodic,
                      "number of payments": count})
    pmt = load(r6, "unrounded periodic payment in dollars", d_type=str, ctx=ctx)
    narrate(ctx, f"Unrounded periodic payment: {pmt}")

    r7 = do(ctx, "Round the periodic payment to the cent with half-up rounding and compute total interest as the rounded payment times the number of payments minus the principal.",
            payloads={"unrounded periodic payment in dollars": pmt,
                      "number of payments": count,
                      "principal in dollars": principal})
    mp = load(r7, "monthly payment in dollars", d_type=str, ctx=ctx)
    ti = load(r7, "total interest in dollars", d_type=str, ctx=ctx)
    narrate(ctx, f"Monthly payment {mp}, total interest {ti}")
    return {"monthly payment in dollars": mp, "total interest in dollars": ti}
