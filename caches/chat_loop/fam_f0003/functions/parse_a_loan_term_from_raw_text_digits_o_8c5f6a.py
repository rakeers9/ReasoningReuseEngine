def semantic_function(ctx):
    """Parse a loan term from raw text (digits or number words, in years or months) into a term in years and a term in months."""
    import re
    from decimal import Decimal
    raw = load(ctx, "the loan term with unit as raw text", d_type=str)
    narrate(ctx, f"Loaded raw term text: {raw!r}")
    text = str(raw).strip().lower().replace('-', ' ')
    text = re.sub(r'\s+', ' ', text)
    ones = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'a': 1, 'an': 1}
    tens = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    unit_years = ('years', 'year', 'yrs', 'yr', 'y')
    unit_months = ('months', 'month', 'mos', 'mo')
    m = re.match(r'^([0-9][0-9,]*\.?[0-9]*|\.[0-9]+)\s*([a-z]+)\.?$', text)
    if m:
        number = Decimal(m.group(1).replace(',', ''))
        unit = m.group(2)
        narrate(ctx, f"Digits number {number} with unit {unit!r}")
    else:
        m2 = re.match(r'^([a-z ]+?)\s+([a-z]+)\.?$', text)
        if not m2:
            narrate(ctx, "The text does not look like a loan term")
            raise ValueError(f"Cannot parse term: {raw!r}")
        words = m2.group(1).split()
        unit = m2.group(2)
        total = 0
        for w in words:
            if w == 'and':
                continue
            if w in ones:
                total += ones[w]
            elif w in tens:
                total += tens[w]
            elif w == 'hundred':
                total = max(total, 1) * 100
            else:
                narrate(ctx, f"Unrecognised number word {w!r}")
                raise ValueError(f"Cannot parse number words in term: {raw!r}")
        number = Decimal(total)
        narrate(ctx, f"Number words give {number} with unit {unit!r}")
    if unit in unit_years:
        years = number
        months = number * Decimal(12)
        narrate(ctx, f"Term is {years} years, which is {months} months")
    elif unit in unit_months:
        months = number
        years = number / Decimal(12)
        narrate(ctx, f"Term is {months} months, which is {years} years")
    else:
        narrate(ctx, f"Unit {unit!r} is not handled")
        raise ValueError(f"Unknown term unit {unit!r} in {raw!r}")
    return {"term in years": years, "term in months": months}