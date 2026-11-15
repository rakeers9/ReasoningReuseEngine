def semantic_function(ctx):
    '''Parse the calendar date text of service into a calendar date value.'''
    import re
    narrate(ctx, 'Parsing the service date text into a calendar date')
    raw = load(ctx, 'the calendar date on which the summons and complaint were served on the defendant, including weekday if given', d_type=str)
    narrate(ctx, f'Loaded service date text: {raw}')
    month_index = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    named = re.search(r'([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})', raw)
    numeric = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw)
    if named and named.group(1)[:3].lower() in month_index:
        year = int(named.group(3))
        month = month_index[named.group(1)[:3].lower()]
        day = int(named.group(2))
    elif numeric:
        year = int(numeric.group(1))
        month = int(numeric.group(2))
        day = int(numeric.group(3))
    else:
        year, month, day = [int(x) for x in smart_convert(ctx, raw, tuple)]
        narrate(ctx, 'Date text was not in a standard form, so the small model normalised it')
    if not 1 <= month <= 12:
        raise ValueError('Month is out of range')
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_lengths = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if not 1 <= day <= month_lengths[month - 1]:
        raise ValueError('Day is out of range for the month')
    iso = '%04d-%02d-%02d' % (year, month, day)
    narrate(ctx, 'Parsed service date as ' + iso)
    return {'the parsed service date': iso}
