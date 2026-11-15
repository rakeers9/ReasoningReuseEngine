def semantic_function(ctx):
    '''Determine the weekday name of a calendar date.'''
    import re
    narrate(ctx, 'Determining the weekday name of the calendar date')
    raw = load(ctx, 'the calendar date whose weekday name is needed', d_type=str)
    narrate(ctx, f'Loaded calendar date text: {raw}')
    month_index = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    numeric = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw)
    named = re.search(r'([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})', raw)
    if numeric:
        year, month, day = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
    elif named and named.group(1)[:3].lower() in month_index:
        year, month, day = int(named.group(3)), month_index[named.group(1)[:3].lower()], int(named.group(2))
    else:
        year, month, day = [int(x) for x in smart_convert(ctx, raw, tuple)]
        narrate(ctx, 'Date text was not in a standard form, so the small model normalised it')
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_lengths = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if not 1 <= month <= 12 or not 1 <= day <= month_lengths[month - 1]:
        raise ValueError('Calendar date is out of range')
    offsets = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    y = year - 1 if month < 3 else year
    index = (y + y // 4 - y // 100 + y // 400 + offsets[month - 1] + day) % 7
    names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekday = names[index]
    narrate(ctx, f'The calendar date {year:04d}-{month:02d}-{day:02d} falls on a {weekday}')
    return {'the weekday name of the date': weekday}