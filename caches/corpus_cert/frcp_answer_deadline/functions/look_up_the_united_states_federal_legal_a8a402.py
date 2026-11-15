def semantic_function(ctx):
    '''Look up the United States federal legal holidays that fall within a given date range.'''
    import re
    narrate(ctx, 'Looking up the federal legal holidays that fall within the date range')
    start_raw = load(ctx, 'the start date of the date range', d_type=str)
    end_raw = load(ctx, 'the end date of the date range', d_type=str)
    narrate(ctx, f'Loaded range start: {start_raw}; range end: {end_raw}')
    month_index = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    def to_ymd(raw):
        text = str(raw)
        numeric = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
        named = re.search(r'([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})', text)
        if numeric:
            return (int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3)))
        if named and named.group(1)[:3].lower() in month_index:
            return (int(named.group(3)), month_index[named.group(1)[:3].lower()], int(named.group(2)))
        values = [int(x) for x in smart_convert(ctx, text, tuple)]
        narrate(ctx, 'Date text was not in a standard form, so the small model normalised it')
        return (values[0], values[1], values[2])
    start = to_ymd(start_raw)
    end = to_ymd(end_raw)
    if start > end:
        start, end = end, start
        narrate(ctx, 'The range bounds were given in reverse order, so they were swapped')
    def is_leap(y):
        return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    def weekday_index(y, m, d):
        yy = y - 1 if m <= 2 else y
        era = (yy if yy >= 0 else yy - 399) // 400
        yoe = yy - era * 400
        mp = (m + 9) % 12
        doy = (153 * mp + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        days = era * 146097 + doe - 719468
        return (days + 3) % 7
    def nth_weekday(y, m, weekday, n):
        offset = (weekday - weekday_index(y, m, 1)) % 7
        return (y, m, 1 + offset + 7 * (n - 1))
    def last_weekday(y, m, weekday):
        lengths = [31, 29 if is_leap(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        last = lengths[m - 1]
        offset = (weekday_index(y, m, last) - weekday) % 7
        return (y, m, last - offset)
    def observed(ymd):
        y, m, d = ymd
        w = weekday_index(y, m, d)
        if w == 5:
            return (y - 1, 12, 31) if d == 1 else (y, m, d - 1)
        if w == 6:
            return (y, m, d + 1)
        return ymd
    def holidays_for_year(y):
        return [
            ('New Year Day', observed((y, 1, 1))),
            ('Martin Luther King Jr. Day', nth_weekday(y, 1, 0, 3)),
            ('Washington Birthday (Presidents Day)', nth_weekday(y, 2, 0, 3)),
            ('Memorial Day', last_weekday(y, 5, 0)),
            ('Juneteenth National Independence Day', observed((y, 6, 19))),
            ('Independence Day', observed((y, 7, 4))),
            ('Labor Day', nth_weekday(y, 9, 0, 1)),
            ('Columbus Day', nth_weekday(y, 10, 0, 2)),
            ('Veterans Day', observed((y, 11, 11))),
            ('Thanksgiving Day', nth_weekday(y, 11, 3, 4)),
            ('Christmas Day', observed((y, 12, 25))),
        ]
    found = []
    for y in range(start[0], end[0] + 2):
        for name, ymd in holidays_for_year(y):
            if start <= ymd <= end:
                label = f'{ymd[0]:04d}-{ymd[1]:02d}-{ymd[2]:02d} {name}'
                found.append(label)
                narrate(ctx, f'Federal legal holiday in range: {label}')
    found.sort()
    if not found:
        narrate(ctx, 'No federal legal holiday falls within the date range')
    return {'the list of federal legal holidays in the date range': found}
