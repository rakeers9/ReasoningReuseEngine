def semantic_function(ctx):
    '''Check whether a calendar date is a weekend day or a federal legal holiday.'''
    import re
    narrate(ctx, 'Checking whether the calendar date is a weekend day or a federal legal holiday')
    raw = load(ctx, 'the calendar date to check', d_type=str)
    narrate(ctx, f'Loaded calendar date to check: {raw}')
    weekday_result = do(ctx, 'Determine the weekday name of a calendar date', payloads={'the calendar date': raw})
    weekday = load(weekday_result, 'the weekday name of the date', d_type=str, ctx=ctx)
    narrate(ctx, f'The date falls on a {weekday}')
    is_weekend = weekday in ('Saturday', 'Sunday')
    if is_weekend:
        narrate(ctx, 'The date is a weekend day')
    numeric = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw)
    if numeric:
        iso = '%04d-%02d-%02d' % (int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3)))
    else:
        parts = [int(x) for x in smart_convert(ctx, raw, tuple)]
        iso = '%04d-%02d-%02d' % (parts[0], parts[1], parts[2])
        narrate(ctx, 'Date text was not in ISO form, so the small model normalised it')
    narrate(ctx, f'Normalised calendar date: {iso}')
    holiday_result = do(ctx, 'Look up the United States federal legal holidays that fall within a given date range', payloads={'the start date of the date range': iso, 'the end date of the date range': iso})
    holidays = load(holiday_result, 'the list of federal legal holidays in the date range', d_type=list, ctx=ctx)
    narrate(ctx, f'Loaded federal legal holidays in range: {holidays}')
    matches = [h for h in holidays if str(h).startswith(iso)]
    is_holiday = len(matches) > 0
    if is_holiday:
        narrate(ctx, f'The date is a federal legal holiday: {matches}')
    else:
        narrate(ctx, 'No federal legal holiday falls on this date')
    non_business = is_weekend or is_holiday
    narrate(ctx, 'The date is a weekend or federal legal holiday' if non_business else 'The date is an ordinary business day')
    return {'whether the date is a weekend or federal legal holiday': non_business, 'the weekday name of the date': weekday}