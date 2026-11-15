def semantic_function(ctx):
    '''Advance a nominal deadline date forward to the next day that is not a weekend day or federal legal holiday.'''
    import re
    narrate(ctx, 'Adjusting the nominal deadline forward past any weekend day or federal legal holiday')
    raw = load(ctx, 'the nominal deadline date', d_type=str)
    narrate(ctx, 'Loaded nominal deadline date ' + str(raw))
    numeric = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw)
    if numeric:
        y, m, d = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
    else:
        y, m, d = [int(x) for x in smart_convert(ctx, raw, tuple)]
        narrate(ctx, 'Deadline text was not in ISO form, so the small model normalised it')
    def days_from_civil(y, m, d):
        y -= 1 if m <= 2 else 0
        era = y // 400
        yoe = y - era * 400
        doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return era * 146097 + doe - 719468
    def civil_from_days(z):
        z += 719468
        era = z // 146097
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        y = yoe + era * 400
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + 3 if mp < 10 else mp - 9
        return (y + 1 if m <= 2 else y), m, d
    serial = days_from_civil(y, m, d)
    iso = '%04d-%02d-%02d' % (y, m, d)
    extension_days = 0
    while True:
        check = do(ctx, 'Check whether a calendar date is a weekend day or a federal legal holiday', payloads={'the calendar date to check': iso})
        non_business = load(check, 'whether the date is a weekend or federal legal holiday', d_type=bool, ctx=ctx)
        if not non_business:
            break
        narrate(ctx, iso + ' is a weekend day or federal legal holiday, so the period continues to the next day')
        serial += 1
        ny, nm, nd = civil_from_days(serial)
        iso = '%04d-%02d-%02d' % (ny, nm, nd)
        extension_days += 1
    narrate(ctx, 'Adjusted last day to serve an answer is ' + iso + ' after ' + str(extension_days) + ' extension day(s)')
    return {'the adjusted last day to serve an answer': iso}
