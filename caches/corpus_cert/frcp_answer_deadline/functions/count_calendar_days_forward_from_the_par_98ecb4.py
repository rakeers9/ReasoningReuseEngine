def semantic_function(ctx):
    '''Count calendar days forward from a service date, excluding the service day, to get the nominal answer deadline.'''
    narrate(ctx, 'Computing the nominal deadline by counting calendar days forward from the service date, excluding the service day')
    service_iso = load(ctx, 'the parsed service date', d_type=str)
    period = load(ctx, 'the response period in days', d_type=int)
    narrate(ctx, 'Loaded service date ' + str(service_iso) + ' and response period of ' + str(period) + ' days')

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

    y, m, d = [int(x) for x in str(service_iso).split('-')]
    start_serial = days_from_civil(y, m, d)
    end_serial = start_serial + period
    ny, nm, nd = civil_from_days(end_serial)
    deadline = '%04d-%02d-%02d' % (ny, nm, nd)
    narrate(ctx, 'Counted ' + str(period) + ' calendar days forward from ' + str(service_iso) + ' with the service day excluded, giving nominal deadline ' + deadline)
    return {'the nominal deadline date': deadline}
