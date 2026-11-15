def semantic_function(ctx):
    '''Look up the spouse base dollar amount and balance fraction for an intestate heir configuration under a named UPC edition.'''
    narrate(ctx, 'Looking up the spouse base dollar amount and balance fraction for the intestate heir configuration')
    edition = load(ctx, 'the intestacy statute edition and its base dollar amounts, i.e. the Uniform Probate Code version in force', d_type=str)
    case_label = load(ctx, 'heir configuration case label stating whether all surviving descendants are also the spouse children or some are not', d_type=str)
    narrate(ctx, f'Intestacy edition named: {edition}')
    narrate(ctx, f'Heir configuration case: {case_label}')
    ed = str(edition).lower()
    label = str(case_label).lower()
    if 'current' in ed or 'without base' in ed:
        edition_key = 'upc_current'
    elif 'base' in ed or '1990' in ed:
        edition_key = 'upc_1990'
    else:
        raise ValueError('unrecognised intestacy edition: ' + str(edition))
    if 'some' in label and 'not' in label:
        case_key = 'some_not_spouse'
    elif 'all' in label and 'spouse' in label:
        case_key = 'all_spouse'
    else:
        raise ValueError('unsupported heir configuration: ' + str(case_label))
    table = {
        ('upc_1990', 'some_not_spouse'): (150000, 0.5),
        ('upc_current', 'some_not_spouse'): (0, 0.5),
    }
    if (edition_key, case_key) not in table:
        raise ValueError('no base amount rule for ' + edition_key + ' and ' + case_key)
    base, fraction = table[(edition_key, case_key)]
    if base == 0:
        narrate(ctx, 'This edition has no base amount for this case, so the spouse takes only the fraction of the estate')
    else:
        narrate(ctx, f'Spouse takes the first {base} dollars plus {fraction} of the balance after the base amount')
    return {
        'base dollar amount the spouse takes first, in dollars': base,
        'fraction of the balance of the estate after the base amount that passes to the spouse': fraction,
    }
