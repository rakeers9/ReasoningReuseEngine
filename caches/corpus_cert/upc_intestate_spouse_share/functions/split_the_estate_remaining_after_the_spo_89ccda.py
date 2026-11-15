def semantic_function(ctx):
    '''Split the estate remaining after the spouse share equally among the surviving descendants and verify that the spouse share and the descendants shares sum to the full estate.'''
    narrate(ctx, 'Distributing the estate remainder equally among the surviving descendants')
    estate = load(ctx, 'the value of the decedent net intestate estate in dollars, with currency unit', d_type=float)
    spouse_total = load(ctx, 'total intestate inheritance of the spouse in dollars', d_type=float)
    count = load(ctx, 'number of surviving descendants of the decedent', d_type=int)
    narrate(ctx, f'Estate value {estate} dollars, spouse inheritance {spouse_total} dollars, surviving descendants {count}')
    remainder = estate - spouse_total
    narrate(ctx, f'Remainder after the spouse share: {remainder} dollars')
    if count > 0:
        per_descendant = remainder / count
        narrate(ctx, f'Each of the {count} surviving descendants takes {per_descendant} dollars')
    else:
        per_descendant = 0.0
        narrate(ctx, 'There are no surviving descendants, so no amount passes to descendants under this rule')
    total_descendants = per_descendant * count
    difference = spouse_total + total_descendants - estate
    balanced = abs(difference) < 1e-6
    narrate(ctx, f'Descendants total {total_descendants} dollars; spouse plus descendants minus estate is {difference}')
    if balanced:
        narrate(ctx, 'Check passed: the spouse share and the descendant shares equal the full estate')
    else:
        narrate(ctx, 'Check failed: the distribution does not equal the full estate')
    return {
        'amount passing to each surviving descendant in dollars': per_descendant,
        'total passing to the surviving descendants in dollars': total_descendants,
        'check that the spouse share and descendants shares equal the full estate': balanced,
    }
