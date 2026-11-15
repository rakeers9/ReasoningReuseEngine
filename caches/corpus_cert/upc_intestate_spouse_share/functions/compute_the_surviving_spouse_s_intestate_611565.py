def semantic_function(ctx):
    '''Compute the surviving spouse's intestate share as the base dollar amount plus a fraction of the estate balance remaining after the base amount, capped at the whole estate.'''
    narrate(ctx, 'Computing the surviving spouse intestate share from the base amount and the balance fraction')
    estate = load(ctx, 'the value of the decedent net intestate estate in dollars, with currency unit', d_type=float)
    base = load(ctx, 'base dollar amount the spouse takes first, in dollars', d_type=float)
    fraction = load(ctx, 'fraction of the balance of the estate after the base amount that passes to the spouse', d_type=float)
    narrate(ctx, f'Estate value {estate} dollars, base amount {base} dollars, spouse fraction of balance {fraction}')
    capped_base = min(base, estate)
    if capped_base < base:
        narrate(ctx, 'The estate is smaller than the base amount, so the spouse takes the whole estate as the base')
    balance = estate - capped_base
    narrate(ctx, f'Balance of the estate after the base amount: {balance} dollars')
    spouse_balance_share = fraction * balance
    narrate(ctx, f'Spouse share of the balance: {spouse_balance_share} dollars')
    total = capped_base + spouse_balance_share
    narrate(ctx, f'Total intestate inheritance of the spouse: {total} dollars')
    return {
        'balance of the estate after the base amount in dollars': balance,
        'spouse share of the balance in dollars': spouse_balance_share,
        'total intestate inheritance of the spouse in dollars': total,
    }