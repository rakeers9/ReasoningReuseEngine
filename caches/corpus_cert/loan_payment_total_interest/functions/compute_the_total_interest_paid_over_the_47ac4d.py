def semantic_function(ctx):
    '''Compute the total interest paid over the life of a loan from the periodic payment, the number of payments and the principal.'''
    narrate(ctx, 'Computing total interest paid over the life of the loan from the periodic payment, number of payments and principal')
    payment = load(ctx, 'the fixed monthly payment amount', d_type=float)
    n_periods = int(round(load(ctx, 'the total number of payment periods', d_type=float)))
    principal = load(ctx, 'the principal amount of the loan with currency unit', d_type=float)
    narrate(ctx, f'Loaded monthly payment M = {payment}, number of payments n = {n_periods}, principal P = {principal}')
    total_paid = payment * n_periods
    narrate(ctx, f'Total of all payments M x n = {payment} x {n_periods} = {total_paid}')
    total_interest = round(total_paid - principal, 2)
    narrate(ctx, f'Total interest = total paid - principal = {total_paid} - {principal} = {total_interest}')
    return {'total interest paid over the life of the loan': total_interest}