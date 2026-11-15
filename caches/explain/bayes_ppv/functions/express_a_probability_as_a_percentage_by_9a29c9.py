def semantic_function(ctx):
    '''Express a probability as a percentage by multiplying it by one hundred.'''
    narrate(ctx, 'Expressing the probability as a percentage')
    probability = load(ctx, 'the posterior probability of disease given a positive test', d_type=float)
    narrate(ctx, f'Loaded posterior probability {probability}')
    percentage = probability * 100
    narrate(ctx, f'Percentage = {probability} x 100 = {percentage}')
    return {'posterior probability as a percentage': percentage}