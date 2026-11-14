def semantic_function(ctx):
    '''Provide the molar gas constant R in SI units of J K-1 mol-1.'''
    narrate(ctx, 'Providing the molar gas constant R in SI units')
    R = 8.314
    narrate(ctx, 'Molar gas constant R = 8.314 J K-1 mol-1')
    return {'the molar gas constant R in J K-1 mol-1': R}