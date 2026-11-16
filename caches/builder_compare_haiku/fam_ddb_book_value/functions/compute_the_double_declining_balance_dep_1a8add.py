def semantic_function(ctx):
    '''Compute the double-declining-balance depreciation rate as two divided by the useful life in years.'''
    from fractions import Fraction
    import re
    narrate(ctx, 'Computing the double-declining-balance rate as 2 divided by the useful life')
    raw = load(ctx, 'the useful life of the machine in years as raw text', d_type=str)
    narrate(ctx, 'Loaded useful life text ' + repr(raw))
    text = str(raw).strip().lower()
    number_words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30}
    life = None
    for tok in re.findall('[a-z]+|[0-9]+', text):
        if tok.isdigit():
            life = int(tok)
            break
        if tok in number_words:
            life = number_words[tok]
            break
    if life is None or life < 1:
        raise ValueError('Cannot parse a positive useful life: ' + repr(raw))
    narrate(ctx, 'Useful life parsed as ' + str(life) + ' years')
    rate = Fraction(2, life)
    narrate(ctx, 'DDB rate = 2 / ' + str(life) + ' = ' + str(rate) + ' (' + str(float(rate) * 100) + ' percent per year)')
    return {'double-declining-balance rate as an exact fraction': rate}
