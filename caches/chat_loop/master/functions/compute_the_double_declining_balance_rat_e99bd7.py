import re
from fractions import Fraction

ONES = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}

def semantic_function(ctx):
    '''Compute the double-declining-balance rate as two divided by the useful life in years.'''
    narrate(ctx, 'Computing the double-declining-balance rate as 2 divided by the useful life in years')
    life_text = load(ctx, 'the useful life of the asset in years, as text copied from the question', d_type=str)
    narrate(ctx, 'Loaded useful life text: ' + str(life_text))
    s = str(life_text).strip().lower().replace('-', ' ').replace(',', ' ')
    s = re.sub(r'\b(years?|yrs?|life|lifetime|useful|long)\b', ' ', s)
    tokens = [w for w in s.split() if w != 'and']
    if not tokens:
        raise ValueError('No useful life found in: ' + str(life_text))
    if len(tokens) == 1 and tokens[0].isdigit():
        life = int(tokens[0])
    else:
        current = 0
        for w in tokens:
            if w in ONES:
                current += ONES[w]
            elif w in TENS:
                current += TENS[w]
            elif w == 'hundred':
                current = (current if current else 1) * 100
            else:
                raise ValueError('Cannot parse useful life: ' + str(life_text))
        life = current
    narrate(ctx, 'Useful life in years: ' + str(life))
    if life < 1:
        raise ValueError('Useful life must be at least 1 year')
    rate = Fraction(2, life)
    narrate(ctx, 'DDB rate = 2/' + str(life) + ' = ' + str(rate))
    return {'double-declining-balance rate as an exact fraction': rate}
