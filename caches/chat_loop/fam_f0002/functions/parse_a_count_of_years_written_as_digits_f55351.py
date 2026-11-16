import re
ONES = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}

def semantic_function(ctx):
    '''Parse a count of years written as digits or English number words into an integer.'''
    text = load(ctx, 'number of years as text copied from the question', d_type=str)
    narrate(ctx, 'Loaded year count text: ' + str(text))
    s = str(text).strip().lower().replace('-', ' ').replace(',', ' ')
    s = re.sub(r'\b(years?|yrs?|lifetime|life|old|period|long|duration)\b', ' ', s)
    tokens = [w for w in s.split() if w != 'and']
    narrate(ctx, 'Tokens left after removing filler words: ' + str(tokens))
    if not tokens:
        narrate(ctx, 'No year count found in the text')
        raise ValueError('No year count found in: ' + str(text))
    if len(tokens) == 1 and tokens[0].isdigit():
        value = int(tokens[0])
        narrate(ctx, 'Parsed digit year count: ' + str(value))
    else:
        current = 0
        for w in tokens:
            if w in ONES:
                current += ONES[w]
                narrate(ctx, 'Added ones word ' + w + ' giving running total ' + str(current))
            elif w in TENS:
                current += TENS[w]
                narrate(ctx, 'Added tens word ' + w + ' giving running total ' + str(current))
            elif w == 'hundred':
                current = (current if current else 1) * 100
                narrate(ctx, 'Applied hundred multiplier giving running total ' + str(current))
            else:
                narrate(ctx, 'Unrecognised token ' + w)
                raise ValueError('Cannot parse year count: ' + str(text))
        value = current
        narrate(ctx, 'Parsed word year count: ' + str(value))
    narrate(ctx, 'Whole number of years: ' + str(value))
    return {'whole number of years as an integer': value}
