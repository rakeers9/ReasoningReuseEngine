def semantic_function(ctx):
    '''Parse a text giving a count of years into a non-negative whole number of years.'''
    import re
    raw = load(ctx, 'the raw text giving a count of years', d_type=str)
    narrate(ctx, 'Loaded raw year-count text ' + repr(raw))
    text = str(raw).strip().lower().replace(',', ' ').replace('-', ' ')
    text = re.sub(r'\b(years?|yrs?|yr)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if re.fullmatch(r'\d+', text):
        value = int(text)
        narrate(ctx, 'Year count is written in digits as ' + str(value))
    else:
        number_words = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
        tokens = [t for t in text.split(' ') if t and t != 'and']
        if not tokens:
            raise ValueError('Cannot parse year count: ' + repr(raw))
        narrate(ctx, 'Year count is written in words, tokens ' + repr(tokens))
        value = 0
        for tok in tokens:
            if tok in number_words:
                value += number_words[tok]
                narrate(ctx, 'Added word ' + tok + ' giving running total ' + str(value))
            elif tok == 'hundred':
                value = (value if value else 1) * 100
                narrate(ctx, 'Applied hundred giving running total ' + str(value))
            else:
                raise ValueError('Cannot parse year count: ' + repr(raw))
    narrate(ctx, 'Parsed ' + repr(raw) + ' into ' + str(value) + ' years')
    return {'count of years as an integer': value}