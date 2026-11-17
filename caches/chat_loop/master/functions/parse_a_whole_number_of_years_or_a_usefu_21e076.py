import re
def semantic_function(ctx):
    '''Parse a whole number of years or a useful life from raw text, handling digits and number words.'''
    raw = load(ctx, 'the raw text containing a whole number of years such as 8 years or an 11-year life', d_type=str)
    narrate(ctx, 'Parsing whole number from raw text: ' + str(raw))
    text = raw.strip().lower()
    NUMBER_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    value = None
    m = re.search(r'(?<![\d.])\d+(?:,\d{3})*(?![\d.])', text)
    if m:
        value = int(m.group(0).replace(',', ''))
        narrate(ctx, 'Found digits in text, whole number: ' + str(value))
    else:
        total = 0
        found = False
        for tok in re.findall(r'[a-z]+', text):
            if tok in NUMBER_WORDS:
                total += NUMBER_WORDS[tok]
                found = True
            elif found:
                break
        if found:
            value = total
            narrate(ctx, 'Found number words in text, whole number: ' + str(value))
    if value is None:
        narrate(ctx, 'Cannot parse a whole number from text: ' + text)
        raise ValueError('Cannot parse whole number from: ' + raw)
    narrate(ctx, 'Whole number parsed: ' + str(value))
    return {'whole number as integer': value}