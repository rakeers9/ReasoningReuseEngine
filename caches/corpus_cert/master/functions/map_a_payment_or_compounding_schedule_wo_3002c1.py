import re

def semantic_function(ctx):
    """Map a payment or compounding schedule word to its number of periods per year."""
    narrate(ctx, 'Mapping the schedule word to the number of periods per year')
    word = load(ctx, 'the schedule word describing the frequency, such as annual, semiannual, quarterly, or monthly', d_type=str)
    narrate(ctx, f'Loaded schedule word: {word}')
    table = {'annual': 1, 'annually': 1, 'yearly': 1, 'semiannual': 2, 'semiannually': 2, 'biannual': 2, 'quarterly': 4, 'monthly': 12}
    key = re.sub(r'[^a-z]', '', str(word).strip().lower())
    if key in table:
        periods = table[key]
    else:
        answer = lm(ctx, f'How many times per year does the schedule {word} occur? Answer with a single integer only.')
        match = re.search(r'\d+', str(answer))
        if match is None:
            raise ValueError(f'Cannot map schedule word {word} to a number of periods')
        periods = int(match.group())
    if periods <= 0:
        raise ValueError('Periods per year must be a positive integer')
    narrate(ctx, f'Periods per year for {word}: {periods}')
    return {'the number of periods per year as a positive integer': periods}
