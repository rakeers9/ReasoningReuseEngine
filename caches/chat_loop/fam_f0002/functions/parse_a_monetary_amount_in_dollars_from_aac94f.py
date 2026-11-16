import re
from decimal import Decimal

def semantic_function(ctx):
    '''Parse a monetary amount in dollars from raw text, handling currency sign, thousands separators and scale words such as k, thousand, million, bn and billion.'''
    raw = load(ctx, 'the raw text of the monetary amount with currency such as $379,000 or $0.42 million', d_type=str)
    narrate(ctx, 'Parsing monetary amount from raw text: ' + str(raw))
    text = raw.strip().lower()
    negative = text.startswith('-')
    if negative:
        narrate(ctx, 'Amount carries a negative sign, so the result will be negated')
    text = text.lstrip('-+ ').replace('usd', '').replace('$', '').replace(',', '').strip()
    scale = Decimal(1)
    for word, factor in [('billion', Decimal(10) ** 9), ('bn', Decimal(10) ** 9), ('million', Decimal(10) ** 6), ('thousand', Decimal(10) ** 3), ('k', Decimal(10) ** 3), ('m', Decimal(10) ** 6)]:
        if text.endswith(word):
            scale = factor
            narrate(ctx, 'Recognised scale word ' + word + ' with factor ' + str(factor))
            text = text[:-len(word)].strip()
            break
    if re.fullmatch(r'(\d+(\.\d+)?|\.\d+)', text) is None:
        narrate(ctx, 'Cannot parse a monetary amount from text: ' + text)
        raise ValueError('Cannot parse monetary amount from: ' + raw)
    value = Decimal(text) * scale
    if negative:
        value = -value
    narrate(ctx, 'Monetary amount in dollars: ' + str(value))
    return {'monetary amount in dollars as exact decimal': value}
