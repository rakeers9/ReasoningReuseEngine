import re
from decimal import Decimal

def semantic_function(ctx):
    """Parse a monetary amount text with currency symbol and scale word into an exact Decimal dollar amount."""
    text = load(ctx, 'monetary amount as text copied from the question with currency symbol and scale word', d_type=str)
    narrate(ctx, 'Parsing monetary amount text: ' + str(text))
    s = str(text).strip().lower()
    s = re.sub(r'usd|us\$|dollars?|[$£€¥]', '', s).strip()
    scale = Decimal(1)
    m = re.fullmatch(r'(.*?)\s*(billion|bn|million|mn|thousand|k|m)', s)
    if m:
        word = m.group(2)
        s = m.group(1).strip()
        scales = {'billion': Decimal(10) ** 9, 'bn': Decimal(10) ** 9, 'million': Decimal(10) ** 6, 'mn': Decimal(10) ** 6, 'm': Decimal(10) ** 6, 'thousand': Decimal(10) ** 3, 'k': Decimal(10) ** 3}
        scale = scales[word]
        narrate(ctx, 'Applying scale word ' + word + ' with factor ' + str(scale))
    compact = s.replace(',', '').replace(' ', '')
    if not re.fullmatch(r'[+-]?(\d+(\.\d+)?|\.\d+)', compact):
        raise ValueError('Cannot parse monetary amount: ' + str(text))
    amount = Decimal(compact) * scale
    narrate(ctx, 'Parsed amount in dollars: ' + str(amount))
    return {'amount in dollars as an exact Decimal': amount}