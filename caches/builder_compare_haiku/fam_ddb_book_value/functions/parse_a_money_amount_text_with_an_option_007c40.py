def semantic_function(ctx):
    '''Parse a money amount text with an optional scale word into an exact decimal number of dollars.'''
    import re
    from decimal import Decimal
    raw = load(ctx, 'the raw money text with optional scale word such as million or k', d_type=str)
    narrate(ctx, 'Loaded raw money text ' + repr(raw))
    text = str(raw).strip().lower()
    text = text.replace('usd', '').replace('dollars', '').replace('dollar', '').replace('$', '').replace(',', '').strip()
    m = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([a-z]*)', text)
    if m is None:
        raise ValueError('Cannot parse money text: ' + repr(raw))
    scales = {'': 1, 'k': 10**3, 'thousand': 10**3, 'm': 10**6, 'mn': 10**6, 'million': 10**6, 'bn': 10**9, 'b': 10**9, 'billion': 10**9}
    suffix = m.group(2)
    if suffix not in scales:
        raise ValueError('Unknown scale word in money text: ' + repr(raw))
    value = Decimal(m.group(1)) * scales[suffix]
    narrate(ctx, 'Parsed ' + repr(raw) + ' into ' + str(value) + ' dollars')
    return {'money amount in dollars as an exact decimal': value}
