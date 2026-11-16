def semantic_function(ctx):
    """Parse a raw quantity text into its numeric magnitude with any scale word applied and its unit symbol."""
    import re
    from decimal import Decimal
    narrate(ctx, 'Parsing a raw quantity text into its numeric magnitude and its unit symbol')
    raw_text = load(ctx, 'quantity as raw text copied from the question', d_type=str)
    narrate(ctx, 'Loaded raw quantity text: ' + repr(raw_text))
    small = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    scales = {'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'bn': 1000000000}
    text = raw_text.strip()
    sign = Decimal(1)
    if text[:1] in ('+', '-'):
        if text[0] == '-':
            sign = Decimal(-1)
            narrate(ctx, 'Negative sign found, so the magnitude will be negated')
        else:
            narrate(ctx, 'Positive sign found, so the magnitude keeps its sign')
        text = text[1:].strip()
    currency = ''
    if text[:1] in ('$', '£', '€'):
        currency = text[0]
        narrate(ctx, 'Currency symbol found: ' + currency)
        text = text[1:].strip()
    m = re.match(r'^([0-9]{1,3}(?:,[0-9]{3})+(?:[.][0-9]+)?|[0-9]+(?:[.][0-9]+)?|[.][0-9]+)', text)
    if m:
        magnitude = Decimal(m.group(1).replace(',', ''))
        after = text[m.end():]
        narrate(ctx, 'Digits found in text: ' + m.group(1))
    else:
        narrate(ctx, 'No digits found, so trying spelled-out number words')
        words = text.replace('-', ' ').split()
        total = 0
        current = 0
        used = 0
        for idx, word in enumerate(words):
            w = word.lower()
            if w in small:
                current += small[w]
                narrate(ctx, 'Number word ' + w + ' added to the running total')
            elif w == 'hundred':
                current = (current or 1) * 100
                narrate(ctx, 'Number word hundred applied to the running total')
            elif w in ('thousand', 'million', 'billion'):
                total += (current or 1) * scales[w]
                current = 0
                narrate(ctx, 'Scale word ' + w + ' applied to the number words so far')
            else:
                narrate(ctx, 'Word ' + w + ' is not a number word, so parsing stops')
                break
            used = idx + 1
        if used == 0:
            narrate(ctx, 'No number found in quantity text')
            raise ValueError('no number found in quantity text: ' + raw_text)
        magnitude = Decimal(total + current)
        after = ' ' + ' '.join(words[used:])
        narrate(ctx, 'Spelled-out number converted to magnitude ' + str(magnitude))
    scale = Decimal(1)
    rest = after.strip()
    if after[:1] == 'k' and not re.match(r'[A-Za-z]', after[1:2]):
        scale = Decimal(1000)
        rest = after[1:].strip()
        narrate(ctx, 'Suffix k attached to the number, scale factor 1000 applied')
    else:
        m_s = re.match(r' *(thousand|million|billion|bn)(?![A-Za-z])', after, re.IGNORECASE)
        if m_s:
            word = m_s.group(1).lower()
            scale = Decimal(scales[word])
            rest = after[m_s.end():].strip()
            narrate(ctx, 'Scale word ' + word + ' applied, factor ' + str(scale))
        else:
            narrate(ctx, 'No scale word present, factor 1')
    unit = (currency + rest).strip()
    value = sign * magnitude * scale
    narrate(ctx, 'Signed magnitude with scale applied: ' + str(value))
    narrate(ctx, 'Unit symbol: ' + repr(unit))
    return {'numeric magnitude of the quantity': value, 'unit symbol of the quantity': unit}
