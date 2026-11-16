import re
from decimal import Decimal

def semantic_function(ctx):
    '''Parse the amount of substance in moles from raw question text, handling scale words and units with exact factors.'''
    narrate(ctx, 'Parsing the amount of substance n from the raw text')
    raw = load(ctx, 'the amount of substance n of the gas with unit (moles)', d_type=str)
    text = str(raw).strip()
    narrate(ctx, 'Raw amount text is: ' + text)
    units = {'mol': Decimal('1'), 'mole': Decimal('1'), 'moles': Decimal('1'), 'kmol': Decimal('1000'), 'kilomol': Decimal('1000'), 'kilomole': Decimal('1000'), 'kilomoles': Decimal('1000'), 'mmol': Decimal('0.001'), 'millimol': Decimal('0.001'), 'millimole': Decimal('0.001'), 'millimoles': Decimal('0.001'), 'umol': Decimal('0.000001'), 'µmol': Decimal('0.000001'), 'μmol': Decimal('0.000001'), 'micromol': Decimal('0.000001'), 'micromole': Decimal('0.000001'), 'micromoles': Decimal('0.000001')}
    small = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    def words_to_number(tokens):
        if not tokens:
            narrate(ctx, 'No number words found before the unit')
            raise ValueError('no number words found before the unit')
        total = 0
        for tok in tokens:
            if tok == 'and':
                continue
            elif tok == 'hundred':
                total = (total or 1) * 100
            elif tok in small:
                total += small[tok]
            else:
                raise ValueError('unknown number word: ' + tok)
        narrate(ctx, 'Number words ' + ' '.join(tokens) + ' read as ' + str(total))
        return Decimal(total)
    numeric = re.match(r'^\s*([+-]?)\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)\s*([A-Za-zµμ]+)', text)
    if numeric is not None:
        narrate(ctx, 'Text starts with a digit-based number followed by a unit')
        sign = numeric.group(1)
        number = Decimal(sign + numeric.group(2).replace(',', ''))
        unit_word = numeric.group(3).lower()
    else:
        narrate(ctx, 'No leading digits found, so reading the amount from number words')
        words = re.findall(r'[A-Za-zµμ]+', text.lower())
        cut = None
        for idx, w in enumerate(words):
            if w in units:
                cut = idx
                break
        if cut is None:
            narrate(ctx, 'No recognised amount unit found in the text')
            raise ValueError('cannot parse an amount of substance with unit from: ' + text)
        number = words_to_number(words[:cut])
        unit_word = words[cut]
    narrate(ctx, 'Number ' + str(number) + ' with unit word ' + unit_word)
    if unit_word not in units:
        narrate(ctx, 'Unsupported amount-of-substance unit: ' + unit_word)
        raise ValueError('unsupported amount-of-substance unit: ' + unit_word)
    amount = number * units[unit_word]
    if amount <= 0:
        raise ValueError('amount of substance must be positive')
    narrate(ctx, 'Amount of substance is ' + str(amount) + ' mol')
    return {'amount of substance n in mol': amount}
