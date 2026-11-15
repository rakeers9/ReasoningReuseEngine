import re

def semantic_function(ctx):
    '''Compute the surviving spouse's intestate inheritance under a named probate code intestacy edition with base dollar amounts, narrating each step.'''
    narrate(ctx, 'Computing the surviving spouse intestate inheritance under the named intestacy edition')
    edition = load(ctx, 'The intestacy statute edition and its base dollar amounts, i.e. the Uniform Probate Code version in force', d_type=str)
    estate_raw = load(ctx, "The value of the decedent's net intestate estate in dollars, with currency unit", d_type=str)
    spouse = load(ctx, 'The surviving spouse of the decedent, with relationship to the decedent', d_type=str)
    descendants_text = load(ctx, 'The surviving descendants of the decedent, with count and whether each is also a child of the surviving spouse', d_type=str)
    heir = load(ctx, 'The heir whose inheritance is being asked for, identified as the surviving spouse', d_type=str)
    narrate(ctx, 'Loaded edition: ' + str(edition) + '; estate: ' + str(estate_raw) + '; spouse: ' + str(spouse))
    narrate(ctx, 'Loaded descendants: ' + str(descendants_text) + '; heir asked for: ' + str(heir))
    estate = float(''.join(ch for ch in str(estate_raw) if ch.isdigit() or ch == '.'))
    narrate(ctx, 'Estate value parsed as ' + str(estate) + ' dollars')
    text = str(descendants_text).lower()
    number_words = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    count = None
    for tok in re.findall('[a-z0-9]+', text):
        if tok.isdigit():
            count = int(tok)
            break
        if tok in number_words:
            count = number_words[tok]
            break
    if count is None:
        reply = str(lm(ctx, 'How many surviving descendants of the decedent does this text describe? Text: ' + text + ' Answer with one integer digit only.'))
        nums = re.findall('[0-9]+', reply)
        count = int(nums[0]) if nums else 0
    not_count = None
    not_reply = str(lm(ctx, 'How many of these surviving descendants are NOT children of the surviving spouse? Text: ' + text + ' Answer with one integer digit only.'))
    nums = re.findall('[0-9]+', not_reply)
    if nums:
        not_count = int(nums[0])
    else:
        for tok in re.findall('[a-z]+', not_reply.lower()):
            if tok in number_words:
                not_count = number_words[tok]
                break
    if not_count is None:
        markers = ['not her', 'not his', 'not their', 'not the spouse', 'prior marriage', 'previous marriage', 'not a child of', 'not of the spouse']
        not_count = count if any(m in text for m in markers) else 0
    not_count = max(0, min(not_count, count))
    narrate(ctx, 'Surviving descendants: ' + str(count) + '; descendants not children of the spouse: ' + str(not_count))
    descendants = [{'child_of_spouse': i < count - not_count} for i in range(count)]
    classify = do(ctx, "Classify the intestate heir configuration by counting the surviving descendants and checking whether each descendant is also a child of the surviving spouse.", payloads={
        "The surviving spouse of the decedent, with relationship to the decedent": spouse,
        "The surviving descendants of the decedent, with count and whether each is also a child of the surviving spouse": descendants,
    })
    case_label = load(classify, 'heir configuration case label', d_type=str, ctx=ctx)
    count_checked = load(classify, 'number of surviving descendants of the decedent', d_type=int, ctx=ctx)
    narrate(ctx, 'Heir configuration: ' + str(case_label) + '; surviving descendants: ' + str(count_checked))
    if count_checked == 0 or not_count == 0:
        spouse_total = estate
        per_descendant = 0.0
        desc_total = 0.0
        narrate(ctx, 'No descendant falls outside the spouse, so the spouse takes the whole estate: ' + str(spouse_total) + ' dollars')
    else:
        lookup = do(ctx, "Look up the spouse's base dollar amount and balance fraction for an intestate heir configuration under a named UPC edition.", payloads={
            'The intestacy statute edition and its base dollar amounts, i.e. the Uniform Probate Code version in force': edition,
            "Heir configuration case label stating whether all surviving descendants are also the spouse's children or some are not": case_label,
        })
        base = load(lookup, 'base dollar amount the spouse takes first, in dollars', d_type=float, ctx=ctx)
        fraction = load(lookup, 'fraction of the balance of the estate after the base amount that passes to the spouse', d_type=float, ctx=ctx)
        narrate(ctx, 'Spouse takes the first ' + str(base) + ' dollars plus ' + str(fraction) + ' of the balance after the base amount')
        share = do(ctx, "Compute the surviving spouse's intestate share as the base dollar amount plus a fraction of the estate balance remaining after the base amount, capped at the whole estate.", payloads={
            "The value of the decedent's net intestate estate in dollars, with currency unit": estate,
            'Base dollar amount the spouse takes first, in dollars': base,
            'Fraction of the balance of the estate after the base amount that passes to the spouse': fraction,
        })
        spouse_total = load(share, 'total intestate inheritance of the spouse in dollars', d_type=float, ctx=ctx)
        split = do(ctx, 'Split the estate remaining after the spouse share equally among the surviving descendants and verify that the spouse share and the descendants shares sum to the full estate.', payloads={
            "The value of the decedent's net intestate estate in dollars, with currency unit": estate,
            'Total intestate inheritance of the spouse in dollars': spouse_total,
            'Number of surviving descendants of the decedent': count_checked,
        })
        per_descendant = load(split, 'amount passing to each surviving descendant in dollars', d_type=float, ctx=ctx)
        desc_total = load(split, 'total passing to the surviving descendants in dollars', d_type=float, ctx=ctx)
        narrate(ctx, 'Descendants receive ' + str(desc_total) + ' dollars in total, ' + str(per_descendant) + ' each')
    narrate(ctx, 'Spouse (' + str(heir) + ') inherits ' + str(spouse_total) + ' dollars')
    narrate(ctx, 'This result applies to edition: ' + str(edition) + '; a revision without a base amount would give a different figure')
    return {
        'amount inherited by the surviving spouse in dollars': spouse_total,
        'explanation of how the spouse inheritance was found, including the edition used': 'Spouse inheritance computed under ' + str(edition) + ' as the base amount plus the fraction of the balance after the base amount; each step is narrated.',
    }
