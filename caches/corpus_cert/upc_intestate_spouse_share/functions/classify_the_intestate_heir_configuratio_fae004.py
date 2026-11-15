def semantic_function(ctx):
    '''Classify the intestate heir configuration by counting the surviving descendants and checking whether each descendant is also a child of the surviving spouse.'''
    narrate(ctx, 'Classifying the intestate heir configuration from the surviving spouse and the surviving descendants')
    spouse = load(ctx, 'the surviving spouse of the decedent, with relationship to the decedent', d_type=str)
    descendants = load(ctx, 'the surviving descendants of the decedent, with count and whether each is also a child of the surviving spouse', d_type=list)
    spouse_present = bool(str(spouse).strip())
    count = len(descendants)
    narrate(ctx, f'Surviving descendants counted: {count}')
    shared = [d for d in descendants if d.get('child_of_spouse')]
    not_shared = count - len(shared)
    narrate(ctx, f'Descendants who are not children of the surviving spouse: {not_shared}')
    if not spouse_present:
        label = 'no surviving spouse, so the spouse rule does not apply'
    elif count == 0:
        label = 'spouse with no surviving descendants'
    elif not_shared == 0:
        label = 'spouse with descendants who are all also children of the spouse'
    else:
        label = 'spouse with descendants some of whom are not children of the spouse'
    narrate(ctx, f'Heir configuration case: {label}')
    return {
        'heir configuration case label': label,
        'number of surviving descendants of the decedent': count,
        'number of descendants who are not children of the surviving spouse': not_shared,
    }
