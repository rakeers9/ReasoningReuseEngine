def semantic_function(ctx):
    '''Determine the number of days allowed to serve an answer given the waiver status of service under the governing rule.'''
    import re
    narrate(ctx, 'Determining the response period in days from the waiver status and the governing rule')
    waiver_text = load(ctx, 'the waiver status of service for the defendant, stating whether the defendant waived formal service or not', d_type=str)
    rule_text = load(ctx, 'the governing procedural rule that sets the time to serve an answer, cited by rule number', d_type=str)
    narrate(ctx, f'Loaded waiver status: {waiver_text}')
    narrate(ctx, f'Loaded governing rule: {rule_text}')
    if '12(a)' not in rule_text:
        raise ValueError('The governing rule does not prescribe a Rule 12(a) answer period')
    text = waiver_text.lower()
    negated = re.search(r'\bnot\b|\bno\b|n.t\b', text) is not None
    waived = (not negated) and re.search(r'\bwaiv', text) is not None
    if waived:
        days = 60
        narrate(ctx, 'Defendant waived service, so the answer period is 60 days after the waiver request was sent under Rule 4(d)(3) and Rule 12(a)(1)(A)(ii)')
    else:
        days = 21
        narrate(ctx, 'Defendant did not waive service, so the answer period is 21 days after service under Rule 12(a)(1)(A)(i)')
    narrate(ctx, f'Response period in days: {days}')
    return {'the response period in days': days}
