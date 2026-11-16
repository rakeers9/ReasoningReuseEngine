from decimal import Decimal

def semantic_function(ctx):
    '''Provide the molar gas constant R expressed in L·atm/(mol·K) for use with volume in litres and pressure in atm.'''
    narrate(ctx, 'Providing the molar gas constant R in L·atm/(mol·K) so that pressure comes out in atm when volume is in litres')
    R = Decimal('0.0820573661')
    narrate(ctx, 'Gas constant R is ' + str(R) + ' L·atm/(mol·K)')
    return {'gas constant R in L·atm/(mol·K)': R}