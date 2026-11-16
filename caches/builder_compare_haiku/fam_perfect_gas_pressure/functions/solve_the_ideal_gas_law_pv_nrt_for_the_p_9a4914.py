from decimal import Decimal

def semantic_function(ctx):
    """Solve the ideal gas law pV = nRT for the pressure of a gas given its amount, volume and absolute temperature."""
    narrate(ctx, 'Solving the ideal gas law pV = nRT for the pressure of the gas, so p = nRT/V')
    amount_text = load(ctx, 'the amount of substance n of the gas with unit (moles)', d_type=str)
    volume_text = load(ctx, 'the volume occupied by the gas with unit', d_type=str)
    temperature_text = load(ctx, 'the absolute temperature of the gas with unit', d_type=str)
    narrate(ctx, 'Raw inputs: amount ' + str(amount_text) + ', volume ' + str(volume_text) + ', temperature ' + str(temperature_text))
    amount_result = do(ctx, 'Parse the amount of substance in moles from raw question text, handling scale words and units with exact factors.', payloads={'raw text giving the amount of substance with unit, e.g. 7.39 mol': amount_text})
    n = Decimal(str(load(amount_result, 'amount of substance n in mol', d_type=str, ctx=ctx)).strip())
    narrate(ctx, 'Amount of substance n = ' + str(n) + ' mol')
    volume_result = do(ctx, 'Parse the gas volume from raw text including its unit and convert it exactly to litres.', payloads={'raw text giving the volume with unit, e.g. 0.03256 m3': volume_text})
    V = Decimal(str(load(volume_result, 'volume in litres', d_type=str, ctx=ctx)).strip())
    narrate(ctx, 'Volume V = ' + str(V) + ' L')
    temperature_result = do(ctx, 'Parse the absolute temperature from raw text and convert it exactly to kelvin, applying the Celsius, Fahrenheit and Rankine offsets and scales.', payloads={'raw text giving the temperature with unit, e.g. 507.15 K': temperature_text})
    T = Decimal(str(load(temperature_result, 'temperature in kelvin', d_type=str, ctx=ctx)).strip())
    narrate(ctx, 'Absolute temperature T = ' + str(T) + ' K')
    R_result = do(ctx, 'Provide the molar gas constant R expressed in L·atm/(mol·K) for use with volume in litres and pressure in atm.', payloads={})
    R = Decimal(str(load(R_result, 'gas constant R in L·atm/(mol·K)', d_type=str, ctx=ctx)).strip())
    narrate(ctx, 'Gas constant R = ' + str(R) + ' L·atm/(mol·K)')
    if n <= 0 or V <= 0 or T <= 0:
        narrate(ctx, 'Amount, volume and temperature must all be positive for the ideal gas law')
        raise ValueError('ideal gas inputs must be positive')
    p_result = do(ctx, 'Compute the ideal gas pressure p = nRT/V from amount, gas constant, temperature and volume using exact decimal arithmetic.', payloads={'amount of substance n in mol': str(n), 'gas constant R in L·atm/(mol·K)': str(R), 'temperature in kelvin': str(T), 'volume in litres': str(V)})
    p = Decimal(str(load(p_result, 'pressure in atm', d_type=str, ctx=ctx)).strip())
    narrate(ctx, 'Pressure p = nRT/V = ' + str(p) + ' atm')
    cross_result = do(ctx, 'Cross-check an ideal gas pressure by recomputing nRT/V in SI units and comparing it with the primary pressure.', payloads={'amount of substance n in mol': str(n), 'temperature in kelvin': str(T), 'volume in litres': str(V), 'pressure in atm': str(p)})
    p_si = Decimal(str(load(cross_result, 'SI cross-check pressure in atm', d_type=str, ctx=ctx)).strip())
    narrate(ctx, 'SI cross-check pressure = ' + str(p_si) + ' atm, consistent with the primary result')
    return {'pressure of the gas in atm': p, 'SI cross-check pressure in atm': p_si}
