def semantic_function(ctx):
    '''Predict the pressure exerted by a perfect gas from its amount, identity, volume and temperature.'''
    narrate(ctx, 'Predicting the pressure of a perfect gas from pV = nRT')
    n_raw = load(ctx, 'the amount of substance of the gas n with unit (mol)', d_type=str)
    narrate(ctx, 'Loaded amount of substance: ' + str(n_raw))
    gas_raw = load(ctx, 'the chemical identity of the gas as a name or formula with state (gas phase)', d_type=str)
    narrate(ctx, 'Gas identity: ' + str(gas_raw) + ' (treated as a perfect gas; identity does not change the ideal gas calculation)')
    volume_raw = load(ctx, 'the volume of the gas sample with unit (dm3, L, cm3 or m3)', d_type=str)
    narrate(ctx, 'Loaded volume: ' + str(volume_raw))
    temp_raw = load(ctx, 'the temperature of the gas with unit (°C or K)', d_type=str)
    narrate(ctx, 'Loaded temperature: ' + str(temp_raw))
    vol_result = do(ctx,
        'Convert a gas sample volume given in dm3, L, cm3 or m3 into cubic metres.',
        payloads={'the volume of the gas sample with unit (dm3, L, cm3 or m3)': volume_raw})
    V_m3 = load(vol_result, 'the volume of the gas sample in cubic metres (m3)', d_type=str, ctx=ctx)
    narrate(ctx, 'Volume in cubic metres: ' + str(V_m3))
    temp_result = do(ctx,
        'Convert a temperature given in degrees Celsius or kelvin into kelvin.',
        payloads={'the temperature of the gas with unit (°C or K)': temp_raw})
    T_K = load(temp_result, 'the temperature of the gas in kelvin (K)', d_type=str, ctx=ctx)
    narrate(ctx, 'Temperature in kelvin: ' + str(T_K))
    R_result = do(ctx,
        'Provide the molar gas constant R in SI units of J K-1 mol-1.',
        payloads={})
    R = load(R_result, 'the molar gas constant R in J K-1 mol-1', d_type=str, ctx=ctx)
    narrate(ctx, 'Molar gas constant R: ' + str(R) + ' J K-1 mol-1')
    pa_result = do(ctx,
        'Compute the pressure in pascals of a perfect gas from pV = nRT given amount, gas constant, kelvin temperature and volume in cubic metres.',
        payloads={
            'the amount of substance of the gas n with unit (mol)': n_raw,
            'the molar gas constant R in J K-1 mol-1': R,
            'the temperature of the gas in kelvin (K)': T_K,
            'the volume of the gas sample in cubic metres (m3)': V_m3,
        })
    p_pa = load(pa_result, 'the perfect gas pressure in pascals (Pa)', d_type=str, ctx=ctx)
    narrate(ctx, 'Perfect gas pressure: ' + str(p_pa) + ' Pa')
    atm_result = do(ctx,
        'Convert a pressure in pascals into atmospheres.',
        payloads={'the perfect gas pressure in pascals (Pa)': p_pa})
    p_atm = load(atm_result, 'the perfect gas pressure in atmospheres (atm)', d_type=str, ctx=ctx)
    narrate(ctx, 'Perfect gas pressure: ' + str(p_atm) + ' atm')
    check_result = do(ctx,
        'Cross-check a perfect gas pressure in atmospheres by recomputing p = nRT/V with R in L atm K-1 mol-1 and volume in litres, then compare it with the primary result.',
        payloads={
            'the amount of substance of the gas n with unit (mol)': n_raw,
            'the temperature of the gas in kelvin (K)': T_K,
            'the volume of the gas sample with unit (dm3, L, cm3 or m3)': volume_raw,
            'the perfect gas pressure in atmospheres (atm)': p_atm,
        })
    agree = load(check_result, 'agreement of the cross-check with the primary pressure', d_type=str, ctx=ctx)
    narrate(ctx, 'Cross-check agreement with primary pressure: ' + str(agree))
    return {
        'the predicted pressure of the perfect gas in atmospheres (atm)': p_atm,
        'the predicted pressure of the perfect gas in pascals (Pa)': p_pa,
    }
