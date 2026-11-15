def semantic_function(ctx):
    """Convert a temperature in degrees Celsius to kelvin by adding 273.15."""
    narrate(ctx, 'Converting the temperature from degrees Celsius to kelvin by adding 273.15')
    t_c = load(ctx, 'temperature in degrees Celsius', d_type=float)
    narrate(ctx, f'Loaded temperature: {t_c} degrees Celsius')
    t_k = t_c + 273.15
    narrate(ctx, f'Absolute temperature: {t_c} + 273.15 = {t_k:.2f} K')
    return {'absolute temperature in kelvin': t_k}