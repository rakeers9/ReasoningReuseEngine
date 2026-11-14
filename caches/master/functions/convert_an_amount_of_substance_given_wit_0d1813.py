def semantic_function(ctx):
    """Convert an amount of substance given with a unit (mol, mmol, kmol, umol, nmol) to moles."""
    value = load(ctx, "the amount value", d_type=float)
    unit = load(ctx, "the amount unit", d_type=str)
    narrate(ctx, f"Converting amount {value} {unit!r} to moles")
    u = str(unit).strip().lower().replace('\u00b5', 'u').replace('\u03bc', 'u').replace(' ', '').rstrip('.')
    factors = {
        'mol': 1.0, 'mole': 1.0, 'moles': 1.0, 'mols': 1.0,
        'mmol': 1e-3, 'millimol': 1e-3, 'millimole': 1e-3, 'millimoles': 1e-3,
        'kmol': 1e3, 'kilomol': 1e3, 'kilomole': 1e3, 'kilomoles': 1e3,
        'umol': 1e-6, 'micromol': 1e-6, 'micromole': 1e-6, 'micromoles': 1e-6,
        'nmol': 1e-9, 'nanomol': 1e-9, 'nanomole': 1e-9, 'nanomoles': 1e-9,
    }
    if u in factors:
        factor = factors[u]
        narrate(ctx, f"Recognised unit {unit!r} directly; conversion factor to mol is {factor}")
    else:
        narrate(ctx, f"Unit {unit!r} not directly recognised; checking whether it begins with a known amount unit (e.g. 'mol C2H6')")
        factor = None
        for key in sorted(factors, key=len, reverse=True):
            if u.startswith(key):
                factor = factors[key]
                narrate(ctx, f"Unit starts with {key!r}; conversion factor to mol is {factor}")
                break
        if factor is None:
            narrate(ctx, f"Asking the language model for the mole equivalent of 1 {unit!r}")
            ans = lm(ctx, f"How many moles is 1 '{unit}' (a unit of amount of substance)? Answer with only a number.")
            factor = smart_convert(ctx, ans, float)
            narrate(ctx, f"Language model gives conversion factor {factor}")
    moles = float(value) * factor
    narrate(ctx, f"Amount in moles: {value} x {factor} = {moles} mol")
    return {"the amount in moles": moles}
