def semantic_function(ctx):
    """Split a physical quantity string into its numeric value and its unit string."""
    import re
    text = load(ctx, "the quantity string with value and unit", d_type=str)
    narrate(ctx, f"Parsing the quantity string: {text!r}")
    s = str(text).strip()
    s_norm = s.replace('\u2212', '-').replace('\u00d7', 'x').replace('\u2009', ' ').replace('\u00a0', ' ')
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*([-+]?\d+))?\s*(.*)$", s_norm)
    if m:
        value = float(m.group(1))
        if m.group(2) is not None:
            value *= 10 ** int(m.group(2))
        unit = m.group(3).strip()
    else:
        num_txt = extract(ctx, s, "the numeric value of this physical quantity", d_type=str)
        value = smart_convert(ctx, num_txt, float)
        unit = extract(ctx, s, "the unit of this physical quantity (without the number)", d_type=str)
        unit = str(unit).strip()
    narrate(ctx, f"Numeric value = {value}, unit = {unit!r}")
    return {"the numeric value": value, "the unit string": unit}
