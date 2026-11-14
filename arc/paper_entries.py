"""The paper's own published cache entries, typed in by hand.

Used in Step 1 to test our runtime independently of our builder: if these
functions reproduce the paper's printed output, the runtime is right.

Sources:
  gas_law: Figure 5 (Kelvin utility), Figure 6 (root). The pressure-conversion
           utility is described but not shown; it is written here from its
           description and from the compiled form in Figure 7.
  thermo:  Figure 11 (process classifier), Figure 12 (root).
Where a figure says "... omitted ..." (extra narrate() calls, extra outputs),
the missing lines are filled in to match the narration traces in Figures 8/14.
The entry-point templates are ours: the paper does not print them.
"""

from __future__ import annotations

from .cache import Cache, EntryPoint, ParamSpec

# ----------------------------------------------------------------------------
# Appendix A: ideal gas law
# ----------------------------------------------------------------------------

GAS_KELVIN = '''
def semantic_function(ctx):
    """Convert a temperature value from Celsius
    or Kelvin to Kelvin."""

    temp_with_unit = load(ctx,
        "the temperature value with unit",
        d_type=str)
    narrate(ctx,
        f"Received temperature specification: {temp_with_unit}")

    temp_value = extract(ctx, temp_with_unit,
        "the numeric temperature value",
        d_type=float)
    narrate(ctx,
        f"Extracted numeric temperature value: {temp_value}")

    temp_unit = lm(ctx,
        f"What is the temperature unit in "
        f"'{temp_with_unit}'? "
        f"Reply with exactly one of: C, K")
    temp_unit = temp_unit.strip().upper()
    narrate(ctx,
        f"Detected temperature unit: {temp_unit}")

    if "C" in temp_unit:
        temp_kelvin = temp_value + 273
        narrate(ctx,
            f"{temp_value} °C + 273 = {temp_kelvin} K")
    else:
        temp_kelvin = temp_value
        narrate(ctx,
            f"Already in Kelvin: {temp_kelvin} K")

    return {
        "the temperature in Kelvin": temp_kelvin
    }
'''

GAS_PRESSURE = '''
def semantic_function(ctx):
    """Convert a pressure value from kPa to multiple
    common units including MPa, bar, and atm."""

    pressure_kpa = load(ctx,
        "the pressure value in kilopascals (kPa)",
        d_type=float)

    pressure_mpa = pressure_kpa / 1000
    narrate(ctx, f"{pressure_kpa:.2f} kPa / 1000 = {pressure_mpa:.2f} MPa")
    pressure_bar = pressure_kpa / 100
    narrate(ctx, f"{pressure_kpa:.2f} kPa / 100 = {pressure_bar:.2f} bar")
    pressure_atm = pressure_kpa / 101.325
    narrate(ctx, f"{pressure_kpa:.2f} kPa / 101.325 = {pressure_atm:.2f} atm")

    return {
        "the pressure in kPa": pressure_kpa,
        "the pressure in MPa": pressure_mpa,
        "the pressure in bar": pressure_bar,
        "the pressure in atm": pressure_atm,
    }
'''

GAS_ROOT = '''
def semantic_function(ctx):
    """Calculate the pressure exerted by a gas
    using PV = nRT, rearranged to P = nRT/V."""

    molar_amount_str = load(ctx,
        "the molar amount of gas with unit",
        d_type=str)
    volume_str = load(ctx,
        "the volume of the container with unit",
        d_type=str)
    temperature_str = load(ctx,
        "the temperature with unit", d_type=str)
    gas_name = load(ctx,
        "the common name of the gas", d_type=str)
    gas_model = load(ctx,
        "the gas model or assumption to use",
        d_type=str)

    narrate(ctx, f"Calculating pressure for "
        f"{gas_name} using the {gas_model} law")
    narrate(ctx, f"Given: n = {molar_amount_str}, "
        f"V = {volume_str}, T = {temperature_str}")

    n = extract(ctx, molar_amount_str,
        "the numeric molar amount", d_type=float)
    narrate(ctx, f"Extracted amount of substance: n = {n} mol")
    volume_value = extract(ctx, volume_str,
        "the numeric volume value", d_type=float)
    narrate(ctx, f"Extracted volume: V = {volume_value} dm^3")

    # Convert temperature via utility function
    narrate(ctx, "Converting temperature...")
    temp_result = do(ctx,
        "Convert a temperature value from Celsius "
        "or Kelvin to Kelvin given a temperature "
        "string with unit",
        payloads={
            "the temperature value with unit":
                temperature_str})
    T = load(temp_result,
        "the temperature in Kelvin",
        d_type=float, ctx=ctx)
    narrate(ctx, f"Temperature in Kelvin: T = {T} K")

    R = 8.314 # kPa.dm^3.mol^-1.K^-1
    narrate(ctx, "Using R = 8.314 kPa·dm^3·mol^-1·K^-1")
    narrate(ctx, f"P = nRT/V = {n} × {R} × {T}"
        f" / {volume_value}")

    pressure_kpa = n * R * T / volume_value
    narrate(ctx, f"P = {pressure_kpa:.2f} kPa")

    # Convert to other units via utility function
    narrate(ctx, "Converting pressure to other units...")
    pressure_result = do(ctx,
        "Convert a pressure value from kPa to "
        "multiple common units including MPa, "
        "bar, and atm",
        payloads={
            "the pressure value in kilopascals "
            "(kPa)": pressure_kpa})

    pressure_mpa = load(pressure_result,
        "the pressure in MPa",
        d_type=float, ctx=ctx)
    pressure_bar = load(pressure_result,
        "the pressure in bar", d_type=float, ctx=ctx)
    pressure_atm = load(pressure_result,
        "the pressure in atm", d_type=float, ctx=ctx)
    narrate(ctx, f"Final: {pressure_kpa:.2f} kPa = {pressure_mpa:.2f} MPa"
        f" = {pressure_bar:.2f} bar = {pressure_atm:.2f} atm")

    return {
        "the calculated pressure in kPa":
            pressure_kpa,
        "the calculated pressure in MPa":
            pressure_mpa,
        "the calculated pressure in bar": pressure_bar,
        "the calculated pressure in atm": pressure_atm,
        "summary of the calculation":
            f"The pressure exerted by {n} mol of {gas_name} in {volume_value} dm^3 "
            f"at {T} K is {pressure_mpa:.2f} MPa (or {pressure_kpa / 1000:.2f} x 10^3 kPa)",
    }
'''

GAS_PARAMS = [
    ParamSpec("the molar amount of gas with unit", ["10.0 mol", "2.5 mol"], "10.0 mol", "> 0 mol"),
    ParamSpec("the volume of the container with unit", ["4.860 dm3", "12 L"], "4.860 dm3", "> 0"),
    ParamSpec("the temperature with unit", ["27°C", "300 K"], "27°C", "above absolute zero"),
    ParamSpec("the common name of the gas", ["ethane", "nitrogen"], "ethane", "any gas"),
    ParamSpec("the gas model or assumption to use", ["perfect gas", "ideal gas"], "perfect gas",
              "perfect/ideal gas"),
]
GAS_TEMPLATE = ("Suppose that ●1 of ●4 is confined to ●2 at ●3. "
                "Predict the pressure exerted by the ●4 from the ●5.")


# ----------------------------------------------------------------------------
# Appendix B: thermodynamics (constant-volume work)
# ----------------------------------------------------------------------------

THERMO_CLASSIFY = '''
def semantic_function(ctx):
    """Classify the type of thermodynamic process
    based on constraints on state variables."""

    # Load the volume constraint
    volume_constraint = load(ctx,
        "constraint on the volume during the process",
        d_type=str)
    narrate(ctx, f"Volume constraint specified: "
           f"'{volume_constraint}'")

    # Load temperature values
    initial_temp = load(ctx,
        "initial temperature of the gas with units",
        d_type=str)
    final_temp = load(ctx,
        "final temperature of the gas with units",
        d_type=str)

    # Determine if volume is constant
    volume_constant_keywords = [
        "fixed", "constant", "unchanging", "unchanged"]
    is_volume_constant = any(
        kw in volume_constraint.lower()
        for kw in volume_constant_keywords)
    narrate(ctx, f"Volume is constant: {is_volume_constant}")

    # Extract numeric temperatures to check if isothermal
    initial_temp_value = extract(ctx, initial_temp,
        "the numeric temperature value, "
        "e.g. 300 or 273.15", d_type=float)
    final_temp_value = extract(ctx, final_temp,
        "the numeric temperature value, "
        "e.g. 341 or 350", d_type=float)
    is_temperature_constant = (
        abs(initial_temp_value - final_temp_value) < 0.01)

    # Classify the process type
    if is_volume_constant:
        process_type = "isochoric"
    elif is_temperature_constant:
        process_type = "isothermal"
    else:
        process_type = "general"
    narrate(ctx, f"Process classified as {process_type.upper()}")

    return {
        "process type": process_type,
        "is volume constant": is_volume_constant,
        "is temperature constant": is_temperature_constant,
    }
'''

THERMO_ROOT = '''
def semantic_function(ctx):
    """Calculate work done by/on a gas during a
    thermodynamic process given system parameters
    and process constraints."""

    narrate(ctx, "Loading thermodynamic system parameters...")
    amount = load(ctx,
        "amount of substance of the gas sample with units",
        d_type=str)
    narrate(ctx, f"Amount of substance: {amount}")
    gas_identity = load(ctx,
        "chemical identity of the gas", d_type=str)
    narrate(ctx, f"Gas identity: {gas_identity}")
    volume_constraint = load(ctx,
        "constraint on the volume during the process",
        d_type=str)
    narrate(ctx, f"Volume constraint: {volume_constraint}")
    volume = load(ctx,
        "volume of the gas sample with units", d_type=str)
    narrate(ctx, f"Volume: {volume}")
    initial_temp = load(ctx,
        "initial temperature of the gas with units",
        d_type=str)
    narrate(ctx, f"Initial temperature: {initial_temp}")
    final_temp = load(ctx,
        "final temperature of the gas with units",
        d_type=str)
    narrate(ctx, f"Final temperature: {final_temp}")
    heat_supplied = load(ctx,
        "heat energy supplied to the system with units",
        d_type=str)
    narrate(ctx, f"Heat energy supplied: {heat_supplied}")
    equation_of_state = load(ctx,
        "equation of state governing the gas behavior",
        d_type=str)
    narrate(ctx, f"Equation of state: {equation_of_state}")
    quantity_to_calculate = load(ctx,
        "thermodynamic quantity to be calculated",
        d_type=str)
    narrate(ctx, f"Quantity to calculate: {quantity_to_calculate}")

    # Classify the thermodynamic process
    narrate(ctx, "Classifying the thermodynamic process "
            "based on constraints...")
    classification_result = do(ctx,
        "Classify the thermodynamic process type "
        "based on volume and temperature constraints,"
        " determining if the process is isochoric, "
        "isothermal, or general",
        payloads={
            "constraint on the volume during the "
            "process": volume_constraint,
            "initial temperature of the gas with "
            "units": initial_temp,
            "final temperature of the gas with "
            "units": final_temp,
        })

    process_type = load(classification_result,
        "process type", d_type=str, ctx=ctx)
    is_volume_constant = load(classification_result,
        "is volume constant", d_type=bool, ctx=ctx)
    narrate(ctx, f"Process classification: {process_type}, "
            f"volume constant: {is_volume_constant}")

    # Calculate work based on process type
    if process_type == "isochoric" or is_volume_constant:
        narrate(ctx, f"Volume is {volume_constraint} "
                f"(constant), dV = 0")
        narrate(ctx, "w = -integral(P_ext * 0) = 0")
        work_value = 0
        work_with_units = "0"
        narrate(ctx, f"Result is independent of "
                f"{equation_of_state}")
        explanation = (
            "For an isochoric (constant volume) "
            "process, w = 0 because dV = 0. "
            "This result is independent of the "
            f"equation of state ({equation_of_state}).")
    else:
        work_value = None
        work_with_units = "requires further calculation"
        explanation = (
            f"For a {process_type} process, work "
            "must be calculated by integration.")

    narrate(ctx, f"Final result: work w = {work_value}")
    return {
        "work value": work_value,
        "work with units": work_with_units,
        "explanation": explanation,
    }
'''

THERMO_PARAMS = [
    ParamSpec("amount of substance of the gas sample with units", ["2.0 mol"], "2.0 mol", "> 0"),
    ParamSpec("chemical identity of the gas", ["CO2", "N2"], "CO2", "any gas"),
    ParamSpec("constraint on the volume during the process", ["fixed", "constant", "expands"],
              "fixed", "a description of how the volume behaves"),
    ParamSpec("volume of the gas sample with units", ["15.0 dm3"], "15.0 dm3", "> 0"),
    ParamSpec("initial temperature of the gas with units", ["300 K"], "300 K", "above absolute zero"),
    ParamSpec("final temperature of the gas with units", ["341 K"], "341 K", "above absolute zero"),
    ParamSpec("heat energy supplied to the system with units", ["2.35 kJ"], "2.35 kJ", "any"),
    ParamSpec("equation of state governing the gas behavior", ["van der Waals", "perfect gas"],
              "van der Waals", "any equation of state"),
    ParamSpec("thermodynamic quantity to be calculated", ["w", "work"], "w", "w"),
]
THERMO_TEMPLATE = (
    "Assume all gases are perfect unless stated otherwise. A sample consisting of ●1 ●2 occupies "
    "a ●3 volume of ●4 at ●5. When it is supplied with ●7 of energy as heat its temperature "
    "increases to ●6. Assume that ●2 is described by the ●8 equation of state, and calculate ●9."
)


def build_paper_cache(problem: dict) -> Cache:
    """A cache holding the paper's hand-typed entry for one problem."""
    cache = Cache(f"paper_{problem['id']}")
    if problem["id"] == "gas_law":
        cache.add_function("Convert a temperature value from Celsius or Kelvin to Kelvin.",
                           GAS_KELVIN, ["the temperature value with unit"],
                           ["the temperature in Kelvin"], origin="paper Figure 5a")
        cache.add_function("Convert a pressure value from kPa to multiple common units "
                           "including MPa, bar, and atm.", GAS_PRESSURE,
                           ["the pressure value in kilopascals (kPa)"],
                           ["the pressure in MPa", "the pressure in bar", "the pressure in atm"],
                           origin="paper (described, not printed)")
        root = cache.add_function("Calculate the pressure exerted by a gas using PV = nRT, "
                                  "rearranged to P = nRT/V.", GAS_ROOT,
                                  [p.description for p in GAS_PARAMS],
                                  ["the calculated pressure in kPa"], origin="paper Figure 6")
        params, template = GAS_PARAMS, GAS_TEMPLATE
    elif problem["id"] == "thermo":
        cache.add_function("Classify the type of thermodynamic process based on constraints "
                           "on state variables.", THERMO_CLASSIFY,
                           ["constraint on the volume during the process"],
                           ["process type"], origin="paper Figure 11a")
        root = cache.add_function("Calculate work done by/on a gas during a thermodynamic "
                                  "process given system parameters and process constraints.",
                                  THERMO_ROOT, [p.description for p in THERMO_PARAMS],
                                  ["work value"], origin="paper Figure 12")
        params, template = THERMO_PARAMS, THERMO_TEMPLATE
    else:
        raise ValueError(f"the paper publishes no cache entry for {problem['id']!r}")
    cache.add_entry_point(EntryPoint(template, "template", root.id, params, problem["id"]))
    cache.add_entry_point(EntryPoint(problem["question"], "original", root.id, params, problem["id"]))
    return cache
