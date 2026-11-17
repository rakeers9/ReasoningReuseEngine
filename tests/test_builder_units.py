"""arc/builder.py units="strict" (DEVIATION, label-free builds): input, money and precision
rules in the builder's instructions. Owner decision 2026-11-15: no unit library; the builder
writes its own conversions, and the hand-typed-factor scan is off unless
config.BUILDER_FACTOR_SCAN is on. units="paper" (the default) must be exactly the paper builder."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from arc import builder, judges
from arc.source import Transcript

PROBLEM = {"id": "p_units", "question": "Convert 25 °C to kelvin.", "golden_label": ""}
T = Transcript("p_units", PROBLEM["question"], "T = 25 + 273.15 = 298.15 K", "", "m", 0.0)
SPEC = builder.PlannedFunction(id="f1", description="Convert a temperature to kelvin",
                               inputs=["the temperature with unit"], outputs=["temperature in K"],
                               calls=[], transcript_fragment="T = 25 + 273.15")
PARAMS = builder.ParameterRecognition(problem_class="c", template="Convert <t> to kelvin.", parameters=[])


def candidate(code):
    return builder.Candidate(reuse_existing_function_id=None, description="Convert a temperature to kelvin",
                             code=code, test_payloads=[builder.Payload(description="the temperature with unit",
                                                                       value="'25 °C'")],
                             expected_result="298.15")


TYPED = ("def semantic_function(ctx):\n    t = load(ctx, 'the temperature in °C', d_type=float)\n"
         "    narrate(ctx, 'adding the offset')\n    return {'temperature in K': t + 273.15}\n")
CLEAN = ("def semantic_function(ctx):\n    t = load(ctx, 'the temperature in °C', d_type=float)\n"
         "    narrate(ctx, 'converting')\n    return {'temperature in K': convert_unit(t, '°C', 'K')}\n")


def test_paper_mode_is_the_paper_builder_byte_for_byte():
    b = builder.Builder(PROBLEM, T, use_golden=False)
    assert b.units == "paper" and b._system[0]["text"] == builder.BUILDER_SYSTEM
    assert "raw text copied from the question" not in builder.BUILDER_SYSTEM
    assert builder.BUILDER_SYSTEM_STRICT.startswith(builder.BUILDER_SYSTEM)


def test_strict_mode_adds_the_input_money_and_precision_rules():
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    text = b._system[0]["text"]
    assert text == builder.BUILDER_SYSTEM + builder.STRICT_UNITS_RULES
    for rule in ("raw text copied from the question", "Convert units yourself", "decimal.Decimal",
                 "Return full precision", "Never round until the end", "Name each output with its unit"):
        assert rule in text
    # owner decision: no unit library to call
    assert "convert_unit" not in builder.STRICT_UNITS_RULES and "substance=" not in builder.STRICT_UNITS_RULES


def test_unknown_units_mode_fails_loudly():
    with pytest.raises(ValueError):
        builder.Builder(PROBLEM, T, units="loose")


@pytest.fixture
def judge_passes(monkeypatch):
    monkeypatch.setattr(judges, "judge", lambda *a, **k: SimpleNamespace(passed=True, critique="",
                                                                            model_dump=lambda: {}))


def test_factor_scan_is_off_by_default_in_strict_mode(fake_llm, judge_passes, monkeypatch):
    from arc import config
    assert config.BUILDER_FACTOR_SCAN is False
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    monkeypatch.setattr(b, "_run_candidate", lambda code, payloads: ({"temperature in K": 298.15}, "ok", None))
    fake_llm.queue(candidate(TYPED))
    fid = b.implement(SPEC, PARAMS, is_root=False)
    assert b.cache.functions[fid].source == TYPED and len(b.report.attempts) == 1


def test_factor_scan_flag_sends_a_typed_factor_back(fake_llm, judge_passes, monkeypatch):
    from arc import config
    monkeypatch.setattr(config, "BUILDER_FACTOR_SCAN", True)
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    monkeypatch.setattr(b, "_run_candidate", lambda code, payloads: ({"temperature in K": 298.15}, "ok", None))
    fake_llm.queue(candidate(TYPED))
    fake_llm.queue(candidate(CLEAN))
    fid = b.implement(SPEC, PARAMS, is_root=False)
    assert b.cache.functions[fid].source == CLEAN
    first, second = b.report.attempts
    assert not first.passed and first.failed_at == "static" and "273.15" in first.feedback
    assert second.passed


def test_paper_mode_does_not_scan_for_factors(fake_llm, judge_passes, monkeypatch):
    b = builder.Builder(PROBLEM, T, use_golden=False)
    monkeypatch.setattr(b, "_run_candidate", lambda code, payloads: ({"temperature in K": 298.15}, "ok", None))
    fake_llm.queue(candidate(TYPED))
    fid = b.implement(SPEC, PARAMS, is_root=False)
    assert b.cache.functions[fid].source == TYPED and len(b.report.attempts) == 1


def test_strict_rules_say_inputs_arrive_as_raw_text():
    rules = builder.STRICT_UNITS_RULES
    for phrase in ('"$150k"', '"$0.42 million"', '"4.860 dm3"', '"27°C"', '"ten moles"',
                   "parse it yourself", "10**3, 10**6, 10**9", "raise an error for text you can't parse"):
        assert phrase in rules


RAW_PARSER = """def semantic_function(ctx):
    import re
    from decimal import Decimal
    raw = load(ctx, 'the loan principal as written', d_type=str)
    narrate(ctx, f'parsing {raw}')
    m = re.search(r'([0-9][0-9,]*\\.?[0-9]*)\\s*(k|thousand|million|bn|billion)?', raw, re.I)
    if not m:
        raise ValueError(f'cannot parse {raw!r}')
    scale = {'k': 10**3, 'thousand': 10**3, 'million': 10**6, 'bn': 10**9, 'billion': 10**9}
    number = Decimal(m.group(1).replace(',', ''))
    suffix = (m.group(2) or '').lower()
    narrate(ctx, f'scale {suffix}')
    return {'principal in dollars': number * scale.get(suffix, 1)}
"""


@pytest.mark.parametrize("raw,want", [("$150k", 150000), ("$0.42 million", 420000), ("$350,000", 350000),
                                      ("$1.25 bn", 1250000000)])
def test_a_raw_text_parser_in_the_strict_style_runs_and_passes_the_scan(raw, want):
    from decimal import Decimal
    from arc import sandbox
    from arc.cache import Cache
    from arc.runtime import PRIMITIVES, Context
    assert builder.typed_factors(RAW_PARSER) == [] and builder.Builder._static_checks(RAW_PARSER) is None
    fn = sandbox.load_function(RAW_PARSER, PRIMITIVES)
    ctx = Context(Cache("t"))
    ctx.set("the loan principal as written", raw)
    assert fn(ctx)["principal in dollars"] == Decimal(want)




# Hidden-set run 2026-11-16: a calorimetry entry answered "equilibrium temperature in K" when
# every temperature in the question was in °C. Rule: an output of the same kind as an input
# comes back in that input's unit (the first one if they differ), unless the question asks for
# another unit, converted back at the end in plain Python; the outputs writer records that unit.

def test_strict_rules_return_an_output_in_its_inputs_unit():
    rules = builder.STRICT_UNITS_RULES
    for phrase in ("If the question asks for the answer in a unit, return it in that unit",
                   "same kind of quantity as an input", "the unit of the first one",
                   "convert the\n  result back at the end of your code in plain Python",
                   "converted back to °C", "put it in the output's name"):
        assert phrase in rules
    # the output-unit rule comes before the naming rule it qualifies
    assert rules.index("Output units.") < rules.index("Name each output with its unit")


def test_output_unit_rule_reaches_only_strict_builds():
    from arc.source import Transcript
    t = Transcript("p", "q", "s", "", "m", 0.0)
    problem = {"id": "p", "question": "q", "golden_label": ""}
    assert "same kind of quantity as an input" not in builder.Builder(problem, t, use_golden=False)._system[0]["text"]
    strict = builder.Builder(problem, t, use_golden=False, units="strict")._system[0]["text"]
    assert "same kind of quantity as an input" in strict


def test_outputs_writer_records_the_unit_the_code_converts_back_to(fake_llm):
    from arc import labelfree
    from arc.cache import Cache, EntryPoint
    from arc.cache import ParamSpec
    c = Cache("fam_cal")
    c.add_function("mix water", "def semantic_function(ctx):\n"
                                "    raw = load(ctx, 'hot water temperature', d_type=str)\n"
                                "    k = float(raw.split()[0]) + 273.15\n"
                                "    return {'equilibrium temperature in °C': k - 273.15}\n")
    c.entry_points = [EntryPoint("q", "original", list(c.functions)[0],
                                 [ParamSpec("hot water temperature", original_value="80 °C")], "fam_cal")]
    fake_llm.queue(labelfree.OutputSpecs(outputs=[
        labelfree.OutputItem(name="equilibrium temperature in °C", kind="measure", unit="°C", decimals=None)]))
    specs = labelfree.write_outputs(c, "fam_cal", Transcript("fam_cal", "Mix water at 80 °C ...", "w", "", "m", 0.0))
    assert specs == {"equilibrium temperature in °C": {"kind": "measure", "unit": "°C", "decimals": None}}
    prompt = fake_llm.calls[0]["prompt"]
    assert "read from the code" in prompt and "the unit is the one it converts back to" in prompt


# ----------------------------------------------------------------------------- as-written parameters

# Regression (hidden-set run 2026-11-16, fam_f0003): these parameter descriptions (Haiku, Sonnet
# and Opus builds) asked for fixed units or forms, so extraction converted "144 months" to
# "8 years" and "eight percent" to "8% APR", and the extraction checks refused 4 of 5 family
# questions. Rule (units="strict"): every description asks for the value as written.
F0003_DESCRIPTIONS = [
    "the investment horizon, meaning the length of time from the start to the evaluation date, "
    "as a duration in years",
    "the nominal annual interest rate of the account with its unit, as a percentage per annum",
    "the starting balance of the account at time zero in dollars",
    "the nominal annual percentage rate (APR) compounded monthly",
    "the starting balance in dollars",
    "the starting account balance with currency",
]
AS_WRITTEN = [
    "the loan term as written, e.g. '144 months' or '12 years'",
    "the annual interest rate as written, e.g. '8%' or 'eight percent APR'",
    "the starting balance as written, with its currency, e.g. '$50k' or '50,000 dollars'",
    "the gas temperature as written, e.g. '27 °C' or 'given in kelvin: 300 K'",   # quoted text is not checked
]


def _param(desc, value="12 years"):
    return builder.Parameter(marker="●1", description=desc, format_examples=["12 years"],
                             original_value=value, validity_range="> 0")


@pytest.mark.parametrize("desc", F0003_DESCRIPTIONS)
def test_f0003_descriptions_break_the_as_written_rule(desc):
    [problem] = builder.fixed_form_problems([_param(desc)])
    assert repr(desc) in problem


@pytest.mark.parametrize("desc", AS_WRITTEN)
def test_as_written_descriptions_pass(desc):
    assert builder.fixed_form_problems([_param(desc)]) == []


@pytest.mark.parametrize("desc", ["the loan term as written, in years", "the loan term as written in months",
                                  "the rate as written, expressed as a decimal",
                                  "the balance as written, converted to dollars",
                                  "the term as written, as a number of years"])
def test_as_written_with_a_fixed_unit_still_fails(desc):
    assert "fixed unit or form" in builder.fixed_form_problems([_param(desc)])[0]


def test_strict_rules_ask_for_parameters_as_written():
    rules = builder.STRICT_UNITS_RULES
    for phrase in ("AS WRITTEN", "the loan term as written, e.g. '144 months' or\n  '12 years'",
                   "not\n  \"the loan term in years\"", 'contains\n  the words "as written"',
                   "your code parses and converts it"):
        assert phrase in rules


def _recognition(*descs):
    return builder.ParameterRecognition(problem_class="c", template="t ●1",
                                        parameters=[_param(d) for d in descs])


def test_strict_phase3_retries_once_with_the_reasons_then_keeps_the_fix(fake_llm):
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    fake_llm.queue(_recognition(F0003_DESCRIPTIONS[0]))
    fake_llm.queue(_recognition(AS_WRITTEN[0]))
    out = b.phase3()
    assert [p.description for p in out.parameters] == [AS_WRITTEN[0]]
    assert len(fake_llm.calls) == 2 and fake_llm.calls[1]["tag"].endswith(":phase3:as_written")
    retry = fake_llm.calls[1]["prompt"][-1]["text"]
    assert fake_llm.calls[0]["prompt"][-1]["text"] in retry                # same instructions, plus why
    assert "as a duration in" in retry
    assert b.report.phases["Phase 3 (parameter recognition)"]["parameters"][0]["description"] == AS_WRITTEN[0]


def test_strict_phase3_failing_twice_is_a_build_error(fake_llm):
    # BuildError is a code failure, so the label-free cascade may try the next builder
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    fake_llm.queue(_recognition(F0003_DESCRIPTIONS[2]))
    fake_llm.queue(_recognition(F0003_DESCRIPTIONS[4]))
    with pytest.raises(builder.BuildError, match="as-written rule"):
        b.phase3()
    assert len(fake_llm.calls) == 2


def test_strict_phase3_that_passes_costs_one_call(fake_llm):
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    fake_llm.queue(_recognition(*AS_WRITTEN[:2]))
    assert len(b.phase3().parameters) == 2 and len(fake_llm.calls) == 1


def test_paper_phase3_is_unchecked_and_unchanged(fake_llm):
    b = builder.Builder(PROBLEM, T, use_golden=False)
    fake_llm.queue(_recognition(F0003_DESCRIPTIONS[0]))
    out = b.phase3()
    assert out.parameters[0].description == F0003_DESCRIPTIONS[0] and len(fake_llm.calls) == 1
    text = fake_llm.calls[0]["prompt"][-1]["text"]
    assert text.endswith("value in the original input copied verbatim, and an analysis of its logical "
                         "validity range.") and "as written" not in text


def test_strict_repair_that_breaks_the_rule_is_a_build_error(fake_llm):
    b = builder.Builder(PROBLEM, T, use_golden=False, units="strict")
    root = b.cache.add_function("root", CLEAN).id
    fake_llm.queue(builder.Repair(parameters=[_param(F0003_DESCRIPTIONS[1])], root_code=None, explanation="x"))
    with pytest.raises(builder.BuildError, match="repaired parameter descriptions"):
        b.repair(root, _recognition(AS_WRITTEN[0]), "failed")
