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


