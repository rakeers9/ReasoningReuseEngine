"""IS2 in number + unit form (arc/extraction.py, ARC_EXTRACT_FORM=number_unit)."""

import pytest
from pydantic import BaseModel

from arc import config, extraction, serve
from arc.cache import ParamSpec

Q = "Calculate the monthly installment for a $150k mortgage at 5.5% annual interest compounded monthly, over 25 years."
PARAMS = [ParamSpec("principal", ["$368,000", "$250,000"]), ParamSpec("rate", ["5.875%"]),
          ParamSpec("term", ["30-year"]), ParamSpec("frequency", ["monthly"])]


def v(evidence, number="", unit="", text=""):
    return {"evidence": evidence, "number": number, "unit": unit, "text": text}


@pytest.fixture(autouse=True)
def number_unit(monkeypatch):
    monkeypatch.setattr(config, "EXTRACT_FORM", "number_unit")


def queue(fake_llm, fields):
    class Value(BaseModel):
        evidence: str
        number: str
        unit: str
        text: str
    Out = type("Out", (BaseModel,), {"__annotations__": {f"p{i}": Value for i in range(len(fields))}})
    fake_llm.queue(Out(**{f"p{i}": Value(**f) for i, f in enumerate(fields)}))


def test_the_150k_question_reaches_the_code_as_full_digits(fake_llm):
    queue(fake_llm, [v("$150k", "150000", "$"), v("5.5%", "5.5", "%"), v("25 years", "25", "years"),
                     v("monthly", text="monthly")])
    out = serve.extract_parameters(Q, PARAMS)
    assert out == {"principal": "$150,000", "rate": "5.5%", "term": "25 years", "frequency": "monthly"}
    prompt = fake_llm.calls[0]["prompt"]
    assert "150000 for \"$150k\"" in prompt and "format examples: '$368,000', '$250,000'" in prompt


def test_a_number_that_disagrees_with_its_evidence_is_never_used(fake_llm):
    queue(fake_llm, [v("$150k", "150", "$"), v("5.5%", "5.5", "%"), v("25 years", "25", "years"),
                     v("monthly", text="monthly")])
    with pytest.raises(serve.ExtractionError, match="does not match the evidence"):
        serve.extract_parameters(Q, PARAMS)


def test_a_dropped_zero_is_caught(fake_llm):
    queue(fake_llm, [v("$150k", "15000", "$"), v("5.5%", "5.5", "%"), v("25 years", "25", "years"),
                     v("monthly", text="monthly")])
    with pytest.raises(serve.ExtractionError):
        serve.extract_parameters(Q, PARAMS)


def test_evidence_must_be_in_the_question(fake_llm):
    queue(fake_llm, [v("$150k", "150000", "$"), v("6.5%", "6.5", "%"), v("25 years", "25", "years"),
                     v("monthly", text="monthly")])
    with pytest.raises(serve.ExtractionError, match="not in the question"):
        serve.extract_parameters(Q, PARAMS)


def test_numbers_written_as_words_are_trusted_to_the_model():
    f = v("Ten moles", "10", "mol")
    assert extraction.check_value(f, "Ten moles of ethane ...", allow_implied=True) is None
    assert extraction.render(f, ParamSpec("amount", ["10.0 mol"])) == "10 mol"


def test_implied_values_need_permission():
    f = v("", "27", "°C")
    assert extraction.check_value(f, "a gas at room temperature", allow_implied=True) is None
    assert extraction.check_value(f, "a gas at room temperature", allow_implied=False) == "no evidence for the value"


@pytest.mark.parametrize("field, examples, want", [
    (v("$150k", "150000", "$"), ["$368,000"], "$150,000"),
    (v("€2.0 million", "2000000", "€"), ["€500,000"], "€2,000,000"),
    (v("-£62,164.50", "-62164.50", "£"), ["£1,000.00"], "-£62,164.50"),
    (v("4.860 dm3", "4.860", "dm3"), ["4.860 dm3"], "4.860 dm3"),          # trailing zero kept
    (v("27°C", "27", "°C"), ["27°C"], "27 °C"),
    (v("1.2 × 10^6 Pa", "1200000", "Pa"), ["101325 Pa"], "1200000 Pa"),   # no separators in examples
    (v("12 people", "12", ""), ["8"], "12"),
    (v("methane", text="methane"), ["ethane"], "methane"),
    (v("USD 4.2m", "4200000", "USD"), ["USD 1,000"], "USD 4,200,000"),
])
def test_values_are_rebuilt_in_the_format_examples_style(field, examples, want):
    assert extraction.render(field, ParamSpec("x", examples)) == want


@pytest.mark.parametrize("field, why", [
    (v("12", "twelve"), "not plain digits"),
    (v("x", ""), "no value extracted"),
])
def test_bad_fields_are_rejected(field, why):
    assert why in extraction.check_value(field, "x 12", allow_implied=True)


def test_scientific_and_percent_evidence_check_against_the_stated_number():
    assert extraction.check_value(v("1.2 × 10^6 Pa", "1200000", "Pa"), "p = 1.2 × 10^6 Pa", True) is None
    assert extraction.check_value(v("5.5%", "0.055", "%"), "at 5.5%", True) is not None   # must keep 5.5


def test_form_out_records_fields_and_problems(fake_llm):
    queue(fake_llm, [v("$150k", "150", "$"), v("5.5%", "5.5", "%"), v("25 years", "25", "years"),
                     v("monthly", text="monthly")])
    out = {}
    with pytest.raises(serve.ExtractionError):
        serve.extract_parameters(Q, PARAMS, form_out=out)
    assert out["fields"]["p0"]["number"] == "150" and out["problems"][0].startswith("principal")


def test_extraction_error_is_a_failed_cache_use_so_the_model_answers():
    assert issubclass(serve.ExtractionError, serve.CacheInvocationError)


def test_copy_mode_still_returns_values_as_written(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "EXTRACT_FORM", "copy")
    monkeypatch.setattr(config, "EXPAND_SHORTHAND", False)

    class Two(BaseModel):
        p0: str
        p1: str
    fake_llm.queue(Two(p0="$150k", p1="25 years"))
    assert serve.extract_parameters(Q, PARAMS[:1] + PARAMS[2:3]) == {"principal": "$150k", "term": "25 years"}
