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


# ---- output_format and extras (what the question asks of the answer) ----

def test_the_prompt_asks_for_output_format_and_extras_and_excludes_assumptions(fake_llm):
    queue(fake_llm, [v("$150k", "150000", "$"), v("5.5%", "5.5", "%"), v("25 years", "25", "years"),
                     v("monthly", text="monthly")])
    serve.extract_parameters(Q, PARAMS)
    prompt = fake_llm.calls[0]["prompt"]
    assert "output_format" in prompt and "extras" in prompt and '"dp:N"' in prompt
    assert "assuming ideal behaviour" in prompt and "are NOT extras" in prompt


def test_rounding_request_is_accepted_when_it_is_in_the_question():
    q = "What is the payment? Round to the nearest dollar."
    assert extraction.check_answer_form({"unit": "", "rounding": "nearest:1", "evidence": "Round to the nearest dollar"},
                                        [], q) == []


def test_output_format_evidence_must_be_in_the_question():
    probs = extraction.check_answer_form({"unit": "kPa", "rounding": "", "evidence": "in kilopascals"}, [], "Find p.")
    assert probs and "not in the question" in probs[0]


def test_a_rounding_the_formatter_cannot_apply_declines():
    assert extraction.check_answer_form({"unit": "", "rounding": "to a nice number", "evidence": ""}, [], "q")


def test_any_extra_instruction_makes_the_cache_decline():
    probs = extraction.check_answer_form({}, [{"kind": "explanation", "text": "explain your reasoning",
                                               "evidence": "explain your reasoning"}], "q, explain your reasoning")
    assert probs == ["the question also asks: 'explain your reasoning' (explanation)"]


def test_extras_raise_an_extraction_error_end_to_end(fake_llm):
    from typing import Literal

    class Value(BaseModel):
        evidence: str
        number: str
        unit: str
        text: str

    class Fmt(BaseModel):
        unit: str
        rounding: str
        evidence: str

    class Extra(BaseModel):
        kind: Literal["explanation", "format", "other"]
        text: str
        evidence: str
    fields = [v("$150k", "150000", "$"), v("5.5%", "5.5", "%"), v("25 years", "25", "years"), v("monthly", text="monthly")]
    Out = type("Out", (BaseModel,), {"__annotations__": {**{f"p{i}": Value for i in range(4)},
                                                          "output_format": Fmt, "extras": list[Extra]}})
    fake_llm.queue(Out(**{f"p{i}": Value(**f) for i, f in enumerate(fields)},
                       output_format=Fmt(unit="", rounding="", evidence=""),
                       extras=[Extra(kind="format", text="as a fraction", evidence="25 years")]))
    out = {}
    with pytest.raises(serve.ExtractionError, match="also asks"):
        serve.extract_parameters(Q, PARAMS, form_out=out)
    assert out["extras"][0]["text"] == "as a fraction"


# ---- a scale word written into the unit (real Haiku output, builder_compare 2026-11-15) ----

@pytest.mark.parametrize("field, number, unit", [
    (v("A machine costs $0.42 million", "0.42", "$ million"), "420000", "$"),
    (v("$150k", "150", "$k"), "150000", "$"),
    (v("€2.0M", "2.0", "€ M"), "2000000", "€"),
    (v("£1.25bn", "1.25", "£bn"), "1250000000", "£"),
    (v("2.5 million people", "2.5", "million people"), "2500000", "people"),
    (v("3 thousand units", "3", "thousand units"), "3000", "units"),
])
def test_a_scale_word_in_the_unit_is_folded_into_the_number(field, number, unit):
    folded = extraction.fold_scale(field)
    assert extraction._decimal(folded["number"]) == extraction._decimal(number) and folded["unit"] == unit
    assert extraction.check_value(folded, field["evidence"] + " ...", allow_implied=True) is None


@pytest.mark.parametrize("unit", ["m", "M", "mm", "km", "K", "kg", "mol", "years", "%", "$"])
def test_real_units_are_never_treated_as_scale(unit):
    f = v("x", "4.2", unit)
    assert extraction.fold_scale(f) == f


def test_the_million_case_now_reaches_the_code_as_full_digits(fake_llm):
    params = [ParamSpec("cost", ["$80,000"])]
    queue(fake_llm, [v("A machine costs $0.42 million", "0.42", "$ million")])
    assert serve.extract_parameters("A machine costs $0.42 million, ...", params) == {"cost": "$420,000"}


def test_the_answer_unit_defaults_to_the_unit_of_same_kind_inputs():
    # hidden run 2026-11-16: a final temperature asked with °C inputs was served in K
    from arc import extraction
    assert "the reader expects the answer in the units they used" in extraction.ANSWER_RULES
    q = "A 260 g piece of titanium at 80 °C is dropped into 410 g of water at 21.5 °C. Final temperature?"
    fmt = {"unit": "°C", "rounding": "", "evidence": "80 °C"}       # evidence is an input, as written
    assert extraction.check_answer_form(fmt, [], q) == []


def test_an_answer_unit_from_the_inputs_that_differs_from_the_entry_goes_to_the_model():
    from arc import present
    spec = {"equilibrium temperature": {"kind": "measure", "unit": "°C"}}
    with pytest.raises(present.PresentError):
        present.apply({"equilibrium temperature": "35.57"}, {"unit": "K", "rounding": "", "evidence": "418.15 K"}, spec)
    assert present.apply({"equilibrium temperature": "35.5712"}, {"unit": "°C", "rounding": "", "evidence": "80 °C"},
                         spec) == {"equilibrium temperature": "35.57"}
