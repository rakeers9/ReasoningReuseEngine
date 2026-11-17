"""IS2 in evidence form (arc/extraction.extract_evidence, ARC_EXTRACT_FORM=evidence)."""

from typing import Literal

import pytest
from pydantic import BaseModel

from arc import config, extraction, serve
from arc.cache import ParamSpec

Q = ("Using double-declining balance, what is the book value after 3 years of a machine that cost $50k, "
     "has a 10-year useful life and a $5k salvage value?")
EXAMPLE = ("Using the double-declining-balance method, find the book value after 8 years of an asset that "
           "cost $379,000, has an 11-year life and a $41,000 salvage value.")
PARAMS = [ParamSpec("years elapsed", ["8 years"], original_value="8 years"),
          ParamSpec("cost", ["$379,000"], original_value="$379,000"),
          ParamSpec("useful life", ["11-year life"], original_value="11-year life"),
          ParamSpec("salvage", ["$41,000"], original_value="$41,000")]
GOOD = [("after 3 years", "3 years"), ("cost $50k", "$50k"), ("a 10-year useful life", "10-year useful life"),
        ("a $5k salvage value", "$5k")]


class Slot(BaseModel):
    evidence: str
    value: str


class Fmt(BaseModel):
    unit: str
    rounding: str
    evidence: str


class Extra(BaseModel):
    kind: Literal["explanation", "format", "other"]
    text: str
    evidence: str


def queue(fake_llm, slots, ambiguous=False, fmt=None, extras=()):
    Out = type("Out", (BaseModel,), {"__annotations__": {**{f"p{i}": Slot for i in range(len(slots))},
                                                          "ambiguous": bool, "output_format": Fmt,
                                                          "extras": list[Extra]}})
    fake_llm.queue(Out(**{f"p{i}": Slot(evidence=e, value=v) for i, (e, v) in enumerate(slots)},
                       ambiguous=ambiguous, output_format=fmt or Fmt(unit="", rounding="", evidence=""),
                       extras=list(extras)))


@pytest.fixture(autouse=True)
def evidence_mode(monkeypatch):
    monkeypatch.setattr(config, "EXTRACT_FORM", "evidence")


def test_values_come_back_as_written_with_rules_in_the_system_prompt(fake_llm):
    queue(fake_llm, GOOD)
    assert serve.extract_parameters(Q, PARAMS, example=EXAMPLE) == {
        "years elapsed": "3 years", "cost": "$50k", "useful life": "10-year useful life", "salvage": "$5k"}
    call = fake_llm.calls[0]
    assert call["role"] == "extractor" and call["system"] == extraction.EVIDENCE_SYSTEM
    assert "respectively" in call["system"] and "are NOT extras" in call["system"]
    assert "calculator" not in call["system"] + call["prompt"]    # general wording, not math-only
    msg = call["prompt"]
    assert "Copy each value" not in msg                         # fixed rules are not in the user message
    assert EXAMPLE in msg and 'p1 (cost): "$379,000"' in msg     # worked example with the original values
    assert msg.index("Inputs to fill") < msg.index(Q)            # the question comes last


def test_no_example_without_every_original_value(fake_llm):
    params = PARAMS[:3] + [ParamSpec("salvage", ["$41,000"])]    # one original value missing
    assert "Example." not in extraction.evidence_message(Q, params, EXAMPLE)
    assert "Example." not in extraction.evidence_message(Q, PARAMS, None)


def test_an_input_the_question_does_not_state_is_refused(fake_llm):
    queue(fake_llm, GOOD[:3] + [("", "")])
    with pytest.raises(serve.ExtractionError, match="salvage: not stated in the question"):
        serve.extract_parameters(Q, PARAMS, example=EXAMPLE)


def test_a_guessed_word_value_without_evidence_is_refused():
    q = "What is the payment on a $300k loan at 6% over 30 years?"
    assert extraction.check_evidence_value("", "monthly", q) is not None            # the text form let this through
    assert extraction.check_evidence_value("compounded monthly", "monthly", q) is not None
    assert extraction.check_text_value("monthly", q) is None


def test_a_value_must_come_from_its_own_evidence():
    # The text form only asks that 5 appear SOMEWHERE in the question, so "$5k" in the cost slot passes there.
    assert extraction.check_text_value("$5k", Q) is None
    assert extraction.check_evidence_value("cost $50k", "$5k", Q) is not None
    assert extraction.check_evidence_value("cost $50k", "$50k", Q) is None
    assert extraction.check_evidence_value("cost $50,000", "$50,000", Q) is not None   # not the words in the question


def test_evidence_ignores_punctuation_and_quote_styles_but_not_words():
    q = "Machine cost: $50k, salvage “$5k”; life — 10 years."
    assert extraction.check_evidence_value("cost $50k", "$50k", q) is None             # colon dropped
    assert extraction.check_evidence_value('salvage "$5k"', "$5k", q) is None          # straight vs curly quotes
    assert extraction.check_evidence_value("life - 10 years", "10 years", q) is None   # dash style
    assert extraction.check_evidence_value("cost $5", "$5", q) is not None             # part of "$50k" is not a match
    assert extraction.check_evidence_value("salvage $50k", "$50k", q) is not None      # words in a different place


def test_a_value_may_reword_its_evidence_but_not_change_a_number():
    q = "A 3.00 L flask holds 0.250 mol of CO2(g) at 25°C. What pressure does it exert?"
    assert extraction.check_evidence_value("CO2(g)", "carbon dioxide", q) is None
    assert extraction.check_evidence_value("3.00 L flask", "3000 mL", q) is not None    # a converted number


def test_ambiguous_assignment_goes_to_the_model(fake_llm):
    queue(fake_llm, GOOD, ambiguous=True)
    with pytest.raises(serve.ExtractionError, match="which value belongs to which input"):
        serve.extract_parameters(Q, PARAMS, example=EXAMPLE)


def test_numbers_written_as_words_pass_on_their_evidence():
    q = "Ten moles of gas sit in a 5 L tank at 300 K."
    assert extraction.check_evidence_value("Ten moles of gas", "Ten moles", q) is None


def test_output_format_and_extras_still_apply(fake_llm):
    queue(fake_llm, GOOD, extras=[Extra(kind="explanation", text="explain", evidence="what is the book value")])
    with pytest.raises(serve.ExtractionError, match="also asks"):
        serve.extract_parameters(Q, PARAMS, example=EXAMPLE)


def test_clef_serving_passes_the_build_question_as_the_example(monkeypatch):
    from arc import clef
    from arc.cache import Cache, EntryPoint
    cache = Cache("t")
    orig = EntryPoint(EXAMPLE, "original", "root", PARAMS, problem_id="ddb")
    cache.entry_points = [EntryPoint("template ●1", "template", "root", PARAMS, problem_id="ddb"), orig]
    monkeypatch.setattr(config, "GATE", "clef")
    monkeypatch.setattr(clef, "candidates", lambda c, r, **k: [(orig, 0.9)])
    monkeypatch.setattr(clef, "decide", lambda c, r, cands, **k: (orig, 0.95, {"ddb": 0.95}))
    seen = {}

    def extract(req, params, tag=None, form_out=None, example=None, **k):
        seen["example"] = example
        return {}
    monkeypatch.setattr(serve, "extract_parameters", extract)

    class Ctx:
        lookups = []

        def narration(self):
            return ""
    monkeypatch.setattr(serve, "dispatch_root", lambda c, root, values: ({"book value": "1"}, Ctx()))
    serve.use(cache, Q)
    assert seen["example"] == EXAMPLE


def test_text_is_still_the_default_form():
    src = (config.ROOT / "arc/config.py").read_text()
    assert 'EXTRACT_FORM = os.environ.get("ARC_EXTRACT_FORM", "text")' in src
