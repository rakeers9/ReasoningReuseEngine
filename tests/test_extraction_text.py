"""IS2 in text form (arc/extraction.extract_text, the default): values copied exactly as written."""

from typing import Literal

import pytest
from pydantic import BaseModel

from arc import config, extraction, serve
from arc.cache import ParamSpec

Q = "Calculate the monthly installment for a $150k mortgage at 5.5% annual interest compounded monthly, over 25 years."
PARAMS = [ParamSpec("principal", ["$368,000"]), ParamSpec("rate", ["5.875%"]), ParamSpec("term", ["30-year"]),
          ParamSpec("frequency", ["monthly"])]


class Fmt(BaseModel):
    unit: str
    rounding: str
    evidence: str


class Extra(BaseModel):
    kind: Literal["explanation", "format", "other"]
    text: str
    evidence: str


def queue(fake_llm, values, fmt=None, extras=()):
    Out = type("Out", (BaseModel,), {"__annotations__": {**{f"p{i}": str for i in range(len(values))},
                                                          "output_format": Fmt, "extras": list[Extra]}})
    fake_llm.queue(Out(**{f"p{i}": v for i, v in enumerate(values)},
                       output_format=fmt or Fmt(unit="", rounding="", evidence=""), extras=list(extras)))


@pytest.fixture(autouse=True)
def text_mode(monkeypatch):
    monkeypatch.setattr(config, "EXTRACT_FORM", "text")


def test_text_is_the_default_form():
    import importlib
    import os
    assert "ARC_EXTRACT_FORM" not in os.environ
    assert 'EXTRACT_FORM = os.environ.get("ARC_EXTRACT_FORM", "text")' in \
        (importlib.import_module("arc.config").ROOT / "arc/config.py").read_text()


def test_values_come_back_exactly_as_written(fake_llm):
    queue(fake_llm, ["$150k", "5.5%", "25 years", "monthly"])
    assert serve.extract_parameters(Q, PARAMS) == {"principal": "$150k", "rate": "5.5%", "term": "25 years",
                                                   "frequency": "monthly"}
    prompt = fake_llm.calls[0]["prompt"]
    assert "Copy each value exactly as it appears" in prompt and '"$150k"' in prompt
    assert "output_format" in prompt and "are NOT extras" in prompt


def test_a_made_up_number_is_refused(fake_llm):
    queue(fake_llm, ["$150k", "6.5%", "25 years", "monthly"])          # 6.5% is not in the question
    with pytest.raises(serve.ExtractionError, match="which is not in the question"):
        serve.extract_parameters(Q, PARAMS)


def test_a_reworded_value_with_the_same_numbers_passes():
    # regression (chat run 2026-11-15): "10 years" for "a 10-year ... loan" was refused, which
    # crashed 2 of 5 fit-check questions and blocked a correct mortgage entry
    q = "What are the monthly payments on a 10-year, $639,000 home loan at 5.875% per year?"
    assert extraction.check_text_value("10 years", q) is None
    assert extraction.check_text_value("$639,000", q) is None and extraction.check_text_value("639000", q) is None
    assert extraction.check_text_value("12 years", q) is not None


@pytest.mark.parametrize("value, ok", [("27°C", True), ("27 °C", True), ("4.860 dm3", True), ("300 K", False),
                                       ("monthly", True), ("", False), ("27 degrees C", True), ("4.86 dm3", True), ("4.87 dm3", False)])
def test_value_check(value, ok):
    q = "10.0 mol is confined to 4.860 dm3 at 27°C, compounded monthly."
    assert (extraction.check_text_value(value, q) is None) is ok


def test_output_format_and_extras_still_apply(fake_llm):
    queue(fake_llm, ["$150k", "5.5%", "25 years", "monthly"],
          extras=[Extra(kind="explanation", text="explain your reasoning", evidence="25 years")])
    out = {}
    with pytest.raises(serve.ExtractionError, match="also asks"):
        serve.extract_parameters(Q, PARAMS, form_out=out)
    queue(fake_llm, ["$150k", "5.5%", "25 years", "monthly"],
          fmt=Fmt(unit="", rounding="nearest:1", evidence="monthly installment"))
    out = {}
    serve.extract_parameters(Q, PARAMS, form_out=out)
    assert out["output_format"]["rounding"] == "nearest:1"


def test_shorthand_expansion_is_off_by_default():
    src = (config.ROOT / "arc/config.py").read_text()
    assert 'EXPAND_SHORTHAND = os.environ.get("ARC_EXPAND_SHORTHAND", "0") == "1"' in src
