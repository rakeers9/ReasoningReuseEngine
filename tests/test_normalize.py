"""Shorthand expansion in extracted values (arc/normalize.py)."""

import pytest

from arc import config, normalize, serve
from arc.cache import ParamSpec

E = normalize.expand_shorthand


@pytest.mark.parametrize("raw, want", [
    ("$150k", "$150,000"),            # the chat-loop bug
    ("$781K", "$781,000"),
    ("$1.5k", "$1,500"),
    ("€2.0 million", "€2,000,000"),
    ("€2.0M", "€2,000,000"),
    ("£1.25bn", "£1,250,000,000"),
    ("$3 mn", "$3,000,000"),
    ("USD 4.2m", "USD 4,200,000"),
    ("2.5 million people", "2,500,000 people"),
    ("3 thousand units", "3,000 units"),
    ("a $1,200.5k budget", "a $1,200,500 budget"),
    ("$0.75 billion", "$750,000,000"),
])
def test_shorthand_is_expanded(raw, want):
    assert E(raw) == want


@pytest.mark.parametrize("raw", [
    "$150,000", "300 K", "15 m", "0.1 M HCl", "5 km", "150 kg", "12 mm", "2.5 MPa", "$368,000",
    "6.5%", "30-year", "monthly", "10.0 mol", "4.860 dm3", "27°C", "1.2e6 Pa", "", "k",
    "Mk II", "the 5k run",   # letter shorthand without a currency sign is left alone
])
def test_everything_else_is_left_exactly_as_written(raw):
    assert E(raw) == raw


def test_non_strings_pass_through():
    assert E(None) is None and normalize.expand_values({"a": "$2k", "b": 5}) == {"a": "$2,000", "b": 5}


def test_extraction_expands_by_default_and_not_when_off(fake_llm, monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        p0: str
        p1: str
    params = [ParamSpec("principal", ["$368,000"]), ParamSpec("term", ["30-year"])]
    monkeypatch.setattr(config, "EXTRACT_FORM", "copy")          # shorthand expansion guards copy mode
    monkeypatch.setattr(config, "EXPAND_SHORTHAND", True)
    fake_llm.queue(Out(p0="$150k", p1="25 years"))
    assert serve.extract_parameters("a $150k loan over 25 years", params) == {"principal": "$150,000", "term": "25 years"}
    monkeypatch.setattr(config, "EXPAND_SHORTHAND", False)      # paper behaviour: values exactly as copied
    fake_llm.queue(Out(p0="$150k", p1="25 years"))
    assert serve.extract_parameters("a $150k loan over 25 years", params) == {"principal": "$150k", "term": "25 years"}
