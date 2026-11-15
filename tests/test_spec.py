"""Spec gate (arc/spec.py): structure text, code checks, field comparison, decisions."""

import pytest

from arc import config, spec
from arc.cache import Cache, EntryPoint, ParamSpec


def make_spec(**over):
    s = {
        "task": "compute",
        "deliverable": "pressure of a single gas",
        "method": "perfect gas law",
        "conditions": ["one gas", "one fixed state"],
        "inputs": [
            {"name": "amount of gas", "value": "10.0 mol", "evidence": "10.0 mol"},
            {"name": "volume", "value": "4.860 dm3", "evidence": "4.860 dm3"},
            {"name": "temperature", "value": "27°C", "evidence": "27°C"},
        ],
        "output_form": "a number with a pressure unit",
    }
    s.update(over)
    return s


QUESTION = "Suppose that 10.0 mol of gas is confined to 4.860 dm3 at 27°C. Find the pressure."


@pytest.fixture
def exact_decider():
    """Deterministic decider: 1.0 if the texts are equal (ignoring case), else 0.0."""
    spec.register_decider("exact", lambda pairs: [float(a.lower() == b.lower()) for a, b in pairs])
    yield "exact"
    spec._DECIDERS.pop("exact", None)


# ----------------------------------------------------------------------------- structure text

def test_spec_text_has_no_values():
    text = spec.spec_text(make_spec())
    for value in ["10.0", "4.860", "27"]:
        assert value not in text
    assert "inputs: amount of gas; volume; temperature" in text


def test_spec_text_marks_empty_lists():
    text = spec.spec_text(make_spec(conditions=[], inputs=[]))
    assert "conditions: none" in text and "inputs: none" in text


# ----------------------------------------------------------------------------- code checks

def test_code_checks_pass_on_grounded_spec():
    assert spec.code_checks(make_spec(), QUESTION) == {"grounded": True, "has_values": True}


def test_made_up_value_fails_grounding():
    s = make_spec()
    s["inputs"][2] = {"name": "temperature", "value": "300 K", "evidence": "300 K"}
    assert spec.code_checks(s, QUESTION)["grounded"] is False


def test_grounding_ignores_case_and_spacing():
    s = make_spec()
    s["inputs"][1]["evidence"] = "4.860   DM3"
    assert spec.code_checks(s, QUESTION)["grounded"] is True


def test_missing_value_fails_has_values():
    s = make_spec()
    s["inputs"][0]["value"] = "not stated"
    assert spec.code_checks(s, QUESTION)["has_values"] is False


# ----------------------------------------------------------------------------- comparison

def test_identical_specs_score_one_everywhere(exact_decider):
    [scores] = spec.compare_many([(make_spec(), make_spec())], exact_decider)
    assert scores == {f: 1.0 for f in spec.FIELDS}


def test_unstated_method_in_query_passes(exact_decider):
    [scores] = spec.compare_many([(make_spec(method="not stated"), make_spec())], exact_decider)
    assert scores["method"] == 1.0


def test_different_method_fails(exact_decider):
    [scores] = spec.compare_many([(make_spec(method="van der Waals equation"), make_spec())], exact_decider)
    assert scores["method"] == 0.0


def test_extra_condition_in_query_fails(exact_decider):
    q = make_spec(conditions=["one gas", "one fixed state", "mixture of two gases"])
    [scores] = spec.compare_many([(q, make_spec())], exact_decider)
    assert scores["conditions"] == 0.0


def test_missing_input_in_query_fails(exact_decider):
    q = make_spec()
    q["inputs"] = q["inputs"][:2]  # no temperature
    [scores] = spec.compare_many([(q, make_spec())], exact_decider)
    assert scores["inputs"] == 0.0


def test_one_side_empty_list_fails_both_empty_passes(exact_decider):
    [one] = spec.compare_many([(make_spec(conditions=[]), make_spec())], exact_decider)
    [both] = spec.compare_many([(make_spec(conditions=[]), make_spec(conditions=[]))], exact_decider)
    assert one["conditions"] == 0.0 and both["conditions"] == 1.0


def test_input_order_does_not_matter(exact_decider):
    q = make_spec()
    q["inputs"] = list(reversed(q["inputs"]))
    [scores] = spec.compare_many([(q, make_spec())], exact_decider)
    assert scores["inputs"] == 1.0


def test_compare_many_keeps_pairs_separate(exact_decider):
    good = (make_spec(), make_spec())
    bad = (make_spec(deliverable="density of a single gas"), make_spec())
    s_good, s_bad = spec.compare_many([good, bad], exact_decider)
    assert s_good["deliverable"] == 1.0 and s_bad["deliverable"] == 0.0


# ----------------------------------------------------------------------------- decide

def test_decide_accepts_a_match(exact_decider):
    d = spec.decide(make_spec(), make_spec(), QUESTION, exact_decider, threshold=0.5)
    assert d["rejected"] == [] and d["score"] == 1.0


def test_decide_names_every_failure(exact_decider):
    q = make_spec(deliverable="change in pressure of a single gas")
    q["inputs"][2] = {"name": "temperature", "value": "300 K", "evidence": "300 K"}
    d = spec.decide(q, make_spec(), QUESTION, exact_decider, threshold=0.5)
    assert d["rejected"] == ["grounded", "deliverable"]


def test_decide_can_limit_fields(exact_decider):
    q = make_spec(output_form="a density")
    d = spec.decide(q, make_spec(), QUESTION, exact_decider, threshold=0.5, fields=["deliverable"])
    assert d["rejected"] == []


# ----------------------------------------------------------------------------- the Haiku call

def test_write_spec_uses_the_small_model(fake_llm):
    fake_llm.queue(spec.Spec(**make_spec()))
    out = spec.write_spec("some question", tag="t")
    [call] = fake_llm.calls
    assert call["role"] == "extractor" and call["model"] == config.TARGET_MODEL
    assert call["schema"] is spec.Spec and "some question" in call["prompt"]
    assert out == make_spec()


def test_index_cache_specs_only_original_entries(fake_llm):
    c = Cache("t")
    c.entry_points = [EntryPoint("seed q", "original", "root", [ParamSpec("x")], "p1"),
                      EntryPoint("template q", "template", "root", [ParamSpec("x")], "p1")]
    fake_llm.queue(spec.Spec(**make_spec()))
    assert spec.index_cache(c) == 1
    assert list(c.specs) == ["p1"] and fake_llm.calls[0]["prompt"].endswith("seed q")
    assert spec.index_cache(c) == 0  # already there, no second call


# ----------------------------------------------------------------------------- local models

@pytest.mark.models
def test_nli_decider_separates_obvious_pair():
    same, diff = spec.decider("nli")([("pressure of the gas", "pressure exerted by the gas"),
                                      ("pressure of the gas", "temperature of the gas")])
    assert same > diff
