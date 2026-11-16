"""Project rule: Haiku only, in every role except re-solving with the customer's own model (see CLAUDE.md)."""

import importlib

import pytest

from arc import config, llm
from conftest import RealAPICallBlocked

HAIKU = "claude-haiku-5-5"
ROLES = ["SOURCE_MODEL", "BUILDER_MODEL", "JUDGE_MODEL", "GENERATOR_MODEL", "TARGET_MODEL"]


def test_every_role_defaults_to_haiku(monkeypatch):
    # user decisions 2026-11-15/16: "builder: Haiku first, Sonnet on code failure, then Opus".
    # BUILDER_MODEL is Haiku; Sonnet and Opus run only as the label-free cascade's later
    # builders (config.LABELFREE_BUILDERS, tests/test_builder_cascade.py).
    for name in ["ARC_PROFILE", "ARC_LABELFREE_BUILDERS", *(f"ARC_{r}" for r in ROLES)]:
        monkeypatch.delenv(name, raising=False)
    try:
        c = importlib.reload(config)
        assert c.PROFILE == "haiku"
        for role in ROLES:
            assert getattr(c, role) == HAIKU, role
        assert c.LABELFREE_BUILDERS == [HAIKU, "claude-sonnet-5-5", "claude-opus-5-5"]
        assert c.EXPLAINER_MODEL == HAIKU
    finally:
        importlib.reload(config)


def test_sonnet_is_priced_and_gets_the_refusal_fallback():
    assert config.PRICES["claude-sonnet-5-5"][:2] == (2.00, 10.00)
    assert "claude-sonnet-5-5" in llm.OPUS_LIKE


def test_sonnet_passes_the_opus_guard():
    with pytest.raises(RealAPICallBlocked):           # past the guard, the call reaches the blocked client
        llm.call("builder", "hi", model="claude-sonnet-5-5")


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-OPUS-anything"])
def test_opus_is_refused_without_opt_in(model):
    with pytest.raises(llm.LLMError, match="Haiku only"):
        llm.call("source", "hi", model=model)          # the builder may use Opus; other roles may not


def test_opus_opt_in_passes_the_guard(monkeypatch):
    monkeypatch.setenv("ARC_ALLOW_OPUS", "1")
    # past the guard, the call reaches the (blocked) client instead of the guard's error
    with pytest.raises(RealAPICallBlocked):
        llm.call("builder", "hi", model="claude-opus-5-5")


def test_haiku_passes_the_guard():
    with pytest.raises(RealAPICallBlocked):
        llm.call("extractor", "hi", model=HAIKU)


def test_resolve_role_may_run_the_customers_opus_model():
    # past the guard: the call reaches the (blocked) client
    with pytest.raises(RealAPICallBlocked):
        llm.call("resolve", "hi", model="claude-opus-5-5")


@pytest.mark.parametrize("role", ["source", "explainer", "judge", "build_judge", "cert_judge", "generator",
                                  "extractor", "target", "production", "decider", "adjudicator"])
def test_every_other_role_still_refuses_opus(role):
    with pytest.raises(llm.LLMError, match="Haiku only"):
        llm.call(role, "hi", model="claude-opus-5-5")


def test_only_resolve_and_the_builder_are_exempt():
    assert llm.CUSTOMER_MODEL_ROLES == {"resolve"} and llm.OPUS_BUILDER_ROLES == {"builder"}


def test_the_builder_may_run_opus_as_the_last_cascade_step():
    with pytest.raises(RealAPICallBlocked):           # past the guard
        llm.call("builder", "hi", model="claude-opus-5-5")
