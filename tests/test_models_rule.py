"""Project rule: Haiku only, in every role except re-solving with the customer's own model (see CLAUDE.md)."""

import importlib

import pytest

from arc import config, llm
from conftest import RealAPICallBlocked

HAIKU = "claude-haiku-5-5"
ROLES = ["SOURCE_MODEL", "BUILDER_MODEL", "JUDGE_MODEL", "GENERATOR_MODEL", "TARGET_MODEL"]


def test_every_role_defaults_to_haiku(monkeypatch):
    for name in ["ARC_PROFILE", *(f"ARC_{r}" for r in ROLES)]:
        monkeypatch.delenv(name, raising=False)
    try:
        c = importlib.reload(config)
        assert c.PROFILE == "haiku"
        for role in ROLES:
            assert getattr(c, role) == HAIKU, role
    finally:
        importlib.reload(config)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-OPUS-anything"])
def test_opus_is_refused_without_opt_in(model):
    with pytest.raises(llm.LLMError, match="Haiku only"):
        llm.call("builder", "hi", model=model)


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


@pytest.mark.parametrize("role", ["source", "builder", "explainer", "judge", "cert_judge", "generator",
                                  "extractor", "target", "production", "decider"])
def test_every_other_role_still_refuses_opus(role):
    with pytest.raises(llm.LLMError, match="Haiku only"):
        llm.call(role, "hi", model="claude-opus-5-5")


def test_only_resolve_is_exempt():
    assert llm.CUSTOMER_MODEL_ROLES == {"resolve"}
