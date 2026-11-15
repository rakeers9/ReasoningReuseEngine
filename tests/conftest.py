"""Shared test setup. Every test is free and offline by default.

- Any real Anthropic API call fails the test (the client is replaced), and so does
  any real Cloudflare (Clef) call.
- `fake_llm` replaces llm.call with a stub that records calls and returns what you queue.
- Tests that need a local model download (embeddings, cross-encoders) are marked
  @pytest.mark.models and skipped unless you pass --models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc import llm  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--models", action="store_true", help="also run tests that load local models")


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs a local model download (run with --models)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--models"):
        return
    skip = pytest.mark.skip(reason="needs local models; run with --models")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


class RealAPICallBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch):
    def blocked():
        raise RealAPICallBlocked("tests must not call the real Anthropic API; use fake_llm")
    monkeypatch.setattr(llm, "client", blocked)

    def blocked_cloudflare(*a, **k):
        raise RealAPICallBlocked("tests must not call Cloudflare (Clef); stub llm.decide or llm._cloudflare_post")
    monkeypatch.setattr(llm, "_cloudflare_post", blocked_cloudflare)
    monkeypatch.delenv("ARC_ALLOW_OPUS", raising=False)


@pytest.fixture
def fake_llm(monkeypatch):
    """Stub for llm.call. Queue return values with fake.queue(obj); obj is a pydantic model
    instance (returned as .parsed) or a str (returned as .text). Calls land in fake.calls."""
    state = SimpleNamespace(calls=[], replies=[])

    def call(role, prompt, *, model, schema=None, **kw):
        state.calls.append({"role": role, "prompt": prompt, "model": model, "schema": schema, **kw})
        reply = state.replies.pop(0) if state.replies else ""
        if isinstance(reply, str):
            return llm.LLMResult(text=reply)
        return llm.LLMResult(text="", parsed=reply)

    state.queue = state.replies.append
    monkeypatch.setattr(llm, "call", call)
    return state
