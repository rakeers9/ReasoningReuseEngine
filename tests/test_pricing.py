"""Pricing of production answers: any model, any provider (arc/pricing.py)."""

import json

import pytest

from arc import config, pricing


def test_list_prices_for_current_models():
    assert pricing.price_of("claude-haiku-5-5") == (0.10, 0.50)
    assert pricing.price_of("claude-sonnet-5-5") == (2.00, 10.00)
    assert pricing.price_of("claude-opus-5-5") == (4.00, 20.00)
    assert pricing.price_of("claude-fable-5-2") == (8.00, 40.00)


def test_provider_prefixed_and_suffixed_ids_match_the_longest_known_model():
    assert pricing.price_of("anthropic.claude-sonnet-5-5") == (2.00, 10.00)
    assert pricing.price_of("claude-opus-5-5-fast") == (4.00, 20.00)
    assert pricing.price_of("claude-fable-5-1") == (10.00, 50.00)   # not confused with claude-fable-5


def test_unknown_or_missing_model_is_unpriced():
    assert pricing.price_of("gpt-something") is None
    assert pricing.price_of(None) is None
    assert pricing.answer_cost({"model": "gpt-x", "input_tokens": 10, "output_tokens": 10}) is None


def test_cost_from_tokens():
    rec = {"model": "claude-opus-5-5", "input_tokens": 1_000_000, "output_tokens": 500_000}
    assert pricing.answer_cost(rec) == pytest.approx(4.0 + 10.0)


def test_missing_token_count_is_unpriced():
    assert pricing.answer_cost({"model": "claude-opus-5-5", "input_tokens": 100}) is None


def test_recorded_cost_wins_over_tokens():
    rec = {"model": "claude-opus-5-5", "input_tokens": 1_000_000, "output_tokens": 0, "cost": 0.5}
    assert pricing.answer_cost(rec) == 0.5
    assert pricing.answer_cost({"cost": 0, "model": "x"}) == 0.0


def test_customer_price_file_adds_and_overrides(tmp_path, monkeypatch):
    f = tmp_path / "prices.json"
    f.write_text(json.dumps({"gpt-x": [1.0, 2.0], "claude-haiku-5-5": [9.0, 9.0]}))
    monkeypatch.setenv("ARC_PRICES_FILE", str(f))
    assert pricing.answer_cost({"model": "gpt-x", "input_tokens": 1e6, "output_tokens": 1e6}) == pytest.approx(3.0)
    assert pricing.price_of("claude-haiku-5-5") == (9.0, 9.0)


def test_config_prices_used_for_models_missing_from_the_list(monkeypatch):
    monkeypatch.setitem(config.PRICES, "claude-test-model", (1.0, 3.0, 0, 0))
    assert pricing.price_of("claude-test-model") == (1.0, 3.0)
