"""What one production answer cost, for any model, so cache-worthiness is model-agnostic.

The break-even rule (arc/families.py) needs the dollar saving of every question the
cache could have answered. That depends on the CUSTOMER'S production model, not ours:
the same question costs ~40x more on Opus than on Haiku. So every backlog record is
priced from its own model and token counts.

Order of preference for one record:
  1. `cost` on the record (the customer already knows what they paid; any provider)
  2. input/output tokens x the model's price from the table below
  3. otherwise unpriced: the question still counts as a sighting, but adds no saving,
     and the report lists the model so a price can be added.

Customer prices (other providers, negotiated rates, in-house models) go in a JSON file
named by ARC_PRICES_FILE: {"model-id": [input_per_M, output_per_M], ...}. They override
the table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config

# $ per million tokens (input, output), Anthropic first-party list prices, 2026-11.
# Prompt-cache rates and long-context tiers are ignored: this is an estimate of savings,
# and a record with an exact `cost` always wins.
LIST_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _customer_prices() -> dict[str, tuple[float, float]]:
    path = os.environ.get("ARC_PRICES_FILE")
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    return {m: (float(p[0]), float(p[1])) for m, p in raw.items()}


def price_of(model: str | None) -> tuple[float, float] | None:
    """(input, output) $ per million tokens, or None if we don't know this model."""
    if not model:
        return None
    table = {**LIST_PRICES, **{m: (p[0], p[1]) for m, p in config.PRICES.items() if m not in LIST_PRICES},
             **_customer_prices()}
    if model in table:
        return table[model]
    for known in sorted(table, key=len, reverse=True):   # "claude-opus-5-5-fast", "anthropic.claude-..."
        if known in model:
            return table[known]
    return None


def answer_cost(record: dict) -> float | None:
    """Dollar cost of the production answer in one backlog record, or None if unpriced."""
    if record.get("cost") is not None:
        return float(record["cost"])
    p = price_of(record.get("model"))
    tin, tout = record.get("input_tokens"), record.get("output_tokens")
    if p is None or tin is None or tout is None:
        return None
    return (tin * p[0] + tout * p[1]) / 1e6
