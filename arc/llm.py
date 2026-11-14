"""One place for every language-model call.

Each call is tagged with a role (source, builder, judge, generator, target,
extractor, converter) so costs can be reported per role, as in the paper's
Appendix C. Every call is also appended to runs/llm_log.jsonl, mirroring the
invocation logs the paper later uses to fine-tune its 1B target model (App. D).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel

from . import config

_client: anthropic.Anthropic | None = None

FALLBACK_BETA = "server-side-fallback-2026-07-01"
OPUS_LIKE = {"claude-opus-5-5"}
SMALL_ROLES = {"target", "extractor", "converter"}   # the target model T at invocation time


class LLMError(RuntimeError):
    pass


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    return _client


# ----------------------------------------------------------------------------
# Cost ledger
# ----------------------------------------------------------------------------

@dataclass
class Record:
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_write: int
    cache_read: int
    cost: float
    tag: str | None = None


@dataclass
class Ledger:
    records: list[Record] = field(default_factory=list)

    def add(self, record: Record) -> None:
        self.records.append(record)

    def cost(self, roles: set[str] | None = None, since: int = 0) -> float:
        return sum(r.cost for r in self.records[since:] if roles is None or r.role in roles)

    def by_role(self, since: int = 0) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records[since:]:
            row = out.setdefault(r.role, {"calls": 0, "input": 0, "output": 0,
                                          "cache_read": 0, "cache_write": 0, "cost": 0.0})
            row["calls"] += 1
            row["input"] += r.input_tokens
            row["output"] += r.output_tokens
            row["cache_read"] += r.cache_read
            row["cache_write"] += r.cache_write
            row["cost"] += r.cost
        return out

    @contextmanager
    def span(self):
        """Collect the records added inside a `with` block."""
        holder = _Span(self, len(self.records))
        yield holder
        holder.end = len(self.records)


@dataclass
class _Span:
    ledger: Ledger
    start: int
    end: int | None = None

    @property
    def records(self) -> list[Record]:
        return self.ledger.records[self.start:self.end]

    def cost(self, roles: set[str] | None = None) -> float:
        return sum(r.cost for r in self.records if roles is None or r.role in roles)

    def by_role(self) -> dict[str, dict[str, float]]:
        tmp = Ledger(self.records)
        return tmp.by_role()


LEDGER = Ledger()


def _price(model: str, usage: Any) -> float:
    p_in, p_out, p_cw, p_cr = config.PRICES.get(model, config.DEFAULT_PRICE)
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (usage.input_tokens * p_in + usage.output_tokens * p_out + cw * p_cw + cr * p_cr) / 1e6


def _log(entry: dict) -> None:
    """Append to the global log and, inside an experiment, to its own calls.jsonl."""
    run_id = os.environ.get("ARC_RUN_ID")
    if run_id:
        entry = {"run_id": run_id, **entry}
    line = json.dumps(entry, default=str) + "\n"
    config.RUNS_DIR.mkdir(exist_ok=True)
    with open(config.RUNS_DIR / "llm_log.jsonl", "a") as f:
        f.write(line)
    if run_id:
        folder = config.RUNS_DIR / "experiments" / run_id
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "calls.jsonl", "a") as f:
            f.write(line)


def _usage_fields(usage: Any) -> dict:
    return {
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


# ----------------------------------------------------------------------------
# The call itself
# ----------------------------------------------------------------------------

@dataclass
class LLMResult:
    text: str
    parsed: Any = None
    thinking: str = ""
    model: str = ""
    cost: float = 0.0


def call(
    role: str,
    prompt: str | list,
    *,
    model: str,
    system: str | list | None = None,
    schema: type[BaseModel] | None = None,
    effort: str | None = None,
    max_tokens: int = 16000,
    show_thinking: bool = False,
    tag: str | None = None,
) -> LLMResult:
    """Make one model call, record its cost, and return text (and parsed JSON)."""
    if "opus" in model.lower() and os.environ.get("ARC_ALLOW_OPUS") != "1":
        # Project rule: Haiku only, in every role. Opus must be an explicit opt-in.
        raise LLMError(f"{role}: refusing to call {model}. This project runs Haiku only; "
                       "set ARC_ALLOW_OPUS=1 to override on purpose.")
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        kwargs["system"] = system

    if role in SMALL_ROLES:
        # Small target model: no thinking, lowest effort, like a plain 1B model.
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["output_config"] = {"effort": effort or "low"}
        api = client().messages
    else:
        # Big roles (source, builder, judges, generator) think adaptively at the role's
        # effort, whichever model fills them; display only controls visibility.
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized" if show_thinking else "omitted"}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if model in OPUS_LIKE:
            # Server-side refusal fallback (on by default for Opus; Haiku 5.5 has none).
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
            api = client().beta.messages
        else:
            api = client().messages

    started = time.time()
    try:
        if schema is not None:
            response = api.parse(output_format=schema, **kwargs)
        else:
            response = api.create(**kwargs)
    except anthropic.APIStatusError as e:
        _log({"time": started, "seconds": round(time.time() - started, 2), "role": role, "model": model,
              "tag": tag, "prompt": prompt, "system": system, "cost": 0.0,
              "error": f"API error {e.status_code}: {e.message}"})
        raise LLMError(f"{role} call failed ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        _log({"time": started, "seconds": round(time.time() - started, 2), "role": role, "model": model,
              "tag": tag, "prompt": prompt, "system": system, "cost": 0.0, "error": "network error"})
        raise LLMError(f"{role} call failed: network error") from e

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        _log({"time": started, "seconds": round(time.time() - started, 2), "role": role, "model": model,
              "tag": tag, "prompt": prompt, "system": system,
              "cost": _price(getattr(response, "model", model) or model, response.usage),
              **_usage_fields(response.usage),
              "refusal_category": category, "refusal_explanation": explanation})
        raise LLMError(f"{role} call was refused by the model "
                       f"(category: {category}; {explanation or 'no explanation'})")
    if response.stop_reason == "max_tokens":
        _log({"time": started, "seconds": round(time.time() - started, 2), "role": role,
              "model": getattr(response, "model", model) or model, "tag": tag, "prompt": prompt,
              "system": system, "response": "".join(b.text for b in response.content if b.type == "text"),
              "cost": _price(getattr(response, "model", model) or model, response.usage),
              **_usage_fields(response.usage), "error": f"hit max_tokens={max_tokens}"})
        raise LLMError(f"{role} call hit max_tokens={max_tokens}")

    text = "".join(b.text for b in response.content if b.type == "text")
    thinking = "".join(getattr(b, "thinking", "") or "" for b in response.content if b.type == "thinking")
    served_model = getattr(response, "model", model) or model
    cost = _price(served_model, response.usage)
    usage = response.usage
    LEDGER.add(Record(
        role=role, model=served_model,
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cost=cost, tag=tag,
    ))
    _log({
        "time": started, "seconds": round(time.time() - started, 2), "role": role,
        "model": served_model, "tag": tag, "prompt": prompt, "system": system,
        "response": text, "cost": cost, **_usage_fields(usage),
        "effort": effort, "max_tokens": max_tokens, "stop_reason": response.stop_reason,
        "visible_text_chars": len(text), "thinking_chars": len(thinking),
    })
    parsed = getattr(response, "parsed_output", None) if schema is not None else None
    if schema is not None and parsed is None:
        raise LLMError(f"{role} call returned no parseable output")
    return LLMResult(text=text, parsed=parsed, thinking=thinking, model=served_model, cost=cost)
