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
OPUS_LIKE = {"claude-opus-5-5", "claude-sonnet-5-5"}   # models that get the server-side refusal fallback
SMALL_ROLES = {"target", "extractor", "converter"}   # the target model T at invocation time
# Roles that run the customer's own production model (which may be Opus). Only these may
# call Opus without ARC_ALLOW_OPUS=1; every other role stays Haiku-only.
CUSTOMER_MODEL_ROLES = {"resolve"}
# The builder cascade's last step may be Opus (user decision 2026-11-16: Haiku, then Sonnet,
# then Opus, for a family neither cheaper model can build). Only the builder role.
OPUS_BUILDER_ROLES = {"builder"}


class LLMError(RuntimeError):
    pass


class OutputError(LLMError):
    """The model answered but its output can't be used: refused, cut off at max_tokens, or
    not parseable into the schema. A stronger model may succeed where this one failed (the
    label-free builder cascade retries on it); API and network errors are plain LLMError."""


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4, timeout=300.0)  # a dead connection must not hang a run
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
    input_tokens: int = 0
    output_tokens: int = 0


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
    if "opus" in model.lower() and os.environ.get("ARC_ALLOW_OPUS") != "1" \
            and role not in CUSTOMER_MODEL_ROLES | OPUS_BUILDER_ROLES:
        # Project rule: Haiku only, in every role. Opus must be an explicit opt-in. One exception
        # (user decision 2026-11-15): "resolve" re-solves a backlog record with the CUSTOMER'S
        # own production model, whatever it is, so the answers can be compared like for like.
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
            # Server-side refusal fallback (on by default for Opus and Sonnet 5.5; Haiku 5.5 has none).
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
        raise OutputError(f"{role} call was refused by the model "
                       f"(category: {category}; {explanation or 'no explanation'})")
    if response.stop_reason == "max_tokens":
        _log({"time": started, "seconds": round(time.time() - started, 2), "role": role,
              "model": getattr(response, "model", model) or model, "tag": tag, "prompt": prompt,
              "system": system, "response": "".join(b.text for b in response.content if b.type == "text"),
              "cost": _price(getattr(response, "model", model) or model, response.usage),
              **_usage_fields(response.usage), "error": f"hit max_tokens={max_tokens}"})
        raise OutputError(f"{role} call hit max_tokens={max_tokens}")

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
        raise OutputError(f"{role} call returned no parseable output")
    return LLMResult(text=text, parsed=parsed, thinking=thinking, model=served_model, cost=cost,
                     input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)


# ----------------------------------------------------------------------------
# Clef (Cloudflare Workers AI): yes/no decisions, used by the cache gate
# ----------------------------------------------------------------------------

def _cloudflare_post(url: str, body: dict, headers: dict, timeout: float = 60, tries: int = 3) -> dict:
    """One HTTP call to Workers AI, retried on rate limits and server errors. Tests replace this."""
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(url, data, {"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"Cloudflare HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"Cloudflare network error: {e}") from None
    raise LLMError("Cloudflare: no response")


def _p_yes(ans: Any) -> float:
    if isinstance(ans, (int, float)):
        return float(ans)
    if isinstance(ans, dict):
        for k in ("noul", "probability", "p", "value", "yes", "true"):
            if isinstance(ans.get(k), (int, float)):
                return float(ans[k])
        probs = ans.get("probabilities")
        if isinstance(probs, dict):
            for k in ("true", "yes", "True", "Yes"):
                if k in probs:
                    return float(probs[k])
    raise LLMError(f"Clef: cannot read P(yes) from {json.dumps(ans)[:200]}")


def decide(state: str, questions: dict[str, str], *, model: str | None = None,
           tag: str | None = None) -> dict[str, float]:
    """Ask Clef one yes/no question per key about `state`. Returns {key: P(yes)}.
    Cost (input tokens x CLEF_PRICE_PER_MTOK) goes into the ledger under role "decider"."""
    model = model or config.CLEF_MODEL
    acct, tok = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
    if not acct or not tok:
        raise LLMError("Clef needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/cloudflare/{model}"
    body = {"model": model, "state": state,
            "questions": {k: {"type": "noul", "instructions": v} for k, v in questions.items()}}
    started = time.time()
    try:
        out = _cloudflare_post(url, body, {"Authorization": f"Bearer {tok}"})
        if isinstance(out.get("result"), dict):
            if out.get("success") is False:
                raise LLMError(f"Clef error: {out.get('errors')}")
            out = out["result"]
        probs = {k: _p_yes(out["answers"][k]) for k in questions}
    except (LLMError, KeyError, TypeError) as e:
        _log({"time": started, "seconds": round(time.time() - started, 2), "role": "decider",
              "model": f"cloudflare/{model}", "tag": tag, "prompt": {"state": state, "questions": questions},
              "cost": 0.0, "error": str(e)[:300]})
        raise LLMError(str(e)) from e
    usage = out.get("usage") or {}
    tin = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    cost = tin * config.CLEF_PRICE_PER_MTOK / 1e6
    LEDGER.add(Record(role="decider", model=f"cloudflare/{model}", input_tokens=tin, output_tokens=0,
                      cache_write=0, cache_read=0, cost=cost, tag=tag))
    _log({"time": started, "seconds": round(time.time() - started, 2), "role": "decider",
          "model": f"cloudflare/{model}", "tag": tag, "prompt": {"state": state, "questions": questions},
          "response": probs, "cost": cost, "input_tokens": tin})
    return probs
