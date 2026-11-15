"""The backlog: a log of real production questions and answers.

Every interaction the customer's system handles is appended here, whether the source
model answered it or the cache did. The backlog feeds three things:
  - cache-worthiness: which task families come up often enough to pay for a build
    (arc/families.py)
  - the explainer: a worked solution for a family's question + production answer
    (arc/explain.py)
  - pruning: entries that stop being used or start getting complaints

One JSON object per line. Fields:
  id, timestamp (unix seconds), question, response, model,
  input_tokens, output_tokens, cost   (for pricing; any of them may be missing),
  reasoning, next_user_message, feedback   (optional),
  session   (stored hashed, never raw),
  served_by   (cache entry id if the cache answered, else null),
  family   (optional: a caller-known family id; otherwise families.py assigns one),
  reusable   (False = not a reusable calculation, e.g. chit-chat: never forms a family;
              None = not checked)

Privacy: session ids are hashed on write. Questions and responses are stored as given;
stripping personal data is the caller's job until a scrubber exists (see NOTES.md).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Iterator

from . import config

FIELDS = ("id", "timestamp", "question", "response", "model", "input_tokens", "output_tokens", "cost",
          "reasoning", "next_user_message", "feedback", "session", "served_by", "family", "reusable")


def default_path() -> Path:
    return config.WORK_DIR / "backlog.jsonl"


def _hash(value: str) -> str:
    return "h:" + hashlib.sha256(value.encode()).hexdigest()[:16]


def make_record(question: str, response: str, model: str, *, input_tokens: int | None = None,
                output_tokens: int | None = None, cost: float | None = None, reasoning: str | None = None,
                next_user_message: str | None = None, feedback: str | None = None, session: str | None = None,
                served_by: str | None = None, family: str | None = None, reusable: bool | None = None,
                id: str | None = None, timestamp: float | None = None) -> dict:
    if not question or not question.strip():
        raise ValueError("a backlog record needs a question")
    if session and not session.startswith("h:"):
        session = _hash(session)
    return {"id": id or uuid.uuid4().hex[:12], "timestamp": time.time() if timestamp is None else float(timestamp),
            "question": question, "response": response, "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost": cost, "reasoning": reasoning,
            "next_user_message": next_user_message, "feedback": feedback, "session": session,
            "served_by": served_by, "family": family, "reusable": reusable}


def log_interaction(question: str, response: str, model: str, path: Path | None = None, **fields) -> dict:
    """Append one production interaction to the backlog and return the stored record."""
    record = make_record(question, response, model, **fields)
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def normalise(raw: dict) -> dict:
    """Fill missing fields and hash the session, for records imported from elsewhere."""
    extra = {k: raw.get(k) for k in FIELDS if k not in ("question", "response", "model")}
    return make_record(raw.get("question", ""), raw.get("response", ""), raw.get("model", ""), **extra)


def read(path: Path | None = None) -> Iterator[dict]:
    """Records in file order. Lines that fail to parse are skipped, not fatal."""
    path = path or default_path()
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield normalise(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
