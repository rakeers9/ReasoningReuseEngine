"""A chatbot wrapped in the cache: the whole loop, end to end.

DEVIATION from the paper, which evaluates a cache built from a fixed problem set. Here
the cache starts empty and learns from use:

  chat(message)
    1. cheap rule: is this a reusable calculation? (chit-chat never forms a family)
    2. try the cache (serve.use, with whatever gate ARC_GATE selects)
         hit  -> reply with the cached code's answer; log it with served_by = entry id
         miss -> the production model answers; log question, answer and token counts
  learn()
    3. read new backlog records into families (arc/families.py)
    4. build every family the break-even rule marks "build" (families.build_ready,
       which calls the label-free builder arc/labelfree.build_family)
    5. reload the live cache, so the next matching question can hit

Single-turn only: every message must be self-contained. Follow-ups that depend on
earlier turns ("what if it were 5%?") are answered by the model and are not
reusable for the cache yet.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import backlog, config, families, llm, serve
from .cache import Cache

from .present import ROUNDING_POLICY

# The wrapper's own chatbot instructions (a customer's chatbot keeps its own). They carry the
# shared rounding policy, so the answers the cache is checked against round the same way.
PRODUCTION_SYSTEM = ("You are a helpful assistant. For calculations, show brief working and end with "
                     "the final answer and its units. " + ROUNDING_POLICY)
# Room for adaptive thinking plus the answer (a 4000 cap cut a mortgage answer off mid-sentence).
PRODUCTION_MAX_TOKENS = 16000
_HAS_NUMBER = re.compile(r"\d")


def is_reusable(message: str, min_words: int = 8) -> bool:
    """Free first-pass filter: a reusable calculation states at least one number and is
    long enough to be a full question. 'hi', 'thanks!', 'what can you do?' are not."""
    return bool(_HAS_NUMBER.search(message)) and len(message.split()) >= min_words


@dataclass
class ChatReply:
    answer: str
    from_cache: bool
    entry: str | None       # cache entry that answered, if any
    reason: str
    cost: float
    record: dict


class Chatbot:
    def __init__(self, cache_name: str = "master", backlog_path: Path | None = None,
                 tracker_path: Path | None = None, model: str | None = None,
                 production: Callable[[str], llm.LLMResult] | None = None):
        self.cache_name = cache_name
        self.backlog_path = backlog_path or backlog.default_path()
        self.tracker = families.FamilyTracker(path=tracker_path)
        self.model = model or config.SOURCE_MODEL
        self._production = production
        self.cache = Cache(cache_name)
        self.reload()
        if config.GATE == "off":
            print("[chat] ARC_GATE=off: near-miss questions can get confident wrong cache answers; "
                  "use ARC_GATE=v2 for real use", file=sys.stderr)

    # ---- the live cache ----
    def reload(self) -> None:
        folder = config.cache_dir(self.cache_name)
        self.cache = Cache.load(folder) if (folder / "registry.json").exists() else Cache(self.cache_name)

    # ---- one turn ----
    def _answer_with_model(self, message: str) -> llm.LLMResult:
        if self._production is not None:
            return self._production(message)
        return llm.call("production", message, model=self.model, system=PRODUCTION_SYSTEM,
                        effort="medium", max_tokens=PRODUCTION_MAX_TOKENS, tag="chat:production")

    def chat(self, message: str, session: str | None = None) -> ChatReply:
        reusable = is_reusable(message)
        reason = "not a reusable calculation" if not reusable else "cache is empty"
        if reusable and self.cache.entry_points:
            trace: dict = {}
            try:
                hit = serve.use(self.cache, message, tag="chat:cache", trace=trace)
            except serve.CacheInvocationError as e:
                hit, reason = None, f"cache error, fell back: {str(e).splitlines()[0][:120]}"
            else:
                if hit is None:
                    reason = (f"cache turned it away ({', '.join(trace.get('rejected_by') or [])})"
                              if trace.get("stage") == "gate_rejected" else "no cache entry close enough")
            if hit is not None:
                answer = serve.format_output(hit.output)
                entry = hit.entry_point.problem_id
                rec = backlog.log_interaction(message, answer, config.TARGET_MODEL, path=self.backlog_path,
                                              cost=hit.cost, session=session, served_by=entry, reusable=True)
                return ChatReply(answer, True, entry, f"cache hit ({entry})", hit.cost, rec)
        r = self._answer_with_model(message)
        rec = backlog.log_interaction(message, r.text, r.model or self.model, path=self.backlog_path,
                                      input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                                      session=session, reusable=reusable)
        return ChatReply(r.text, False, None, reason, r.cost, rec)

    # ---- learning between turns ----
    def learn(self, build_fn: Callable | None = None, max_builds: int | None = None,
              can_start: Callable[[], bool] | None = None) -> list:
        """Read new backlog records, build every family that has reached break-even, and
        reload the cache. PAID when build_fn is the real builder."""
        self.tracker.ingest(self.backlog_path)
        self.tracker.save()
        results = families.build_ready(self.tracker, build_fn, max_builds=max_builds, path=self.backlog_path,
                                       can_start=can_start)
        self.reload()
        return results
