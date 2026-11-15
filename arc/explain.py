"""Explainer: get a worked solution when production gives only a question and an answer.

DEVIATION from the paper, which builds from the source model's own reasoning trace. In
real use the production model may show no reasoning at all (and a model's visible
reasoning is not a faithful record of how it got the answer anyway). The builder only
needs A correct worked method, so we write one at build time:

  E1  blind solve: the explainer model solves the question with full working, without
      seeing the production answer (so it cannot just justify that answer).
  E2  compare its final answer with the production answer.
        agree    -> this worked solution becomes the transcript; build as usual.
        disagree -> E3.
  E3  two more blind solves. Use one only if BOTH agree with production: our own solves
      contradicting each other means we are unsure, so one lucky match is not enough.
  E4  otherwise -> one hinted call sees the production answer and says which side
      is wrong. It is diagnostic only: we never build from a solution written after
      seeing the answer, and the entry is not cached either way.

Input is model-agnostic: a backlog record {id, question, response[, reasoning]}. Any
reasoning in the response is optional extra context, never required.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from . import config, llm
from .certify import Comparison, _compare
from .source import SOLVE_PROMPT, Transcript


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Worked(_Strict):
    working: str
    final_answer: str


class Diagnosis(_Strict):
    analysis: str
    verdict: Literal["production answer is right", "production answer is wrong", "unclear"]


@dataclass
class Attempt:
    kind: str              # "blind 1", "blind 2"
    final_answer: str
    agrees: bool
    comparison: dict


@dataclass
class Explanation:
    record_id: str
    question: str
    production_answer: str
    decision: str = "not explained"    # "explained" | "not explained"
    reason: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    diagnosis: dict | None = None
    transcript: dict | None = None
    cost: float = 0.0


def _blind_solve(question: str, tag: str) -> Worked:
    prompt = SOLVE_PROMPT.format(question=question) + (
        "\n\nPut the full working in `working` and only the final answer, with units, in `final_answer`.")
    return llm.call("explainer", prompt, model=config.EXPLAINER_MODEL, schema=Worked,
                    effort=config.SOURCE_EFFORT, max_tokens=16000, tag=tag).parsed


def _diagnose(question: str, production: str, ours: list[str], tag: str) -> Diagnosis:
    prompt = (f"## Question\n{question}\n\n## Answer A (from another system)\n{production}\n\n"
              + "\n".join(f"## Answer {chr(66 + i)} (independent solve)\n{a}\n" for i, a in enumerate(ours))
              + "\nThese answers disagree. Work the problem carefully and decide whether answer A is right. "
                "Say \"unclear\" if the question is ambiguous or you cannot tell.")
    return llm.call("explainer", prompt, model=config.EXPLAINER_MODEL, schema=Diagnosis,
                    effort=config.SOURCE_EFFORT, max_tokens=16000, tag=tag).parsed


def explain(record: dict, blind_tries: int = 3) -> Explanation:
    """record: {"id", "question", "response"[, "reasoning"]} from a production log / backlog."""
    rid, question, production = record["id"], record["question"], record["response"]
    ex = Explanation(rid, question, production)
    context = f"\n\n(Reasoning shown by the original system, may be incomplete or wrong:)\n{record['reasoning']}" \
        if record.get("reasoning") else ""
    with llm.LEDGER.span() as span:
        try:
            ours, works = [], []
            for i in range(1, blind_tries + 1):
                w = _blind_solve(question, f"{rid}:explain:blind{i}")
                c: Comparison = _compare(f"production vs blind {i}", question, production, w.final_answer,
                                         f"{rid}:explain:cmp{i}")
                ex.attempts.append(Attempt(f"blind {i}", w.final_answer, c.agree, asdict(c)))
                ours.append(w.final_answer)
                works.append(w)
                first_agrees = i == 1 and c.agree
                later_all_agree = i == blind_tries and i > 1 and all(a.agrees for a in ex.attempts[1:])
                if i > 1 and not c.agree:
                    break  # a later solve disagreed too: no clean majority, go to diagnosis
                if first_agrees or later_all_agree:
                    w = works[0] if first_agrees else works[1]
                    i = 1 if first_agrees else 2
                    solution = f"{w.working}\n\nFinal answer: {w.final_answer}"
                    t = Transcript(rid, question, solution,
                                   f"explainer ({config.EXPLAINER_MODEL}), blind solve {i}, agreed with production"
                                   + (" (production reasoning was available but not used)" if context else ""),
                                   config.EXPLAINER_MODEL, 0.0)
                    ex.transcript = asdict(t)
                    ex.decision = "explained"
                    ex.reason = ("blind solve 1 agrees with the production answer" if first_agrees else
                                 f"blind solve 1 disagreed; blind solves 2-{blind_tries} all agree with production")
                    return ex
            d = _diagnose(question + context, production, ours, f"{rid}:explain:diagnose")
            ex.diagnosis = d.model_dump()
            ex.reason = (f"blind solves {[a.agrees for a in ex.attempts]} (agree with production?) "
                         f"is not a clean majority; diagnosis: {d.verdict}")
        finally:
            span.end = len(llm.LEDGER.records)
            ex.cost = span.cost()
            if ex.transcript:
                ex.transcript["cost"] = ex.cost
    return ex


def transcript_of(ex: Explanation) -> Transcript:
    assert ex.transcript is not None, "only explained records have a transcript"
    return Transcript(**ex.transcript)


def save(ex: Explanation) -> None:
    folder = config.problem_dir(ex.record_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "explanation.json").write_text(json.dumps(asdict(ex), indent=2, default=str))
