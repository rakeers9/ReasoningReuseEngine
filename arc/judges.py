"""LM judges J: language models asked to verify correctness and faithfulness."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from . import config, llm


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    critique: str


def judge(prompt: str, role: str = "judge", tag: str | None = None) -> Verdict:
    system = ("You are a meticulous verifier. Judge only what you are asked. Treat answers as "
              "equivalent when they agree up to rounding and unit conversion. Keep the critique "
              "short and specific.")
    return llm.call(role, prompt, model=config.JUDGE_MODEL, system=system, schema=Verdict,
                    effort=config.JUDGE_EFFORT, max_tokens=8000, tag=tag).parsed


def committee(prompt: str, role: str = "judge", tag: str | None = None,
              size: int = config.JUDGE_COMMITTEE_SIZE) -> tuple[bool, list[Verdict]]:
    """A committee of independent judges; acceptance requires unanimity (paper Sec. 3.4)."""
    verdicts = [judge(prompt, role=role, tag=tag) for _ in range(size)]
    return all(v.passed for v in verdicts), verdicts
