"""Step 3: the source model S solves the original problem, producing S(q(o)).

Opus 5.5 keeps its raw thinking private, so we ask it to write its full
working in the visible answer. That written solution is the transcript the
builder consumes, the same kind of artifact as the paper's Figure 4. The
thinking summary is saved separately, for humans only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from . import config, llm

SOLVE_PROMPT = """Solve the following problem.

In your answer, write out your full working step by step: list the given information, \
state the formulas or principles you use, show every substitution and intermediate \
calculation, and finish with a clearly stated final answer.

Problem:
{question}"""


@dataclass
class Transcript:
    problem_id: str
    question: str
    solution: str            # visible worked solution: the transcript the builder uses
    thinking_summary: str    # for human reading only
    model: str
    cost: float


def solve(problem: dict) -> Transcript:
    result = llm.call("source", SOLVE_PROMPT.format(question=problem["question"]),
                      model=config.SOURCE_MODEL, effort=config.SOURCE_EFFORT,
                      show_thinking=True, tag=f"source:{problem['id']}")
    return Transcript(problem["id"], problem["question"], result.text, result.thinking,
                      result.model, result.cost)


def save(t: Transcript) -> None:
    folder = config.problem_dir(t.problem_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transcript.json").write_text(json.dumps(asdict(t), indent=2))
    (folder / "transcript.md").write_text(
        f"# Source transcript: {t.problem_id}\n\n**Model:** {t.model}  \n"
        f"**Cost:** ${t.cost:.4f}\n\n## Question\n\n{t.question}\n\n"
        f"## Worked solution (used by the builder)\n\n{t.solution}\n\n"
        f"## Thinking summary (for humans only)\n\n{t.thinking_summary or '(none returned)'}\n"
    )


def load(problem_id: str) -> Transcript:
    data = json.loads((config.problem_dir(problem_id) / "transcript.json").read_text())
    return Transcript(**data)
