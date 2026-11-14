"""Gate cards: per-entry metadata the verify gate checks requests against.

DEVIATION from the paper. The paper stores an entry point (the original question
plus a template) and serves anything whose embedding is close enough. A card adds
what the entry is FOR, so the gate can turn away near-misses:

    solves        one line: what the solver computes and how
    target        the quantity it returns           method   the model / rule it uses
    slots         per input slot: does it change the answer? what forms does it accept?
    out_of_scope  nearby questions it must NOT answer
    output_units  units the result can be reported in
    index_texts   8-10 builder-written rewordings (new numbers, units, wording) + intent,
                  embedded INSTEAD of the verbatim original question (fix #2)

The builder model writes it once per entry, from the same material the builder
saw (question, transcript, golden label) plus the finished code. It never sees
any evaluation question.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import config, llm
from .cache import Cache, EntryPoint
from .source import Transcript


class SlotNote(BaseModel):
    slot: str = Field(description="the slot key, e.g. p0")
    affects_answer: bool = Field(description="false only if the solver's numeric result does not depend on this slot")
    accepts: str = Field(description="units or forms of value the code can parse for this slot")


class Card(BaseModel):
    solves: str
    target: str
    method: str
    slots: list[SlotNote]
    out_of_scope: list[str]
    output_units: list[str]
    index_texts: list[str]
    intent: str


CARD_PROMPT = """\
You built a cached solver from the worked example below. Write the card the runtime \
uses to decide which incoming questions this solver may answer. A wrong "yes" serves a \
wrong answer, so be precise about the solver's limits. Base everything on what the code \
actually does.

## Example question
{question}

## Worked solution
{solution}
{golden}
## Solver code (root function first)
{code}

## Input slots
{slots}

## Solver outputs
{outputs}

Fill in:
- solves: one sentence, what it computes, from which inputs, by which method.
- target: the quantity (or quantities) it returns, in a few words.
- method: the model, equation or rule it uses, in a few words.
- slots: one note per slot key above. affects_answer is false only if the numeric result \
does not depend on the slot (for example a name used only in narration). accepts lists \
the units or forms the code can parse.
- out_of_scope: 4-8 kinds of questions that look almost the same but which this solver \
must NOT answer. Cover each that applies: a different target quantity, a different method \
or model, an extra given or condition the code ignores, a process with two states, more \
items than the code handles, a different rule or jurisdiction, a multi-part question \
asking for more than it returns.
- output_units: the units the result is reported in (empty list if the result has no unit).
- index_texts: 8 to 10 questions this solver answers correctly. Vary the wording, clause \
order, numbers, and the input units the code accepts. Each must be a question the solver \
would answer right with no changes.
- intent: one short line, e.g. "Find X of Y given A, B, C using M."
"""


def _entry_code(cache: Cache, root_id: str) -> str:
    root = cache.functions[root_id]
    others = [f for fid, f in cache.functions.items() if fid != root_id]
    parts = [f"# {root.description}\n{root.source}"] + [f"# {f.description}\n{f.source}" for f in others]
    return "\n\n".join(parts)


def make_card(cache: Cache, problem: dict, transcript: Transcript, use_golden: bool = True) -> dict:
    """One builder-model call. `cache` is the per-sample cache for this problem."""
    ep: EntryPoint = next(e for e in cache.entry_points
                          if e.problem_id == problem["id"] and e.kind == "original")
    root = cache.functions[ep.root_id]
    slots = "\n".join(f"p{i}: {p.description} (format examples: {', '.join(map(repr, p.format_examples))})"
                      for i, p in enumerate(ep.params))
    golden = f"\n## Golden label\n{problem['golden_label']}\n" if use_golden else ""
    prompt = CARD_PROMPT.format(question=transcript.question, solution=transcript.solution, golden=golden,
                                code=_entry_code(cache, ep.root_id), slots=slots,
                                outputs=", ".join(root.outputs) or "(not listed)")
    result = llm.call("builder", prompt, model=config.BUILDER_MODEL, schema=Card,
                      effort=config.BUILDER_EFFORT, max_tokens=8000, tag=f"{problem['id']}:card")
    card = result.parsed.model_dump()
    card["root_id"] = ep.root_id
    return card
