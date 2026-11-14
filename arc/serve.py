"""Cache invocation (paper Section 2.2.2) and gated serving (Figure 1).

    IS1 entry-point search    embed the request, cosine vs entry points
    IS2 parameter extraction  T extracts each expected parameter
    IS3 dispatch              fresh Context, run the root function
    IS4 output                named results + narration trace

`serve()` tries the cache first and falls through to the source LLM on a miss
or on any error, exactly like the paper's Figure 1.
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ConfigDict, create_model

from . import config, llm, sandbox
from .cache import Cache, EntryPoint, ParamSpec
from .runtime import CacheInvocationError, Context

CACHE_ROLES = {"target", "extractor", "converter"}


@dataclass
class CacheResult:
    output: dict[str, Any]
    narration: str
    entry_point: EntryPoint
    similarity: float
    parameters: dict[str, str]
    cost: float
    lookups: list = field(default_factory=list)


@dataclass
class Served:
    answer: str
    from_cache: bool
    cache_result: CacheResult | None
    reason: str
    cost: float


def extract_parameters(request: str, params: list[ParamSpec], tag: str | None = None) -> dict[str, str]:
    """IS2: the target model extracts each expected parameter from the raw input."""
    fields = {f"p{i}": (str, ...) for i in range(len(params))}
    Model = create_model("ExtractedParameters", __config__=ConfigDict(extra="forbid"), **fields)
    lines = []
    for i, p in enumerate(params):
        examples = ", ".join(repr(x) for x in p.format_examples) or "none"
        lines.append(f"p{i}: {p.description} (format examples: {examples})")
    prompt = (
        "Extract the following values from the input text. Copy each value as it appears in "
        "the text, including its unit when it has one. If a value is implied rather than "
        "stated, write it in the style of the format examples.\n\n"
        "Values to extract:\n" + "\n".join(lines) + f"\n\nInput text:\n{request}"
    )
    result = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Model,
                      max_tokens=1024, tag=tag)
    data = result.parsed.model_dump()
    return {p.description: data[f"p{i}"] for i, p in enumerate(params)}


GATE_RULES = """\
Rules:
- For each input, copy the value as it appears in the request, including its unit when it has \
one. If a value is implied rather than stated, write it in the style of the format examples. If the request does not give the value and \
it cannot be clearly implied, return null. Never fill a value from the example question.
- asked_for: in a few words, what the request asks you to find.
- unstated_facts: list every condition, constant, quantity or instruction in the request that \
would change the correct answer but has no matching input above (for example a different \
equation or model, a correction factor, a second state or step, an extra item, a different \
target quantity). Leave out details that do not change the answer, such as names, labels \
or background.
- fits: true only if running the procedure on the extracted inputs gives exactly what the \
request asks for: the same target quantity, the same method and assumptions, every needed input \
present, and nothing in unstated_facts. Different numbers, units, wording or numbers written as \
words are fine."""


@dataclass
class GateVerdict:
    fits: bool
    asked_for: str
    unstated_facts: list[str]
    empty_slots: list[str]
    reason: str

    @property
    def rejected_by(self) -> list[str]:
        why = []
        if self.empty_slots:
            why.append("empty_slot")
        if self.unstated_facts:
            why.append("unstated_facts")
        if not self.fits:
            why.append("does_not_fit")
        return why


def _original_text(cache: Cache, ep: EntryPoint) -> str:
    """The question the entry was built from (what the gate shows as the example)."""
    for e in cache.entry_points:
        if e.root_id == ep.root_id and e.kind == "original":
            return e.text
    return ep.text


def extract_and_check(cache: Cache, request: str, ep: EntryPoint,
                      tag: str | None = None) -> tuple[dict[str, str | None], GateVerdict]:
    """IS2 with the turn-away gate (DEVIATION, config.GATE == "on").

    One Haiku call, same as plain IS2, but the form now lets the model leave a slot empty,
    list facts the procedure would ignore, and say whether the procedure fits at all.
    Reasoning fields come before the verdict so the model writes them first."""
    root = cache.functions[ep.root_id]
    params = ep.params
    fields: dict[str, Any] = {"asked_for": (str, ...)}
    fields.update({f"p{i}": (str | None, ...) for i in range(len(params))})
    fields.update({"unstated_facts": (list[str], ...), "reason": (str, ...), "fits": (bool, ...)})
    Model = create_model("GatedExtraction", __config__=ConfigDict(extra="forbid"), **fields)
    lines = []
    for i, p in enumerate(params):
        examples = ", ".join(repr(x) for x in p.format_examples) or "none"
        lines.append(f"p{i}: {p.description} (format examples: {examples})")
    outputs = ", ".join(root.outputs) or "(not listed)"
    prompt = (
        "A cached procedure answers one kind of question. Decide whether it answers the request "
        "below exactly, and extract its inputs.\n\n"
        f"Procedure: {root.description}\n"
        f"It returns: {outputs}\n"
        f"Example question it was built from: {_original_text(cache, ep)}\n\n"
        "Inputs it needs:\n" + "\n".join(lines) + "\n\n" + GATE_RULES +
        f"\n\nRequest:\n{request}"
    )
    result = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Model,
                      max_tokens=1024, tag=tag)
    data = result.parsed.model_dump()
    values = {p.description: data[f"p{i}"] for i, p in enumerate(params)}
    empty = [k for k, v in values.items() if v is None or not str(v).strip()]
    facts = [f for f in data["unstated_facts"] if f and f.strip()]
    return values, GateVerdict(bool(data["fits"]), data["asked_for"], facts, empty, data["reason"])


def dispatch_root(cache: Cache, root_id: str, values: dict[str, Any]) -> tuple[dict, Context]:
    """IS3: a fresh Context populated with the parameter values, then the root runs."""
    ctx = Context(cache)
    for k, v in values.items():
        ctx.set(k, v)
    record = cache.functions[root_id]
    with sandbox.time_limit(config.FUNCTION_TIMEOUT_SECONDS):
        output = record.fn(ctx)
    if not isinstance(output, dict):
        raise CacheInvocationError(f"root returned {type(output).__name__}, not dict")
    return output, ctx


def _find(cache: Cache, request: str) -> tuple[EntryPoint | None, float]:
    if config.INDEX == "cards" and cache.cards:
        from .verify import rank_entries
        top = rank_entries(cache, request, k=1)
        return (top[0][0], top[0][1]) if top else (None, -1.0)
    return cache.find_entry_point(request)


def use(cache: Cache, request: str, tag: str | None = None,
        trace: dict | None = None, shadow: bool = False) -> CacheResult | None:
    """Try the cache. Returns None on a miss; raises CacheInvocationError on failure.

    If `trace` is a dict, it is filled with what happened at each step (for evaluation).
    shadow=True (evaluation only): run the gate and the cached code even when the request
    would be turned away, and record what the code would have answered in trace, so any
    threshold or check combination can be scored afterwards from one run. Still returns
    None for anything the live policy turns away."""
    trace = trace if trace is not None else {}
    before = len(llm.LEDGER.records)
    ep, sim = _find(cache, request)                                # IS1
    trace.update(entry=ep.problem_id if ep else None, entry_kind=ep.kind if ep else None,
                 similarity=sim, gate=config.GATE, index=config.INDEX)
    below = ep is None or sim < config.ENTRY_THRESHOLD
    if ep is None or (below and not shadow):
        trace["stage"] = "below_threshold"
        return None
    rejected: list[str] = []
    try:
        if config.GATE == "v1":                                    # IS2 + turn-away gate v1
            values, verdict = extract_and_check(cache, request, ep, tag=tag)
            rejected = verdict.rejected_by
            trace.update(values=values, verdict=asdict(verdict))
        elif config.GATE == "v2":                                  # IS2 + verify gate v2
            from . import verify
            if ep.problem_id not in cache.cards:
                rejected = ["no_card"]
                values = {}
            else:
                v = verify.verify(cache, request, ep, tag=tag)
                values = v["values"]
                rejected = verify.failed(v["checks"])
                trace.update(values=values, checks=v["checks"], uncovered_numbers=v["uncovered_numbers"],
                             form=v["form"])
        else:
            values = extract_parameters(request, ep.params, tag=tag)   # IS2
            trace["values"] = values
        trace["rejected_by"] = rejected
        turned_away = below or bool(rejected)
        trace["stage"] = "below_threshold" if below else "gate_rejected" if rejected else "dispatch"
        if turned_away and not shadow:
            return None
        if turned_away and not values:
            return None
        try:
            output, ctx = dispatch_root(cache, ep.root_id, values)     # IS3
        except Exception as e:
            if not turned_away:
                raise
            trace["shadow_error"] = f"{type(e).__name__}: {e}"[:300]
            return None
        if turned_away:
            trace["shadow_output"] = format_output(output)
            return None
    except CacheInvocationError:
        raise
    except Exception as e:  # any error inside cached code counts as a failed invocation
        raise CacheInvocationError(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}") from e
    cost = llm.LEDGER.cost(CACHE_ROLES, since=before)
    trace["cost"] = cost
    return CacheResult(output, ctx.narration(), ep, sim, values, cost, ctx.lookups)  # IS4


def format_output(output: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in output.items())


def serve(cache: Cache, request: str) -> Served:
    """Figure 1: try the cache; on a miss or error fall through to the full LLM."""
    trace: dict = {}
    try:
        result = use(cache, request, trace=trace)
        if result is not None:
            return Served(format_output(result.output), True, result, "cache hit", result.cost)
        if trace.get("stage") == "gate_rejected":
            reason = f"cache turned the request away ({', '.join(trace['rejected_by'])})"
        else:
            reason = "cache miss (no entry point close enough)"
    except CacheInvocationError as e:
        reason = f"cache error, fell back: {str(e).splitlines()[0]}"
    response = llm.call("source", request, model=config.SOURCE_MODEL, effort=config.SOURCE_EFFORT)
    return Served(response.text, False, None, reason, response.cost)
