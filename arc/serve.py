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


class ExtractionError(CacheInvocationError):
    """IS2 output failed a code check (number doesn't match its evidence, evidence not in the
    question, nothing extracted). The question goes to the model instead."""


def extract_parameters(request: str, params: list[ParamSpec], tag: str | None = None,
                       allow_implied: bool = True, form_out: dict | None = None,
                       example: str | None = None) -> dict[str, str]:
    """IS2: the target model extracts each expected parameter from the raw input.

    config.EXTRACT_FORM "copy" (paper): each value copied as written, e.g. "$150k".
    config.EXTRACT_FORM "number_unit" (DEVIATION, default): each value comes back as separate
    fields, number (full plain digits, never shorthand), unit, text (non-numbers) and evidence
    (the exact words). Code checks the number against its evidence, then rebuilds one value in
    the style of the format examples ("$150,000"), so cached code never sees shorthand.

    allow_implied=False (spec gate) drops the paper-style "implied value" rule, which is
    where made-up values such as "300 K" came from.

    config.EXTRACT_FORM "evidence" (DEVIATION, opt-in): the text form plus the exact words that
    state each value (checked in code), no value without words in the question behind it, and `example` (the question the entry
    was built from) shown with its original values as a worked example."""
    if config.EXTRACT_FORM == "evidence":
        from .extraction import extract_evidence
        return extract_evidence(request, params, tag=tag, example=example, form_out=form_out)
    if config.EXTRACT_FORM == "text":
        from .extraction import extract_text
        return extract_text(request, params, tag=tag, allow_implied=allow_implied, form_out=form_out)
    if config.EXTRACT_FORM == "number_unit":
        from .extraction import extract_number_unit
        return extract_number_unit(request, params, tag=tag, allow_implied=allow_implied, form_out=form_out)
    fields = {f"p{i}": (str, ...) for i in range(len(params))}
    Model = create_model("ExtractedParameters", __config__=ConfigDict(extra="forbid"), **fields)
    lines = []
    for i, p in enumerate(params):
        examples = ", ".join(repr(x) for x in p.format_examples) or "none"
        lines.append(f"p{i}: {p.description} (format examples: {examples})")
    prompt = (
        "Extract the following values from the input text. Copy each value as it appears in "
        "the text, including its unit when it has one. " +
        ("If a value is implied rather than stated, write it in the style of the format examples.\n\n"
         if allow_implied else
         "Write each value in the style of its format examples. Never assume or compute a value.\n\n") +
        "Values to extract:\n" + "\n".join(lines) + f"\n\nInput text:\n{request}"
    )
    result = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Model,
                      max_tokens=1024, tag=tag)
    data = result.parsed.model_dump()
    values = {p.description: data[f"p{i}"] for i, p in enumerate(params)}
    if config.EXPAND_SHORTHAND:   # DEVIATION: "$150k" -> "$150,000" before cached code parses it
        from .normalize import expand_values
        values = expand_values(values)
    return values


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


def _find(cache: Cache, request: str, qspec: dict | None = None) -> tuple[EntryPoint | None, float]:
    if config.INDEX == "spec" and qspec is not None and cache.specs:
        from .spec import rank
        top = rank(cache, qspec, k=1)
        return (top[0][0], top[0][1]) if top else (None, -1.0)
    if config.INDEX == "cards" and cache.cards:
        from .verify import rank_entries
        top = rank_entries(cache, request, k=1)
        return (top[0][0], top[0][1]) if top else (None, -1.0)
    return cache.find_entry_point(request)


def _entry_similarity(cache: Cache, request: str, ep: EntryPoint) -> float:
    """Best cosine between the request and the texts IS1 would use for this entry."""
    texts = [e.text for e in cache.entry_points if e.problem_id == ep.problem_id] or [ep.text]
    if config.INDEX == "cards" and ep.problem_id in cache.cards:
        card = cache.cards[ep.problem_id]
        texts = [t for t in [*card.get("index_texts", []), card.get("intent", "")] if t] or texts
    from . import embed
    return embed.best_match(request, texts)[1]


def _use_clef(cache: Cache, request: str, tag: str | None, trace: dict, shadow: bool,
              entry_point: EntryPoint | None, before: int) -> CacheResult | None:
    """ARC_GATE=clef (DEVIATION): structure retrieval -> Clef decides -> IS2 -> IS3 (arc/clef.py)."""
    from . import clef
    trace.update(gate="clef", index=config.CLEF_RETRIEVAL, clef_threshold=config.CLEF_THRESHOLD)
    if entry_point is not None:                                    # forced entry: Clef judges only it
        cands = [(entry_point, _entry_similarity(cache, request, entry_point))]
        trace["forced_entry"] = True
    else:
        cands = clef.candidates(cache, request, tag=tag, trace=trace)
    if not cands:
        trace.update(stage="below_threshold", entry=None, similarity=-1.0, rejected_by=[])
        return None
    try:
        ep, p, probs = clef.decide(cache, request, cands, tag=tag, trace=trace,
                                   exclude_self=entry_point is not None)   # checks: no self-vouching
    except llm.LLMError as e:                                       # never serve on a gate error
        trace.update(stage="gate_rejected", rejected_by=["clef_error"], clef_error=str(e)[:300],
                     entry=cands[0][0].problem_id, similarity=cands[0][1])
        return None
    best = ep or max(cands, key=lambda c: probs.get(c[0].problem_id, 0.0))[0]
    sim = next(s for e, s in cands if e.problem_id == best.problem_id)
    trace.update(entry=best.problem_id, entry_kind=best.kind, similarity=sim, clef_p=round(p, 4))
    if ep is None:
        trace.update(stage="gate_rejected", rejected_by=["clef"])
        if not shadow:
            return None
    try:
        values = extract_parameters(request, best.params, tag=tag,
                                    form_out=trace.setdefault("extraction", {}),
                                    example=_original_text(cache, best))   # IS2
        trace["values"] = values
        if ep is not None:
            trace.update(stage="dispatch", rejected_by=[])
        try:
            output, ctx = dispatch_root(cache, best.root_id, values)  # IS3
        except Exception as e:
            if ep is not None:
                raise
            trace["shadow_error"] = f"{type(e).__name__}: {e}"[:300]
            return None
        if ep is None:
            trace["shadow_output"] = format_output(output)
            return None
        from . import present   # DEVIATION: rounding and output units happen here, once (arc/present.py)
        output = present.apply(output, (trace.get("extraction") or {}).get("output_format"),
                               (getattr(cache, "outputs", None) or {}).get(best.problem_id))
        trace["presented"] = output
    except CacheInvocationError:
        raise
    except Exception as e:  # any error inside cached code counts as a failed invocation
        raise CacheInvocationError(f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}") from e
    cost = llm.LEDGER.cost(CACHE_ROLES | {"decider"}, since=before)
    trace["cost"] = cost
    return CacheResult(output, ctx.narration(), best, sim, values, cost, ctx.lookups)   # IS4


def use(cache: Cache, request: str, tag: str | None = None,
        trace: dict | None = None, shadow: bool = False,
        entry_point: EntryPoint | None = None) -> CacheResult | None:
    """Try the cache. Returns None on a miss; raises CacheInvocationError on failure.

    If `trace` is a dict, it is filled with what happened at each step (for evaluation).
    shadow=True (evaluation only): run the gate and the cached code even when the request
    would be turned away, and record what the code would have answered in trace, so any
    threshold or check combination can be scored afterwards from one run. Still returns
    None for anything the live policy turns away.

    entry_point (label-free certification only): skip IS1 and test this entry directly.
    The gate and the code run as usual; the cosine IS1 would have seen is still put in
    trace["similarity"], but the threshold is not applied."""
    trace = trace if trace is not None else {}
    before = len(llm.LEDGER.records)
    if config.GATE == "clef":
        return _use_clef(cache, request, tag, trace, shadow, entry_point, before)
    qspec = None
    if config.GATE == "spec":                                      # spec gate: structure first
        from . import spec as specmod
        qspec = specmod.write_spec(request, tag=tag)
        trace["spec"] = qspec
    if entry_point is not None:                                    # forced entry: no IS1
        ep, sim = entry_point, _entry_similarity(cache, request, entry_point)
        trace["forced_entry"] = True
    else:
        ep, sim = _find(cache, request, qspec)                     # IS1
    trace.update(entry=ep.problem_id if ep else None, entry_kind=ep.kind if ep else None,
                 similarity=sim, gate=config.GATE, index=config.INDEX)
    below = ep is None or (entry_point is None and sim < config.ENTRY_THRESHOLD)
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
        elif config.GATE == "spec":                                # spec gate
            if ep.problem_id not in cache.specs:
                rejected, values = ["no_spec"], {}
            else:
                d = specmod.decide(qspec, cache.specs[ep.problem_id], request)
                rejected = d["rejected"]
                trace.update(spec_checks=d["checks"], spec_scores=d["scores"], spec_score=d["score"])
                values = extract_parameters(request, ep.params, tag=tag, allow_implied=False,
                                            example=_original_text(cache, ep)) \
                    if (not rejected and not below) or shadow else {}
                trace["values"] = values
        else:
            values = extract_parameters(request, ep.params, tag=tag, example=_original_text(cache, ep))   # IS2
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


def answer_text(cache: Cache, result: "CacheResult") -> str:
    """The answer shown to a user: the outputs, plus the entry's stated conventions (how it
    rounds, which payment it totals from) so an exact amount says how it was computed.
    Comparisons use format_output alone; the conventions are for the reader."""
    text = format_output(result.output)
    rules = cache.conventions.get(result.entry_point.problem_id) or []
    return text + ("\nConventions: " + "; ".join(rules) if rules else "")


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
