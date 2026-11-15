"""Spec gate: rewrite every request into one fixed structure, then compare field by field.

DEVIATION from the paper (config.GATE == "spec"). Embedding a whole question measures
what it is ABOUT, so a near-miss (same words, one changed qualifier) scores higher than a
real rewording. Here Haiku rewrites BOTH sides into the same structure with the values
taken out, using the same prompt and model:

    task          compute | write | summarize | decide | extract | explain | translate | other
    deliverable   what must be produced, every qualifier kept, no values
    method        the method / model / rule the request names or requires, or "not stated"
    conditions    every condition that changes the answer, one per line, no values
    inputs        every given the answer depends on: name (no value), value, evidence
    output_form   the form of the answer, plus any unit or format the request asks for

Entry side: the same call on the entry's seed question, stored in cache.specs.
Query side: one call per request. If the request is accepted, the paper's IS2 extraction
fills the entry's slots (without the "implied value" rule).

Decision = code checks + a field-by-field comparison by a pluggable decider:
    grounded      every input's evidence is really in the request (catches made-up values)
    has_values    every input has a value
    deliverable / method / conditions / inputs / output_form
                  decider score per field; lists are matched both ways, so a missing or
                  an extra condition or input pulls the field score down
A request passes when every code check holds and every field score >= SPEC_THRESHOLD.
Every field score is recorded, so thresholds and field subsets can be tuned offline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import config, embed, llm
from .cache import Cache, EntryPoint

NOT_STATED = "not stated"
FIELDS = ["deliverable", "method", "conditions", "inputs", "output_form"]
CODE_CHECKS = ["grounded", "has_values"]


class SpecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="general name of the given, with no value")
    value: str = Field(description="the value as written in the request")
    evidence: str = Field(description="exact text copied from the request")


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["compute", "write", "summarize", "decide", "extract", "explain", "translate", "other"]
    deliverable: str
    method: str
    conditions: list[str]
    inputs: list[SpecInput]
    output_form: str


PROMPT = """\
Rewrite the request below into a fixed structure. Describe what the request asks; do \
not solve it. Write in short, general, plain words, so two requests that ask for the \
same thing come out the same even if they are worded differently.

Fields:
- task: the kind of work asked for.
- deliverable: what must be produced, with NO numbers, names or specific values. Keep \
every word that changes what is produced, such as "change in", "total", "maximum", \
"per year", "not", "top" or "bottom", "after", "remaining".
- method: the method, model, equation, rule or assumption the request names or clearly \
requires, in a few words. Write "not stated" if it names none.
- conditions: every condition that changes the answer, one short line each, with no \
values. For example: one item or several, one state or a process between two states, \
a mixture, timing, ordering, a jurisdiction, an extra step, a constraint. Do not repeat \
the inputs here. Do not name specific substances, people, places or items: write the \
kind of thing instead ("a single gas", not "ethane"). Leave out background that does \
not change the answer. Use an empty list if there are none.
- inputs: every given value the answer depends on. name is a general name for it with no \
value (for example "amount of gas", "loan principal", "number of people"). value is the \
value as written. evidence is the exact text copied from the request that states it. \
Never assume, convert or compute a value, and never add a value the request does not state.
- output_form: the form of the answer (for example "a number with a pressure unit", \
"a date", "one SQL SELECT statement", "a short paragraph"), plus any unit or format the \
request asks for.

Request:
{request}"""


def write_spec(text: str, tag: str | None = None) -> dict:
    """One Haiku call. The same function writes entry specs and query specs."""
    result = llm.call("extractor", PROMPT.format(request=text), model=config.TARGET_MODEL,
                      schema=Spec, max_tokens=1500, tag=tag)
    return result.parsed.model_dump()


def spec_text(spec: dict) -> str:
    """The structure as one string with every value removed (what retrieval embeds)."""
    return "\n".join([
        f"task: {spec['task']}",
        f"deliverable: {spec['deliverable']}",
        f"method: {spec['method']}",
        f"conditions: {'; '.join(spec['conditions']) or 'none'}",
        f"inputs: {'; '.join(i['name'] for i in spec['inputs']) or 'none'}",
        f"output: {spec['output_form']}",
    ])


# ----------------------------------------------------------------------------- deciders
# A decider scores short text pairs: list[(a, b)] -> scores, higher = more likely the same
# meaning. Register new ones with register_decider(name, fn).

Decider = Callable[[list[tuple[str, str]]], list[float]]
_DECIDERS: dict[str, Decider] = {}
_LOADED: dict[str, object] = {}


def register_decider(name: str, fn: Decider) -> None:
    _DECIDERS[name] = fn


def _embed_decider(pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    texts = sorted({t for p in pairs for t in p})
    vecs = dict(zip(texts, embed.embedder().encode(texts)))
    return [float(vecs[a] @ vecs[b]) for a, b in pairs]


NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def _nli_decider(pairs: list[tuple[str, str]]) -> list[float]:
    """Equivalence = entailment both ways: min(P(a entails b), P(b entails a))."""
    if not pairs:
        return []
    if "nli" not in _LOADED:
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(NLI_MODEL, device="cpu")
        labels = {v.lower(): int(k) for k, v in m.model.config.id2label.items()}
        _LOADED["nli"] = (m, labels["entailment"])
    m, ent = _LOADED["nli"]

    def p(xs):
        logits = np.asarray(m.predict(xs, batch_size=32), dtype=float)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, ent]

    return np.minimum(p(pairs), p([(b, a) for a, b in pairs])).tolist()


register_decider("embed", _embed_decider)
register_decider("nli", _nli_decider)


def decider(name: str | None = None) -> Decider:
    return _DECIDERS[name or config.SPEC_DECIDER]


# ----------------------------------------------------------------------------- comparison

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def _stated(s: str) -> bool:
    return _norm(s).strip(" .") not in {"", "not stated", "none", "n/a"}


def code_checks(spec: dict, request: str) -> dict[str, bool]:
    q = _norm(request)
    return {
        "grounded": all(_norm(i["evidence"]) in q for i in spec["inputs"] if _stated(i["evidence"])),
        "has_values": all(_stated(i["value"]) and _stated(i["evidence"]) for i in spec["inputs"]),
    }


def _field_jobs(q: dict, e: dict) -> dict[str, list]:
    """What each field compares. A string field is one pair; a list field is every
    (entry item, query item) pair, matched both ways later. None = passes without a score."""
    names = lambda s: [i["name"] for i in s["inputs"]]
    jobs: dict[str, list] = {
        "deliverable": [(e["deliverable"], q["deliverable"])],
        # an unstated method in the request is fine: the entry's method is the default
        "method": None if not _stated(q["method"]) else [(e["method"], q["method"])],
        "conditions": ("list", e["conditions"], q["conditions"]),
        "inputs": ("list", names(e), names(q)),
        "output_form": [(e["output_form"], q["output_form"])],
    }
    return jobs


def compare_many(items: list[tuple[dict, dict]], decider_name: str | None = None) -> list[dict[str, float]]:
    """Field scores for many (query spec, entry spec) pairs, batched through the decider."""
    fn = decider(decider_name)
    plan, pairs = [], []
    for q, e in items:
        jobs = _field_jobs(q, e)
        rec = {}
        for f, job in jobs.items():
            if job is None:
                rec[f] = None
            elif isinstance(job, tuple):
                _, a, b = job
                idx = []
                for x in a:
                    for y in b:
                        idx.append(len(pairs))
                        pairs.append((x, y))
                rec[f] = ("list", len(a), len(b), idx)
            else:
                rec[f] = ("one", len(pairs))
                pairs.extend(job)
        plan.append(rec)
    scores = fn(pairs) if pairs else []
    out = []
    for rec in plan:
        row = {}
        for f, how in rec.items():
            if how is None:
                row[f] = 1.0
            elif how[0] == "one":
                row[f] = float(scores[how[1]])
            else:
                _, na, nb, idx = how
                if na == 0 and nb == 0:
                    row[f] = 1.0
                elif na == 0 or nb == 0:
                    row[f] = 0.0  # one side has items the other lacks entirely
                else:
                    m = np.array([scores[i] for i in idx]).reshape(na, nb)
                    # every entry item must have a match in the query, and vice versa
                    row[f] = float(min(m.max(axis=1).min(), m.max(axis=0).min()))
        out.append(row)
    return out


def decide(query: dict, entry: dict, request: str, decider_name: str | None = None,
           threshold: float | None = None, fields: list[str] | None = None) -> dict:
    t = config.SPEC_THRESHOLD if threshold is None else threshold
    use = fields or FIELDS
    checks = code_checks(query, request)
    scores = compare_many([(query, entry)], decider_name)[0]
    rejected = [c for c in CODE_CHECKS if not checks[c]] + [f for f in use if scores[f] < t]
    return {"checks": checks, "scores": scores, "score": min(scores[f] for f in use),
            "rejected": rejected}


# ----------------------------------------------------------------------------- cache side

def index_cache(cache: Cache, force: bool = False) -> int:
    """Write a spec for every entry's original question (same prompt as the query side).
    Returns how many were written. Caller saves the cache."""
    n = 0
    for ep in cache.entry_points:
        if ep.kind != "original" or (ep.problem_id in cache.specs and not force):
            continue
        cache.specs[ep.problem_id] = write_spec(ep.text, tag=f"{ep.problem_id}:spec")
        n += 1
    return n


def rank(cache: Cache, query: dict, k: int = 1) -> list[tuple[EntryPoint, float]]:
    """Retrieval on spec text (config.INDEX == "spec"): best entries by cosine."""
    originals = {e.problem_id: e for e in cache.entry_points if e.kind == "original"}
    pids = [p for p in cache.specs if p in originals]
    if not pids:
        return []
    ranked = embed.ranked(spec_text(query), [spec_text(cache.specs[p]) for p in pids])
    return [(originals[pids[i]], sim) for i, sim in ranked[:k]]


if __name__ == "__main__":
    # .venv/bin/python -m arc.spec <cache folder>   (PAID: one Haiku call per entry)
    import sys
    from pathlib import Path
    folder = Path(sys.argv[1])
    c = Cache.load(folder)
    written = index_cache(c, force="--force" in sys.argv)
    c.save(folder)
    print(f"wrote {written} entry specs to {folder}, cost ${llm.LEDGER.cost():.4f}")
