"""Verify gate v2: extract-and-verify in one Haiku call, then checks in code.

DEVIATION from the paper (config.GATE == "v2"). The idea: similarity tells you what a
question is ABOUT, not what it ASKS, so embeddings only find candidates and this module
decides whether the candidate applies. The model is asked for structure (what is asked,
which method, where each given goes, the exact text each slot came from), and plain
code checks that structure. We do not trust a bare yes/no from a small model.

Checks (each recorded separately so runs can be ablated offline):
    asked_for     the question asks for the solver's target (or part of it)
    method        the method is the solver's, or not specified
    out_of_scope  none of the card's out-of-scope cases applies
    no_unused     every given in the question has a slot
    stated        every slot that affects the answer has a value from the text
    grounded      every evidence string is really in the question (catches made-up values)
    numbers       every number in the question sits inside some slot's evidence (no model)
    unit          a requested output unit is one the solver reports
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, create_model

from . import config, embed, llm
from .cache import Cache, EntryPoint

NOT_STATED = "NOT_STATED"
ASKED = ["exactly the solver's target", "part of the solver's target", "a different quantity",
         "a change or difference in the target", "the target plus something the solver does not compute"]
ASKED_OK = set(ASKED[:2])
METHOD = ["the solver's method", "a different model, equation or rule is required", "not specified"]
METHOD_OK = {METHOD[0], METHOD[2]}
ALL_CHECKS = ["asked_for", "method", "out_of_scope", "no_unused", "stated", "grounded", "numbers", "unit"]

PROMPT = """\
You are checking whether a cached solver can answer a question, and extracting its inputs.

SOLVER: {solves}
TARGET: {target}
METHOD: {method}
IT CANNOT HANDLE:
{out_of_scope}
INPUT SLOTS:
{slots}

QUESTION: {question}

Rules:
1. List EVERY number, quantity, substance and condition in the question under "items", \
copied exactly. Mark each with the slot it fills, or UNUSED if no slot fits.
2. Fill a slot only with text that appears in the question, and copy that text into \
"evidence". "value" is the same value written the way the slot's format examples are \
written (numbers written as words become digits). If a value is not written in the \
question, use NOT_STATED for both. Never assume or compute a value.
3. A slot can only be filled once. If the question gives two values for the same kind of \
thing (two volumes, two temperatures, two gases), mark the extra ones UNUSED.
4. For asked_for_match and method_match, pick the option that is literally true, not the \
closest one.
5. out_of_scope_hit: the item from IT CANNOT HANDLE that applies, copied, or "none".
6. requested_unit: the unit the question asks the answer to be given in, copied, or "none".
"""


# ----------------------------------------------------------------------------- retrieval

def rank_entries(cache: Cache, request: str, k: int = 3) -> list[tuple[EntryPoint, float, str]]:
    """Best (entry point, similarity, matched text) per problem, best first.

    config.INDEX == "cards": embed against each card's index_texts + intent only (fix #2).
    Otherwise the paper's index: the stored original question and the template."""
    originals = {e.problem_id: e for e in cache.entry_points if e.kind == "original"}
    if config.INDEX == "cards" and cache.cards:
        table = [(pid, t) for pid, c in cache.cards.items() if pid in originals
                 for t in [*c["index_texts"], c["intent"]]]
        eps = {pid: originals[pid] for pid in originals}
    else:
        table = [(i, e.text) for i, e in enumerate(cache.entry_points)]
        eps = dict(enumerate(cache.entry_points))
    best: dict[Any, tuple[EntryPoint, float, str]] = {}
    for i, sim in embed.ranked(request, [t for _, t in table]):
        key, text = table[i]
        ep = eps[key]
        if ep.problem_id not in best:
            best[ep.problem_id] = (ep, sim, text)
        if len(best) == k:
            break
    return sorted(best.values(), key=lambda x: -x[1])


# ----------------------------------------------------------------------------- the call

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _covered_numbers(question: str, evidences: list[str]) -> list[str]:
    """Numbers in the question that do NOT fall inside any evidence span (by position)."""
    q = _norm(question)
    spans = []
    for ev in evidences:
        ev = _norm(ev)
        if not ev:
            continue
        start = q.find(ev)
        while start != -1:
            spans.append((start, start + len(ev)))
            start = q.find(ev, start + 1)
    return [m.group() for m in NUM.finditer(q)
            if not any(a <= m.start() and m.end() <= b for a, b in spans)]


def _unit_ok(requested: str, output_units: list[str]) -> bool:
    r = _norm(requested).strip(" .")
    if r in {"", "none"} or not output_units:
        return True
    alias = {"dollars": "$", "usd": "$", "dollar": "$", "percent": "%", "litres": "l", "liters": "l"}
    r = alias.get(r, r)
    units = {alias.get(_norm(u), _norm(u)) for u in output_units}
    return r in units or any(r == u.split("(")[0].strip() for u in units)


def verify(cache: Cache, request: str, ep: EntryPoint, tag: str | None = None) -> dict:
    """One Haiku call + code checks. Returns values for the code, every check, and the raw form."""
    card = cache.cards[ep.problem_id]
    keys = [f"p{i}" for i in range(len(ep.params))]
    notes = {n["slot"]: n for n in card["slots"]}
    slot_lines = []
    for key, p in zip(keys, ep.params):
        n = notes.get(key, {})
        examples = ", ".join(map(repr, p.format_examples)) or "none"
        slot_lines.append(f"{key}: {p.description} (accepts: {n.get('accepts', 'see examples')}; "
                          f"format examples: {examples})")

    SlotFill = create_model("SlotFill", __config__=ConfigDict(extra="forbid"), value=(str, ...), evidence=(str, ...))
    Slots = create_model("Slots", __config__=ConfigDict(extra="forbid"), **{k: (SlotFill, ...) for k in keys})
    Use = Literal.__getitem__(tuple(keys + ["UNUSED"]))
    Item = create_model("Item", __config__=ConfigDict(extra="forbid"), span=(str, ...), use=(Use, ...))
    Form = create_model(
        "VerifyForm", __config__=ConfigDict(extra="forbid"),
        asked_for_text=(str, ...),
        asked_for_match=(Literal.__getitem__(tuple(ASKED)), ...),
        method_match=(Literal.__getitem__(tuple(METHOD)), ...),
        items=(list[Item], ...),
        slots=(Slots, ...),
        out_of_scope_hit=(str, ...),
        requested_unit=(str, ...),
    )
    prompt = PROMPT.format(solves=card["solves"], target=card["target"], method=card["method"],
                           out_of_scope="\n".join(f"- {x}" for x in card["out_of_scope"]),
                           slots="\n".join(slot_lines), question=request)
    form = llm.call("extractor", prompt, model=config.TARGET_MODEL, schema=Form,
                    max_tokens=2048, tag=tag).parsed.model_dump()

    fills = form["slots"]
    stated = lambda f: f["value"].strip() not in {"", NOT_STATED} and f["evidence"].strip() not in {"", NOT_STATED}
    needed = [k for k in keys if notes.get(k, {}).get("affects_answer", True)]
    q = _norm(request)
    evidences = [fills[k]["evidence"] for k in keys if stated(fills[k])]
    uncovered = _covered_numbers(request, evidences)
    checks = {
        "asked_for": form["asked_for_match"] in ASKED_OK,
        "method": form["method_match"] in METHOD_OK,
        "out_of_scope": _norm(form["out_of_scope_hit"]).strip(" .\"'") in {"none", ""},
        "no_unused": not any(i["use"] == "UNUSED" for i in form["items"]),
        "stated": all(stated(fills[k]) for k in needed),
        "grounded": all(_norm(fills[k]["evidence"]) in q for k in keys if stated(fills[k])),
        "numbers": not uncovered,
        "unit": _unit_ok(form["requested_unit"], card.get("output_units", [])),
    }
    values = {p.description: (fills[k]["value"] if stated(fills[k]) else "not stated")
              for k, p in zip(keys, ep.params)}
    return {"values": values, "checks": checks, "uncovered_numbers": uncovered, "form": form}


def enabled_checks() -> list[str]:
    return ALL_CHECKS if config.GATE_CHECKS == "all" else [c for c in config.GATE_CHECKS.split(",") if c]


def failed(checks: dict, enabled: list[str] | None = None) -> list[str]:
    return [c for c in (enabled if enabled is not None else enabled_checks()) if not checks.get(c, True)]
