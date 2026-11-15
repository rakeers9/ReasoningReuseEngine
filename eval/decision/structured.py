"""Structured input for the decision suite: compare structure to structure, not raw text.

eval/spec writes Haiku "structures" for the 17 seeds and the dev questions: one
fixed, task-agnostic form with the values taken out of the text fields:

    task, deliverable, method, conditions, inputs [{name, value, evidence}], output_form

This module reads that file (tolerant of a few shapes), renders a structure as text, and
supplies the prompts and the code checks. Nothing here calls a model.

Input kinds (run_decision.py --input)
    raw     the question and seed question as written (the original suite)
    struct  whole structure vs whole structure, one yes/no per candidate
    fields  one yes/no per field per candidate; a candidate passes only if every field does
            (P = min over fields)

Code checks (--code-checks, applied at scoring time, free): a question whose structure has an
input with no value, or with evidence that is not in the question text, is rejected outright.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

FIELDS = ["task", "deliverable", "method", "conditions", "output_form"]
# fields mode also compares the input list: a request missing an input the solver needs
# (e.g. Ka) simply lists fewer inputs, which only this comparison can see
COMPARE = FIELDS + ["inputs"]
UNSTATED = {"", "not stated", "not_stated", "none", "null", "unknown", "n/a", "unspecified"}
ALIASES = {"output form": "output_form", "outputform": "output_form", "output": "output_form",
           "condition": "conditions", "qualifiers": "conditions"}

STRUCT_PROMPT = (
    "A cached solver handles requests with this structure (input values left out):\n"
    "<<<{seed}>>>\n"
    "The state is a new request written in the same structure. Answer yes only if the solver "
    "answers the new request exactly: the same task and deliverable, the same method (or no "
    "method stated), the same conditions with none added, dropped or changed, the same inputs, "
    "and the same output form. Different input values do not matter."
)

FIELD_PROMPT = (
    "A cached solver's {label} is: <<<{seed}>>>. Is the {label} of the request in the state "
    "exactly the same thing? Wording may differ, meaning may not.{extra}"
)
FIELD_EXTRA = {
    "method": " If the request states no method, answer yes.",
    "conditions": " Every qualifier must match: none added, dropped or changed.",
    "inputs": " The request must give every input the solver needs, and no extra input that would "
              "change the answer. Names may be worded differently.",
}
FIELD_LABEL = {"task": "task", "deliverable": "deliverable (what it produces)", "method": "method",
               "conditions": "set of conditions", "output_form": "output form",
               "inputs": "list of inputs"}


def _norm_key(k: str) -> str:
    k = k.strip().lower().replace("-", "_")
    return ALIASES.get(k, ALIASES.get(k.replace("_", " "), k.replace(" ", "_")))


def _unwrap(s: dict) -> dict:
    for k in ("structure", "spec", "form", "struct"):
        if isinstance(s.get(k), dict):
            return s[k]
    return s


def normalize(s: dict) -> dict:
    s = {_norm_key(k): v for k, v in _unwrap(s).items()}
    out = {}
    for f in FIELDS:
        v = s.get(f)
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v if str(x).strip())
        out[f] = (str(v).strip() if v is not None else "")
    inputs = s.get("inputs") or []
    if isinstance(inputs, dict):
        inputs = [{"name": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in inputs.items()]
    out["inputs"] = [i if isinstance(i, dict) else {"name": str(i)} for i in inputs]
    return out


def load(path: str | Path) -> dict[str, dict]:
    """{corpus row id: normalized structure}. Accepts a JSON dict keyed by id, a JSON list,
    {"rows"/"structures"/"items": ...}, or JSONL; each item carries "id" (or "row_id")."""
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl":
        items = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        data = json.loads(text)
        for k in ("rows", "structures", "items", "data"):
            if isinstance(data, dict) and k in data:
                data = data[k]
                break
        items = (data if isinstance(data, list)
                 else [{"id": k, **(v if isinstance(v, dict) else {})} for k, v in data.items()])
    # eval/spec/specs.jsonl keys dev rows by corpus id and seeds as "seed:<entry>"
    corpus = Path(__file__).resolve().parents[1] / "corpus" / "cache_eval_corpus.jsonl"
    seed_row = {r["seed_entry"]: r["id"] for r in map(json.loads, corpus.read_text().splitlines())
                if r.get("split") == "seed"}
    out = {}
    for it in items:
        rid = str(it.get("id") or it.get("row_id") or it.get("qid") or it.get("key") or "")
        if rid.startswith("seed:"):
            rid = seed_row.get(rid[5:], rid)
        if rid:
            out[rid] = normalize(it)
    if not out:
        raise SystemExit(f"no structures with an id found in {p}")
    return out


def render(s: dict) -> str:
    names = ", ".join(str(i.get("name", "?")) for i in s["inputs"]) or "none"
    return "\n".join([f"{f.replace('_', ' ')}: {s[f] or 'not stated'}" for f in FIELDS] + [f"inputs: {names}"])


def field_text(s: dict, f: str) -> str:
    if f == "inputs":
        return "; ".join(str(i.get("name", "?")) for i in s["inputs"]) or "not stated"
    return s[f] or "not stated"


def unstated(v: str) -> bool:
    return v.strip().lower() in UNSTATED


def _n(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t)).lower()).strip()


def code_check(s: dict, question: str) -> list[str]:
    """Reasons to reject without asking a model: an input with no value, or ungrounded evidence."""
    q, why = _n(question), []
    for i in s["inputs"]:
        name = i.get("name", "?")
        val = str(i.get("value", "") if i.get("value") is not None else "")
        if unstated(val):
            why.append(f"missing:{name}")
            continue
        ev = i.get("evidence")
        if ev and not unstated(str(ev)) and _n(ev) not in q:
            why.append(f"ungrounded:{name}")
    return why
