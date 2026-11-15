"""Clef gate: which cache entry, if any, really fits a question (ARC_GATE=clef).

DEVIATION from the paper, which serves the entry whose embedding is closest if it clears
a threshold. Near-miss questions (same wording, different reasoning) clear it too and
get a confident wrong answer. Here:

  1. candidates  Haiku rewrites the question into its value-free structure (arc/spec.py),
                 and the top CLEF_TOP_K entries by structure cosine are taken, above a
                 threshold or not. Structure text roughly doubles retrieval quality on
                 near-misses (AUC 0.33 -> 0.80 on dev). Entries without a stored structure
                 are ranked on the raw text instead.
  2. decide      Clef 27B (Cloudflare Workers AI) reads the RAW question (state) against
                 one yes/no question per candidate, built from that entry's original
                 question. Raw text measured better for Clef than structures.
  3. serve       the best candidate if P >= CLEF_THRESHOLD; otherwise nothing, and the
                 question goes to the model. A Clef error never serves.

Prompts:
  v1  the first prompt measured in eval/decision.
  v2  written for this use case: adds what the entry computes, its method, conditions and
      inputs (from its stored structure), and spells out each near-miss type.
  v3  v2 plus a "read every word" rule for small wording changes. Default (user decision
      2026-11-15): dev top-3 at 0.8, 113/119 right and 0/214 wrong (v2: 113 right, 2 wrong).
  Dev, top-3, serve the best candidate at P >= 0.8 (eval/decision --prompt v1|v2):
      v1  110/119 true matches served right, 2/214 served wrong, AUC 0.997, $0.00019/question
      v2  113/119 right, the same 2 wrong, AUC 0.999, $0.00034/question
  Both wrongly accept "less than or equal to 1000" (boundary) and "8 numbered seats"
  (rotations distinct). Below 0.8, v2 also lets a mixture and a regime change through.
"""

from __future__ import annotations

from . import config, embed, llm
from .cache import Cache, EntryPoint

PROMPT_V1 = (
    "A cached solver was built from this example question:\n"
    "<<<{seed}>>>\n"
    "It re-runs exactly the same procedure on new input values. Answer yes only if running that "
    "procedure on the values given in the question above answers that question exactly: it asks "
    "for the same result, needs the same method and assumptions, gives every input the procedure "
    "needs, and adds no condition, step or item the procedure would ignore. Different numbers, "
    "units, names or wording are fine."
)

PROMPT_V2 = (
    "A cached solver was built from this example question:\n"
    "<<<{seed}>>>\n"
    "{profile}"
    "The solver re-runs exactly that calculation on new input values and can do nothing else: it "
    "cannot change its method, add a step, handle a second item, or answer a different question.\n"
    "Answer yes only if running it on the values given in the question above answers that question "
    "completely and exactly. Answer no if any of these is true:\n"
    "1. The question asks for a different result: another quantity, only an intermediate value, a "
    "change or difference in the result, or working backwards to find one of the inputs.\n"
    "2. It needs a different method or assumption: another equation, model, rule, convention, legal "
    "test, jurisdiction, compounding or payment scheme.\n"
    "3. An input the solver needs is not stated in the question.\n"
    "4. It adds something the solver would ignore: an extra condition, constraint, step, time "
    "period, second object, mixture or extra quantity that changes the answer.\n"
    "Different numbers, units, names, currencies or wording are fine, and so are background "
    "details that do not change the answer."
)


# v3 (default since 2026-11-15): v2 plus a read-every-word rule. The examples name general kinds
# of small wording change; two of them (inclusive bounds, labelled seats) are the kinds behind
# v2's two dev errors, so a dev gain must be confirmed on the test split.
CAREFUL_READING = (
    "\nRead every word of both questions and compare them closely. A small wording change can "
    "change the expected answer: an inclusive or exclusive bound (\"less than\" vs \"less than or "
    "equal to\", \"at least\" vs \"more than\"), whether items or positions are labelled or "
    "identical, \"and\" vs \"or\", \"with\" vs \"without\" replacement, or a changed time, order "
    "or direction. If any difference, however small, would change the correct output, answer no."
)
PROMPT_V3 = PROMPT_V2 + CAREFUL_READING

# v4 (2026-11-15): judge the task, not the wording. v3 rejected true variants that were the
# same task said differently ("from 1 to 999" vs "less than 1000", "multiples of" vs
# "divisible by", mg/dL vs umol/L, a fact-list style, an unused tax rate). v4 keeps every "no"
# rule and adds an explicit list of differences that do NOT change the task.
PROMPT_V4 = (
    "A cached solver was built from this example question:\n"
    "<<<{seed}>>>\n"
    "{profile}"
    "The solver re-runs exactly that calculation on new input values and can do nothing else: it "
    "cannot change its method, add a step, handle a second item, or answer a different question.\n"
    "Decide whether running it on the values given in the question above answers that question "
    "completely and exactly. Judge the task, not the wording.\n"
    "The same task can be written very differently. None of these makes it a different task:\n"
    "- different numbers, names, currencies, or units of the same quantity (mg/dL or umol/L, "
    "km/h or m/s, L or dm3);\n"
    "- a different style or order: terse notes, a list of facts, a story, a question first or last;\n"
    "- synonyms: \"multiples of\" means \"divisible by\", \"at least one of A and B\" means "
    "\"A or B\", \"carrying amount\" means \"book value\";\n"
    "- the same bound written another way: \"from 1 to 999\" is the same set as \"less than 1000\";\n"
    "- background details the calculation does not use, such as a tax rate when after-tax amounts "
    "are already given.\n"
    "Answer no if any of these is true:\n"
    "1. The question asks for a different result: another quantity, only an intermediate value, a "
    "change or difference in the result, or working backwards to find one of the inputs.\n"
    "2. It needs a different method or assumption: another equation, model, rule, convention, legal "
    "test, jurisdiction, compounding or payment scheme.\n"
    "3. An input the solver needs is not stated in the question.\n"
    "4. It adds something that changes the answer and the solver would ignore: an extra condition, "
    "constraint, step, time period, second object or mixture.\n"
    "5. A small wording change alters the correct output: an inclusive or exclusive bound on the same "
    "number (\"less than 1000\" vs \"less than or equal to 1000\"), labelled or identical items or "
    "positions, \"and\" vs \"or\" where it changes which cases count, with or without replacement, "
    "or a changed time, order or direction.\n"
    "Read every word of both questions, then decide by asking: would the solver's answer be the "
    "correct answer to this question?"
)

# v5 (2026-11-15): v4 made Clef doubt that the solver converts units (creatinine in umol/L,
# a beam in mm and N/mm dropped below 0.8). v5 states that the solver reads each input in
# any unit of the same quantity and converts it, and that amounts like "$1.2M" are just numbers.
PROMPT_V5 = PROMPT_V4.replace(
    "- different numbers, names, currencies, or units of the same quantity (mg/dL or umol/L, "
    "km/h or m/s, L or dm3);\n",
    "- different numbers, names or currencies, and amounts written as \"$1.2M\" or \"150k\";\n"
    "- an input given in a different unit of the same quantity: the solver reads each input in any "
    "unit of that quantity and converts it itself (creatinine in umol/L or mg/dL, a length in mm or "
    "m, a load in N/mm or kN/m, a speed in km/h or m/s, a volume in L or dm3);\n")
assert PROMPT_V5 != PROMPT_V4


def _profile(spec: dict | None) -> str:
    if not spec:
        return ""
    conds = "; ".join(spec.get("conditions") or []) or "none"
    inputs = "; ".join(i["name"] for i in spec.get("inputs") or []) or "none"
    return (f"What it computes: {spec.get('deliverable', 'not stated')}. Method: {spec.get('method', 'not stated')}. "
            f"Conditions it assumes: {conds}. Inputs it needs: {inputs}.\n")


def prompt_for(cache: Cache, ep: EntryPoint, version: str | None = None, seed: str | None = None) -> str:
    """The yes/no question Clef answers for one candidate entry, built around one of its
    seed questions (default: the original)."""
    version = version or config.CLEF_PROMPT
    seed = seed if seed is not None else original_question(cache, ep)
    if version == "v1":
        return PROMPT_V1.format(seed=seed)
    if version in ("v2", "v3", "v4", "v5"):
        template = {"v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4, "v5": PROMPT_V5}[version]
        return template.format(seed=seed, profile=_profile(cache.specs.get(ep.problem_id)))
    raise ValueError(f"unknown ARC_CLEF_PROMPT {version!r} (v1, v2, v3, v4 or v5)")


def original_question(cache: Cache, ep: EntryPoint) -> str:
    """The question the entry was built from (Clef compares against it, as in the eval)."""
    for e in cache.entry_points:
        if e.problem_id == ep.problem_id and e.kind == "original":
            return e.text
    return ep.text


def _same(a: str, b: str) -> bool:
    return " ".join((a or "").lower().split()) == " ".join((b or "").lower().split())


def seeds_for(cache: Cache, ep: EntryPoint, exclude: str | None = None) -> list[str]:
    """The questions Clef compares a request with for this entry: the original first, then
    the most recent verified family questions (entry points of kind "family"), up to
    CLEF_SEEDS in all. `exclude` drops a seed identical to the question being checked, so a
    family question never vouches for itself in a pre-publish check."""
    original = original_question(cache, ep)
    family = [e.text for e in cache.entry_points if e.problem_id == ep.problem_id and e.kind == "family"]
    out = []
    for s in [original] + list(reversed(family)):
        if exclude is not None and _same(s, exclude):
            continue
        if not any(_same(s, t) for t in out):
            out.append(s)
        if len(out) >= max(1, config.CLEF_SEEDS):
            break
    return out


def _originals(cache: Cache) -> dict[str, EntryPoint]:
    out: dict[str, EntryPoint] = {}
    for e in cache.entry_points:
        if e.kind == "original":
            out.setdefault(e.problem_id, e)
    for e in cache.entry_points:     # entries saved without an "original" still count
        out.setdefault(e.problem_id, e)
    return out


def candidates(cache: Cache, request: str, k: int | None = None, tag: str | None = None,
               trace: dict | None = None) -> list[tuple[EntryPoint, float]]:
    """Top-k distinct entries for the request.
    CLEF_RETRIEVAL "raw" (default): cosine of the raw question against each entry's texts
        (original, template, family questions). No model call. This is how the Clef dev
        numbers were measured: Clef, not retrieval, separates near-misses.
    CLEF_RETRIEVAL "structure": one Haiku call writes the question's value-free structure,
        compared with each entry's stored structure (raw text for entries without one)."""
    k = k or config.CLEF_TOP_K
    originals = _originals(cache)
    if not originals:
        return []
    by_structure = config.CLEF_RETRIEVAL == "structure"
    with_spec = [p for p in originals if p in cache.specs] if by_structure else []
    without = [p for p in originals if p not in with_spec]
    ranked: list[tuple[str, float, str]] = []
    if with_spec:
        from . import spec as specmod
        qspec = specmod.write_spec(request, tag=tag)
        if trace is not None:
            trace["spec"] = qspec
        sims = embed.ranked(specmod.spec_text(qspec), [specmod.spec_text(cache.specs[p]) for p in with_spec])
        ranked += [(with_spec[i], s, "structure") for i, s in sims]
    if without:
        # raw text: best over all of the entry's texts (original, template, family questions)
        texts = {p: [e.text for e in cache.entry_points if e.problem_id == p] for p in without}
        for p in without:
            ranked.append((p, embed.best_match(request, texts[p])[1], "raw"))
    ranked.sort(key=lambda x: -x[1])
    out = ranked[:k]
    if trace is not None:
        trace["candidates"] = [{"entry": p, "similarity": round(s, 4), "by": by} for p, s, by in out]
    return [(originals[p], s) for p, s, _ in out]


def decide(cache: Cache, request: str, cands: list[tuple[EntryPoint, float]], tag: str | None = None,
           trace: dict | None = None, exclude_self: bool = False) -> tuple[EntryPoint | None, float, dict[str, float]]:
    """One Clef call for all candidates, each compared with up to CLEF_SEEDS of its seed
    questions (original + verified family questions); an entry's P is its best seed's.
    Returns (best entry or None, its P, P per entry). exclude_self (pre-publish checks):
    never use a seed identical to the request. Raises llm.LLMError on a Clef failure (the
    caller must not serve)."""
    if not cands:
        return None, 0.0, {}
    questions, seed_of = {}, {}
    for ep, _ in cands:
        seeds = seeds_for(cache, ep, exclude=request if exclude_self else None)
        for k, seed in enumerate(seeds):
            key = f"{ep.problem_id}#{k}"
            questions[key], seed_of[key] = prompt_for(cache, ep, seed=seed), seed
    if not questions:   # every seed was the request itself: nothing independent to compare with
        if trace is not None:
            trace["clef"] = {}
        return None, 0.0, {}
    raw = llm.decide(request, questions, tag=tag)
    probs: dict[str, float] = {}
    for key, p in raw.items():
        pid = key.split("#")[0]
        probs[pid] = max(probs.get(pid, 0.0), p)
    if trace is not None:
        trace["clef"] = {k: round(v, 4) for k, v in probs.items()}
        trace["clef_seeds"] = {k: {"p": round(raw[k], 4), "seed": seed_of[k][:120]} for k in raw}
        trace["clef_prompt"] = config.CLEF_PROMPT
    best_ep, best_p = max(((ep, probs.get(ep.problem_id, 0.0)) for ep, _ in cands), key=lambda x: x[1])
    return (best_ep if best_p >= config.CLEF_THRESHOLD else None), best_p, probs


# ----------------------------------------------------------------------------- families

_PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4, "v5": PROMPT_V5}


def fits(question: str, anchors: dict[str, str], tag: str | None = None) -> dict[str, float]:
    """P(question asks the same task as each anchor question), for family membership
    (config.FAMILY_CHECK == "clef"). Same prompt as the gate, with the anchor as the "example
    question" and no profile (a family has no built entry yet). One Clef call for all anchors.
    Raises llm.LLMError on a Clef failure."""
    if not anchors:
        return {}
    template = _PROMPTS[config.CLEF_PROMPT]
    questions = {k: template.format(seed=a, profile="") for k, a in anchors.items()}
    return llm.decide(question, questions, tag=tag)
