"""Label-free builds: make and certify a cache entry from the backlog alone.

Two flows, chosen by config.LABELFREE_CERTIFY:

  off (default, the simple flow): take the family's newest record, re-solve it with the
      customer's own production model (the record's `model`), compare final answers.
      Match -> build from that working and publish. No match -> one more re-solve. Still
      no match -> status "mismatch", nothing published. No held-out checks.
  on (the certified flow, everything below): build from the most typical record and
      certify on held-out family questions before publishing.

DEVIATION from the paper, which gives the builder and the acceptance judge the
benchmark's golden label. Production has no label. It has a backlog: real questions
from one task family, each with the answer the production model gave. That is the
evidence used here, through three checks with different blind spots:

  1. Backlog agreement. The new entry answers held-out family questions, through the
     real serving path (gate included). Its answer is compared with production's.
  2. Adjudication. When they disagree, fresh blind solves (no answer shown) break the
     tie. The cache is cleared only if every blind solve sides with it. It is blamed if a
     blind solve sides with production and none with it. Anything else is "unresolved"
     and counts against the entry. Production's own mistakes are found this way and
     don't count against the cache.
  3. Metamorphic relations (arc/metamorphic.py). The same input written another way
     (km/h vs m/s, "€2.0 million" vs "€2,000,000") must give the same output. No model
     is involved, so this catches unit-handling bugs that every model might agree on.

Acceptance is strict. Any confirmed-wrong answer, unresolved disagreement or violated
relation rejects the entry. A cache may always decline, and a declined question costs
one production answer, while a wrong cached answer is served to every matching
question. With zero failures in n checks, the one-sided 95% upper bound on the
error rate is 1 - 0.05**(1/n), which is reported as error_upper_bound (about 3/n).

Too few usable questions -> status "need_more_data". The built entry is kept pending
and re-certified on newer records later, without rebuilding.

Why these checks (see the research notes in eval/labelfree/README.md): agreement
between samples of one model is not verification, because same-family models share
errors (CodeT / MBR-exec show agreement helps, and "Consensus is Not Verification"
2026 shows its limit). So we mix in an outside reference (the production answers),
a decorrelated model-free check (metamorphic relations), and keep checking after launch
(audit_hit, in the style of vCache / Krites).

Known limit: if production, the explainer and the blind solves all make the same
mistake, and the mistake doesn't break a metamorphic relation, nothing here catches it.
For high-stakes families, add a human or stronger-model audit before serving widely.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from pydantic import BaseModel, ConfigDict

from . import config, embed, explain, llm
from . import metamorphic as mm
from .cache import Cache, EntryPoint
from .certify import _compare
from .runtime import CacheInvocationError
from .source import Transcript

STATUSES = ("accepted", "rejected", "need_more_data", "mismatch")
MISMATCH_REASON = "caching tried but answer didn't match"
MISFIT_REASON = "built entry doesn't fit its family"
CODE_WRONG_REASON = "cached code got a family question wrong"
RELATION_REASON = "cached code changed its answer when an input was rewritten"
CONVENTION_REASON = "entry's stated conventions don't match its code"
RESOLVE_ROLE = "resolve"   # the only role that may run the customer's own model (see arc/llm.py)


@dataclass
class BuildResult:
    """What arc.families.build_ready expects back from a build_fn."""
    accepted: bool
    entry_id: str | None
    reason: str
    cost: float
    parts: dict = field(default_factory=dict)
    certificate: dict | None = None
    serve_rate: float | None = None
    status: str = "rejected"
    record_id: str | None = None     # simple flow: the record the build was attempted from


@dataclass
class Check:
    record_id: str
    question: str
    production: str
    served: str                      # served | turned_away | below_threshold | crashed
    cache_answer: str = ""
    outcome: str = ""                # agree | cache_right | cache_wrong | unresolved | crashed | not_served
    detail: str = ""
    blind_answers: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    similarity: float | None = None  # IS1 cosine to the entry's own index texts


@dataclass
class Certificate:
    entry_id: str
    family_id: str
    status: str = "rejected"
    reason: str = ""
    checks: list = field(default_factory=list)
    cost: float = 0.0

    def count(self, outcome: str) -> int:
        return sum(c["outcome"] == outcome for c in self.checks)

    @property
    def served(self) -> int:
        return sum(c["served"] == "served" for c in self.checks)

    @property
    def verified(self) -> int:
        """Answers we have evidence are right: agreed with production, or cleared on adjudication."""
        return self.count("agree") + self.count("cache_right")

    @property
    def failures(self) -> int:
        return self.count("cache_wrong") + self.count("unresolved")

    @property
    def violations(self) -> int:
        return sum(r["status"] == "violated" for c in self.checks for r in c["relations"])

    @property
    def serve_rate(self) -> float | None:
        return self.verified / len(self.checks) if self.checks else None

    def summary(self) -> dict:
        n = self.verified
        return {"entry_id": self.entry_id, "family_id": self.family_id, "status": self.status,
                "reason": self.reason, "questions_checked": len(self.checks), "served": self.served,
                "verified": n, "agree": self.count("agree"), "production_errors_found": self.count("cache_right"),
                "cache_wrong": self.count("cache_wrong"), "unresolved": self.count("unresolved"),
                "crashed": self.count("crashed"), "not_served": self.count("not_served"),
                "relations_held": sum(r["status"] == "held" for c in self.checks for r in c["relations"]),
                "relations_violated": self.violations,
                "relations_crashed": sum(r["status"] == "crashed" for c in self.checks for r in c["relations"]),
                "error_upper_bound": error_upper_bound(n) if not self.failures and not self.violations else None,
                "serve_rate": self.serve_rate, "cost": self.cost,
                "verified_below_threshold": sum(
                    c["outcome"] in ("agree", "cache_right") and c.get("similarity") is not None
                    and c["similarity"] < config.ENTRY_THRESHOLD for c in self.checks)}


def error_upper_bound(n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on the error rate after 0 failures in n."""
    return 1.0 if n <= 0 else 1.0 - (1.0 - confidence) ** (1.0 / n)


# ----------------------------------------------------------------------------- records

def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q or "").strip().lower()


def usable(records: list[dict]) -> list[dict]:
    """Records with a question and a production answer, one per distinct question, oldest first."""
    seen, out = set(), []
    for r in sorted(records, key=lambda r: r.get("timestamp") or 0):
        key = _norm(r.get("question", ""))
        if key and (r.get("response") or "").strip() and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def typical_first(records: list[dict]) -> list[dict]:
    """Most typical question first (highest mean similarity to the rest, numbers masked).
    Building from a typical member, not an outlier or a near-miss, gives a better entry."""
    if len(records) <= 2:
        return list(records)
    from .families import mask_numbers
    v = embed.embedder().encode([mask_numbers(r["question"]) for r in records])
    score = (v @ v.T).mean(axis=1)
    return [records[i] for i in np.argsort(-score, kind="stable")]


# ----------------------------------------------------------------------------- adjudication

class Program(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program: str


DEFAULT_CONVENTION = ("money that is actually paid (each payment or installment) is rounded to the "
                      "cent before any total is computed from it")

PROGRAM_PROMPT = """\
Write a short, self-contained Python 3 program that computes the answer to the question below \
and prints it. It will be run as is, and its printed output is the answer.

Rules:
- Work only from the question{and_rules}. Put each given value in a variable, with a comment \
naming its unit.
- Keep exact arithmetic: full precision through every step (use decimal.Decimal with a high \
precision for money). Round only where a rule below or the question says to, and round printed \
money to the cent.
- Conventions, in order of priority: (1) anything the question itself says; (2) the rules \
listed below; (3) by default, {default}.
- Print one line per quantity the question asks for, with its name and unit, for example \
"area of the plot: 12.50 square metres".
- Allowed imports: math, fractions, decimal, statistics, itertools, functools. No input, files \
or network.
{rules}
Question:
{question}
"""


def _program_solve(question: str, tag: str, conventions: list[str] | None = None) -> tuple[str | None, str]:
    """One tie-break solve: Haiku writes a program from the question (and the entry's stated
    conventions, never its code or any answer), and it runs in a separate process.
    Returns (printed answer, "") or (None, why it failed)."""
    from . import sandbox
    rules = "".join(f"\n- {r}" for r in conventions or [])
    prompt = PROGRAM_PROMPT.format(
        question=question, default=DEFAULT_CONVENTION,
        and_rules=" and the convention rules listed below" if conventions else "",
        rules=f"\nConvention rules for this kind of question:{rules}\n" if conventions else "")
    try:
        prog = llm.call("adjudicator", prompt, model=config.EXPLAINER_MODEL, schema=Program,
                        effort=config.SOURCE_EFFORT, max_tokens=4000, tag=tag).parsed
    except llm.LLMError as e:
        return None, f"model error: {e}"[:200]
    run = sandbox.run_program(prog.program, seconds=config.LABELFREE_PROGRAM_SECONDS)
    if not run.ok:
        return None, f"program failed: {run.error}"[:200]
    if not run.stdout.strip():
        return None, "program printed nothing"
    return run.stdout.strip(), ""


def _tie_break_solve(question: str, tag: str, conventions: list[str] | None) -> tuple[str | None, str]:
    if config.LABELFREE_ADJUDICATOR == "blind":
        return explain._blind_solve(question, tag).final_answer, ""
    return _program_solve(question, tag, conventions)


def adjudicate(question: str, production: str, cache_answer: str, tag: str,
               max_solves: int = 2, conventions: list[str] | None = None) -> tuple[str, list[str], str]:
    """Cache and production disagree; independent solves decide. Returns (outcome, answers, detail).

    Each solve is a program written from the question and the entry's conventions only
    (LABELFREE_ADJUDICATOR=program, default), so its arithmetic is exact.

    cache_right  every solve agrees with the cache (production was wrong)
    cache_wrong  some solve agrees with production and none with the cache
    unresolved   anything else (the solves disagree with each other or with both, or a
                 program failed: a failed solve sides with no one, never with the cache)"""
    answers, with_cache, with_prod = [], 0, 0
    for i in range(1, max_solves + 1):
        ans, why = _tie_break_solve(question, f"{tag}:adj{i}", conventions)
        if ans is None:
            answers.append(f"(no answer: {why})")
            break
        answers.append(ans)
        c = _compare(f"solve {i} vs cache", question, ans, cache_answer, f"{tag}:adj{i}:c").agree
        p = _compare(f"solve {i} vs production", question, ans, production, f"{tag}:adj{i}:p").agree
        with_cache += c and not p
        with_prod += p and not c
        if with_prod and not with_cache:
            return "cache_wrong", answers, f"blind solve {i} agrees with production, not the cache"
        if not c:  # a solve that doesn't side with the cache means we can't clear it
            break
    if with_cache == len(answers) == max_solves:
        return "cache_right", answers, f"all {max_solves} independent solves agree with the cache; production looks wrong"
    if answers and answers[-1].startswith("(no answer"):
        return "unresolved", answers, answers[-1][1:-1]
    return "unresolved", answers, "independent solves don't settle it"


# ----------------------------------------------------------------------------- certify

def _run_serving(cache: Cache, question: str, tag: str, entry_point=None, trace: dict | None = None):
    """Serve one question. entry_point given (family index on): test that entry directly."""
    from .serve import use
    trace = trace if trace is not None else {}
    kw = {"entry_point": entry_point} if entry_point is not None else {}
    try:
        result = use(cache, question, tag=tag, trace=trace, **kw)
    except CacheInvocationError as e:
        return None, "crashed", str(e).splitlines()[0][:200]
    if result is None:
        return None, "below_threshold" if trace.get("stage") == "below_threshold" else "turned_away", \
            ", ".join(trace.get("rejected_by") or [])
    return result, "served", ""


def _entry_under_test(cache: Cache, entry_id: str):
    """The entry point certification forces, or None to let IS1 search (flag off)."""
    if not config.LABELFREE_FAMILY_INDEX:
        return None
    return next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None)


def certify(cache: Cache, entry_id: str, family_id: str, records: list[dict],
            fail_fast: bool | None = None, prior: list[dict] | None = None,
            budget: float | None = None) -> Certificate:
    """Check a built entry against held-out family records. `prior` carries checks from
    an earlier pass (a pending entry being re-certified), so evidence accumulates.
    `budget` ($) stops checking once spent; the decision then uses what was checked."""
    from .serve import dispatch_root, format_output
    fail_fast = config.LABELFREE_FAIL_FAST if fail_fast is None else fail_fast
    cert = Certificate(entry_id, family_id, checks=list(prior or []))
    relations_left = config.LABELFREE_RELATIONS - sum(len(c["relations"]) for c in cert.checks)
    root = {e.problem_id: e.root_id for e in cache.entry_points}.get(entry_id)
    forced = _entry_under_test(cache, entry_id)
    with llm.LEDGER.span() as span:
        try:
            start = len(llm.LEDGER.records)
            for r in records[: max(0, config.LABELFREE_MAX_CHECKED - len(cert.checks))]:
                if budget is not None and llm.LEDGER.cost(since=start) >= budget:
                    break
                tag = f"{entry_id}:lf:{r['id']}"
                trace: dict = {}
                result, served, why = _run_serving(cache, r["question"], tag, forced, trace)
                ck = Check(r["id"], r["question"], r["response"], served, detail=why,
                           similarity=trace.get("similarity"))
                if served == "crashed":
                    ck.outcome = "crashed"
                elif served != "served":
                    ck.outcome = "not_served"
                else:
                    ck.cache_answer = format_output(result.output)
                    cmp = _compare("production vs cache", r["question"], r["response"], ck.cache_answer, f"{tag}:cmp")
                    if cmp.agree:
                        ck.outcome = "agree"
                    else:
                        ck.outcome, ck.blind_answers, ck.detail = adjudicate(
                            r["question"], r["response"], ck.cache_answer, tag,
                            conventions=cache.conventions.get(entry_id))
                    if ck.outcome in ("agree", "cache_right") and relations_left > 0 and root:
                        run = lambda v: dispatch_root(cache, root, v)[0]
                        rels = mm.check(run, result.parameters, result.output, max_relations=relations_left)
                        ck.relations = [asdict(x) for x in rels]
                        relations_left -= len(rels)
                cert.checks.append(asdict(ck))
                bad = ck.outcome in ("cache_wrong", "unresolved") or any(
                    x["status"] == "violated" for x in ck.relations)
                if bad and fail_fast:
                    break
        finally:
            span.end = len(llm.LEDGER.records)
            cert.cost = span.cost()
    _decide(cert)
    return cert


def _decide(cert: Certificate) -> None:
    s = cert.summary()
    if s["cache_wrong"] or s["unresolved"]:
        first = next(c for c in cert.checks if c["outcome"] in ("cache_wrong", "unresolved"))
        cert.status = "rejected"
        cert.reason = (f"{s['cache_wrong']} wrong, {s['unresolved']} unresolved of {s['served']} served; "
                       f"first: {first['detail']} on {first['record_id']}")
    elif s["relations_violated"]:
        first = next(r for c in cert.checks for r in c["relations"] if r["status"] == "violated")
        cert.status = "rejected"
        cert.reason = (f"metamorphic relation violated: {first['slot']} {first['original']!r} -> "
                       f"{first['rewritten']!r} {first['detail']}")[:400]
    elif s["verified"] < config.LABELFREE_MIN_CHECKED:
        cert.status = "need_more_data"
        cert.reason = (f"only {s['verified']} verified answers (need {config.LABELFREE_MIN_CHECKED}); "
                       f"{s['not_served']} not served, {s['crashed']} crashed")
    else:
        cert.status = "accepted"
        cert.reason = (f"{s['verified']} verified answers, 0 wrong, {s['relations_held']} relations held; "
                       f"error rate <= {s['error_upper_bound']:.1%} at 95%")


# ----------------------------------------------------------------------------- build

def _state_dir(entry_id: str) -> Path:
    return config.WORK_DIR / "labelfree" / entry_id


def _pending_cache_dir(entry_id: str) -> Path:
    return config.CACHE_ROOT / "pending" / entry_id


def _explain_and_build(entry_id: str, candidates: list[dict], parts: dict) -> tuple[Cache | None, list[str], str]:
    """Try the most typical records in turn until one is explained and builds."""
    from .builder import Builder, BuildError
    tried, why = [], "no record could be explained"
    for rec in candidates[: config.LABELFREE_BUILD_TRIES]:
        tried.append(rec["id"])
        with llm.LEDGER.span() as sp:
            ex = explain.explain({k: rec.get(k) for k in ("id", "question", "response", "reasoning")})
            sp.end = len(llm.LEDGER.records)
        parts["explain"] = parts.get("explain", 0.0) + sp.cost()
        if ex.decision != "explained":
            why = f"explainer: {ex.reason}"[:300]
            continue
        t = Transcript(**{**ex.transcript, "problem_id": entry_id})
        problem = {"id": entry_id, "question": t.question, "golden_label": ""}
        with llm.LEDGER.span() as sp:
            try:
                b = Builder(problem, t, use_golden=False)
                cache = b.build()
                _add_gate_data(cache, problem, t)
            except BuildError as e:
                why = f"build failed: {e}"[:300]
                cache = None
            finally:
                sp.end = len(llm.LEDGER.records)
        parts["build"] = parts.get("build", 0.0) + sp.cost()
        if cache is not None:
            return cache, tried, ""
    return None, tried, why


def _add_gate_data(cache: Cache, problem: dict, t: Transcript) -> None:
    """The gate in use needs per-entry data: a structure (spec gate) or a card (v2)."""
    if config.GATE in ("spec", "clef"):     # clef retrieves by structure too
        from . import spec
        spec.index_cache(cache)
    elif config.GATE == "v2":
        from .gatecard import make_card
        cache.cards[problem["id"]] = make_card(cache, problem, t, use_golden=False)


def index_family(cache: Cache, entry_id: str, checks: list[dict]) -> list[str]:
    """Add verified family questions as extra entry points (and card index_texts).

    Only questions the entry was shown to answer right (agree / cache_right) are added, so
    a near-miss the family tracker grouped in by mistake never becomes an index point.
    Lowest-cosine questions go first: they widen what IS1 can reach the most. Returns the
    texts added."""
    orig = next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None)
    if orig is None:
        return []
    have = {e.text for e in cache.entry_points}
    good = [c for c in checks if c["outcome"] in ("agree", "cache_right")
            and not any(r["status"] == "violated" for r in c.get("relations", []))]
    good.sort(key=lambda c: c.get("similarity") if c.get("similarity") is not None else 0.0)
    added: list[str] = []
    for c in good:
        if len(added) >= config.LABELFREE_INDEX_MAX:
            break
        q = c["question"].strip()
        if not q or q in have:
            continue
        cache.add_entry_point(EntryPoint(q, "family", orig.root_id, orig.params, entry_id))
        have.add(q)
        added.append(q)
    card = cache.cards.get(entry_id)
    if card is not None and added:
        card["index_texts"] = [*card.get("index_texts", []),
                               *[q for q in added if q not in card.get("index_texts", [])]]
    return added


def _publish(cache: Cache, entry_id: str, checks: list[dict] | None = None) -> list[str]:
    added = index_family(cache, entry_id, checks or []) if config.LABELFREE_FAMILY_INDEX else []
    cache.save(config.CACHE_ROOT / entry_id)
    master_dir = config.cache_dir("master")
    master = Cache.load(master_dir) if (master_dir / "registry.json").exists() else Cache("master")
    master.merge(cache)
    master.save(master_dir)
    return added


def _build_certified(family_id: str, records: list[dict]) -> BuildResult:
    """LABELFREE_CERTIFY=on: build from the most typical record, certify on held-out ones."""
    entry_id = f"fam_{family_id}"
    state_dir = _state_dir(entry_id)
    state_file = state_dir / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    recs = usable(records)
    parts: dict[str, float] = {}

    start_all = len(llm.LEDGER.records)
    with llm.LEDGER.span() as total:
        try:
            pending = state.get("status") == "need_more_data" and \
                (_pending_cache_dir(entry_id) / "registry.json").exists()
            if pending:  # re-certify on records not used before; no rebuild
                cache = Cache.load(_pending_cache_dir(entry_id))
                used = set(state["used_ids"])
                prior = state["checks"]
                build_ids = state["build_ids"]
            else:
                if len(recs) < config.LABELFREE_MIN_CHECKED + 1:
                    return BuildResult(False, None, f"{len(recs)} usable records; need "
                                       f"{config.LABELFREE_MIN_CHECKED + 1} before building", 0.0,
                                       status="need_more_data")
                cache, build_ids, why = _explain_and_build(entry_id, typical_first(recs), parts)
                if cache is None:
                    state = {"status": "rejected", "reason": why, "build_ids": build_ids, "time": time.time()}
                    return BuildResult(False, None, why, 0.0, parts, None, None, "rejected")
                used, prior = set(build_ids), []
            holdout = [r for r in recs if r["id"] not in used and r["id"] not in set(build_ids)]
            with llm.LEDGER.span() as sp:
                left = config.LABELFREE_MAX_COST - llm.LEDGER.cost(since=start_all)
                cert = certify(cache, entry_id, family_id, holdout, prior=prior, budget=max(0.0, left))
                sp.end = len(llm.LEDGER.records)
            parts["certify"] = parts.get("certify", 0.0) + sp.cost()

            indexed: list[str] = []
            if cert.status == "accepted":
                indexed = _publish(cache, entry_id, cert.checks)
            elif cert.status == "need_more_data":
                cache.save(_pending_cache_dir(entry_id))
            state = {"status": cert.status, "reason": cert.reason, "build_ids": build_ids,
                     "used_ids": sorted(used | {c["record_id"] for c in cert.checks}),
                     "checks": cert.checks, "indexed": len(indexed), "time": time.time()}
            return BuildResult(cert.status == "accepted", entry_id if cert.status == "accepted" else None,
                               cert.reason, 0.0, parts, cert.summary(), cert.serve_rate, cert.status)
        finally:
            total.end = len(llm.LEDGER.records)
            if state:
                state_dir.mkdir(parents=True, exist_ok=True)
                state["cost_this_run"] = total.cost()
                state_file.write_text(json.dumps(state, indent=1, default=str))


# ----------------------------------------------------------------------------- simple flow

def _resolve(question: str, model: str, tag: str):
    """Re-solve with the customer's production model, full working kept for the builder."""
    prompt = explain.SOLVE_PROMPT.format(question=question) + (
        "\n\nPut the full working in `working` and only the final answer, with units, in `final_answer`.")
    return llm.call(RESOLVE_ROLE, prompt, model=model, schema=explain.Worked,
                    effort=config.SOURCE_EFFORT, max_tokens=16000, tag=tag).parsed


# ----------------------------------------------------------------------------- conventions

class Conventions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: list[str]


CONVENTIONS_PROMPT = """\
You built a cached solver from the worked example below. List every rounding or convention \
choice the solver's CODE makes that changes the numeric answer, as plain one-line rules a person \
could apply by hand. Describe exactly what the code does, not what it should do: if it totals \
interest from the unrounded payment, say so. Cover each that applies: rounding of intermediate \
amounts (for example "total interest = monthly payment rounded to the cent x number of payments - \
principal"), compounding or day-count conventions, which formula variant or rate, inclusive or \
exclusive bounds, how the final result is rounded and in what unit. Leave out anything that does \
not change the answer. Return an empty list if there is nothing.

## Example question
{question}

## Worked solution
{solution}

## Solver code (root function first)
{code}
"""


def write_conventions(cache: Cache, entry_id: str, transcript: Transcript) -> list[str]:
    """One builder-model call: the entry's conventions, stored in cache.conventions."""
    from .gatecard import _entry_code
    ep = next(e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original")
    prompt = CONVENTIONS_PROMPT.format(question=transcript.question, solution=transcript.solution,
                                       code=_entry_code(cache, ep.root_id))
    rules = llm.call("builder", prompt, model=config.BUILDER_MODEL, schema=Conventions,
                     effort=config.BUILDER_EFFORT, max_tokens=2000, tag=f"{entry_id}:conventions").parsed.rules
    rules = [r.strip() for r in rules if r and r.strip()]
    cache.conventions[entry_id] = rules
    return rules


def _convention_check(cache: Cache, entry_id: str, rec: dict) -> dict | None:
    """The entry's stated conventions must describe its code: a program written from the
    build question plus those conventions must give exactly the code's answer on the build
    question. None = check off."""
    from .serve import dispatch_root, format_output
    if not config.LABELFREE_CONVENTION_CHECK:
        return None
    ep = next(e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original")
    rules = cache.conventions.get(entry_id) or []
    out = {"conventions": rules, "cache_answer": "", "program_answer": "", "passed": False, "detail": ""}
    try:
        output, _ = dispatch_root(cache, ep.root_id, {p.description: p.original_value for p in ep.params})
        out["cache_answer"] = format_output(output)
    except Exception as e:  # noqa: BLE001  (any failure of cached code)
        out["detail"] = f"code failed on its own build question: {type(e).__name__}: {e}"[:200]
        return out
    ans, why = _program_solve(rec["question"], f"{entry_id}:conv", rules)
    if ans is None:
        out["detail"] = why
        return out
    out["program_answer"] = ans
    cmp = _compare("conventions program vs cache", rec["question"], ans, out["cache_answer"], f"{entry_id}:conv:cmp")
    out["passed"] = bool(cmp.agree)
    if not cmp.agree:
        out["detail"] = f"program under the stated conventions said {ans[:100]!r}; the code said {out['cache_answer'][:100]!r}"
    return out


# ----------------------------------------------------------------------------- pre-publish

def _add_seed_questions(cache: Cache, entry_id: str, others: list[dict], n_checked: int) -> list[str]:
    """Give the gate more wordings to compare with during the pre-publish check: up to
    CLEF_SEEDS-1 other family questions, newest first, preferring ones not being checked
    (the gate drops a seed identical to the question it is checking). Temporary: removed
    again before publishing, when only verified questions are kept."""
    k = max(0, getattr(config, "CLEF_SEEDS", 1) - 1)
    if k == 0 or not others:
        return []
    checked = others[-n_checked:] if n_checked else []
    pool = [r for r in reversed(others) if r not in checked] + list(reversed(checked))
    orig = next(e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original")
    have = {e.text for e in cache.entry_points}
    picks = []
    for r in pool:
        q = r["question"].strip()
        if q and q not in have and q not in picks:
            picks.append(q)
        if len(picks) == k:
            break
    for q in reversed(picks):                       # oldest first, so the newest is "most recent"
        cache.add_entry_point(EntryPoint(q, "family", orig.root_id, orig.params, entry_id))
    return picks


def _remove_seed_questions(cache: Cache, entry_id: str, seeds: list[str]) -> None:
    cache.entry_points = [e for e in cache.entry_points
                          if not (e.problem_id == entry_id and e.kind == "family" and e.text in seeds)]


def _prepublish_check(cache: Cache, entry_id: str, others: list[dict], budget: float) -> dict | None:
    """Before publishing (simple flow): run the newest other family questions through the
    real serving path, forced onto the new entry, and check three things on each:

      fit       the gate serves it (Clef P >= CLEF_THRESHOLD with ARC_GATE=clef). At least
                LABELFREE_FIT_MIN of them must fit (4 of 5).
      answer    the cached code's answer equals production's under the strict comparison
                (certify._compare: exact to the precision written). On a disagreement, blind
                re-solves decide (adjudicate): production's own mistakes don't count against
                the cache, but a wrong or unresolved cache answer blocks publishing.
      relations the same quantity written another way ("$150k" vs "$150,000", km/h vs m/s)
                must give the same output (arc/metamorphic.py, code only, no model).

    A wrong or unresolved answer, or a violated relation, blocks publishing. A crash or an
    extraction refusal (serve.ExtractionError) is a safe decline in production (the question
    falls back to the model), so it counts as "not served" toward the fit rule. Clef
    saying a question fits proves the entry is for it, not that the code reads it right.
    Gate or model errors count as "doesn't fit", so an outage never publishes unchecked.
    None = skipped (LABELFREE_FIT_CHECKS=0 or no other family questions)."""
    from .serve import dispatch_root, format_output, use
    n = config.LABELFREE_FIT_CHECKS
    ep = next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None)
    if n <= 0 or not others or ep is None:
        return None
    rows, start, relations_left, stopped = [], len(llm.LEDGER.records), config.LABELFREE_RELATIONS, False
    for r in others[-n:]:                                 # usable() sorts oldest -> newest
        if llm.LEDGER.cost(since=start) >= budget:
            stopped = True
            break
        tag = f"{entry_id}:pre:{r['id']}"
        row = {"record_id": r["id"], "question": r["question"], "fits": False, "p": None, "outcome": "",
               "cache_answer": "", "detail": "", "relations": [], "similarity": None}
        trace: dict = {}
        try:
            result = use(cache, r["question"], tag=tag, trace=trace, entry_point=ep)
        except CacheInvocationError as e:   # code crash or extraction refusal: a safe decline
            row.update(outcome="crashed", detail=str(e).splitlines()[0][:200])
            result = None
        except llm.LLMError as e:
            row.update(outcome="error", detail=f"error: {e}"[:200])
            result = None
        row["p"] = trace.get("clef_p")
        row["similarity"] = trace.get("similarity")
        if result is None and not row["outcome"]:
            row.update(outcome="turned_away", detail=", ".join(trace.get("rejected_by") or []))
        if result is not None:
            row["fits"] = True
            row["cache_answer"] = format_output(result.output)
            try:
                cmp = _compare("production vs cache", r["question"], r["response"], row["cache_answer"], f"{tag}:cmp")
                if cmp.agree:
                    row["outcome"] = "agree"
                else:
                    outcome, _, detail = adjudicate(r["question"], r["response"], row["cache_answer"], tag,
                                                    conventions=cache.conventions.get(entry_id))
                    row.update(outcome=outcome, detail=detail)
            except llm.LLMError as e:
                row.update(outcome="unresolved", detail=f"error: {e}"[:200])
            if row["outcome"] in ("agree", "cache_right") and relations_left > 0:
                rels = mm.check(lambda v: dispatch_root(cache, ep.root_id, v)[0], result.parameters,
                                result.output, max_relations=relations_left)
                row["relations"] = [asdict(x) for x in rels]
                relations_left -= len(rels)
        rows.append(row)
    fits = sum(x["fits"] for x in rows)
    need = math.ceil(round(config.LABELFREE_FIT_MIN * min(n, len(others)), 9))
    bad = next((x for x in rows if x["outcome"] in ("cache_wrong", "unresolved")), None)
    violated = next((rel for x in rows for rel in x["relations"] if rel["status"] == "violated"), None)
    if stopped:
        reason = f"cost cap ${budget:.2f} reached after {len(rows)} of {min(n, len(others))} pre-publish checks"
    elif fits < need:
        reason = f"{MISFIT_REASON}: {fits} of {len(rows)} family questions fit (need {need})"
    elif bad:
        reason = (f"{CODE_WRONG_REASON}: {bad['outcome']} on record {bad['record_id']} "
                  f"(cache said {bad['cache_answer'][:80]!r}; {bad['detail'][:120]})")
    elif violated:
        reason = (f"{RELATION_REASON}: {violated['slot']} {violated['original']!r} -> "
                  f"{violated['rewritten']!r} {violated['detail']}")[:300]
    else:
        reason = ""
    return {"checked": len(rows), "fits": fits, "needed": need,
            "verified": sum(x["outcome"] in ("agree", "cache_right") for x in rows),
            "passed": not reason, "reason": reason, "rows": rows}


def _build_simple(family_id: str, records: list[dict]) -> BuildResult:
    """LABELFREE_CERTIFY=off: newest record, re-solved by its own production model.

    Final answers match -> build from that working and publish. No match -> re-solve once
    more. Still no match -> status "mismatch", nothing published. After a build,
    _prepublish_check runs the newest other family questions through the entry (gate fit,
    the code's answer vs production under the strict comparison, metamorphic relations).
    Any failure -> status "mismatch" with the reason, nothing published."""
    from .builder import Builder, BuildError
    entry_id = f"fam_{family_id}"
    recs = usable(records)
    if not recs:
        return BuildResult(False, None, "no usable records", 0.0, status="need_more_data")
    rec = recs[-1]                                       # usable() sorts oldest -> newest
    model = rec.get("model") or config.SOURCE_MODEL
    parts: dict[str, float] = {}
    state: dict = {"record_id": rec["id"], "model": model, "attempts": [], "time": time.time()}
    start = len(llm.LEDGER.records)

    def result(status, reason, accepted=False):
        state.update(status=status, reason=reason, cost=llm.LEDGER.cost(since=start), parts=parts)
        d = _state_dir(entry_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps(state, indent=1, default=str))
        return BuildResult(accepted, entry_id if accepted else None, reason, 0.0, parts,
                           None, None, status, rec["id"])

    worked = None
    for i in range(1, config.LABELFREE_RESOLVE_TRIES + 1):
        if llm.LEDGER.cost(since=start) >= config.LABELFREE_MAX_COST:
            return result("rejected", f"cost cap ${config.LABELFREE_MAX_COST} reached after {i - 1} re-solves")
        with llm.LEDGER.span() as sp:
            try:
                w = _resolve(rec["question"], model, f"{entry_id}:resolve{i}")
                cmp = _compare("production vs re-solve", rec["question"], rec["response"], w.final_answer,
                               f"{entry_id}:resolve{i}:cmp")
            except llm.LLMError as e:
                sp.end = len(llm.LEDGER.records)
                parts["resolve"] = parts.get("resolve", 0.0) + sp.cost()
                return result("rejected", f"re-solve failed: {e}"[:300])
            sp.end = len(llm.LEDGER.records)
        parts["resolve"] = parts.get("resolve", 0.0) + sp.cost()
        state["attempts"].append({"final_answer": w.final_answer, "agree": cmp.agree})
        if cmp.agree:
            worked = w
            break
    if worked is None:
        return result("mismatch", MISMATCH_REASON)

    t = Transcript(entry_id, rec["question"], f"{worked.working}\n\nFinal answer: {worked.final_answer}",
                   f"re-solve by the production model ({model}), agreed with production", model, 0.0)
    problem = {"id": entry_id, "question": rec["question"], "golden_label": ""}
    with llm.LEDGER.span() as sp:
        try:
            cache = Builder(problem, t, use_golden=False).build()
            _add_gate_data(cache, problem, t)
        except (BuildError, llm.LLMError) as e:
            cache, why = None, f"build failed: {e}"[:300]
        finally:
            sp.end = len(llm.LEDGER.records)
    parts["build"] = sp.cost()
    if cache is None:
        return result("rejected", why)
    # conventions: written from the code, then checked against it on the build question
    with llm.LEDGER.span() as sp:
        try:
            state["conventions"] = write_conventions(cache, entry_id, t)
            conv = _convention_check(cache, entry_id, rec)
        except llm.LLMError as e:
            conv = {"passed": False, "detail": f"model error: {e}"[:200]}
        finally:
            sp.end = len(llm.LEDGER.records)
    parts["conventions"] = sp.cost()
    if conv is not None:
        state["convention_check"] = conv
        if not conv["passed"]:
            return result("mismatch", f"{CONVENTION_REASON}: {conv['detail']}"[:400])

    others = [r for r in recs if r["id"] != rec["id"]]
    seeds = _add_seed_questions(cache, entry_id, others, min(config.LABELFREE_FIT_CHECKS, len(others)))
    with llm.LEDGER.span() as sp:
        try:
            left = max(0.0, config.LABELFREE_MAX_COST - llm.LEDGER.cost(since=start))
            pre = _prepublish_check(cache, entry_id, others, left)
        finally:
            sp.end = len(llm.LEDGER.records)
            _remove_seed_questions(cache, entry_id, seeds)    # only verified questions are kept
    state["seed_questions"] = seeds
    if pre is not None:
        parts["prepublish"] = sp.cost()
        state["prepublish"] = pre
        if not pre["passed"]:      # same status as a mismatch: the family retries on a newer record
            return result("mismatch", pre["reason"])
    state["indexed"] = len(_publish(cache, entry_id, pre["rows"] if pre else None))
    return result("accepted", f"re-solve {len(state['attempts'])} by {model} matched production; built and published",
                  accepted=True)


def build_family(family_id: str, records: list[dict]) -> BuildResult:
    """build_fn for arc.families.build_ready. records: the family's source-model answers.
    Simple flow by default; LABELFREE_CERTIFY=on runs the certified flow."""
    return _build_certified(family_id, records) if config.LABELFREE_CERTIFY else _build_simple(family_id, records)


def _with_cost(fn):
    """Fill BuildResult.cost with the full spend of the call, rejected attempts included."""
    def wrapped(family_id: str, records: list[dict]) -> BuildResult:
        before = len(llm.LEDGER.records)
        res = fn(family_id, records)
        res.cost = llm.LEDGER.cost(since=before)
        return res
    wrapped.__doc__ = fn.__doc__
    wrapped.__name__ = fn.__name__
    return wrapped


build_family = _with_cost(build_family)


# ----------------------------------------------------------------------------- after launch

def audit_hit(question: str, cache_answer: str, production_answer: str, tag: str,
              conventions: list[str] | None = None) -> dict:
    """Check one served cache answer against a production answer for the same question
    (the serving layer samples a share of hits for this, LABELFREE_AUDIT_RATE). Returns
    {"outcome": agree | cache_right | cache_wrong | unresolved, ...}. A cache_wrong or
    unresolved outcome should take the entry out of service until it's rebuilt."""
    cmp = _compare("production vs cache (audit)", question, production_answer, cache_answer, f"{tag}:audit")
    if cmp.agree:
        return {"outcome": "agree", "blind_answers": [], "detail": ""}
    outcome, answers, detail = adjudicate(question, production_answer, cache_answer, f"{tag}:audit",
                                          conventions=conventions)
    return {"outcome": outcome, "blind_answers": answers, "detail": detail}


def audit_should_disable(outcome: str) -> bool:
    return outcome in ("cache_wrong", "unresolved")


__all__ = ["CONVENTION_REASON", "write_conventions", "MISMATCH_REASON", "MISFIT_REASON", "CODE_WRONG_REASON", "RELATION_REASON", "BuildResult", "Certificate", "build_family", "certify", "adjudicate", "audit_hit",
           "audit_should_disable", "error_upper_bound", "usable", "typical_first", "STATUSES"]
