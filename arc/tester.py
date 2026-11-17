"""Independent acceptance tests for a built entry, written by a different model than the builder.

DEVIATION (owner decisions 2026-11-17): the builder used to write the code AND most of the
checks it was judged by, so the two shared blind spots. Here a separate test-maker model writes
the tests. The line it never crosses: it may see WHAT the entry is meant to do, never HOW it
was built (no code, no builder transcript or reasoning, no builder tests, no cached answers).

What the tester sees (Context; every part degrades gracefully when it doesn't exist yet):
  examples      family questions WITH the production model's answers, spread across the family.
                A build only starts once a family has FAMILY_MIN_SIGHTINGS answered questions,
                so there are always some; on a young cache there are just a few.
  contract      the entry's inputs (names, accepted formats, valid ranges) and outputs (names,
                units): what it claims to handle, so tests aim at its edges.
  conventions   rules the entry states. Ranked BELOW the practice visible in past answers, so a
                wrong builder convention gets caught instead of copied into the tests.
  near misses   real backlog questions that were close to this family but judged a different
                task (families.near_misses). None on a young cache: the tester invents them.
  history       what earlier build attempts for this family got wrong. None the first time.

The suite is written ONCE per tester model per build and split into two halves. A half is
"spent" once it produced repair feedback; it never judges a build that got that feedback. So a
repair faces unseen tests without paying for a second suite, and the cascade's next builder
reuses an unspent half. TESTER_MAX_SUITES caps suites written per build.

Verdicts per case, through the real serving path forced onto the new entry:
  in-scope, served, matches the program's answer      pass (confirmed)
  in-scope, served, differs                           program tie-breaks (labelfree.adjudicate):
                                                      cache_right -> pass; cache_wrong -> FAIL;
                                                      unresolved -> unchecked (no verdict)
  in-scope, not served                                declined (safe)
  look-alike, not served                              pass
  look-alike, served, and its own correct answer
    (tester's program) equals the cache's             harmless (the cache was right anyway)
  look-alike, served, otherwise                       FAIL (gate)

Publishing needs: no FAIL, at least TESTER_MIN_CASES confirmed passes, and at least half the
checkable in-scope cases served. Look-alike failures are the gate's, not the code's: they block
publishing but produce no builder feedback, so no repair rebuild is wasted on them.

Feedback: on in-scope failures the tester writes GENERAL rules ("treats 'per month' rates as
yearly"), never the tests. Backstops drop lines quoting a failing question and blank numbers
taken from the failing tests.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from . import config, llm, sandbox
from .cache import Cache

ROLE = "tester"
FAILED_REASON = "independent tests failed"
GATE_REASON = "independent tests: entry answered a look-alike wrongly"
TOO_FEW_REASON = "independent tests: too few confirmed passes"
DECLINED_REASON = "independent tests: entry declined most in-scope cases"
LIMIT_REASON = "independent tests: suite limit reached for this build"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TestCase(_Strict):
    question: str
    purpose: str        # what this case tries to break
    program: str        # computes and prints the expected answer


class DeclineCase(_Strict):
    question: str
    why_different: str
    program: str        # its OWN correct answer if answerable, "" if a needed value is missing


class TestSuite(_Strict):
    cases: list[TestCase]
    declines: list[DeclineCase]


class Feedback(_Strict):
    problems: list[str]     # one general description per distinct failure, no test values


# ----------------------------------------------------------------------------- context

@dataclass
class Context:
    examples: list[dict] = field(default_factory=list)     # {"question", "answer"}
    near_misses: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    @property
    def questions(self) -> list[str]:
        return [e["question"] for e in self.examples]


def pick_examples(records: list[dict], k: int | None = None) -> list[dict]:
    """Up to k answered records spread evenly across the family (oldest -> newest), newest always
    included, so the tester sees the family's range rather than its latest cluster."""
    k = config.TESTER_EXAMPLES if k is None else k
    recs = [r for r in records if (r.get("response") or "").strip()]
    if len(recs) > k:
        idx = sorted({round(i * (len(recs) - 1) / (k - 1)) for i in range(k)}) if k > 1 else [len(recs) - 1]
        recs = [recs[i] for i in idx]
    return [{"question": r["question"], "answer": _tail(r["response"])} for r in recs]


def _tail(text: str, n: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= n else "..." + text[-n:]       # the final answer is at the end


def context_from(records: list[dict], near_misses: list[str] | None = None,
                 history: list[str] | None = None) -> Context:
    return Context(pick_examples(records), list(near_misses or [])[:config.TESTER_NEAR_MISSES],
                   list(dict.fromkeys(h for h in history or [] if h))[:8])


# ----------------------------------------------------------------------------- suites

@dataclass
class Half:
    cases: list[TestCase]
    declines: list[DeclineCase]
    spent: bool = False           # produced repair feedback: never judges a build that got it


@dataclass
class Bank:
    """Suites written during one build_family call, by tester model."""
    halves: dict[str, list[Half]] = field(default_factory=dict)
    written: int = 0

    def next_half(self, model: str) -> Half | None:
        return next((h for h in self.halves.get(model, []) if not h.spent), None)

    def can_test(self, builder_model: str) -> bool:
        return self.next_half(pick_model(builder_model)) is not None or self.written < config.TESTER_MAX_SUITES


def pick_model(builder_model: str | None) -> str:
    """The tester must not be the model that built the entry."""
    if builder_model and builder_model == config.TESTER_MODEL:
        return config.TESTER_FALLBACK_MODEL
    return config.TESTER_MODEL


PROMPT = """\
You write acceptance tests for an automated calculator that was built for ONE kind of question. \
You will never see its code. Your job is to catch it being wrong, so write tests a careless \
implementation would fail.

## Past questions of this kind, with the answers our production assistant gave
Use these for the expected format, units and rounding practice. They are usually, not always, right.
{examples}

## What the calculator takes and returns
Inputs:
{inputs}
Outputs (name: unit):
{outputs}
Conventions it states:
{conventions}
{near_misses}{history}
## Write two kinds of test

1. {n_cases} new in-scope questions of exactly this kind, with new values. Each must be fully \
answerable from its own text. Vary them on purpose: the edges of each input's valid range, zero \
or one where they make sense, very large and very small numbers, values written in other formats \
or units ("$1.5k", "1,500 dollars", "90 km/h" vs "25 m/s"), values in a different order, natural \
but different wording.{history_ask} For each, a short self-contained Python 3 program that \
computes the answer and prints it.

2. {n_declines} questions the calculator must REFUSE: very similar to the examples, but needing \
different reasoning (a different formula, an extra step, a different quantity asked for) or \
missing a value the calculation needs.{near_miss_ask} Say in one line what makes each one \
different. If it is still answerable, also give a program for ITS correct answer; if a needed \
value is missing, leave program empty.

Program rules:
- Put each given value in a variable, with a comment naming its unit.
- Exact arithmetic: full precision through every step (decimal.Decimal with high precision for \
money). Round only at the end, money to the cent.
- Conventions, in priority order: what the question says; the practice visible in the past \
answers; the conventions the calculator states; otherwise {default}.
- Print one line per output above, in exactly those units, for example "area of the plot: \
12.50 square metres".
- Allowed imports: math, fractions, decimal, statistics, itertools, functools. No input, files \
or network.
"""


def _bullets(items) -> str:
    items = [str(i) for i in items if str(i).strip()]
    return "\n".join(f"- {i}" for i in items) if items else "- (none stated)"


def _outputs(cache: Cache, entry_id: str) -> str:
    specs = cache.outputs.get(entry_id) or {}
    return _bullets(f"{name}: {s.get('unit') or 'no unit'}" for name, s in specs.items())


def _inputs(cache: Cache, entry_id: str) -> str:
    """The entry's input contract. original_value is left out: it is the build question's value."""
    ep = next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None)
    lines = []
    for p in ep.params if ep else []:
        bits = [p.description]
        if p.format_examples:
            bits.append("formats: " + ", ".join(map(str, p.format_examples[:4])))
        if p.validity_range:
            bits.append(f"valid range: {p.validity_range}")
        lines.append("; ".join(bits))
    return _bullets(lines)


def write_suite(cache: Cache, entry_id: str, ctx: Context, tag: str, model: str | None = None) -> TestSuite:
    """One call to the test-maker model for BOTH halves (twice TESTER_CASES / TESTER_DECLINES)."""
    from .labelfree import DEFAULT_CONVENTION
    examples = "\n".join(f"Q: {e['question']}\nA: {e['answer']}\n" for e in ctx.examples) or "(none)"
    near = ("\n## Real questions that were judged a DIFFERENT task\n" + _bullets(ctx.near_misses) + "\n"
            if ctx.near_misses else "")
    hist = ("\n## Problems earlier attempts at this calculator had\n" + _bullets(ctx.history) + "\n"
            if ctx.history else "")
    prompt = PROMPT.format(
        examples=examples, inputs=_inputs(cache, entry_id), outputs=_outputs(cache, entry_id),
        conventions=_bullets(cache.conventions.get(entry_id) or []), near_misses=near, history=hist,
        n_cases=2 * config.TESTER_CASES, n_declines=2 * config.TESTER_DECLINES, default=DEFAULT_CONVENTION,
        history_ask=" Include cases that probe the earlier problems listed above." if ctx.history else "",
        near_miss_ask=(" Model them on the real different-task questions above where you can (new values,"
                       " same twist)." if ctx.near_misses else ""))
    return llm.call(ROLE, prompt, model=model or config.TESTER_MODEL, schema=TestSuite,
                    effort=config.TESTER_EFFORT, max_tokens=16000, tag=tag).parsed


def _get_half(bank: Bank, model: str, cache: Cache, entry_id: str, ctx: Context, out: dict) -> Half | None:
    half = bank.next_half(model)
    if half is not None:
        out["suite"] = "reused"
        return half
    if bank.written >= config.TESTER_MAX_SUITES:
        out["reason"] = f"{LIMIT_REASON} ({config.TESTER_MAX_SUITES} suites)"
        return None
    bank.written += 1
    suite = write_suite(cache, entry_id, ctx, f"{entry_id}:tests{bank.written}", model=model)
    n, m = config.TESTER_CASES, config.TESTER_DECLINES
    out["dropped"] = max(0, len(suite.cases) - 2 * n) + max(0, len(suite.declines) - 2 * m)   # hard cap
    halves = [Half(suite.cases[:n], suite.declines[:m]), Half(suite.cases[n:2 * n], suite.declines[m:2 * m])]
    bank.halves.setdefault(model, []).extend(h for h in halves if h.cases or h.declines)
    out["suite"] = "written"
    return bank.next_half(model)


# ----------------------------------------------------------------------------- feedback

FEEDBACK_PROMPT = """\
An automated calculator, built for the kind of question shown below, failed some of your \
acceptance tests. Its developer will fix it but must NOT see the tests themselves.

Examples of the questions it was built for:
{examples}

Failed tests (for your eyes only):
{failures}

Describe what the calculator gets wrong as GENERAL RULES about this kind of question, so the \
developer fixes the underlying behaviour rather than particular cases: which rule, case, input \
form or kind of question it mishandles, and what correct behaviour looks like (for example \
"rates stated per month are treated as yearly; convert by the stated period"). One line per \
distinct problem. Do NOT quote or paraphrase the test questions, and do NOT include any \
specific numbers from them or from the answers.
"""

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _redact(text: str, leaked: set[str]) -> str:
    """Replace any number that appeared in a failing test with <value>."""
    return _NUM.sub(lambda m: "<value>" if m.group(0).replace(",", "") in leaked else m.group(0), text)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _quotes(line: str, question: str, run: int = 5) -> bool:
    """True if `line` repeats a run of `run` consecutive words from `question`."""
    q = _words(question)
    grams = {tuple(q[i:i + run]) for i in range(len(q) - run + 1)}
    w = _words(line)
    return any(tuple(w[i:i + run]) in grams for i in range(len(w) - run + 1))


def write_feedback(rows: list[dict], questions: list[str], model: str, tag: str) -> list[str]:
    """Semantic description of the in-scope failures, for the builder's repair round."""
    failed = [r for r in rows if r["outcome"] == "fail"]
    if not failed:
        return []
    lines = [f"- {r['question']} | expected: {r['expected']} | calculator said: {r['cache_answer']}"
             f" | what the test probed: {r.get('purpose', '')}" for r in failed]
    prompt = FEEDBACK_PROMPT.format(examples=_bullets(questions[-config.TESTER_EXAMPLES:]),
                                    failures="\n".join(lines))
    try:
        got = llm.call(ROLE, prompt, model=model, schema=Feedback, effort=config.TESTER_EFFORT,
                       max_tokens=4000, tag=tag).parsed
    except llm.LLMError:
        return []
    leaked = {n.replace(",", "") for r in failed
              for n in _NUM.findall(f"{r['question']} {r['expected']} {r['cache_answer']}")}
    return [_redact(p, leaked) for p in got.problems
            if p.strip() and not any(_quotes(p, r["question"]) for r in failed)]


# ----------------------------------------------------------------------------- run

def _program(source: str) -> str | None:
    if not source.strip():
        return None
    ran = sandbox.run_program(source, seconds=config.LABELFREE_PROGRAM_SECONDS)
    return ran.stdout.strip() if ran.ok and ran.stdout.strip() else None


def run(cache: Cache, entry_id: str, ctx: Context, bank: Bank, budget: float,
        builder_model: str | None = None) -> dict:
    """Judge the entry on an unspent half for pick_model(builder_model), writing a suite if none is
    left. Shape matches labelfree._prepublish_check: passed, reason, rows, code_error, stopped;
    plus gate_error (a look-alike was answered wrongly) and feedback (for the builder's repair)."""
    from . import labelfree
    from .serve import format_output
    ep = next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None)
    model = pick_model(builder_model)
    out = {"model": model, "passed": False, "reason": "", "rows": [], "code_error": False, "gate_error": False,
           "stopped": False, "checkable": 0, "served": 0, "confirmed": 0, "feedback": [],
           "context": {"examples": len(ctx.examples), "near_misses": len(ctx.near_misses),
                       "history": len(ctx.history)}}
    if builder_model and builder_model == model:
        out["warning"] = "tester and builder are the same model: the tests are not independent"
        print(f"[tester] warning: {out['warning']} ({builder_model})", file=sys.stderr)
    if ep is None:
        out["reason"] = f"{TOO_FEW_REASON}: entry has no original entry point"
        return out
    start = len(llm.LEDGER.records)
    try:
        half = _get_half(bank, model, cache, entry_id, ctx, out)
    except llm.LLMError as e:
        out["reason"] = f"test-maker failed: {e}"[:300]
        return out
    if half is None:
        return out
    conventions = cache.conventions.get(entry_id)

    jobs = [("case", c) for c in half.cases] + [("decline", d) for d in half.declines]
    for i, (kind, case) in enumerate(jobs):
        if llm.LEDGER.cost(since=start) >= budget:
            out["stopped"] = True
            break
        tag = f"{entry_id}:test{bank.written}.{i}"
        row = {"kind": kind, "question": case.question, "outcome": "", "expected": "", "cache_answer": "",
               "purpose": getattr(case, "purpose", ""), "detail": getattr(case, "why_different", "")}
        out["rows"].append(row)
        expected = _program(case.program)
        if kind == "case":
            if expected is None:
                row.update(outcome="no_reference", detail="the tester's program failed")
                continue
            out["checkable"] += 1
        row["expected"] = expected or ""
        try:
            result, status, detail = labelfree._run_serving(cache, case.question, tag, entry_point=ep)
        except llm.LLMError as e:     # gate or extraction outage: no verdict on the entry
            row.update(outcome="error", detail=f"error: {e}"[:200])
            continue
        if result is None:
            row.update(outcome="declined" if kind == "case" else "pass", detail=f"{status} {detail}".strip()[:200])
            out["code_error"] |= status == "crashed" and kind == "case"
            continue
        row["cache_answer"] = format_output(result.output)
        try:
            agree = bool(expected) and labelfree._compare("tester vs cache", case.question, expected,
                                                          row["cache_answer"], f"{tag}:cmp").agree
            if kind == "decline":
                # answering a look-alike is only harmful if the answer is wrong for IT
                row["outcome"] = "harmless" if agree else "fail_served"
                continue
            out["served"] += 1
            if agree:
                row["outcome"] = "pass"
            else:
                verdict, _, why = labelfree.adjudicate(case.question, expected, row["cache_answer"], tag,
                                                       conventions=conventions)
                row.update(outcome={"cache_right": "pass", "cache_wrong": "fail"}.get(verdict, "unchecked"),
                           detail=f"{verdict}: {why}"[:200])
        except llm.LLMError as e:
            row.update(outcome="unchecked" if kind == "case" else "fail_served", detail=f"error: {e}"[:200])
        out["confirmed"] += row["outcome"] == "pass"

    fail = next((r for r in out["rows"] if r["outcome"] == "fail"), None)
    gate = next((r for r in out["rows"] if r["outcome"] == "fail_served"), None)
    if out["stopped"]:
        out["reason"] = f"cost cap ${budget:.2f} reached after {len(out['rows'])} of {len(jobs)} tests"
    elif fail:
        out["reason"] = (f"{FAILED_REASON}: expected {fail['expected'][:60]!r}, cache said "
                         f"{fail['cache_answer'][:60]!r} ({fail['question'][:100]!r}; {fail['detail'][:100]})")
        out["code_error"] = True
        if llm.LEDGER.cost(since=start) < budget:
            out["feedback"] = write_feedback(out["rows"], ctx.questions, model, f"{entry_id}:feedback")
            half.spent = bool(out["feedback"])
    elif gate:
        out["reason"] = f"{GATE_REASON}: {gate['question'][:100]!r} ({gate['detail'][:100]})"
        out["gate_error"] = True
    elif out["confirmed"] < config.TESTER_MIN_CASES:
        out["reason"] = f"{TOO_FEW_REASON}: {out['confirmed']} (need {config.TESTER_MIN_CASES})"
    elif out["served"] < math.ceil(out["checkable"] / 2):
        out["reason"] = f"{DECLINED_REASON}: served {out['served']} of {out['checkable']}"
    out["passed"] = not out["reason"]
    out["cost"] = llm.LEDGER.cost(since=start)
    return out
