"""Step 5: evaluation (paper Section 3).

Correctness   S Acc (transcript vs golden), C Acc (cache vs golden),
              Fidelity (cache vs transcript)
Robustness    phrasal (reworded, same values), parametric (same wording, new
              values), combined (both). Each follows PT1 generate -> PT2
              validation by a unanimous judge committee -> PT3 run through
              the cache and grade with a second unanimous committee.
              Only run when the cache is golden-correct on the original.
Cost          source, builder and per-invocation cache cost; Build x,
              Savings x and break-even B / (S - C).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from . import config, llm
from .cache import Cache
from .checks import number_check
from .judges import committee
from .runtime import CacheInvocationError
from .serve import format_output, use
from .source import Transcript


def log(msg: str) -> None:
    print(msg, flush=True)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Rephrasings(_Strict):
    variants: list[str]


class ParametricVariant(_Strict):
    question: str
    changed_parameters: list[str]
    original_solution: str     # the paper's "original narration"
    new_solution: str          # the paper's "final narration"
    final_answer: str


class ParametricSet(_Strict):
    solution_steps: list[str]
    parameters_and_constraints: list[str]
    variants: list[ParametricVariant]


@dataclass
class VariantResult:
    mode: str
    question: str
    reference: str
    outcome: str            # pass | fail | miss | error
    cache_answer: str = ""
    detail: str = ""
    # Evidence for human review (no effect on the outcome):
    reference_source: str = ""      # "golden label" or "generator model"
    reference_answer: str = ""      # the reference's final answer only
    verdicts: list = field(default_factory=list)   # each grading judge: passed, critique
    number_check: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)  # what the small model extracted
    output: dict = field(default_factory=dict)      # the root function's raw output
    narration: str = ""
    entry_point: str = ""
    similarity: float | None = None


@dataclass
class EvalReport:
    problem_id: str
    source_correct: bool = False
    cache_correct: bool = False
    fidelity: bool = False
    original_cache_answer: str = ""
    original_narration: str = ""
    variants: list[VariantResult] = field(default_factory=list)
    rejected_candidates: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    # Evidence for human review: the three correctness checks and every generated candidate.
    correctness_checks: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)


def _gen(prompt: str, schema, tag: str):
    return llm.call("generator", prompt, model=config.GENERATOR_MODEL, schema=schema,
                    effort=config.GENERATOR_EFFORT, max_tokens=16000, tag=tag).parsed


class Evaluator:
    def __init__(self, problem: dict, transcript: Transcript, cache: Cache, build_cost: float):
        self.p = problem
        self.t = transcript
        self.cache = cache
        self.build_cost = build_cost
        self.report = EvalReport(problem["id"])
        self.invocation_costs: list[float] = []
        self.k = config.VARIANTS_PER_MODE
        self.done: list[str] = []
        self.progress_path = config.problem_dir(problem["id"]) / "eval_progress.json"
        if self.progress_path.exists():
            self._restore()

    # -- checkpointing (a crash or refusal never throws away paid-for results) --

    def _checkpoint(self) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(json.dumps({
            "done": self.done, "invocation_costs": self.invocation_costs,
            "report": asdict(self.report)}, indent=2, default=str))

    def _restore(self) -> None:
        data = json.loads(self.progress_path.read_text())
        self.done = data["done"]
        self.invocation_costs = data["invocation_costs"]
        r = data["report"]
        r["variants"] = [VariantResult(**v) for v in r["variants"]]
        self.report = EvalReport(**r)
        log(f"  [resume] already done: {', '.join(self.done) or 'nothing'}")

    # -- running one query through the cache ---------------------------------

    def _run(self, question: str) -> tuple[str, str, str]:
        """(outcome, answer, narration) for one cache invocation; details in self.last_run."""
        self.last_run: dict[str, Any] = {}
        try:
            r = use(self.cache, question, tag=f"{self.p['id']}:eval")
        except CacheInvocationError as e:
            return "error", "", str(e).splitlines()[0]
        if r is None:
            ep, sim = self.cache.find_entry_point(question)
            self.last_run = {"entry_point": ep.text if ep else "", "similarity": sim}
            return "miss", "", ""
        self.invocation_costs.append(r.cost)
        self.last_run = {"parameters": dict(r.parameters), "output": {k: str(v) for k, v in r.output.items()},
                         "narration": r.narration, "entry_point": r.entry_point.text,
                         "similarity": r.similarity}
        return "ok", format_output(r.output), r.narration

    def _grade(self, question: str, answer: str, reference: str, tag: str) -> tuple[bool, str]:
        ok, verdicts = committee(
            f"## Question\n{question}\n\n## Reference answer\n{reference}\n\n## Cache output\n"
            f"{answer}\n\nIs the cache output's final answer correct, i.e. equivalent to the "
            "reference answer (up to rounding and unit conversion)?",
            role="test_judge", tag=f"{self.p['id']}:{tag}")
        self.last_verdicts = [v.model_dump() for v in verdicts]
        return ok, " | ".join(v.critique for v in verdicts)

    # -- correctness ---------------------------------------------------------

    def correctness(self) -> bool:
        golden = self.p["golden_label"]
        checks = self.report.correctness_checks
        self.report.source_correct, verdicts = committee(
            f"## Question\n{self.t.question}\n\n## Golden label\n{golden}\n\n## Transcript\n"
            f"{self.t.solution}\n\nIs the transcript's final answer correct per the golden label?",
            role="test_judge", tag=f"{self.p['id']}:s_acc")
        checks["source_correct"] = {
            "question": "Is the source model's answer correct per the golden label?",
            "passed": self.report.source_correct, "verdicts": [v.model_dump() for v in verdicts],
            "evidence": {"Golden label": golden, "Source transcript": self.t.solution},
            "number_check": number_check(self.t.solution, golden)}
        outcome, answer, narration = self._run(self.t.question)
        self.report.original_cache_answer = answer
        self.report.original_narration = narration
        run = dict(self.last_run)
        if outcome == "ok":
            self.report.cache_correct, _ = self._grade(self.t.question, answer, golden, "c_acc")
            checks["cache_correct"] = {
                "question": "Is the cache's answer correct per the golden label?",
                "passed": self.report.cache_correct, "verdicts": self.last_verdicts,
                "evidence": {"Golden label": golden, "Cache output": answer},
                "number_check": number_check(answer, golden), "run": run}
            self.report.fidelity, _ = self._grade(self.t.question, answer,
                                                  self.t.solution, "fidelity")
            checks["fidelity"] = {
                "question": "Does the cache's answer agree with the source transcript?",
                "passed": self.report.fidelity, "verdicts": self.last_verdicts,
                "evidence": {"Source transcript": self.t.solution, "Cache output": answer},
                "number_check": None}
        else:
            checks["cache_correct"] = {"question": "Is the cache's answer correct per the golden label?",
                                       "passed": False, "verdicts": [], "evidence": {},
                                       "note": f"cache {outcome}", "run": run}
        log(f"  [correctness] source correct: {self.report.source_correct} | cache correct: "
            f"{self.report.cache_correct} | fidelity: {self.report.fidelity}"
            + ("" if outcome == "ok" else f" | cache {outcome}"))
        return self.report.cache_correct

    # -- perturbation generation (PT1 + PT2) -------------------------------

    def phrasal_candidates(self, question: str, n: int, mode: str) -> list[str]:
        gen = _gen(f"Write {n} rephrasings of the problem below. Each must ask the substantively "
                   "same question with every parameter value kept exactly the same, but with "
                   "sufficiently different phrasing (restructure sentences, change wording and "
                   f"order).\n\nProblem:\n{question}", Rephrasings, f"{self.p['id']}:gen:{mode}")
        accepted = []
        for cand in gen.variants[:n]:
            ok, verdicts = committee(
                f"## Original problem\n{question}\n\n## Candidate rephrasing\n{cand}\n\nAccept only "
                "if all three hold: (i) every parameter value from the original is preserved "
                "exactly, (ii) the phrasing is sufficiently different from the original, (iii) the "
                "substance of the question is unchanged.", role="gen_judge",
                tag=f"{self.p['id']}:gen_judge:{mode}")
            self.report.candidates.append({"mode": mode, "kind": "rephrasing", "base_question": question,
                                           "question": cand, "accepted": ok,
                                           "verdicts": [v.model_dump() for v in verdicts]})
            if ok:
                accepted.append(cand)
        self.report.rejected_candidates[mode] = self.report.rejected_candidates.get(mode, 0) + \
            len(gen.variants[:n]) - len(accepted)
        return accepted

    def parametric_candidates(self, question: str, n: int, mode: str) -> list[ParametricVariant]:
        gen = _gen(
            f"## Problem\n{question}\n\n## Known correct answer\n{self.p['golden_label']}\n\n"
            "## Worked solution\n" + self.t.solution + "\n\nWe are writing test variants of this "
            "textbook problem. List the solution steps and the problem's parameters with their "
            f"valid ranges and constraints. Then write {n} variants with new parameter values "
            "that differ from the original and respect every constraint. For each variant give the "
            "question using exactly the original wording with only the values changed, an "
            "original_solution (a short worked solution of the ORIGINAL problem that reaches the "
            "known correct answer), a new_solution (a short worked solution of the NEW problem), "
            "and its final_answer.",
            ParametricSet, f"{self.p['id']}:gen:{mode}")
        accepted = []
        for cand in gen.variants[:n]:
            ok, verdicts = committee(
                f"## Original problem\n{question}\n\n## Known correct answer\n"
                f"{self.p['golden_label']}\n\n## Candidate reparameterization\n{cand.question}\n\n"
                f"## Worked solution of the original\n{cand.original_solution}\n\n## Worked solution of "
                f"the new problem\n{cand.new_solution}\nFinal answer: {cand.final_answer}\n\nAccept only if all "
                "hold: (i) the question structure is preserved, (ii) at least one parameter value "
                "actually changed, (iii) all domain constraints are satisfied, (iv) the original "
                "solution correctly reproduces the known correct answer, (v) the new solution "
                "is logically consistent and arrives at a coherent answer for the new problem.",
                role="gen_judge", tag=f"{self.p['id']}:gen_judge:{mode}")
            self.report.candidates.append({"mode": mode, "kind": "new values", "base_question": question,
                                           "question": cand.question, "accepted": ok,
                                           "changed_parameters": cand.changed_parameters,
                                           "final_answer": cand.final_answer,
                                           "new_solution": cand.new_solution,
                                           "original_solution": cand.original_solution,
                                           "verdicts": [v.model_dump() for v in verdicts]})
            if ok:
                accepted.append(cand)
        self.report.rejected_candidates[mode] = self.report.rejected_candidates.get(mode, 0) + \
            len(gen.variants[:n]) - len(accepted)
        return accepted

    # -- PT3: run and grade --------------------------------------------------

    def _test(self, mode: str, question: str, reference: str, reference_source: str,
              reference_answer: str) -> None:
        outcome, answer, detail = self._run(question)
        run = dict(self.last_run)
        self.last_verdicts = []
        if outcome == "ok":
            ok, critique = self._grade(question, answer, reference, f"test:{mode}")
            outcome, detail = ("pass" if ok else "fail"), critique
        self.report.variants.append(VariantResult(
            mode, question, reference, outcome, answer, detail,
            reference_source=reference_source, reference_answer=reference_answer,
            verdicts=self.last_verdicts,
            number_check=number_check(answer, reference_answer) if answer else {},
            **run))
        log(f"  [{mode}] {outcome.upper():5s} {question[:90]}...")

    def _phrasal(self) -> None:
        log("  [phrasal] generating and validating rephrasings...")
        for q in self.phrasal_candidates(self.t.question, self.k, "phrasal"):
            self._test("phrasal", q, self.p["golden_label"], "golden label", self.p["golden_label"])

    def _parametric(self) -> None:
        log("  [parametric] generating and validating new parameter values...")
        for v in self.parametric_candidates(self.t.question, self.k, "parametric"):
            self._test("parametric", v.question, f"{v.final_answer}\n\n{v.new_solution}",
                       "generator model", v.final_answer)

    def _combined(self) -> None:
        # Stage 1 rephrases (fresh, independently judged); stage 2 reparameterizes each.
        log("  [combined] rephrasing, then changing values...")
        for q in self.phrasal_candidates(self.t.question, self.k, "combined-phrasal"):
            for v in self.parametric_candidates(q, 1, "combined-parametric"):
                self._test("combined", v.question, f"{v.final_answer}\n\n{v.new_solution}",
                           "generator model", v.final_answer)

    def robustness(self) -> None:
        for mode, step in (("phrasal", self._phrasal), ("parametric", self._parametric),
                           ("combined", self._combined)):
            if mode in self.done:
                continue
            # Drop partial rows from an interrupted earlier attempt at this mode.
            self.report.variants = [v for v in self.report.variants if v.mode != mode]
            self.report.candidates = [c for c in self.report.candidates
                                      if c["mode"] not in (mode, f"{mode}-phrasal", f"{mode}-parametric")]
            try:
                step()
                self.done.append(mode)
                self.report.errors.pop(mode, None)
            except llm.LLMError as e:
                self.report.errors[mode] = str(e)
                log(f"  [{mode}] stopped: {e}")
            self._checkpoint()

    # -- costs ---------------------------------------------------------------

    def costs(self) -> None:
        S, B = self.t.cost, self.build_cost
        C = sum(self.invocation_costs) / len(self.invocation_costs) if self.invocation_costs else float("nan")
        self.report.costs = {
            "source_cost": S, "builder_cost": B, "cache_cost_per_invocation": C,
            "build_multiplier": B / S if S else None,
            "savings_multiplier": S / C if C and C == C else None,
            "break_even_invocations": math.ceil(B / (S - C)) if C == C and S > C else None,
            "invocations_measured": len(self.invocation_costs),
        }

    def run(self) -> EvalReport:
        if "correctness" not in self.done:
            self.correctness()
            self.done.append("correctness")
            self._checkpoint()
        if self.report.cache_correct:
            self.robustness()
        else:
            log("  [robustness] skipped: the paper only perturbs problems the cache gets right")
        self.costs()
        return self.report

    def save(self) -> None:
        folder = config.problem_dir(self.p["id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "eval_report.json").write_text(json.dumps(asdict(self.report), indent=2, default=str))


def pass_rates(report: EvalReport) -> dict[str, str]:
    out = {}
    for mode in ("phrasal", "parametric", "combined"):
        rows = [v for v in report.variants if v.mode == mode]
        passed = sum(v.outcome == "pass" for v in rows)
        out[mode] = f"{passed}/{len(rows)}" if rows else "n/a"
    return out
