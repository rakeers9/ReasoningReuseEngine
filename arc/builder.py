"""Step 4: cache construction (paper Section 2.2.1).

Given q(o) and the source transcript S(q(o)), the builder model B constructs
the program p in five phases:

    1  input-output isolation
    2  decomposition
    3  parameter recognition (template q(●1..●n) + validity ranges)
    4  planning (function tree, bottom-up order, granularity)
    5  implementation: write -> static checks -> run -> judge -> fix, per
       function; then the root end-to-end in two stages (correctness against
       transcript + golden label; extraction robustness repeated N times)

The paper does not publish its prompts; the prompts below are ours, written
from the paper's description of each phase.
"""

from __future__ import annotations

import ast
import json
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from . import config, judges, llm, sandbox
from .cache import Cache, EntryPoint, ParamSpec
from .checks import number_check
from .runtime import PRIMITIVES, Context
from .serve import dispatch_root, extract_parameters, format_output
from .source import Transcript


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# What the builder is told about the runtime
# ----------------------------------------------------------------------------

RUNTIME_DOC = '''
## Semantic functions

A semantic function is a Python function with exactly this shape:

    def semantic_function(ctx):
        """<one-sentence imperative description of the job it does>"""
        ...
        return {"<natural-language description of an output>": value, ...}

It implements one logical step of a reasoning process, and it must be reusable
for every problem that needs the same step, however that problem is worded and
whatever its values are.

Rules:
- Never rely on parameter names. Get every input with
  load(ctx, "<description>", d_type=<type>). Descriptions are matched by
  meaning, so make them specific, e.g. "the temperature of the gas with unit".
- Return a dict whose keys are natural-language descriptions of the outputs.
- Call another semantic function only through
  do(ctx, "<imperative description of the job>", payloads={"<input description>": value})
  and read its outputs with load(result, "<output description>", d_type=<type>, ctx=ctx).
- Use lm / extract / smart_convert only for genuine language tasks (reading a
  number out of text, recognising a unit, a short classification). Do all
  arithmetic and logic in plain Python.
- Call narrate(ctx, "...") for every loaded value and every meaningful step;
  the narration is the human-readable explanation of how the answer was found.
- Nothing may depend on the specific values of the original problem instance.
- Allowed imports: math, cmath, fractions, decimal, itertools, functools, re,
  statistics, collections, operator. No file, network, or OS access.

## Runtime primitives (already in scope; do not import them)

- load(ctx, query, d_type=None): value from the context by description.
- load(result, query, d_type=None, ctx=ctx): value from a do() result by description.
- do(ctx, query, payloads): find and run another semantic function by description;
  returns its output dict.
- narrate(ctx, message): record a human-readable reasoning step.
- lm(ctx, prompt) -> str: ask a small language model an elementary question.
- extract(ctx, text, prompt, d_type=None): small model extracts a value verbatim from text.
- convert(ctx, value, type) / smart_convert(ctx, text, type): type coercion; plain
  Python first, then the small model rewrites the value as a Python literal.
- invoke_tool(ctx, name, args): call a registered external tool.

## Example of the style (from an unrelated problem)

def semantic_function(ctx):
    """Check whether a valid rotation exists that maps blue vertices to red positions."""
    narrate(ctx, "Checking if a valid rotation exists that "
                 "maps blue vertices to red positions")
    vertex_set = load(ctx, "the blue vertex set B", d_type=list)
    modulus = load(ctx, "the vertex count n", d_type=int)
    B = set(vertex_set)
    diff_result = do(ctx,
        "Compute the difference set D(B) for the vertex subset B with modulus n",
        payloads={
            "the set of vertices B": list(B),
            "the modulus n for modular arithmetic": modulus,
        })
    diff_set = load(diff_result, "difference set", d_type=set, ctx=ctx)
    all_rotations = set(range(modulus))
    valid_k_values = all_rotations - diff_set
    narrate(ctx, f"Valid rotations: {sorted(valid_k_values)}")
    return {
        "valid rotation exists": len(valid_k_values) > 0,
        "valid k values": sorted(valid_k_values),
    }
'''

BUILDER_SYSTEM = (
    "You are the builder of an agentic reasoning cache. You read a capable model's reasoning "
    "transcript for one problem and turn it into a network of small, reusable, executable "
    "semantic functions that solve that problem and every problem solvable with the same "
    "reasoning. Follow the runtime contract below exactly.\n" + RUNTIME_DOC
)


# DEVIATION (label-free builds, units="strict"): input, money and precision rules added to
# the builder's instructions. The paper prompt above is unchanged when units="paper".
# Owner decision 2026-11-15: no unit library or table; the builder model writes its own
# conversions. The hand-typed-factor scan (arc/factorscan.py) stays available behind
# config.BUILDER_FACTOR_SCAN, off by default.
from .factorscan import typed_factors  # noqa: E402

STRICT_UNITS_RULES = '''
## Inputs, units, money and precision (these rules override the style example)

- Every input arrives as the raw text copied from the question, e.g. "$150k", "$0.42 million",
  "4.860 dm3", "27°C", "ten moles", "1 in 200". Load it as a string and parse it yourself:
  the number (digits or number words, thousands separators, a sign), any scale word or suffix
  ("k", "thousand", "million", "bn", "billion"; multiply by 10**3, 10**6, 10**9) and the unit.
  Never assume a value is already a bare number, and raise an error for text you can't parse.
- Convert units yourself in plain Python with exact factors. Check every prefix (mL vs L,
  cm3 vs dm3 vs m3, kPa vs Pa) and every temperature offset, and handle each unit the
  question family could use, not only the one in this example. Raise an error for a unit
  you don't handle rather than guessing.
- Do money arithmetic with decimal.Decimal (Decimal(str(x))), never with float.
- Return full precision. Never round until the end: do not round any intermediate value or
  any result (a monthly payment used in a total stays unrounded), unless the question itself
  states a rounding rule. Rounding for display happens once, later, outside your code.
- Name each output with its unit or currency, e.g. "monthly payment in dollars",
  "pressure in kPa", and return the value in exactly that unit.
'''

BUILDER_SYSTEM_STRICT = BUILDER_SYSTEM + STRICT_UNITS_RULES


# ----------------------------------------------------------------------------
# Structured outputs for each phase
# ----------------------------------------------------------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IOIsolation(_Strict):
    user_input: str
    expected_output: str


class Operation(_Strict):
    id: str
    description: str
    inputs: list[str]
    outputs: list[str]
    parent_id: str | None
    reusable: bool


class Decomposition(_Strict):
    operations: list[Operation]


class Parameter(_Strict):
    marker: str
    description: str
    format_examples: list[str]
    original_value: str
    validity_range: str


class ParameterRecognition(_Strict):
    problem_class: str
    template: str
    parameters: list[Parameter]


class PlannedFunction(_Strict):
    id: str
    description: str
    inputs: list[str]
    outputs: list[str]
    calls: list[str]
    transcript_fragment: str


class Plan(_Strict):
    functions: list[PlannedFunction]
    root_id: str
    implementation_order: list[str]


class Payload(_Strict):
    description: str
    value: str


class Candidate(_Strict):
    reuse_existing_function_id: str | None
    description: str
    code: str
    test_payloads: list[Payload]
    expected_result: str


class Repair(_Strict):
    parameters: list[Parameter]
    root_code: str | None
    explanation: str


# ----------------------------------------------------------------------------
# Build state
# ----------------------------------------------------------------------------

@dataclass
class Attempt:
    function_id: str
    attempt: int
    code: str
    passed: bool
    feedback: str
    # Evidence for human review (no effect on the build):
    failed_at: str | None = None        # static | run | judge | None (passed)
    description: str = ""
    test_inputs: dict = field(default_factory=dict)
    expected_result: str = ""           # the builder's own claim, from the transcript
    transcript_fragment: str = ""
    output: dict | None = None
    narration: str = ""
    verdict: dict | None = None         # the build judge's passed + critique


@dataclass
class BuildReport:
    problem_id: str
    phases: dict[str, Any] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    end_to_end: list[dict] = field(default_factory=list)
    extraction_rounds: list[dict] = field(default_factory=list)
    reused: dict[str, str] = field(default_factory=dict)
    success: bool = False
    cost: float = 0.0
    cost_by_role: dict = field(default_factory=dict)


# DEVIATION (label-free builds, Builder(reference="executed")): the transcript's numbers come
# from a program that was run, so a judge that redoes the arithmetic by hand can only add errors
# (fam_f0003, 2026-11-15: a hand slip made the judges reject correct code 8 times).
EXECUTED_STEP_RULE = (
    "The transcript's numbers come from an executed program (its source and printed output are in "
    "the transcript), so they are exact. Do not recompute any arithmetic by hand. Pass only if the "
    "output agrees with the executed values for this step (equal to their printed precision, or "
    "more precise) and the code uses its inputs rather than hard-coding the original values. If the "
    "builder's expected result disagrees with the executed values, the executed values win. Fail "
    "only for a real disagreement with an executed value, or for hard-coding. If no printed value "
    "covers this step, check that the code does what the program does for it; never fail it on a "
    "number you worked out yourself.")
EXECUTED_FINAL_RULE = (
    " The transcript's values come from an executed program, so they are exact. Do not recompute "
    "them by hand. Pass only if the cache output agrees with the executed final answer (equal to "
    "its printed precision, or more precise).")


class BuildError(RuntimeError):
    pass


class Builder:
    def __init__(self, problem: dict, transcript: Transcript, use_golden: bool = True,
                 granularity: str = config.GRANULARITY, units: str = "paper", model: str | None = None,
                 reference: str = "transcript"):
        """units="paper": the paper's builder, unchanged. units="strict" (label-free builds,
        DEVIATION): raw-text inputs parsed by the code, Decimal money, full precision, output
        names with units; with config.BUILDER_FACTOR_SCAN on, a scan also sends hand-typed
        conversion factors back to the builder. model: the builder model (default
        config.BUILDER_MODEL; the label-free cascade passes each of LABELFREE_BUILDERS).
        reference="transcript": the paper's build judges, unchanged. reference="executed"
        (label-free builds, DEVIATION): the transcript's values come from an executed program,
        so the judges compare with those values and never redo the arithmetic by hand."""
        if units not in ("paper", "strict"):
            raise ValueError(f"units must be 'paper' or 'strict', not {units!r}")
        if reference not in ("transcript", "executed"):
            raise ValueError(f"reference must be 'transcript' or 'executed', not {reference!r}")
        self.units = units
        self.reference = reference
        self.model = model or config.BUILDER_MODEL
        self.problem = problem
        self.t = transcript
        self.use_golden = use_golden
        self.granularity = granularity
        self.cache = Cache(problem["id"])          # isolated per-sample cache (Sec. 3.4)
        self.report = BuildReport(problem["id"])
        golden = (f"\n\n## Golden label (correct final answer)\n\n{problem['golden_label']}"
                  if use_golden else "")
        self._context_block = {
            "type": "text",
            "text": (f"## User query q(o)\n\n{transcript.question}\n\n"
                     f"## Source model transcript S(q(o))\n\n{transcript.solution}{golden}"),
            "cache_control": {"type": "ephemeral"},
        }
        system = BUILDER_SYSTEM_STRICT if units == "strict" else BUILDER_SYSTEM
        self._system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    # -- helpers -------------------------------------------------------------

    def _ask(self, instructions: str, schema: type[BaseModel], tag: str):
        prompt = [self._context_block, {"type": "text", "text": instructions}]
        return llm.call("builder", prompt, model=self.model, system=self._system,
                        schema=schema, effort=config.BUILDER_EFFORT, max_tokens=16000,
                        tag=f"{self.problem['id']}:{tag}").parsed

    def _so_far(self) -> str:
        return "\n\n".join(f"## {name} result\n\n{json.dumps(val, indent=2)}"
                           for name, val in self.report.phases.items())

    # -- phases 1-4 ----------------------------------------------------------

    def phase1(self) -> IOIsolation:
        out = self._ask(
            "# Phase 1: input-output isolation\n\nExtract the user input (the problem statement "
            "exactly as the user posed it) and isolate the expected output: the final answer "
            "the transcript arrives at, copied verbatim with its units.", IOIsolation, "phase1")
        self.report.phases["Phase 1 (input-output isolation)"] = out.model_dump()
        log(f"  [phase 1] expected output: {out.expected_output}")
        return out

    def phase2(self) -> Decomposition:
        out = self._ask(
            self._so_far() + "\n\n# Phase 2: decomposition\n\nDecompose the reasoning in the "
            "transcript into a hierarchical list of logical operations. Describe each operation "
            "imperatively and name the basic individual inputs it consumes (values, entities, "
            "relationships, quantities); never refer to the problem as a monolithic description. "
            "Aim for function-sized units: small enough to be one semantic function, large "
            "enough to be a coherent reasoning step. Nest operations with parent_id (null at the "
            "top level). Set reusable=true for sub-computations likely to recur across variants "
            "of this problem or in entirely different problems.", Decomposition, "phase2")
        self.report.phases["Phase 2 (decomposition)"] = out.model_dump()
        log(f"  [phase 2] {len(out.operations)} operations: " +
            "; ".join(o.description for o in out.operations))
        return out

    def phase3(self) -> ParameterRecognition:
        out = self._ask(
            self._so_far() + "\n\n# Phase 3: parameter recognition\n\nIdentify the class of "
            "problems this query belongs to and which parts of the user input are the natural "
            "input parameters that vary across instances of that class. Every parameter must be "
            "a basic, self-contained piece of information the reasoning directly consumes (a "
            "number with its unit, a name, a formula, a relationship, a list, a table), never an "
            "entire problem description or multi-sentence narrative. Keep variable-size "
            "collections as single parameters.\n\nProduce the generic input template: the user "
            "input with every parameter replaced by a marker ●1, ●2, ... For each parameter give "
            "its marker, a descriptive natural-language name (this is how the value is looked "
            "up, e.g. 'the temperature of the gas with unit'), 2-3 format examples, its concrete "
            "value in the original input copied verbatim, and an analysis of its logical "
            "validity range.", ParameterRecognition, "phase3")
        self.report.phases["Phase 3 (parameter recognition)"] = out.model_dump()
        log(f"  [phase 3] class: {out.problem_class}")
        log(f"  [phase 3] template: {out.template}")
        for p in out.parameters:
            log(f"            {p.marker} {p.description!r} = {p.original_value!r}  "
                f"(valid: {p.validity_range})")
        return out

    def phase4(self) -> Plan:
        out = self._ask(
            self._so_far() + f"\n\n# Phase 4: planning\n\nProduce a structural plan for the "
            f"program. Granularity: {self.granularity} (fine = nearly every decomposition step "
            "becomes its own function; medium = merge small adjacent steps but keep reusable "
            "sub-computations separate; coarse = keep only the major reasoning stages).\n\n"
            "Requirements:\n- Exactly one root function. Its inputs are exactly the parameter "
            "descriptions from phase 3, copied verbatim, and it returns the final answer.\n"
            "- For every function: an id, an imperative description (callers find it by this "
            "description), inputs and outputs as natural-language descriptions, the ids of the "
            "functions it calls, and the transcript fragment it corresponds to (what it computes "
            "for the original instance).\n- implementation_order is bottom-up: every function "
            "comes after all the functions it calls; the root comes last.\n- The structure must "
            "not depend on the specific parameter values of the original instance.",
            Plan, "phase4")
        ids = {f.id for f in out.functions}
        if out.root_id not in ids or set(out.implementation_order) != ids:
            raise BuildError("plan is inconsistent: root or implementation order does not match")
        self.report.phases["Phase 4 (planning)"] = out.model_dump()
        log(f"  [phase 4] {len(out.functions)} functions, order: {' -> '.join(out.implementation_order)}")
        for f in out.functions:
            log(f"            {f.id}: {f.description}" + (f"  (calls {', '.join(f.calls)})" if f.calls else ""))
        return out

    # -- phase 5: write, check, run, judge, fix ------------------------------

    @staticmethod
    def _static_checks(code: str) -> str | None:
        try:
            tree = sandbox.check_source(code)
        except (SyntaxError, sandbox.UnsafeCodeError) as e:
            return f"Static check failed: {type(e).__name__}: {e}"
        narrations = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "narrate")
        branches = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.If, ast.For, ast.While)))
        needed = 1 + branches // 2
        if narrations < needed:
            return (f"Narration check failed: {narrations} narrate() call(s) for {branches} "
                    f"branch/loop statements; at least {needed} are needed.")
        return None

    @staticmethod
    def _payload_value(raw: str) -> Any:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw

    def _run_candidate(self, code: str, payloads: dict[str, Any]) -> tuple[dict | None, str, str | None]:
        """Execute a candidate function against test payloads in a fresh context."""
        try:
            fn = sandbox.load_function(code, PRIMITIVES)
        except Exception as e:
            return None, "", f"{type(e).__name__}: {e}"
        ctx = Context(self.cache)
        for k, v in payloads.items():
            ctx.set(k, v)
        try:
            with sandbox.time_limit(config.FUNCTION_TIMEOUT_SECONDS):
                output = fn(ctx)
            if not isinstance(output, dict):
                return None, ctx.narration(), f"returned {type(output).__name__}, not a dict"
            return output, ctx.narration(), None
        except Exception as e:
            return None, ctx.narration(), f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"

    def _registered_catalogue(self) -> str:
        if not self.cache.functions:
            return "(none yet)"
        return "\n".join(f"- id {r.id}: description {r.description!r}; inputs {r.inputs}; "
                         f"outputs {r.outputs}" for r in self.cache.functions.values())

    def implement(self, spec: PlannedFunction, params: ParameterRecognition, is_root: bool) -> str:
        similar = [(r, s) for r, s in self.cache.search_functions(spec.description, k=3) if s >= 0.5]
        similar_text = "\n\n".join(f"### id {r.id} (similarity {s:.2f})\n{r.source}" for r, s in similar) \
            or "(none)"
        root_note = ""
        if is_root:
            root_note = ("\nThis is the ROOT function. Load each problem parameter with exactly "
                         "these descriptions: " + json.dumps([p.description for p in params.parameters]) +
                         ". Its test payloads must be exactly these parameters with their original "
                         "values (as Python string literals).")
        feedback = ""
        for attempt in range(1, config.MAX_FUNCTION_ATTEMPTS + 1):
            cand = self._ask(
                self._so_far() + f"\n\n# Phase 5: implementation of function `{spec.id}`\n\n"
                f"Planned function:\n{json.dumps(spec.model_dump(), indent=2)}\n{root_note}\n\n"
                f"Functions already registered (call them with do() using their descriptions):\n"
                f"{self._registered_catalogue()}\n\nSemantically similar registered functions "
                f"(set reuse_existing_function_id to one of these ids only if it already does "
                f"exactly this job):\n{similar_text}\n\nWrite the semantic function. Also give "
                "test_payloads (the inputs it receives for the original instance; each value is a "
                "Python literal, so strings need quotes) and expected_result (what it should "
                "return for those inputs, according to the transcript)." + feedback,
                Candidate, f"phase5:{spec.id}:{attempt}")

            if cand.reuse_existing_function_id and cand.reuse_existing_function_id in self.cache.functions:
                fid = cand.reuse_existing_function_id
                self.cache.add_alias(fid, spec.description)
                self.report.reused[spec.id] = fid
                log(f"  [phase 5] {spec.id}: reusing existing function {fid}")
                return fid

            payloads = {p.description: self._payload_value(p.value) for p in cand.test_payloads}
            if is_root:
                payloads = {p.description: p.original_value for p in params.parameters}
            problem = self._static_checks(cand.code)
            if problem is None and self.units == "strict" and config.BUILDER_FACTOR_SCAN:
                factors = typed_factors(cand.code)
                if factors:
                    problem = "Hand-typed unit conversion found:\n" + "\n".join(factors)
            failed_at = "static" if problem else None
            output, narration, verdict = None, "", None
            if problem is None:
                output, narration, error = self._run_candidate(cand.code, payloads)
                if error:
                    problem = f"Running the function failed:\n{error}"
                    failed_at = "run"
            if problem is None:
                verdict = judges.judge(
                    f"A cache builder wrote a function for one step of this transcript.\n\n"
                    f"## Transcript\n{self.t.solution}\n\n## Step\n{spec.description}\n"
                    f"Transcript fragment: {spec.transcript_fragment}\nExpected result (builder's "
                    f"claim): {cand.expected_result}\n\n## Inputs given\n{payloads}\n\n## Output\n"
                    f"{output}\n\n## Narration\n{narration}\n\n## Code\n{cand.code}\n\n"
                    + (EXECUTED_STEP_RULE if self.reference == "executed" else
                       "Pass only if the output matches what the transcript computes for this step "
                       "and the code uses its inputs rather than hard-coding the original values."),
                    role="build_judge", tag=f"{self.problem['id']}:judge:{spec.id}")
                if not verdict.passed:
                    problem = f"The judge rejected the output: {verdict.critique}"
                    failed_at = "judge"

            self.report.attempts.append(Attempt(
                spec.id, attempt, cand.code, problem is None, problem or "passed",
                failed_at=failed_at, description=cand.description,
                test_inputs={k: repr(v) for k, v in payloads.items()},
                expected_result=cand.expected_result, transcript_fragment=spec.transcript_fragment,
                output={k: str(v) for k, v in output.items()} if output else None,
                narration=narration, verdict=verdict.model_dump() if verdict else None))
            if problem is None:
                rec = self.cache.add_function(cand.description, cand.code, spec.inputs, spec.outputs,
                                              origin=f"builder:{self.problem['id']}")
                log(f"  [phase 5] {spec.id}: passed on attempt {attempt} -> registered as {rec.id}")
                return rec.id
            first_line = problem.splitlines()[0][:160]
            log(f"  [phase 5] {spec.id}: attempt {attempt} failed - {first_line}")
            feedback = (f"\n\n## Your previous attempt failed\n\n```python\n{cand.code}\n```\n\n"
                        f"Feedback:\n{problem}\n\nFix the problem and return a corrected function.")
        raise BuildError(f"function {spec.id} did not pass after {config.MAX_FUNCTION_ATTEMPTS} attempts")

    # -- end-to-end tests ----------------------------------------------------

    def _judge_final(self, output: dict, narration: str, stage: str) -> judges.Verdict:
        golden = f"\nGolden label: {self.problem['golden_label']}" if self.use_golden else ""
        return judges.judge(
            f"## Problem\n{self.t.question}\n\n## Transcript answer\n{self.t.solution}\n{golden}\n\n"
            f"## Cache output\n{format_output(output)}\n\n## Narration\n{narration}\n\n"
            "Does the cache output give the correct final answer to the problem (agreeing with "
            "the transcript" + (" and the golden label" if self.use_golden else "") + ")?"
            + (EXECUTED_FINAL_RULE if self.reference == "executed" else ""),
            role="build_judge", tag=f"{self.problem['id']}:judge:{stage}")

    def end_to_end(self, root_id: str, params: ParameterRecognition) -> tuple[bool, str]:
        values = {p.description: p.original_value for p in params.parameters}
        try:
            output, ctx = dispatch_root(self.cache, root_id, values)
        except Exception as e:
            return False, f"root failed: {type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
        v = self._judge_final(output, ctx.narration(), "end_to_end")
        self.report.end_to_end.append({"output": {k: str(x) for k, x in output.items()},
                                       "passed": v.passed, "critique": v.critique,
                                       "inputs": {k: str(x) for k, x in values.items()},
                                       "narration": ctx.narration(),
                                       "number_check": number_check(format_output(output),
                                                                    self.problem["golden_label"])})
        return v.passed, v.critique

    def extraction_rounds(self, root_id: str, params: ParameterRecognition) -> list[dict]:
        specs = [ParamSpec(p.description, p.format_examples, p.original_value, p.validity_range)
                 for p in params.parameters]
        results = []
        for i in range(config.EXTRACTION_ROUNDS):
            row: dict[str, Any] = {"round": i + 1}
            try:
                values = extract_parameters(self.t.question, specs, tag=f"{self.problem['id']}:build")
                row["extracted"] = values
                output, ctx = dispatch_root(self.cache, root_id, values)
                v = self._judge_final(output, ctx.narration(), f"extraction{i + 1}")
                row.update(passed=v.passed, critique=v.critique,
                           expected={p.description: p.original_value for p in params.parameters},
                           output={k: str(x) for k, x in output.items()},
                           number_check=number_check(format_output(output), self.problem["golden_label"]))
            except Exception as e:
                row.update(passed=False, critique=f"{type(e).__name__}: {e}")
            results.append(row)
            log(f"  [phase 5] extraction round {i + 1}: {'pass' if row['passed'] else 'FAIL - ' + row['critique'][:120]}")
        self.report.extraction_rounds.extend(results)
        return results

    def repair(self, root_id: str, params: ParameterRecognition, failure: str) -> ParameterRecognition:
        fix = self._ask(
            self._so_far() + f"\n\n# Repair\n\nThe root function currently is:\n```python\n"
            f"{self.cache.functions[root_id].source}\n```\n\nIt failed this end-to-end test:\n"
            f"{failure}\n\nReturn corrected parameters (same markers; you may improve descriptions "
            "and format examples) and, if the root code needs to change, the corrected root code "
            "(else null). Explain the fix briefly.", Repair, "repair")
        log(f"  [repair] {fix.explanation}")
        params = ParameterRecognition(problem_class=params.problem_class, template=params.template,
                                      parameters=fix.parameters)
        self.report.phases["Phase 3 (parameter recognition)"] = params.model_dump()
        if fix.root_code:
            problem = self._static_checks(fix.root_code)
            if problem:
                raise BuildError(f"repaired root failed static checks: {problem}")
            rec = self.cache.functions[root_id]
            self.cache.add_function(rec.description, fix.root_code, [p.description for p in params.parameters],
                                    rec.outputs, origin=rec.origin, replace_id=root_id)
        return params

    # -- the whole build -----------------------------------------------------

    def build(self) -> Cache:
        with llm.LEDGER.span() as span:
            try:
                self.phase1()
                self.phase2()
                params = self.phase3()
                plan = self.phase4()
                by_id = {f.id: f for f in plan.functions}
                ids: dict[str, str] = {}
                for fid in plan.implementation_order:
                    ids[fid] = self.implement(by_id[fid], params, is_root=(fid == plan.root_id))
                root_id = ids[plan.root_id]

                # Stage 1: correctness on the original parameters.
                ok, critique = self.end_to_end(root_id, params)
                repairs = 0
                while not ok and repairs < config.MAX_ROOT_REPAIRS:
                    log(f"  [phase 5] end-to-end FAIL - {critique[:160]}")
                    params = self.repair(root_id, params, critique)
                    ok, critique = self.end_to_end(root_id, params)
                    repairs += 1
                if not ok:
                    raise BuildError(f"end-to-end correctness failed: {critique}")
                log("  [phase 5] end-to-end correctness: pass")

                # Stage 2: extraction robustness, N rounds.
                rounds = self.extraction_rounds(root_id, params)
                if not all(r["passed"] for r in rounds) and repairs < config.MAX_ROOT_REPAIRS:
                    failure = next(r for r in rounds if not r["passed"])
                    params = self.repair(root_id, params, json.dumps(failure, default=str))
                    ok, critique = self.end_to_end(root_id, params)
                    rounds = self.extraction_rounds(root_id, params) if ok else []
                if not rounds or not all(r["passed"] for r in rounds):
                    raise BuildError("extraction robustness failed")

                # Register both entry points: the template and the original input.
                specs = [ParamSpec(p.description, p.format_examples, p.original_value, p.validity_range)
                         for p in params.parameters]
                self.cache.add_entry_point(EntryPoint(params.template, "template", root_id, specs,
                                                      self.problem["id"]))
                self.cache.add_entry_point(EntryPoint(self.t.question, "original", root_id, specs,
                                                      self.problem["id"]))
                self.report.success = True
            finally:
                span.end = len(llm.LEDGER.records)
                self.report.cost = span.cost()
                self.report.cost_by_role = span.by_role()
        return self.cache

    def save_report(self) -> None:
        folder = config.problem_dir(self.problem["id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "build_report.json").write_text(json.dumps(asdict(self.report), indent=2, default=str))
