"""Agreement certificate: a label-free stand-in for the golden label at build time.

DEVIATION from the paper. The paper builds every entry with the benchmark's golden
label in the builder's context and in the acceptance judge. Production has no label.
Building without it lets a wrong transcript become a wrong cache entry that is then
served to every matching question.

A cache may always decline to cache (the question just goes to the source model, as
today), so we only need high precision on "this entry is right". The certificate is
an execution-based agreement test in the style of CodeT / MBR-exec:

  C1  the generator writes N new instances of the question (template wording, new
      values). It writes questions only, never answers.
  C2  the source model solves the original and each new instance from scratch, in
      fresh calls that never see the transcript or the code.
  C3  the cached code runs on the same instances (target-model extraction + dispatch).
  C4  every pair must agree: the transcript vs the fresh solve of the original, and
      the cache vs the fresh solve on the original and on every new instance.
      Agreement = an equivalence judge says yes AND the number check is not a mismatch.

The entry is accepted only if every comparison agrees. The golden label is never
used to decide; it is recorded afterwards for diagnostics only, so we can measure
whether the certificate's decisions were right.

Known limit: the fresh solves come from the same model family as the transcript, so
an error the model makes consistently can pass. This catches random slips and code
that is right only at the original values, not shared misconceptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from . import config, judges, llm
from .cache import Cache, EntryPoint
from .checks import number_check
from .serve import dispatch_root, extract_parameters, format_output
from .source import SOLVE_PROMPT, Transcript


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewInstance(_Strict):
    question: str
    changed_values: list[str]


class NewInstances(_Strict):
    instances: list[NewInstance]


class Solved(_Strict):
    working: str
    final_answer: str


@dataclass
class Comparison:
    name: str              # e.g. "transcript vs fresh solve (original)"
    question: str
    reference: str         # the fresh independent answer
    candidate: str         # transcript answer or cache output
    agree: bool
    judge_said: bool
    number_status: str
    critique: str


@dataclass
class Certificate:
    problem_id: str
    accepted: bool = False
    reason: str = ""
    instances: list[str] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    fresh_original: str = ""
    cache_original: str = ""
    golden_diagnostic: dict = field(default_factory=dict)   # never used to decide
    cost: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


def _generate_instances(question: str, template: str, ep: EntryPoint, n: int, tag: str) -> list[str]:
    params = "\n".join(f"- {p.description}: original {p.original_value!r}; other accepted forms: "
                       f"{', '.join(map(repr, p.format_examples)) or 'none'}; valid range: "
                       f"{p.validity_range or 'not stated'}" for p in ep.params)
    prompt = (
        f"## Original question\n{question}\n\n## Template (●n marks each value)\n{template}\n\n"
        f"## Values\n{params}\n\n"
        f"Write {n} new versions of the original question for testing. Use exactly the original "
        "wording with only the values changed. Change at least two values in each version, and across "
        "the versions change every value at least once, so each input is tested with a new value. "
        "Where a value has a unit, also write it in a different accepted unit in at least one version "
        "(see the other accepted forms), so each unit conversion is exercised. "
        "Use realistic values inside the valid ranges, and make the versions differ from each other. "
        "Write only the questions; do not solve them."
    )
    out = llm.call("generator", prompt, model=config.GENERATOR_MODEL, schema=NewInstances,
                   effort=config.GENERATOR_EFFORT, max_tokens=4000, tag=tag).parsed
    return [i.question for i in out.instances[:n]]


def _fresh_solve(question: str, tag: str) -> str:
    """The source model solves from scratch: no transcript, no code, no label."""
    prompt = SOLVE_PROMPT.format(question=question) + (
        "\n\nPut the full working in `working` and only the final answer, with units, in `final_answer`.")
    out = llm.call("source", prompt, model=config.SOURCE_MODEL, schema=Solved,
                   effort=config.SOURCE_EFFORT, max_tokens=16000, tag=tag).parsed
    return out.final_answer


# Owner rule (2026-11-15): an exact amount must match exactly. The judge is told the same
# rule the number check applies (arc/checks.py): equal to the precision written.
EXACT_JUDGE_SYSTEM = (
    "You are a meticulous verifier. Judge only what you are asked. Two answers agree only if "
    "they give the same value to the precision written: any difference beyond the last digit "
    "stated in the less precise of the two is a mismatch. Money given to the cent must match to "
    "the cent. A unit conversion of the same value is fine (5.13 MPa = 5130 kPa). Keep the "
    "critique short and specific.")


def _compare(name: str, question: str, reference: str, candidate: str, tag: str) -> Comparison:
    """Agree only if the judge says the final answers are the same value to the precision
    written AND the number check finds no mismatch (no percentage tolerance)."""
    nc = number_check(candidate, reference)
    v = judges.judge(
        f"## Question\n{question}\n\n## Answer A\n{reference}\n\n## Answer B\n{candidate}\n\n"
        "Do A and B give the same final answer to the question? They must be the same value to the "
        "precision written: any difference beyond the last digit stated in the less precise of the "
        "two is a mismatch, and money given to the cent must match to the cent. A unit conversion "
        "of the same value is fine. Compare only the final answers, not the working. If the question "
        "asks for several values, every one of them must match.",
        role="cert_judge", tag=tag, system=EXACT_JUDGE_SYSTEM)
    agree = v.passed and nc["status"] != "mismatch"
    return Comparison(name, question, reference, candidate, agree, v.passed, nc["status"], v.critique)


def _run_cache(cache: Cache, ep: EntryPoint, question: str, tag: str) -> str:
    values = extract_parameters(question, ep.params, tag=tag)
    output, _ = dispatch_root(cache, ep.root_id, values)
    return format_output(output)


def certify(cache: Cache, problem: dict, transcript: Transcript, n: int = 3) -> Certificate:
    """Run C1-C4 on a freshly built per-sample cache. Stops at the first disagreement."""
    pid = problem["id"]
    cert = Certificate(pid)
    with llm.LEDGER.span() as span:
        try:
            ep = next(e for e in cache.entry_points if e.problem_id == pid and e.kind == "original")
            tmpl = next(e for e in cache.entry_points if e.problem_id == pid and e.kind == "template")
            cert.instances = _generate_instances(transcript.question, tmpl.text, ep, n, f"{pid}:cert:gen")
            if len(cert.instances) < n:
                cert.reason = f"generator wrote {len(cert.instances)} of {n} instances"
                return cert

            fresh = cert.fresh_original = _fresh_solve(transcript.question, f"{pid}:cert:solve0")
            checks = [("transcript vs fresh solve (original)", transcript.question, fresh,
                       transcript.solution)]
            try:
                cert.cache_original = _run_cache(cache, ep, transcript.question, f"{pid}:cert:run0")
                checks.append(("cache vs fresh solve (original)", transcript.question, fresh,
                               cert.cache_original))
            except Exception as e:
                cert.reason = f"cache failed on the original: {type(e).__name__}: {e}"[:300]
                return cert
            for i, q in enumerate(cert.instances, 1):
                ref = _fresh_solve(q, f"{pid}:cert:solve{i}")
                try:
                    out = _run_cache(cache, ep, q, f"{pid}:cert:run{i}")
                except Exception as e:
                    cert.comparisons.append(Comparison(f"cache vs fresh solve (instance {i})", q, ref,
                                                       "", False, False, "error", f"cache failed: {e}"[:300]))
                    cert.reason = f"cache failed on instance {i}"
                    return cert
                checks.append((f"cache vs fresh solve (instance {i})", q, ref, out))

            for k, (name, q, ref, cand) in enumerate(checks):
                c = _compare(name, q, ref, cand, f"{pid}:cert:judge{k}")
                cert.comparisons.append(c)
                if not c.agree:
                    cert.reason = f"disagreement: {name}"
                    return cert
            cert.accepted = True
            cert.reason = f"all {len(checks)} comparisons agree"
        finally:
            span.end = len(llm.LEDGER.records)
            cert.cost = span.cost()
            # Diagnostics only: did the cache match the real label on the original?
            g = problem.get("golden_label")
            if g:
                cert.golden_diagnostic = {
                    "transcript_vs_golden": number_check(transcript.solution[-1500:], g)["status"],
                    "fresh_vs_golden": number_check(cert.fresh_original, g)["status"] if cert.fresh_original else "n/a",
                    "cache_vs_golden": number_check(cert.cache_original, g)["status"] if cert.cache_original else "n/a",
                }
    return cert


def save(cert: Certificate) -> None:
    folder = config.problem_dir(cert.problem_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "certificate.json").write_text(json.dumps(cert.to_json(), indent=2, default=str))
