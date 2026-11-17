"""BuildResult.failure: why a label-free build failed, for the families' wait-and-retry logic.

  "code"      the built code failed its own tests or a code check (counts toward the
              families' cap of failed builds)
  "api"       a model/API error (the family waits; not counted)
  "cost_cap"  the cost cap stopped the build (the family waits; not counted)
  None        accepted, or a mismatch that is no failure (the re-solve disagrees with
              production, too few questions fit)

For the cascade it is why the LAST attempt ended. Re-solve, builder, gate and judges are faked.
"""

from __future__ import annotations

import pytest

from arc import config, labelfree, llm
from test_builder_cascade import HAIKU, SONNET, cascade  # noqa: F401
from test_labelfree_simple import FAM, FakeBuilder, family, pre, sim, state  # noqa: F401


def build():
    return labelfree.build_family(FAM, family(8))


# ----------------------------------------------------------------------------- None

def test_accepted_has_no_failure(cascade):
    res = build()
    assert res.status == "accepted" and res.failure is None
    assert state(cascade)["failure"] is None and res.cascade[0]["failure"] is None


def test_resolve_mismatch_is_not_a_failure(cascade):
    cascade.answers = ["7", "9"]
    res = build()
    assert res.status == "mismatch" and res.failure is None


def test_too_few_questions_fit_is_not_a_failure(cascade):
    cascade.p = {f"double {i} m": 0.2 for i in range(1, 8)}
    res = build()
    assert res.status == "mismatch" and res.reason.startswith(labelfree.MISFIT_REASON) and res.failure is None


# ----------------------------------------------------------------------------- "code"

def test_both_builders_failing_their_own_tests_is_code(cascade):
    FakeBuilder.fail = True
    res = build()
    assert res.status == "rejected" and res.failure == "code"
    assert [r["failure"] for r in res.cascade] == ["code", "code"]
    assert state(cascade)["failure"] == "code"


def test_wrong_code_in_the_prepublish_check_is_code(cascade):
    cascade.wrong = {"double 5 m"}
    res = build()
    assert res.status == "mismatch" and res.reason.startswith(labelfree.CODE_WRONG_REASON)
    assert res.failure == "code"


def test_convention_mismatch_is_code(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_CONVENTION_CHECK", True)
    monkeypatch.setattr(labelfree, "_convention_check", lambda c, e, r: {"passed": False, "detail": "differ"})
    res = build()
    assert res.reason.startswith(labelfree.CONVENTION_REASON) and res.failure == "code"


def test_unusable_conventions_output_is_code(cascade, monkeypatch):
    def boom(cache, eid, t, model=None):
        raise ValueError("truncated JSON")
    monkeypatch.setattr(labelfree, "write_outputs", boom)
    res = build()
    assert res.reason.startswith(labelfree.OUTPUTS_REASON) and res.failure == "code"


def test_last_attempt_decides(cascade, monkeypatch):
    # Haiku's code fails, Sonnet hits an API error: the result is Sonnet's "api"
    err = llm.LLMError("builder call failed (529): overloaded")
    FakeBuilder.fail = {HAIKU}
    real_build = FakeBuilder.build

    def build_(self):
        if self.model == SONNET:
            FakeBuilder.models.append(self.model)
            raise err
        return real_build(self)
    monkeypatch.setattr(FakeBuilder, "build", build_)
    res = build()
    assert [r["failure"] for r in res.cascade] == ["code", "api"] and res.failure == "api"


def test_cap_after_a_code_failure_keeps_code(cascade, monkeypatch):
    # a real code failure happened; the cap only stopped the rebuild, so it still counts
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.05)
    FakeBuilder.cost = {HAIKU: 0.05, SONNET: 0.03}
    FakeBuilder.fail = {HAIKU}
    res = build()
    assert res.cascade[1]["status"] == "skipped" and res.failure == "code"


# ----------------------------------------------------------------------------- "api"

def test_api_error_in_the_build_is_api(cascade):
    FakeBuilder.fail = llm.LLMError("builder call failed (400): credit balance is too low")
    res = build()
    assert res.status == "rejected" and res.failure == "api"


def test_api_error_in_the_resolve_is_api(cascade):
    cascade.error = llm.LLMError("resolve call failed (500)")
    res = build()
    assert res.status == "rejected" and res.failure == "api"


def test_api_error_writing_the_output_specs_is_api(cascade, monkeypatch):
    def down(cache, eid, t, model=None):
        raise llm.LLMError("outputs call failed (503)")
    monkeypatch.setattr(labelfree, "write_outputs", down)
    assert build().failure == "api"


def test_gate_outage_that_costs_the_fit_is_api(cascade):
    cascade.gate_error = {f"double {i} m" for i in range(1, 8)}
    res = build()
    assert res.status == "mismatch" and res.failure == "api"


# ----------------------------------------------------------------------------- "cost_cap"

def test_cap_before_any_resolve_is_cost_cap(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.0)
    res = build()
    assert res.status == "rejected" and res.failure == "cost_cap" and FakeBuilder.models == []


def test_cap_during_the_prepublish_check_is_cost_cap(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.0122)
    res = build()
    assert "cost cap" in res.reason and res.failure == "cost_cap"


def test_cap_before_the_first_builder_is_cost_cap(cascade, monkeypatch):
    monkeypatch.setattr(config, "LABELFREE_MAX_COST", 0.01)     # the re-solve alone uses it up
    res = build()
    assert FakeBuilder.models == [] and res.cascade[0]["status"] == "skipped" and res.failure == "cost_cap"


@pytest.mark.parametrize("e,want", [(llm.LLMError("x"), "api"), (llm.OutputError("x"), "code"),
                                    (ValueError("x"), "code")])
def test_failure_of(e, want):
    assert labelfree._failure_of(e) == want


def test_extraction_refusals_that_cost_the_fit_are_not_a_failure(cascade):
    # an extraction refusal is a safe decline (the question goes to the model), not a code fault
    cascade.decline = {f"double {i} m": "extraction failed its checks: no value extracted" for i in range(3, 7)}
    res = build()
    assert res.status == "mismatch" and res.reason.startswith(labelfree.MISFIT_REASON)
    assert res.failure is None and state(cascade)["failure"] is None
