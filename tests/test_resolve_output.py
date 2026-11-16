"""Re-solve robustness (peer report 2026-11-16, fam_f0003 mortgage run):

1. The re-solve thought for 7-16k tokens at "high" effort and once hit max_tokens=16000. It now
   runs at config.RESOLVE_EFFORT (medium) with config.RESOLVE_MAX_TOKENS (32000).
2. A structured reply with a missing or mistyped field raised a bare pydantic ValidationError out
   of the SDK's parse(), uncharged and unlogged, and ended the build ("build error:
   ValidationError ... for ResolveOut"). llm.call now validates the reply itself: an unusable
   reply is charged, logged and raised as OutputError, and the simple flow counts it as a failed
   re-solve try, so the next try runs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pydantic
import pytest

from arc import config, labelfree, llm
from test_labelfree_simple import ENTRY, FAM, FakeBuilder, rec, sim, state  # noqa: F401  (sim: fixture)

REAL_RESOLVE = labelfree._resolve          # taken before the `sim` fixture replaces it
HAIKU = "claude-haiku-5-5"
GOOD = json.dumps({"working": "x times two", "program": "print('FINAL ANSWER')\nprint('result: 2 m')",
                   "final_answer": "2 m"})
MISSING = json.dumps({"working": "x times two", "program": "print('result: 2 m')"})   # no final_answer
MISTYPED = json.dumps({"working": "x times two", "program": ["print(2)"], "final_answer": "2 m"})


def _response(text, stop="end_turn", out_tokens=1000):
    usage = SimpleNamespace(input_tokens=500, output_tokens=out_tokens, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)
    return SimpleNamespace(stop_reason=stop, usage=usage, model=HAIKU, stop_details=None,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)])


@pytest.fixture
def api(monkeypatch):
    """A fake Anthropic client. Queue reply texts in api.replies; create() kwargs land in
    api.calls. parse() behaves like the SDK's (validates inside the call and raises a bare
    ValidationError), so code that still used it would fail these tests."""
    s = SimpleNamespace(replies=[], calls=[], logs=[])

    def create(**kw):
        s.calls.append(kw)
        text, stop = s.replies.pop(0)
        return _response(text, stop)

    def parse(output_format=None, **kw):
        r = create(**kw)
        pydantic.TypeAdapter(output_format).validate_json(r.content[-1].text)
        return r
    msgs = SimpleNamespace(create=create, parse=parse)
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)))
    monkeypatch.setattr(llm, "_log", s.logs.append)
    monkeypatch.setattr(llm, "LEDGER", llm.Ledger())
    s.queue = lambda text, stop="end_turn": s.replies.append((text, stop))
    return s


# ----------------------------------------------------------------------------- llm.call

def test_valid_structured_reply_is_parsed_and_charged(api):
    api.queue(GOOD)
    out = llm.call("resolve", "q", model=HAIKU, schema=labelfree.ResolveOut, effort="medium", tag="t")
    assert isinstance(out.parsed, labelfree.ResolveOut) and out.parsed.final_answer == "2 m"
    [kw] = api.calls
    assert kw["output_config"]["effort"] == "medium"                    # effort kept beside the format
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and set(fmt["schema"]["required"]) == {"working", "program", "final_answer"}
    assert len(llm.LEDGER.records) == 1 and llm.LEDGER.cost() == pytest.approx(out.cost) and out.cost > 0


@pytest.mark.parametrize("text,field", [(MISSING, "final_answer"), (MISTYPED, "program"),
                                        ('{"working": "x", "progr', "(root)")])
def test_unparseable_reply_is_an_output_error_that_is_charged_and_logged(api, text, field):
    api.queue(text)
    with pytest.raises(llm.OutputError, match="no parseable output") as e:
        llm.call("resolve", "q", model=HAIKU, schema=labelfree.ResolveOut, tag="t")
    assert field in str(e.value)                                        # the reason names the field
    assert not isinstance(e.value, pydantic.ValidationError)
    assert len(llm.LEDGER.records) == 1 and llm.LEDGER.cost() > 0       # the paid reply counts
    [log] = api.logs
    assert "no parseable output" in log["error"] and log["cost"] == pytest.approx(llm.LEDGER.cost())


@pytest.mark.parametrize("stop,match", [("max_tokens", "max_tokens"), ("refusal", "refused")])
def test_cut_off_or_refused_replies_are_charged(api, stop, match):
    api.queue('{"working": "x", "pro', stop)
    with pytest.raises(llm.OutputError, match=match):
        llm.call("resolve", "q", model=HAIKU, schema=labelfree.ResolveOut, tag="t")
    assert len(llm.LEDGER.records) == 1 and llm.LEDGER.cost() > 0


def test_plain_text_calls_are_unchanged(api):
    api.queue("just text")
    out = llm.call("source", "q", model=HAIKU, effort="high")
    assert out.text == "just text" and out.parsed is None
    assert api.calls[0]["output_config"] == {"effort": "high"}           # no format without a schema


# ----------------------------------------------------------------------------- resolve settings

def test_resolve_defaults_are_medium_effort_and_32k_tokens():
    assert config.RESOLVE_EFFORT == "medium" and config.RESOLVE_MAX_TOKENS == 32000
    assert config.SOURCE_EFFORT == "high"                               # other roles keep theirs


def test_resolve_call_uses_the_resolve_settings(fake_llm, monkeypatch):
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('result: 2 m')", final_answer="2 m"))
    labelfree._resolve("double 1 m", "cust-model", "t")
    [c] = fake_llm.calls
    assert c["effort"] == "medium" and c["max_tokens"] == 32000
    monkeypatch.setattr(config, "RESOLVE_EFFORT", "low")
    monkeypatch.setattr(config, "RESOLVE_MAX_TOKENS", 24000)
    fake_llm.queue(labelfree.ResolveOut(working="w", program="print('result: 2 m')", final_answer="2 m"))
    labelfree._resolve("double 1 m", "cust-model", "t")
    assert fake_llm.calls[1]["effort"] == "low" and fake_llm.calls[1]["max_tokens"] == 24000


# ----------------------------------------------------------------------------- the build retries

def test_unusable_resolve_reply_counts_as_a_try_and_the_next_one_runs(sim):
    replies = [llm.OutputError("resolve call returned no parseable output (final_answer: Field required)"), "2 m"]
    real = labelfree._resolve

    def resolve(question, model, tag, unit_hint=None):
        r = replies.pop(0)
        if isinstance(r, Exception):
            sim.calls.append((question, model))
            raise r
        sim.answers = [r]
        return real(question, model, tag, unit_hint)
    labelfree._resolve = resolve
    try:
        res = labelfree.build_family(FAM, [rec(0)])
    finally:
        labelfree._resolve = real
    assert res.status == "accepted" and len(sim.calls) == 2
    a1, a2 = state(sim)["attempts"]
    assert "Field required" in a1["output_error"] and not a1["executed"] and a2["agree"]


def test_every_resolve_reply_unusable_is_rejected_after_all_tries(sim):
    sim.error = llm.OutputError("resolve call hit max_tokens=32000")
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "rejected" and res.failure == "output" and len(sim.calls) == config.LABELFREE_RESOLVE_TRIES
    assert res.reason.startswith(labelfree.RESOLVE_OUTPUT_REASON) and "max_tokens=32000" in res.reason
    assert FakeBuilder.built == [] and not (sim.path / "caches" / ENTRY).exists()
    assert res.cost == pytest.approx(0.02)                              # both paid tries are counted


def test_unusable_reply_then_crashed_program_is_a_program_failure(sim):
    errors = [llm.OutputError("resolve call was refused by the model")]
    real = labelfree._resolve

    def resolve(question, model, tag, unit_hint=None):
        if errors:
            sim.calls.append((question, model))
            raise errors.pop(0)
        sim.answers = [None]                                            # the program crashes
        return real(question, model, tag, unit_hint)
    labelfree._resolve = resolve
    try:
        res = labelfree.build_family(FAM, [rec(0)])
    finally:
        labelfree._resolve = real
    assert res.status == "mismatch" and res.reason.startswith(labelfree.PROGRAM_FAILED_REASON)
    assert "refused" in res.reason and "ZeroDivisionError" in res.reason


def test_api_errors_still_reject_without_a_retry(sim):
    sim.error = llm.LLMError("resolve call failed (529): overloaded")
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "rejected" and res.failure == "api" and len(sim.calls) == 1


def test_regression_validation_error_from_the_model_no_longer_ends_the_build(sim, api, monkeypatch):
    """fam_f0003, 2026-11-16: the first re-solve's structured reply lacked a field. Through the
    real _resolve and llm.call, that try now fails cleanly and the second try builds."""
    monkeypatch.setattr(labelfree, "_resolve", REAL_RESOLVE)
    api.queue(MISSING)
    api.queue(GOOD)
    res = labelfree.build_family(FAM, [rec(0)])
    assert res.status == "accepted" and len(api.calls) == 2
    assert all(c["max_tokens"] == 32000 and c["output_config"]["effort"] == "medium" for c in api.calls)
    a1, a2 = state(sim)["attempts"]
    assert "final_answer" in a1["output_error"] and a2["agree"] and a2["final_answer"] == "result: 2 m"
    spent = llm.LEDGER.cost(roles={"resolve"})
    assert spent > 0 and res.parts["resolve"] == pytest.approx(spent)    # the failed try is charged too
