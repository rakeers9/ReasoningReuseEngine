"""The Clef gate (arc/clef.py, ARC_GATE=clef) and the Clef call (llm.decide). Stubs only."""

import ast
from pathlib import Path

import pytest

from arc import clef, config, embed, llm, serve
from arc.cache import Cache, EntryPoint, ParamSpec

ROOT = Path(__file__).resolve().parents[1]
GAS = "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure from the perfect gas equation."
LOAN = "A $350,000 mortgage at 6.25% over 30 years: what is the monthly payment?"
SPEC = {"task": "compute", "deliverable": "pressure of a gas", "method": "perfect gas equation",
        "conditions": ["a single gas"], "inputs": [{"name": "amount of gas", "value": "10.0 mol", "evidence": "10.0 mol"},
                                                   {"name": "volume", "value": "4.860 dm3", "evidence": "4.860 dm3"}],
        "output_form": "a number with a pressure unit"}


@pytest.fixture(autouse=True)
def clef_gate(monkeypatch):
    monkeypatch.setattr(config, "GATE", "clef")
    monkeypatch.setattr(config, "CLEF_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "CLEF_TOP_K", 3)
    monkeypatch.setattr(config, "CLEF_PROMPT", "v1")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    # fake meaning: texts share a direction when they share their first word
    monkeypatch.setattr(embed, "ranked", lambda q, cands: sorted(
        ((i, 0.9 if c.split()[0] == q.split()[0] else 0.3) for i, c in enumerate(cands)), key=lambda x: -x[1]))
    monkeypatch.setattr(embed, "best_match", lambda q, cands: (0, 0.9 if any(
        c.split()[0] == q.split()[0] for c in cands) else 0.3))


def cache(with_specs=True):
    c = Cache("t")
    c.entry_points += [EntryPoint(GAS, "original", "gas_root", [ParamSpec("amount", ["10.0 mol"])], "gas"),
                       EntryPoint("Suppose that ●1 ...", "template", "gas_root", [], "gas"),
                       EntryPoint(LOAN, "original", "loan_root", [ParamSpec("principal", ["$350,000"])], "loan")]
    if with_specs:
        c.specs = {"gas": SPEC, "loan": {**SPEC, "deliverable": "monthly loan payment"}}
    return c


def stub_decide(monkeypatch, probs, calls=None):
    def fake(state, questions, *, model=None, tag=None):
        if calls is not None:
            calls.append((state, questions))
        return {k: probs.get(k, probs.get(k.split("#")[0], 0.0)) for k in questions}   # keys "<entry>#<seed>"
    monkeypatch.setattr(llm, "decide", fake)


def stub_run(monkeypatch, fake_llm=None):
    monkeypatch.setattr(serve, "extract_parameters", lambda req, params, tag=None, **k: {"amount": "2 mol"})

    class Ctx:
        lookups = []

        def narration(self):
            return "ran"
    monkeypatch.setattr(serve, "dispatch_root", lambda c, root, values: ({"pressure": f"{root} ok"}, Ctx()))


def stub_spec(monkeypatch, fake_llm):
    from arc import spec
    monkeypatch.setattr(spec, "write_spec", lambda text, tag=None: {**SPEC, "deliverable": text.split()[0]})


# ---- prompts ----

def test_v1_prompt_is_the_measured_eval_prompt_byte_for_byte():
    src = (ROOT / "eval/decision/run_decision.py").read_text()
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "PROMPT")
    assert clef.PROMPT_V1 == ast.literal_eval(node.value)


def test_v1_prompt_quotes_the_entrys_original_question_not_the_template():
    c = cache()
    template = next(e for e in c.entry_points if e.kind == "template")
    assert f"<<<{GAS}>>>" in clef.prompt_for(c, template, "v1")


def test_v2_prompt_adds_the_entry_profile_and_near_miss_rules():
    text = clef.prompt_for(cache(), cache().entry_points[0], "v2")
    assert f"<<<{GAS}>>>" in text and "What it computes: pressure of a gas" in text
    assert "Inputs it needs: amount of gas; volume" in text and "working backwards" in text
    no_spec = clef.prompt_for(cache(with_specs=False), cache().entry_points[0], "v2")
    assert "What it computes" not in no_spec and "working backwards" in no_spec


def test_v3_is_v2_plus_the_careful_reading_rule():
    v2 = clef.prompt_for(cache(), cache().entry_points[0], "v2")
    v3 = clef.prompt_for(cache(), cache().entry_points[0], "v3")
    assert v3 == v2 + clef.CAREFUL_READING
    assert "Read every word" in v3 and "less than or equal to" in v3 and "answer no" in v3
    assert "{" not in clef.CAREFUL_READING        # no stray format fields


def test_default_is_v6_and_v2_text_is_unchanged():
    import hashlib
    src = (ROOT / "arc/config.py").read_text()      # the fixture above pins v1, so read the source
    assert 'CLEF_PROMPT = os.environ.get("ARC_CLEF_PROMPT", "v6")' in src   # best on dev, 2026-11-15
    assert hashlib.sha256(clef.PROMPT_V2.encode()).hexdigest() == "465b86c6f2665903d69da874f6d4122925b918add0e09c1b5e1f893061cf358e"


def test_unknown_prompt_version_fails_loudly():
    with pytest.raises(ValueError):
        clef.prompt_for(cache(), cache().entry_points[0], "v9")


# ---- candidates ----

def test_candidates_use_structure_and_one_entry_per_problem(monkeypatch, fake_llm):
    monkeypatch.setattr(config, "CLEF_RETRIEVAL", "structure")
    stub_spec(monkeypatch, fake_llm)
    trace = {}
    cands = clef.candidates(cache(), "Suppose 3 mol of argon ...", trace=trace)
    assert [e.problem_id for e, _ in cands] == ["gas", "loan"]          # no duplicate for the template
    assert all(c["by"] == "structure" for c in trace["candidates"]) and "spec" in trace


def test_candidates_without_structures_rank_raw_text_and_skip_the_haiku_call(monkeypatch, fake_llm):
    from arc import spec
    monkeypatch.setattr(spec, "write_spec", lambda *a, **k: pytest.fail("no structure call needed"))
    cands = clef.candidates(cache(with_specs=False), "A $200,000 loan ...")
    assert cands[0][0].problem_id == "loan"


def test_candidates_capped_at_top_k(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    monkeypatch.setattr(config, "CLEF_TOP_K", 1)
    assert len(clef.candidates(cache(), "Suppose ...")) == 1


def test_empty_cache_has_no_candidates():
    assert clef.candidates(Cache("empty"), "anything") == []


# ---- serving through ARC_GATE=clef ----

def test_serves_the_best_candidate_clef_accepts(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    stub_run(monkeypatch)
    calls = []
    stub_decide(monkeypatch, {"gas": 0.93, "loan": 0.85}, calls)
    trace = {}
    r = serve.use(cache(), "Suppose that 2 mol N2 ...", trace=trace)
    assert r is not None and r.entry_point.problem_id == "gas" and r.output == {"pressure": "gas_root ok"}
    assert trace["stage"] == "dispatch" and trace["clef"] == {"gas": 0.93, "loan": 0.85}
    state, questions = calls[0]
    assert state == "Suppose that 2 mol N2 ..." and set(questions) == {"gas#0", "loan#0"}   # raw question, one call


def test_below_threshold_is_turned_away_with_a_reason(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    stub_run(monkeypatch)
    stub_decide(monkeypatch, {"gas": 0.79})
    trace = {}
    assert serve.use(cache(), "Suppose a mixture ...", trace=trace) is None
    assert trace["stage"] == "gate_rejected" and trace["rejected_by"] == ["clef"] and trace["clef_p"] == 0.79


def test_threshold_is_inclusive(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    stub_run(monkeypatch)
    stub_decide(monkeypatch, {"gas": 0.8})
    assert serve.use(cache(), "Suppose ...") is not None


def test_clef_error_never_serves(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    stub_run(monkeypatch)

    def broken(*a, **k):
        raise llm.LLMError("Cloudflare HTTP 503")
    monkeypatch.setattr(llm, "decide", broken)
    trace = {}
    assert serve.use(cache(), "Suppose ...", trace=trace) is None
    assert trace["rejected_by"] == ["clef_error"] and "503" in trace["clef_error"]


def test_forced_entry_asks_clef_about_that_entry_only(monkeypatch, fake_llm):
    from arc import spec
    monkeypatch.setattr(spec, "write_spec", lambda *a, **k: pytest.fail("forced entry needs no retrieval"))
    stub_run(monkeypatch)
    calls = []
    stub_decide(monkeypatch, {"loan": 0.9}, calls)
    c = cache()
    trace = {}
    r = serve.use(c, "Suppose ...", trace=trace, entry_point=c.entry_points[2])
    assert r.entry_point.problem_id == "loan" and list(calls[0][1]) == ["loan#0"] and trace["forced_entry"]


def test_shadow_runs_the_code_but_still_turns_away(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    stub_run(monkeypatch)
    stub_decide(monkeypatch, {"gas": 0.2})
    trace = {}
    assert serve.use(cache(), "Suppose ...", trace=trace, shadow=True) is None
    assert trace["shadow_output"] == "pressure: gas_root ok"


def test_cached_code_error_is_a_cache_invocation_error(monkeypatch, fake_llm):
    stub_spec(monkeypatch, fake_llm)
    stub_decide(monkeypatch, {"gas": 0.95})
    monkeypatch.setattr(serve, "extract_parameters", lambda *a, **k: {})

    def boom(*a):
        raise ZeroDivisionError("x")
    monkeypatch.setattr(serve, "dispatch_root", boom)
    with pytest.raises(serve.CacheInvocationError):
        serve.use(cache(), "Suppose ...")


def test_other_gates_never_call_clef(monkeypatch, fake_llm):
    monkeypatch.setattr(config, "GATE", "off")
    monkeypatch.setattr(llm, "decide", lambda *a, **k: pytest.fail("clef called with ARC_GATE=off"))
    stub_run(monkeypatch)
    serve.use(cache(), "Suppose ...")


# ---- the Clef call itself ----

def test_decide_reads_answers_and_charges_input_tokens(monkeypatch):
    monkeypatch.setattr(llm, "LEDGER", llm.Ledger())
    seen = {}

    def post(url, body, headers, **k):
        seen.update(url=url, body=body, headers=headers)
        return {"success": True, "result": {"answers": {"gas": {"noul": 0.91}, "loan": 0.1},
                                            "usage": {"input_tokens": 1000}}}
    monkeypatch.setattr(llm, "_cloudflare_post", post)
    monkeypatch.setattr(llm, "_log", lambda e: None)
    probs = llm.decide("question", {"gas": "is it gas?", "loan": "is it a loan?"}, tag="t")
    assert probs == {"gas": 0.91, "loan": 0.1}
    assert seen["url"].endswith("/accounts/acct/ai/run/@cf/cloudflare/clef")
    assert seen["body"]["questions"]["gas"] == {"type": "noul", "instructions": "is it gas?"}
    [rec] = llm.LEDGER.records
    assert rec.role == "decider" and rec.cost == pytest.approx(1000 * 0.24 / 1e6)


def test_decide_without_credentials_fails(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    with pytest.raises(llm.LLMError, match="CLOUDFLARE"):
        llm.decide("q", {"a": "b"})


def test_decide_on_a_bad_response_raises_and_logs(monkeypatch):
    logged = []
    monkeypatch.setattr(llm, "_cloudflare_post", lambda *a, **k: {"success": False, "result": {}, "errors": ["x"]})
    monkeypatch.setattr(llm, "_log", logged.append)
    with pytest.raises(llm.LLMError):
        llm.decide("q", {"a": "b"})
    assert logged and "error" in logged[0]


def test_tests_cannot_reach_cloudflare():
    from conftest import RealAPICallBlocked
    with pytest.raises(RealAPICallBlocked):
        llm._cloudflare_post("https://example", {}, {})


def test_raw_retrieval_is_the_default_and_makes_no_model_call(monkeypatch):
    from arc import spec
    monkeypatch.setattr(config, "CLEF_RETRIEVAL", "raw")
    monkeypatch.setattr(spec, "write_spec", lambda *a, **k: pytest.fail("raw retrieval writes no structure"))
    trace = {}
    cands = clef.candidates(cache(), "A $200,000 loan ...", trace=trace)
    assert [e.problem_id for e, _ in cands][0] == "loan" and len(cands) == 2      # one per entry
    assert all(c["by"] == "raw" for c in trace["candidates"]) and "spec" not in trace


def test_raw_retrieval_default_in_config():
    src = (ROOT / "arc/config.py").read_text()
    assert 'CLEF_RETRIEVAL = os.environ.get("ARC_CLEF_RETRIEVAL", "raw")' in src


def test_serving_with_raw_retrieval_still_uses_the_entry_profile_in_the_prompt(monkeypatch, fake_llm):
    from arc import spec
    monkeypatch.setattr(config, "CLEF_RETRIEVAL", "raw")
    monkeypatch.setattr(config, "CLEF_PROMPT", "v3")
    monkeypatch.setattr(spec, "write_spec", lambda *a, **k: pytest.fail("no per-question structure"))
    stub_run(monkeypatch)
    calls = []
    stub_decide(monkeypatch, {"gas": 0.95}, calls)
    assert serve.use(cache(), "Suppose that 2 mol N2 ...") is not None
    assert "What it computes: pressure of a gas" in calls[0][1]["gas#0"]   # built-once entry structure


# ---- several seed questions per entry ----

def _with_family(c, texts, pid="gas"):
    root = next(e.root_id for e in c.entry_points if e.problem_id == pid)
    c.entry_points += [EntryPoint(t, "family", root, [], pid) for t in texts]
    return c


def test_seeds_are_the_original_then_the_latest_family_questions(monkeypatch):
    monkeypatch.setattr(config, "CLEF_SEEDS", 3)
    c = _with_family(cache(), ["fam one", "fam two", "fam three"])
    gas = c.entry_points[0]
    assert clef.seeds_for(c, gas) == [GAS, "fam three", "fam two"]
    assert clef.seeds_for(c, gas, exclude="FAM  three") == [GAS, "fam two", "fam one"]   # never itself
    monkeypatch.setattr(config, "CLEF_SEEDS", 1)
    assert clef.seeds_for(c, gas) == [GAS]


def test_an_entry_scores_its_best_seed_in_one_call(monkeypatch, fake_llm):
    stub_run(monkeypatch)
    monkeypatch.setattr(config, "CLEF_SEEDS", 3)
    c = _with_family(cache(), ["A flask holds ... what pressure?"])
    calls = []
    stub_decide(monkeypatch, {"gas#0": 0.4, "gas#1": 0.9, "loan#0": 0.2}, calls)
    trace = {}
    r = serve.use(c, "A 2 L flask holds 1 mol ... what pressure?", trace=trace)
    assert r is not None and r.entry_point.problem_id == "gas" and trace["clef"]["gas"] == 0.9
    assert len(calls) == 1 and set(calls[0][1]) == {"gas#0", "gas#1", "loan#0"}
    assert trace["clef_seeds"]["gas#1"]["seed"].startswith("A flask holds")


def test_a_pre_publish_check_never_uses_the_checked_question_as_a_seed(monkeypatch, fake_llm):
    stub_run(monkeypatch)
    q = "A 2 L flask holds 1 mol ... what pressure?"
    c = _with_family(cache(), [q])
    calls = []
    stub_decide(monkeypatch, {"gas#0": 0.95}, calls)
    serve.use(c, q, entry_point=c.entry_points[0])
    assert all(q not in prompt for prompt in calls[0][1].values())        # only the original was used


def test_no_independent_seed_means_no_serve(monkeypatch, fake_llm):
    stub_run(monkeypatch)
    from arc import llm
    monkeypatch.setattr(llm, "decide", lambda *a, **k: pytest.fail("nothing to compare with"))
    c = Cache("t")
    c.entry_points.append(EntryPoint(GAS, "original", "gas_root", [], "gas"))
    assert serve.use(c, GAS, entry_point=c.entry_points[0]) is None      # the only seed is itself


def test_default_seed_count():
    assert 'CLEF_SEEDS = int(os.environ.get("ARC_CLEF_SEEDS", 1))' in (ROOT / "arc/config.py").read_text()   # extra seeds hurt on dev


def test_v4_keeps_every_no_rule_and_lists_what_does_not_change_the_task():
    c = cache()
    v4 = clef.prompt_for(c, c.entry_points[0], "v4")
    assert f"<<<{GAS}>>>" in v4 and "What it computes: pressure of a gas" in v4
    for must in ("Judge the task, not the wording", '"from 1 to 999" is the same set as "less than 1000"',
                 '"multiples of" means "divisible by"', "units of the same quantity",
                 "working backwards", "different method or assumption", "not stated in the question",
                 "second object or mixture", '"less than or equal to 1000"', "Read every word"):
        assert must in v4, must
    assert "{" not in clef.PROMPT_V4.replace("{seed}", "").replace("{profile}", "")


def test_v4_family_prompt_has_no_profile(monkeypatch):
    monkeypatch.setattr(config, "CLEF_PROMPT", "v4")
    assert "What it computes" not in clef._PROMPTS["v4"].format(seed="x", profile="")


def test_v5_is_v4_with_the_unit_conversion_rule():
    v5 = clef.PROMPT_V5
    assert "converts it itself" in v5 and "umol/L or mg/dL" in v5 and '"$1.2M"' in v5
    assert v5.replace(clef.PROMPT_V5.split("- different numbers")[1].split("- a different style")[0], "") \
        == clef.PROMPT_V4.replace(clef.PROMPT_V4.split("- different numbers")[1].split("- a different style")[0], "")


def test_v6_adds_the_labelled_seats_example_and_the_limit_rule():
    v6 = clef.PROMPT_V6
    assert "numbered seats around a table make every rotation a different seating" in v6
    assert "6. The values given push the problem past a limit" in v6
    assert v6.startswith(clef.PROMPT_V5.split("5. A small wording change")[0])


def test_family_checks_use_their_own_measured_prompt(monkeypatch):
    from arc import llm
    seen = {}
    monkeypatch.setattr(llm, "decide", lambda state, qs, **k: seen.update(qs) or {q: 0.9 for q in qs})
    monkeypatch.setattr(config, "CLEF_PROMPT", "v6")
    monkeypatch.setattr(config, "FAMILY_CLEF_PROMPT", "v3")
    clef.fits("q", {"f#0": "anchor"})
    assert seen["f#0"] == clef.PROMPT_V3.format(seed="anchor", profile="")
    assert 'FAMILY_CLEF_PROMPT = os.environ.get("ARC_FAMILY_CLEF_PROMPT", "v3")' in (ROOT / "arc/config.py").read_text()


def test_served_answers_go_through_the_formatter(monkeypatch, fake_llm):
    stub_decide(monkeypatch, {"loan": 0.95})

    def extract(req, params, tag=None, form_out=None, **k):
        if form_out is not None:
            form_out["output_format"] = {"unit": "", "rounding": "", "evidence": ""}
        return {"principal": "$200,000"}
    monkeypatch.setattr(serve, "extract_parameters", extract)

    class Ctx:
        lookups = []

        def narration(self):
            return ""
    monkeypatch.setattr(serve, "dispatch_root", lambda c, root, values: ({"monthly payment": 1199.101006}, Ctx()))
    c = cache()
    c.outputs = {"loan": {"monthly payment": {"kind": "money", "unit": "$"}}}
    trace = {}
    r = serve.use(c, "A $200,000 loan ...", trace=trace)
    assert r.output == {"monthly payment": "1199.10"} and trace["presented"] == r.output


def test_an_answer_that_cannot_be_presented_falls_back(monkeypatch, fake_llm):
    stub_decide(monkeypatch, {"loan": 0.95})

    def extract(req, params, tag=None, form_out=None, **k):
        form_out["output_format"] = {"unit": "yen", "rounding": "", "evidence": "in yen"}
        return {"principal": "$200,000"}
    monkeypatch.setattr(serve, "extract_parameters", extract)

    class Ctx:
        lookups = []

        def narration(self):
            return ""
    monkeypatch.setattr(serve, "dispatch_root", lambda c, root, values: ({"monthly payment": 1199.1}, Ctx()))
    with pytest.raises(serve.CacheInvocationError):          # chat.py turns this into a model answer
        serve.use(cache(), "A $200,000 loan ... in yen?")
