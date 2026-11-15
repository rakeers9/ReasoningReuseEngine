"""Cache persistence of gate data, and the IS2 extraction prompt."""

from arc import serve
from arc.cache import Cache, ParamSpec

from pydantic import BaseModel


def test_specs_and_cards_round_trip(tmp_path):
    c = Cache("t")
    c.specs = {"p1": {"task": "compute", "deliverable": "x"}}
    c.cards = {"p1": {"target": "x"}}
    c.save(tmp_path)
    loaded = Cache.load(tmp_path)
    assert loaded.specs == c.specs and loaded.cards == c.cards


def test_cache_without_specs_writes_no_file(tmp_path):
    Cache("t").save(tmp_path)
    assert not (tmp_path / "specs.json").exists()
    assert Cache.load(tmp_path).specs == {}


def test_merge_keeps_specs():
    a, b = Cache("a"), Cache("b")
    b.specs = {"p2": {"task": "compute"}}
    a.merge(b)
    assert a.specs == {"p2": {"task": "compute"}}


class _Two(BaseModel):
    p0: str
    p1: str


PARAMS = [ParamSpec("amount", ["10.0 mol"]), ParamSpec("volume", ["4.860 dm3"])]
PAPER_SENTENCE = "If a value is implied rather than stated, write it in the style of the format examples."


def test_paper_extraction_prompt_is_unchanged(fake_llm, monkeypatch):
    monkeypatch.setattr(serve.config, "EXTRACT_FORM", "copy")   # the paper's IS2 form
    fake_llm.queue(_Two(p0="1 mol", p1="2 L"))
    out = serve.extract_parameters("q", PARAMS)
    assert PAPER_SENTENCE in fake_llm.calls[0]["prompt"]
    assert out == {"amount": "1 mol", "volume": "2 L"}


def test_spec_extraction_drops_implied_rule(fake_llm, monkeypatch):
    monkeypatch.setattr(serve.config, "EXTRACT_FORM", "copy")   # the paper's IS2 form
    fake_llm.queue(_Two(p0="1 mol", p1="2 L"))
    serve.extract_parameters("q", PARAMS, allow_implied=False)
    prompt = fake_llm.calls[0]["prompt"]
    assert "implied" not in prompt and "Never assume or compute a value" in prompt


# ----------------------------------------------------------------------------- forced entry (label-free)

def _two_entry_cache():
    from arc.cache import EntryPoint, ParamSpec
    c = Cache("t")
    c.add_function("return two", "def semantic_function(ctx):\n    return {'y': 2}\n")
    c.add_function("return three", "def semantic_function(ctx):\n    return {'y': 3}\n")
    two, three = list(c.functions)
    c.entry_points = [EntryPoint("give two", "original", two, [ParamSpec("x")], "A"),
                      EntryPoint("give three", "original", three, [ParamSpec("x")], "B")]
    return c


def test_forced_entry_skips_search_and_threshold(monkeypatch):
    from arc import config, embed, serve
    monkeypatch.setattr(config, "GATE", "off")
    monkeypatch.setattr(config, "INDEX", "paper")
    monkeypatch.setattr(serve, "extract_parameters", lambda *a, **k: {"x": "1"})
    monkeypatch.setattr(embed, "best_match", lambda q, texts: (0, 0.30))   # far below threshold
    c = _two_entry_cache()
    trace = {}
    assert serve.use(c, "something", trace=trace) is None and trace["stage"] == "below_threshold"
    trace = {}
    r = serve.use(c, "something", trace=trace, entry_point=c.entry_points[1])
    assert r.output == {"y": 3} and r.entry_point.problem_id == "B"
    assert trace["forced_entry"] and trace["similarity"] == 0.30 and trace["stage"] == "dispatch"


def test_forced_entry_still_runs_the_gate(monkeypatch):
    from arc import config, embed, serve, verify
    monkeypatch.setattr(config, "GATE", "v2")
    monkeypatch.setattr(embed, "best_match", lambda q, texts: (0, 0.99))
    c = _two_entry_cache()                     # no cards -> v2 turns everything away
    trace = {}
    assert serve.use(c, "something", trace=trace, entry_point=c.entry_points[0]) is None
    assert trace["rejected_by"] == ["no_card"]


def test_entry_similarity_uses_only_that_entrys_texts(monkeypatch):
    from arc import config, embed, serve
    from arc.cache import EntryPoint
    monkeypatch.setattr(config, "INDEX", "paper")
    seen = []
    monkeypatch.setattr(embed, "best_match", lambda q, texts: seen.append(texts) or (0, 0.5))
    c = _two_entry_cache()
    c.entry_points.append(EntryPoint("two please", "family", c.entry_points[0].root_id, [], "A"))
    serve._entry_similarity(c, "q", c.entry_points[0])
    assert seen[-1] == ["give two", "two please"]
    monkeypatch.setattr(config, "INDEX", "cards")
    c.cards["A"] = {"index_texts": ["deux"], "intent": ""}
    serve._entry_similarity(c, "q", c.entry_points[0])
    assert seen[-1] == ["deux"]
