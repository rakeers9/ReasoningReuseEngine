"""eval/decision: the Laya backend, offline (a stub stands in for the laya package)."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "decision"))

import backends  # noqa: E402
from models import MODELS  # noqa: E402

QS = {"gas": "Does the solver answer this question?", "loan": "Does the other solver answer it?"}


def test_noul_form_is_our_plain_yes_no_question():
    assert backends.laya_questions(QS) == backends.noul_questions(QS)


def test_choice_form_has_two_neutral_options_and_keeps_the_instructions():
    qs = backends.laya_questions(QS, "choice")
    for qid, text in QS.items():
        assert qs[qid]["type"] == "choice" and qs[qid]["instructions"] == text
        assert set(qs[qid]["criteria"]) == {"a", "b"}


def test_unknown_form_is_refused():
    with pytest.raises(ValueError):
        backends.laya_questions(QS, "score")


def test_p_yes_reads_noul_and_choice_answers():
    assert backends.laya_p_yes({"noul": 0.83}) == 0.83
    assert backends.laya_p_yes({"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}, "choice") == 0.2


def test_laya_models_need_allow_remote_code():
    for name in ("laya", "laya-typed", "laya-multilingual"):
        assert MODELS[name]["kind"] == "laya" and MODELS[name]["remote_code"]
        with pytest.raises(SystemExit):
            backends.make_backend(name)


def test_laya_downloads_never_fetch_the_repo_python_files():
    for name in ("laya", "laya-typed", "laya-multilingual"):
        allow = MODELS[name]["allow"]
        assert not any(p.endswith(".py") or p == "*" for p in allow)
        sub = MODELS[name].get("subfolder")
        if sub:   # a sub-checkpoint fetches only its own folder, not its siblings' weights
            assert all(p.startswith(f"{sub}/") for p in allow)
        else:
            assert "model.safetensors" in allow and not any("/" in p and not p.startswith(("tokenizer/", "encoder/"))
                                                            for p in allow)


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}   # one token per word


class _Agent:
    made = []

    def __init__(self, root, subfolder=None, device=None):
        self.root, self.subfolder, self.device = root, subfolder, device
        self.cfg = {"max_len": 512, "head_max_len": 192}
        self.tok = _Tok()
        self.calls = []
        _Agent.made.append(self)

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {q: {"noul": 0.9 if q == "gas" else 0.1} for q in questions},
                "usage": {"input_tokens": 123, "truncated": False}}


@pytest.fixture
def fake_laya(monkeypatch):
    _Agent.made.clear()
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Agent=_Agent))
    import huggingface_hub
    asked = []

    def snap(repo, local_files_only=False, allow_patterns=None):
        asked.append((repo, local_files_only, allow_patterns))
        return "/fake/snap"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
    _Agent.asked = asked
    return _Agent


def test_backend_looks_up_only_its_own_files_offline(fake_laya):
    backends.make_backend("laya-multilingual", allow_remote_code=True)
    assert fake_laya.asked == [("convaiinnovations/laya", True, MODELS["laya-multilingual"]["allow"])]


def test_backend_loads_the_local_snapshot_and_widens_the_budget(fake_laya):
    b = backends.make_backend("laya-typed", allow_remote_code=True)
    a = fake_laya.made[0]
    assert a.root == "/fake/snap" and a.subfolder == "typed-decisions"
    assert a.cfg["max_len"] == MODELS["laya-typed"]["max_len"]
    assert a.cfg["head_max_len"] == MODELS["laya-typed"]["head_max_len"] == b.head


def test_backend_returns_p_yes_per_question_and_usage(fake_laya):
    b = backends.make_backend("laya", allow_remote_code=True)
    probs, usage = b.decide("What is the pressure of 2 mol in 3 L at 300 K?", QS)
    assert probs == {"gas": 0.9, "loan": 0.1}
    assert usage["input_tokens"] == 123 and usage["state_truncated"] is False
    assert usage["instruction_cut"] == []
    state, sent = fake_laya.made[0].calls[0]
    assert sent == backends.noul_questions(QS)


def test_backend_reports_an_instruction_too_long_for_the_head(fake_laya):
    b = backends.make_backend("laya", allow_remote_code=True)
    long_q = {"gas": "word " * (b.head + 10), "loan": "short"}
    _, usage = b.decide("state", long_q)
    assert usage["instruction_cut"] == ["gas"]


def test_backend_not_downloaded_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Agent=_Agent))
    import huggingface_hub

    def missing(repo, local_files_only=False, allow_patterns=None):
        raise FileNotFoundError(repo)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(SystemExit, match="not downloaded"):
        backends.make_backend("laya", allow_remote_code=True)


def test_adding_laya_leaves_the_other_backends_alone():
    assert MODELS["clef"]["kind"] == "cloudflare" and "laya" not in backends.RAW_SEED
