"""The production backlog log (arc/backlog.py)."""

import json

import pytest

from arc import backlog


def test_log_interaction_appends_a_full_record(tmp_path):
    path = tmp_path / "b.jsonl"
    r = backlog.log_interaction("What is 2+2?", "4", "claude-opus-5-5", path=path, input_tokens=10,
                                output_tokens=2, timestamp=5.0, id="r1")
    backlog.log_interaction("Again?", "yes", "claude-opus-5-5", path=path)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 2 and lines[0] == r
    assert set(lines[0]) == set(backlog.FIELDS)
    assert lines[0]["timestamp"] == 5.0 and lines[1]["id"] and lines[1]["id"] != "r1"


def test_session_is_hashed_and_rehashing_is_stable():
    r = backlog.make_record("q", "a", "m", session="user-42@example.com")
    assert r["session"].startswith("h:") and "user-42" not in r["session"]
    assert backlog.make_record("q", "a", "m", session=r["session"])["session"] == r["session"]
    assert backlog.make_record("q", "a", "m", session="user-42@example.com")["session"] == r["session"]


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        backlog.make_record("  ", "a", "m")


def test_read_normalises_and_skips_bad_lines(tmp_path):
    path = tmp_path / "b.jsonl"
    path.write_text('{"question": "q1", "response": "a", "model": "m", "session": "s"}\n'
                    "not json\n\n"
                    '{"question": "", "response": "a"}\n'
                    '{"question": "q2"}\n')
    rows = list(backlog.read(path))
    assert [r["question"] for r in rows] == ["q1", "q2"]
    assert rows[0]["session"].startswith("h:") and rows[1]["model"] == "" and rows[1]["cost"] is None


def test_read_missing_file_yields_nothing(tmp_path):
    assert list(backlog.read(tmp_path / "none.jsonl")) == []
