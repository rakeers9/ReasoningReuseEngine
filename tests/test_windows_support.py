"""Windows support, tested on any OS by forcing the Windows code paths.

- arc/sandbox.time_limit: Windows has no SIGALRM, so a timer thread interrupts the main thread.
- web/server.py: the venv's Python lives in .venv/Scripts on Windows, and a run is stopped
  with Ctrl+Break (Windows can't send SIGINT to another process).
- arc/__main__.py: Ctrl+Break raises KeyboardInterrupt, like Ctrl+C.
"""

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from arc import sandbox

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_alarm(monkeypatch):
    monkeypatch.setattr(sandbox, "_HAS_ALARM", False)


def busy(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def test_timer_fallback_stops_runaway_code(no_alarm):
    started = time.monotonic()
    with pytest.raises(sandbox.FunctionTimeout):
        with sandbox.time_limit(0.2):
            busy(5)
    assert time.monotonic() - started < 2
    assert sandbox._timer_active is False


def test_timer_fallback_leaves_no_interrupt_behind(no_alarm):
    with sandbox.time_limit(0.2):
        busy(0.01)
    busy(0.4)   # well past the deadline: a leftover timer would raise KeyboardInterrupt here


def test_timer_fallback_keeps_the_outer_deadline(no_alarm):
    with pytest.raises(sandbox.FunctionTimeout):
        with sandbox.time_limit(0.2):
            with sandbox.time_limit(30):   # inner limit must not replace the outer one
                busy(5)


def test_timer_fallback_passes_a_real_ctrl_c_on(no_alarm):
    with pytest.raises(KeyboardInterrupt):
        with sandbox.time_limit(30):
            raise KeyboardInterrupt


def test_timer_fallback_lets_a_real_ctrl_c_signal_through(no_alarm):
    import _thread
    with pytest.raises(KeyboardInterrupt):
        with sandbox.time_limit(30):
            _thread.interrupt_main()   # Ctrl+C from the user, not the timer
            busy(1)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler   # handler restored


def test_timer_fallback_never_leaks_an_interrupt_near_the_deadline(no_alarm):
    # Blocks that end just before, at or after the deadline: each must either finish or time
    # out, and no interrupt may surface after the block (that would read as the user's Ctrl+C).
    outcomes = {"finished": 0, "timeout": 0}
    for i in range(300):
        try:
            with sandbox.time_limit(0.01):
                busy(0.005 + (i % 30) * 0.0005)
            busy(0.015)
            outcomes["finished"] += 1
        except sandbox.FunctionTimeout:
            outcomes["timeout"] += 1
    assert outcomes["finished"] and outcomes["timeout"]


def test_alarm_path_is_unchanged_where_sigalrm_exists():
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("no SIGALRM on this OS")
    with pytest.raises(sandbox.FunctionTimeout):
        with sandbox.time_limit(0.2):
            busy(5)


# ---------------------------------------------------------------------------- web/server.py

@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("web_server_under_test", ROOT / "web" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_venv_python_on_each_os(server, tmp_path):
    assert server.venv_python(tmp_path) == sys.executable              # no venv: the server's own Python
    posix = tmp_path / ".venv" / "bin" / "python"
    posix.parent.mkdir(parents=True)
    posix.write_text("")
    assert server.venv_python(tmp_path) == str(posix)
    win = tmp_path / ".venv" / "Scripts" / "python.exe"
    win.parent.mkdir(parents=True)
    win.write_text("")
    assert server.venv_python(tmp_path) == str(win)


class FakeProc:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send_signal(self, sig):
        if self.fail:
            raise ValueError("Unsupported signal")
        self.sent.append(sig)

    def poll(self):
        return None

    def kill(self):
        pass


def test_stop_uses_ctrl_break_on_windows_and_sigint_elsewhere(server, monkeypatch):
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)
    monkeypatch.setattr(server, "WINDOWS", True)
    proc = FakeProc()
    server.interrupt(proc)
    assert proc.sent == [signal.CTRL_BREAK_EVENT]
    monkeypatch.setattr(server, "WINDOWS", False)
    proc = FakeProc()
    server.interrupt(proc)
    assert proc.sent == [signal.SIGINT]


def test_stop_still_succeeds_when_the_signal_fails(server, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "EXP_DIR", tmp_path)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "meta.json").write_text("{}")
    monkeypatch.setitem(server.RUNNING, "r1", FakeProc(fail=True))
    assert server.stop_run("r1") == {"ok": True}                       # the force-kill fallback is scheduled
    assert json.loads((tmp_path / "r1" / "meta.json").read_text())["stop_requested"] is True


def test_runs_get_their_own_process_group_on_windows():
    src = (ROOT / "web" / "server.py").read_text()
    assert "CREATE_NEW_PROCESS_GROUP} if WINDOWS" in src and "**group)" in src


# ---------------------------------------------------------------------------- arc/__main__.py

@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="needs a spare signal to stand in for SIGBREAK")
def test_ctrl_break_raises_keyboard_interrupt(monkeypatch):
    from arc import __main__ as cli
    monkeypatch.setattr(signal, "SIGBREAK", signal.SIGUSR1, raising=False)
    previous = signal.getsignal(signal.SIGUSR1)
    try:
        cli.stop_on_ctrl_break()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGUSR1)
            busy(1)
    finally:
        signal.signal(signal.SIGUSR1, previous)
