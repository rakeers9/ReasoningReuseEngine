"""Local experiment server for the ARC pipeline.

    python web/server.py [port]      (default port 8765, binds 127.0.0.1 only; macOS, Linux or Windows)

Starts pipeline commands as subprocesses with ARC_RUN_ID set, and serves every
experiment's calls, console output, settings and reports as JSON to web/index.html.
Python standard library only.

Files per experiment, in runs/experiments/<run_id>/:
    meta.json      written here: how the UI started it, status, exit code
    output.txt     written here: the live console output
    settings.json  written by arc/__main__.py: command, args, config snapshot
    calls.jsonl    written by arc/llm.py: every model call
    result.json    written by arc/__main__.py: status, duration, copied files
    files/         copies of the report files the run changed
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc import config  # noqa: E402  (also loads .env into this process, for the subprocesses)
from arc.checks import number_check, prompt_sections  # noqa: E402
from arc.walkthrough import STAGE_NAMES, _prompt_text, _stage  # noqa: E402

WEB = ROOT / "web"
EXP_DIR = config.RUNS_DIR / "experiments"
WINDOWS = os.name == "nt"


def venv_python(root: Path) -> str:
    """The project's virtualenv Python (Scripts\\python.exe on Windows, bin/python elsewhere),
    else the Python running this server."""
    for path in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if path.exists():
            return str(path)
    return sys.executable


PYTHON = venv_python(ROOT)
LEGACY_ID = "legacy-gas_law"
LEGACY_PID = "gas_law"

COMMANDS = ["paper", "ask", "solve", "build", "eval", "compile", "all"]
CONFIRM = {"solve", "build", "eval", "all"}
# Used only when no earlier run of the command exists to average over.
STATIC_ESTIMATE = {"paper": 0.00003, "ask": 0.012, "solve": 0.025, "build": 0.80,
                   "eval": 0.75, "compile": 0.0, "all": 1.60}  # Opus prices; ask: worst case (fallback)
# Opus 5.5 costs 40x Haiku 5.5 per token ($4/$20 vs $0.10/$0.50), used to scale Opus-based guesses.
PROFILE_COST_FACTOR = {"paper": 1.0, "haiku": 1 / 40}
JUDGE_ROLES = {"judge", "build_judge", "gen_judge", "test_judge"}
CMD_STAGE = {"paper": "step1", "ask": "step2", "solve": "source", "build": "build",
             "eval": "eval", "compile": "compile"}

RUNNING: dict[str, subprocess.Popen] = {}
LOCK = threading.Lock()
_cache: dict[str, tuple] = {}


# ----------------------------------------------------------------------------
# Reading files
# ----------------------------------------------------------------------------

def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass  # a line being written right now
    except OSError:
        pass
    return out


def sig(*paths: Path) -> tuple:
    return tuple((p.stat().st_mtime, p.stat().st_size) if p.exists() else None for p in paths)


def cached(key: str, signature: tuple, build):
    hit = _cache.get(key)
    if hit and hit[0] == signature:
        return hit[1]
    value = build()
    _cache[key] = (signature, value)
    return value


# ----------------------------------------------------------------------------
# Stages and steps for each call
# ----------------------------------------------------------------------------

def step_of(tag: str | None, pid: str | None) -> str:
    """Group key for 'totals by step': the tag without the problem id, cut to its kind."""
    tag = tag or ""
    if pid and tag == f"source:{pid}":
        return "source"
    rest = tag[len(pid) + 1:] if pid and tag.startswith(pid + ":") else tag
    if not rest:
        return "(untagged)"
    parts = rest.split(":")
    if parts[0] in ("gen", "gen_judge", "test") and len(parts) > 1:
        return f"{parts[0]}:{parts[1]}"
    if parts[0] == "eval":
        return "cache invocation"
    return parts[0]


def stage_new(cmd: str, pid: str | None, e: dict, after_last_eval: bool) -> str:
    tag = e.get("tag") or ""
    if pid and tag == f"source:{pid}":
        return "source"
    rest = tag[len(pid) + 1:] if pid and tag.startswith(pid + ":") else ""
    if rest.startswith(("phase", "judge", "build", "repair")):
        return "build"
    if rest:
        return "eval"
    if cmd in CMD_STAGE:
        return CMD_STAGE[cmd]
    return "compile" if after_last_eval else "eval"


def light_call(i: int, e: dict, stage: str, pid: str | None, t0: float) -> dict:
    prompt = _prompt_text(e.get("prompt"))
    has_tokens = "input_tokens" in e
    return {
        "i": i, "time": e.get("time"), "offset": round((e.get("time") or t0) - t0, 2),
        "role": e.get("role"), "tag": e.get("tag"), "stage": stage, "step": step_of(e.get("tag"), pid),
        "model": e.get("model"), "seconds": e.get("seconds"), "cost": e.get("cost", 0) or 0,
        "input_tokens": e.get("input_tokens") if has_tokens else None,
        "output_tokens": e.get("output_tokens") if has_tokens else None,
        "cache_read_tokens": e.get("cache_read_tokens") if has_tokens else None,
        "cache_write_tokens": e.get("cache_write_tokens") if has_tokens else None,
        "error": e.get("error"), "refusal": e.get("refusal_category") if "refusal_category" in e else None,
        "refused": "refusal_category" in e,
        "preview": prompt[:180].replace("\n", " "),
    }


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

def legacy_raw() -> list[dict]:
    path = config.RUNS_DIR / "llm_log.jsonl"
    return cached("legacy_raw", sig(path),
                  lambda: [e for e in read_jsonl(path) if not e.get("run_id")])


def snapshot_legacy_files() -> None:
    """Copy the first gas_law run's reports once, so later runs that overwrite them don't change it."""
    folder = EXP_DIR / LEGACY_ID / "files"
    if folder.exists():
        return
    folder.mkdir(parents=True)
    src = config.RUNS_DIR / LEGACY_PID
    for name in ("transcript.json", "transcript.md", "build_report.json", "eval_report.json",
                 "eval_progress.json", "build_log.txt", "eval_log.txt", "eval_log2.txt"):
        if (src / name).exists():
            shutil.copy2(src / name, folder / name)
    compiled = config.CACHES_DIR / "master" / f"compiled_{LEGACY_PID}.py"
    if compiled.exists():
        shutil.copy2(compiled, folder / compiled.name)


def legacy_calls() -> list[dict]:
    log = legacy_raw()
    pid = LEGACY_PID
    if not log:
        return []
    try:
        first_source = next(i for i, e in enumerate(log) if e.get("tag") == f"source:{pid}")
    except StopIteration:
        first_source = len(log)
    eval_idx = [i for i, e in enumerate(log) if (e.get("tag") or "").startswith(pid + ":") and
                not (e.get("tag") or "")[len(pid) + 1:].startswith(("phase", "judge", "build", "repair"))]
    first_eval, last_eval = (min(eval_idx), max(eval_idx)) if eval_idx else (len(log), len(log))
    t0 = log[0].get("time") or 0
    return [light_call(i, e, _stage(i, e, pid, first_source, first_eval, last_eval), pid, t0)
            for i, e in enumerate(log)]


def exp_folders() -> list[Path]:
    if not EXP_DIR.exists():
        return []
    return [p for p in EXP_DIR.iterdir() if p.is_dir() and p.name != LEGACY_ID]


def exp_problem(settings: dict, meta: dict) -> str | None:
    args = settings.get("args") or meta.get("args") or {}
    return args.get("problem")


def exp_command(settings: dict, meta: dict) -> str:
    return settings.get("command") or meta.get("command") or "?"


def exp_profile(settings: dict) -> str:
    """Runs from before profiles existed all used Opus, i.e. the paper profile."""
    values = (settings.get("settings") or {}).get("values") or {}
    return values.get("PROFILE") or "paper"


def models_of(values: dict) -> dict:
    return {k: values.get(k) for k in ("SOURCE_MODEL", "BUILDER_MODEL", "JUDGE_MODEL",
                                       "GENERATOR_MODEL", "TARGET_MODEL") if values.get(k)}


def golden_of(problem: str | None) -> str | None:
    if not problem:
        return None
    return (read_json(config.PROBLEMS_DIR / f"{problem}.json", {}) or {}).get("golden_label")


def raw_calls(run_id: str) -> list[dict]:
    if run_id == LEGACY_ID:
        return legacy_raw()
    path = EXP_DIR / run_id / "calls.jsonl"
    return cached(f"raw:{run_id}", sig(path), lambda: read_jsonl(path))


def calls_for(run_id: str) -> list[dict]:
    if run_id == LEGACY_ID:
        return cached("legacy_calls", sig(config.RUNS_DIR / "llm_log.jsonl"), legacy_calls)
    folder = EXP_DIR / run_id
    settings = read_json(folder / "settings.json", {}) or {}
    meta = read_json(folder / "meta.json", {}) or {}
    cmd, pid = exp_command(settings, meta), exp_problem(settings, meta)
    log = raw_calls(run_id)
    tagged_eval = [i for i, e in enumerate(log) if stage_new(cmd, pid, e, False) == "eval" and e.get("tag")]
    last_eval = max(tagged_eval) if tagged_eval else -1
    t0 = settings.get("started") or (log[0].get("time") if log else 0) or 0
    return [light_call(i, e, stage_new(cmd, pid, e, i > last_eval), pid, t0) for i, e in enumerate(log)]


def totals(calls: list[dict]) -> dict:
    t = {"calls": len(calls), "cost": 0.0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
         "calls_with_tokens": 0, "errors": 0, "refusals": 0}
    for c in calls:
        t["cost"] += c["cost"] or 0
        if c["input_tokens"] is not None:
            t["calls_with_tokens"] += 1
            t["input"] += c["input_tokens"] or 0
            t["output"] += c["output_tokens"] or 0
            t["cache_read"] += c["cache_read_tokens"] or 0
            t["cache_write"] += c["cache_write_tokens"] or 0
        t["errors"] += bool(c["error"])
        t["refusals"] += bool(c["refused"])
    return t


def group(calls: list[dict], key: str) -> dict:
    out: dict = defaultdict(list)
    for c in calls:
        out[c[key] or "?"].append(c)
    return {k: totals(v) for k, v in out.items()}


def files_dir(run_id: str) -> Path:
    return EXP_DIR / run_id / "files"


def eval_metrics(ev: dict | None) -> dict | None:
    if not ev:
        return None
    modes = {}
    for mode in ("phrasal", "parametric", "combined"):
        rows = [v for v in ev.get("variants", []) if v.get("mode") == mode]
        modes[mode] = {"passed": sum(v.get("outcome") == "pass" for v in rows), "total": len(rows)}
    return {"source_correct": ev.get("source_correct"), "cache_correct": ev.get("cache_correct"),
            "fidelity": ev.get("fidelity"), "modes": modes, "costs": ev.get("costs") or {},
            "errors": ev.get("errors") or {}, "rejected": ev.get("rejected_candidates") or {}}


def build_metrics(b: dict | None) -> dict | None:
    if not b:
        return None
    attempts = b.get("attempts", [])
    return {"success": b.get("success"), "cost": b.get("cost"),
            "functions": len({a.get("function_id") for a in attempts}),
            "attempts": len(attempts), "failed_attempts": sum(not a.get("passed") for a in attempts),
            "end_to_end": [x.get("passed") for x in b.get("end_to_end", [])],
            "extraction": [x.get("passed") for x in b.get("extraction_rounds", [])],
            "cost_by_role": b.get("cost_by_role") or {}}


def output_path(run_id: str) -> Path:
    return EXP_DIR / run_id / "output.txt"


def ask_path(run_id: str) -> str | None:
    try:
        text = output_path(run_id).read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"^Path: (CACHE|FULL LLM) \(([^)]*)\)", text, re.M)
    return f"{m.group(1)} ({m.group(2)})" if m else None


def default_settings() -> dict:
    return config.snapshot()["values"]


def summary(run_id: str) -> dict:
    if run_id == LEGACY_ID:
        calls = calls_for(LEGACY_ID)
        raw = legacy_raw()
        start = raw[0].get("time") if raw else None
        end = (raw[-1].get("time") or 0) + (raw[-1].get("seconds") or 0) if raw else None
        fd = files_dir(LEGACY_ID)
        return {
            "id": LEGACY_ID, "label": "legacy gas_law run", "command": "legacy",
            "problem": LEGACY_PID, "args": {"problem": LEGACY_PID},
            "argv": ["paper/ask/solve/build/eval/compile", LEGACY_PID],
            "status": "ok", "started": start, "finished": end,
            "seconds": round(end - start, 1) if start and end else None,
            "totals": totals(calls), "by_stage": group(calls, "stage"),
            "settings": default_settings(), "overrides": {}, "settings_recorded": False,
            "build": build_metrics(read_json(fd / "build_report.json")),
            "eval": eval_metrics(read_json(fd / "eval_report.json")),
            "files": sorted(p.name for p in fd.iterdir()) if fd.exists() else [],
            "legacy": True, "profile": "paper",
            "models": {"SOURCE_MODEL": "claude-opus-5-5", "BUILDER_MODEL": "claude-opus-5-5",
                       "JUDGE_MODEL": "claude-opus-5-5", "GENERATOR_MODEL": "claude-opus-5-5",
                       "TARGET_MODEL": "claude-haiku-5-5"},
            "golden": golden_of(LEGACY_PID), "reviews": review_counts(LEGACY_ID),
        }
    folder = EXP_DIR / run_id
    settings = read_json(folder / "settings.json", {}) or {}
    meta = read_json(folder / "meta.json", {}) or {}
    result = read_json(folder / "result.json")
    calls = calls_for(run_id)
    with LOCK:
        proc = RUNNING.get(run_id)
    if proc is not None and proc.poll() is None:
        status = "running"
    elif result:
        status = result.get("status", "ok")
        if status == "ok" and meta.get("exit_code") not in (None, 0):
            status = "failed"
    elif meta.get("status") == "running":
        status = "interrupted"
    else:
        status = meta.get("status") or "unknown"
    started = settings.get("started") or meta.get("created")
    finished = (result or {}).get("finished") or meta.get("finished")
    now = time.time()
    fd = files_dir(run_id)
    snap = settings.get("settings") or {}
    cmd = exp_command(settings, meta)
    out = {
        "id": run_id, "label": meta.get("label") or "", "command": cmd,
        "problem": exp_problem(settings, meta),
        "args": settings.get("args") or meta.get("args") or {},
        "argv": settings.get("argv") or meta.get("argv") or [],
        "status": status, "started": started, "finished": finished,
        "seconds": round((finished or now) - started, 1) if started else None,
        "exit_code": meta.get("exit_code", (result or {}).get("exit_code")),
        "error": (result or {}).get("error"),
        "totals": totals(calls), "by_stage": group(calls, "stage"),
        "settings": snap.get("values") or {}, "overrides": snap.get("overrides") or {},
        "settings_recorded": bool(snap), "source": meta.get("source", "cli"),
        "estimate": meta.get("estimate"),
        "build": build_metrics(read_json(fd / "build_report.json")),
        "eval": eval_metrics(read_json(fd / "eval_report.json")),
        "files": sorted(p.name for p in fd.iterdir()) if fd.exists() else [],
        "profile": exp_profile(settings), "models": models_of(snap.get("values") or {}),
        "golden": golden_of(exp_problem(settings, meta)), "reviews": review_counts(run_id),
    }
    if cmd == "ask":
        out["ask_path"] = ask_path(run_id)
    return out


def all_summaries() -> list[dict]:
    rows = [summary(p.name) for p in exp_folders()]
    rows.sort(key=lambda r: r.get("started") or 0, reverse=True)
    if legacy_raw():
        rows.append(summary(LEGACY_ID))
    return rows


def estimate(cmd: str, problem: str | None, extract: bool = False, profile: str | None = None) -> dict:
    """Average cost of earlier finished runs of the same command and profile; else a scaled guess."""
    profile = profile or config.PROFILE
    out = _estimate(cmd, problem, extract, profile)
    out["profile"] = profile
    return out


def _estimate(cmd: str, problem: str | None, extract: bool, profile: str) -> dict:
    past = []
    for p in exp_folders():
        s = summary(p.name)
        if s["command"] == cmd and s["status"] == "ok" and s["profile"] == profile and \
                (cmd == "ask" or s["problem"] == problem):
            past.append(s["totals"]["cost"])
    if past:
        return {"cost": sum(past) / len(past), "basis": f"average of {len(past)} earlier {profile}-profile run(s)",
                "min": min(past), "max": max(past)}
    if profile != "paper":
        opus = _estimate(cmd, problem, extract, "paper")
        factor = PROFILE_COST_FACTOR.get(profile, 1.0)
        if cmd in ("paper", "compile"):  # only the small model runs; same cost on every profile
            return opus
        return {"cost": opus["cost"] * factor,
                "basis": f"{opus['basis']}, scaled by 1/{round(1 / factor)} for {profile} prices"}
    if problem == LEGACY_PID and cmd in ("solve", "build", "eval"):
        stage = {"solve": "source", "build": "build", "eval": "eval"}[cmd]
        st = group(calls_for(LEGACY_ID), "stage").get(stage)
        if st:
            note = " (a fresh eval regenerates variants; a resumed one costs ~$0)" if cmd == "eval" else ""
            return {"cost": st["cost"], "basis": f"legacy gas_law run's {cmd} stage{note}"}
    if problem == LEGACY_PID and cmd == "all":
        by = group(calls_for(LEGACY_ID), "stage")
        c = sum(by.get(s, {}).get("cost", 0) for s in ("source", "build", "eval", "compile"))
        if c:
            return {"cost": c, "basis": "legacy gas_law run's solve + build + eval + compile"}
    cost = STATIC_ESTIMATE.get(cmd, 0)
    if cmd == "compile" and extract:
        cost = 0.0001
    basis = "rough guess"
    if cmd == "ask":
        basis = "rough guess: ~$0.0001 on a cache hit, ~$0.01 when it falls back to the source model (Opus, high effort)"
    return {"cost": cost, "basis": basis}


# ----------------------------------------------------------------------------
# Starting and stopping runs
# ----------------------------------------------------------------------------

def new_run_id(cmd: str) -> str:
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{cmd}"
    rid, n = base, 1
    while (EXP_DIR / rid).exists():
        n += 1
        rid = f"{base}-{n}"
    return rid


def start_run(body: dict) -> dict:
    cmd = body.get("command")
    if cmd not in COMMANDS:
        raise ValueError(f"unknown command {cmd!r}")
    with LOCK:
        busy = [rid for rid, p in RUNNING.items() if p.poll() is None]
    if busy:
        raise RuntimeError(f"experiment {busy[0]} is still running; stop it or wait")

    argv = [cmd]
    problem = None
    if cmd == "ask":
        cache, question = body.get("cache") or "", (body.get("question") or "").strip()
        if not cache or not (config.cache_dir(cache) / "registry.json").exists():
            raise ValueError(f"no cache named {cache!r}")
        if not question:
            raise ValueError("the question is empty")
        argv += [cache, question]
    else:
        problem = body.get("problem") or ""
        if not (config.PROBLEMS_DIR / f"{problem}.json").exists():
            raise ValueError(f"no problem named {problem!r}")
        argv.append(problem)
        if cmd in ("build", "all") and body.get("no_golden"):
            argv.append("--no-golden")
        if cmd in ("eval", "compile", "all"):
            cache = body.get("cache") or "master"
            if not re.fullmatch(r"[\w.-]+", cache):
                raise ValueError("bad cache name")
            argv += ["--cache", cache]
        if cmd in ("compile", "all") and body.get("extract"):
            argv.append("--extract")

    overrides = {}
    for name, value in (body.get("overrides") or {}).items():
        if name not in config.SETTING_NAMES:
            raise ValueError(f"unknown setting {name}")
        value = str(value).strip()
        if value != "":
            overrides[name] = value

    run_id = new_run_id(cmd)
    folder = EXP_DIR / run_id
    folder.mkdir(parents=True)

    # eval resumes from runs/<problem>/eval_progress.json; a fresh eval moves it into this
    # experiment's folder first so the run regenerates and re-grades everything.
    moved = None
    profile = overrides.get("PROFILE", config.PROFILE)
    ns = config.namespace_for(profile)
    if cmd in ("eval", "all") and body.get("fresh_eval") and problem:
        prog = (config.RUNS_DIR / ns if ns else config.RUNS_DIR) / problem / "eval_progress.json"
        if prog.exists():
            shutil.move(str(prog), str(folder / "prior_eval_progress.json"))
            moved = "prior_eval_progress.json"

    env = dict(os.environ)
    env["ARC_RUN_ID"] = run_id
    env["PYTHONUNBUFFERED"] = "1"
    for name, value in overrides.items():
        env[f"ARC_{name}"] = value

    est = estimate(cmd, problem, bool(body.get("extract")), profile)
    meta = {"run_id": run_id, "command": cmd, "argv": argv,
            "args": used_args(cmd, problem, body),
            "overrides": overrides, "label": (body.get("label") or "").strip(), "source": "ui",
            "created": time.time(), "status": "running", "estimate": est, "moved_progress": moved}
    out_file = open(folder / "output.txt", "wb")
    out_file.write(("$ python -m arc " + " ".join(_shell_quote(a) for a in argv) + "\n").encode())
    if overrides:
        out_file.write(("  overrides: " + ", ".join(f"ARC_{k}={v}" for k, v in overrides.items()) + "\n").encode())
    if moved:
        out_file.write(f"  fresh eval: moved eval_progress.json into this experiment's folder\n".encode())
    out_file.flush()
    # Windows: the run gets its own process group, so Stop can send it Ctrl+Break (see interrupt).
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {}
    proc = subprocess.Popen([PYTHON, "-m", "arc", *argv], cwd=ROOT, env=env, stdout=out_file,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **group)
    out_file.close()
    meta["pid"] = proc.pid
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    with LOCK:
        RUNNING[run_id] = proc
    threading.Thread(target=_watch, args=(run_id, proc), daemon=True).start()
    return {"run_id": run_id}


def used_args(cmd: str, problem: str | None, body: dict) -> dict:
    """Only the options that apply to this command."""
    if cmd == "ask":
        return {"cache": body.get("cache"), "question": (body.get("question") or "").strip()}
    out = {"problem": problem}
    if cmd in ("eval", "compile", "all"):
        out["cache"] = body.get("cache") or "master"
    if cmd in ("build", "all"):
        out["no_golden"] = bool(body.get("no_golden"))
    if cmd in ("compile", "all"):
        out["extract"] = bool(body.get("extract"))
    if cmd in ("eval", "all"):
        out["fresh_eval"] = bool(body.get("fresh_eval"))
    return out


def _shell_quote(a: str) -> str:
    return a if re.fullmatch(r"[\w./:=-]+", a) else "'" + a.replace("'", "'\\''") + "'"


def _watch(run_id: str, proc: subprocess.Popen) -> None:
    code = proc.wait()
    path = EXP_DIR / run_id / "meta.json"
    meta = read_json(path, {}) or {}
    meta.update({"status": "ok" if code == 0 else ("stopped" if meta.get("stop_requested") else "failed"),
                 "exit_code": code, "finished": time.time()})
    path.write_text(json.dumps(meta, indent=2))
    with open(output_path(run_id), "ab") as f:
        f.write(f"\n[process exited with code {code}]\n".encode())


def interrupt(proc: subprocess.Popen) -> None:
    """Ask a run to stop as Ctrl+C would, so arc/__main__.py still writes result.json. Windows
    can't send SIGINT to another process; it gets Ctrl+Break, which arc/__main__.py turns into
    the same KeyboardInterrupt."""
    if WINDOWS:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


def stop_run(run_id: str) -> dict:
    with LOCK:
        proc = RUNNING.get(run_id)
    if proc is None or proc.poll() is not None:
        raise ValueError("that experiment is not running")
    path = EXP_DIR / run_id / "meta.json"
    meta = read_json(path, {}) or {}
    meta["stop_requested"] = True
    path.write_text(json.dumps(meta, indent=2))
    try:
        interrupt(proc)
    except (OSError, ValueError):   # the force-kill below still runs
        pass

    def hard_stop():
        time.sleep(8)
        if proc.poll() is None:
            proc.kill()
    threading.Thread(target=hard_stop, daemon=True).start()
    return {"ok": True}


def mark_orphans() -> None:
    """Runs left 'running' by an earlier server process."""
    for p in exp_folders():
        meta = read_json(p / "meta.json")
        if meta and meta.get("status") == "running":
            meta["status"] = "interrupted"
            (p / "meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------
# Reports and single calls
# ----------------------------------------------------------------------------

def reports(run_id: str) -> dict:
    fd = files_dir(run_id)
    out = {}
    for key, name in (("build", "build_report.json"), ("eval", "eval_report.json"),
                      ("transcript", "transcript.json"), ("eval_progress", "eval_progress.json")):
        data = read_json(fd / name)
        if data is not None:
            out[key] = data
    for f in fd.glob("compiled_*.py") if fd.exists() else []:
        out["compiled"] = f.read_text()
    return out


def full_call(run_id: str, i: int) -> dict:
    log = raw_calls(run_id)
    if not 0 <= i < len(log):
        raise ValueError("no such call")
    e = log[i]
    light = calls_for(run_id)[i]
    resp = e.get("response")
    pretty = resp
    if isinstance(resp, str):
        try:
            pretty = json.dumps(json.loads(resp), indent=2, ensure_ascii=False)
        except ValueError:
            pass
    return {**light, "prompt": _prompt_text(e.get("prompt")), "system": _prompt_text(e.get("system")),
            "response": pretty or "", "refusal_explanation": e.get("refusal_explanation"),
            "extra": {k: v for k, v in e.items() if k in ("effort", "max_tokens", "stop_reason",
                                                         "visible_text_chars", "thinking_chars")}}


def search(run_id: str, q: str) -> list[int]:
    q = q.lower()
    hits = []
    for i, e in enumerate(raw_calls(run_id)):
        text = " ".join([e.get("role") or "", e.get("tag") or "", _prompt_text(e.get("prompt")),
                         _prompt_text(e.get("system")), str(e.get("response") or ""), str(e.get("error") or "")])
        if q in text.lower():
            hits.append(i)
    return hits


def read_output(run_id: str, offset: int) -> dict:
    if run_id == LEGACY_ID:
        fd = files_dir(LEGACY_ID)
        parts = []
        for name in ("build_log.txt", "eval_log.txt", "eval_log2.txt"):
            if (fd / name).exists():
                parts.append(f"===== {name} =====\n" + (fd / name).read_text(errors="replace"))
        text = "\n".join(parts) or "(no console logs were kept for this run)"
        data = text.encode()
        return {"text": data[offset:].decode(errors="replace"), "offset": len(data), "status": "ok"}
    path = output_path(run_id)
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:
        chunk = b"" if offset else b"(no console output was captured: started outside the UI)\n"
        return {"text": chunk.decode(), "offset": offset + len(chunk), "status": summary(run_id)["status"]}
    return {"text": chunk.decode(errors="replace"), "offset": offset + len(chunk),
            "status": summary(run_id)["status"]}


def meta_info() -> dict:
    problems = []
    for p in sorted(config.PROBLEMS_DIR.glob("*.json")):
        d = read_json(p, {}) or {}
        pd = config.problem_dir(p.stem)
        problems.append({"id": p.stem, "question": d.get("question", ""), "ran": pd.is_dir(),
                         "has_transcript": (pd / "transcript.json").exists(),
                         "has_build": (pd / "build_report.json").exists(),
                         "eval_progress": (pd / "eval_progress.json").exists()})
    names = set()
    if config.CACHE_ROOT.exists():
        names |= {p.name for p in config.CACHE_ROOT.iterdir() if (p / "registry.json").exists()}
    if config.CACHES_DIR.exists():  # the paper's hand-typed caches are shared by every profile
        names |= {p.name for p in config.CACHES_DIR.iterdir() if p.name.startswith("paper_") and (p / "registry.json").exists()}
    caches = sorted(names)
    with LOCK:
        running = [rid for rid, p in RUNNING.items() if p.poll() is None]
    return {"commands": COMMANDS, "confirm": sorted(CONFIRM), "problems": problems, "caches": caches,
            "settings": default_settings(), "setting_names": config.SETTING_NAMES,
            "stage_names": STAGE_NAMES, "running": running, "profile": config.PROFILE,
            "profiles": sorted(config.PROFILE_BIG_MODEL)}


# ----------------------------------------------------------------------------
# Judge verdicts and human reviews
# ----------------------------------------------------------------------------

JUDGE_KIND = {"build_judge": "Build check", "gen_judge": "Variant validation",
              "test_judge": "Grading", "judge": "Judge"}


def _kind(role: str, step: str, tag: str | None = None) -> str:
    if role == "build_judge":
        last = (tag or "").split(":")[-1]
        if last == "end_to_end":
            return "Build: end-to-end"
        if last.startswith("extraction"):
            return "Build: extraction round"
        return "Build: function check"
    if role == "test_judge" and step in ("s_acc", "c_acc", "fidelity"):
        return "Correctness"
    return JUDGE_KIND.get(role, role)


def _verdict(response) -> dict:
    try:
        data = json.loads(response) if isinstance(response, str) else (response or {})
        return {"passed": bool(data.get("passed")), "critique": data.get("critique", "")}
    except (ValueError, AttributeError):
        return {"passed": None, "critique": str(response or "")}


def _evidence_check(sections: list[dict], golden: str | None) -> dict | None:
    """The no-LLM number check for a verdict: the answer under review vs its reference."""
    by = {x["title"].lower(): x["text"] for x in sections}
    answer = by.get("cache output") or by.get("output") or (by.get("transcript") if "golden label" in by else None)
    reference = by.get("reference answer") or by.get("golden label")
    against = "reference answer" if by.get("reference answer") else "golden label"
    if answer and not reference and golden and ("cache output" in by):
        reference, against = golden, "golden label (from the problem file)"
    if not answer or not reference:
        return None
    return {**number_check(answer, reference), "against": against}


def judgements(run_id: str) -> list[dict]:
    """Every judge decision, with the committee's verdicts grouped and the prompt split into evidence."""
    log = raw_calls(run_id)
    calls = calls_for(run_id)
    s = summary(run_id)
    golden = s.get("golden")
    groups: list[dict] = []
    for i, e in enumerate(log):
        if e.get("role") not in JUDGE_ROLES or e.get("error") or "refusal_category" in e:
            continue
        prompt = _prompt_text(e.get("prompt"))
        last = groups[-1] if groups else None
        verdict = {**_verdict(e.get("response")), "call": i, "model": e.get("model")}
        if last and last["prompt"] == prompt and last["role"] == e.get("role") and last["tag"] == e.get("tag"):
            last["verdicts"].append(verdict)
            last["calls"].append(i)
            continue
        c = calls[i]
        sections = prompt_sections(prompt)
        groups.append({"key": f"call:{i}", "calls": [i], "role": e.get("role"), "tag": e.get("tag"),
                       "stage": c["stage"], "step": c["step"], "kind": _kind(e.get("role"), c["step"], e.get("tag")),
                       "prompt": prompt, "sections": sections, "verdicts": [verdict],
                       "number_check": _evidence_check(sections, golden)})
    for g in groups:
        vs = [v["passed"] for v in g["verdicts"]]
        g["passed"] = all(v is True for v in vs)
        g["split"] = len(set(vs)) > 1
        g.pop("prompt")
    return groups


def reviews_path(run_id: str) -> Path:
    return EXP_DIR / run_id / "reviews.json"


def read_reviews(run_id: str) -> dict:
    return read_json(reviews_path(run_id), {}) or {}


def save_review(run_id: str, body: dict) -> dict:
    key = str(body.get("key") or "")
    if not re.fullmatch(r"[\w:.-]{1,80}", key):
        raise ValueError("bad review key")
    verdict = body.get("verdict")
    if verdict not in ("correct", "wrong", "unsure", None):
        raise ValueError("verdict must be correct, wrong, unsure or null")
    with LOCK:
        data = read_reviews(run_id)
        if verdict is None and not (body.get("note") or "").strip():
            data.pop(key, None)
        else:
            data[key] = {"verdict": verdict, "note": (body.get("note") or "").strip()[:4000],
                         "judge_passed": body.get("judge_passed"), "time": time.time()}
        reviews_path(run_id).parent.mkdir(parents=True, exist_ok=True)
        reviews_path(run_id).write_text(json.dumps(data, indent=2))
    return data


def review_counts(run_id: str) -> dict:
    data = read_reviews(run_id)
    out = {"reviewed": 0, "disagree": 0, "correct": 0, "wrong": 0, "unsure": 0}
    for r in data.values():
        v = r.get("verdict")
        if not v:
            continue
        out["reviewed"] += 1
        out[v] += 1
        jp = r.get("judge_passed")
        if jp is not None and v in ("correct", "wrong") and (v == "correct") != bool(jp):
            out["disagree"] += 1
    return out


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json"}


def valid_id(run_id: str) -> bool:
    return run_id == LEGACY_ID or (bool(re.fullmatch(r"[\w.-]+", run_id)) and (EXP_DIR / run_id).is_dir())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet except for errors
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            if not parts or parts[0] != "api":
                return self.static(url.path)
            if parts == ["api", "meta"]:
                return self.send_json(meta_info())
            if parts == ["api", "estimate"]:
                return self.send_json(estimate(q.get("command", [""])[0], q.get("problem", [None])[0],
                                               q.get("extract", ["0"])[0] == "1",
                                               q.get("profile", [None])[0] or None))
            if parts == ["api", "experiments"]:
                return self.send_json(all_summaries())
            if len(parts) >= 3 and parts[1] == "experiments":
                rid = parts[2]
                if not valid_id(rid):
                    return self.send_json({"error": "no such experiment"}, 404)
                if len(parts) == 3:
                    calls = calls_for(rid)
                    return self.send_json({**summary(rid), "calls": calls, "by_role": group(calls, "role"),
                                           "by_step": group(calls, "step")})
                if parts[3] == "calls" and len(parts) == 5:
                    return self.send_json(full_call(rid, int(parts[4])))
                if parts[3] == "search":
                    return self.send_json(search(rid, q.get("q", [""])[0]))
                if parts[3] == "output":
                    return self.send_json(read_output(rid, int(q.get("offset", ["0"])[0])))
                if parts[3] == "reports":
                    return self.send_json(reports(rid))
                if parts[3] == "judgements":
                    return self.send_json(judgements(rid))
                if parts[3] == "reviews":
                    return self.send_json(read_reviews(rid))
            self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:  # report, don't crash the server
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if parts == ["api", "run"]:
                return self.send_json(start_run(body))
            if len(parts) == 4 and parts[1] == "experiments" and parts[3] == "stop":
                return self.send_json(stop_run(parts[2]))
            if len(parts) == 4 and parts[1] == "experiments" and parts[3] == "reviews":
                if not valid_id(parts[2]):
                    return self.send_json({"error": "no such experiment"}, 404)
                return self.send_json(save_review(parts[2], body))
            self.send_json({"error": "not found"}, 404)
        except (ValueError, RuntimeError) as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def static(self, path: str):
        if path in ("", "/"):
            path = "/index.html"
        target = (WEB / path.lstrip("/")).resolve()
        if WEB.resolve() not in target.parents or not target.is_file() or target.suffix not in MIME:
            return self.send_json({"error": "not found"}, 404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME[target.suffix])
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("ARC_WEB_PORT", 8765))
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_legacy_files()
    mark_orphans()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ARC experiments UI on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
