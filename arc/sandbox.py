"""Load and run model-written semantic-function source code more safely.

The paper does not describe sandboxing. Since the builder writes code that we
then execute, we add two light guards:
  1. a static check that rejects imports outside a small allow-list and any use
     of file, process, or reflection builtins; and
  2. a wall-clock timeout around every execution.
This keeps the embedding model in-process (loading it per subprocess would be
slow) while blocking the obvious ways generated code could misbehave.
"""

from __future__ import annotations

import ast
import builtins
import math
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass

ALLOWED_MODULES = {
    "math", "cmath", "fractions", "decimal", "itertools", "functools", "re",
    "statistics", "collections", "operator", "string", "datetime", "numbers",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit", "help", "memoryview", "setattr",
    "delattr",
}


class UnsafeCodeError(ValueError):
    pass


class FunctionTimeout(TimeoutError):
    pass


def check_source(source: str, require_function: bool = True,
                 forbidden: set[str] | None = None) -> ast.Module:
    """Parse `source` and reject anything outside the allowed surface."""
    forbidden = FORBIDDEN_NAMES if forbidden is None else forbidden
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    raise UnsafeCodeError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise UnsafeCodeError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in forbidden:
            raise UnsafeCodeError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(f"access to '{node.attr}' is not allowed")
    if require_function and not any(isinstance(n, ast.FunctionDef) and n.name == "semantic_function"
                                    for n in tree.body):
        raise UnsafeCodeError("source must define `def semantic_function(ctx):`")
    return tree


def _safe_builtins() -> dict:
    allowed = {k: getattr(builtins, k) for k in dir(builtins)
               if not k.startswith("_") and k not in FORBIDDEN_NAMES}
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError(f"import of '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    allowed["__import__"] = guarded_import
    allowed["__build_class__"] = builtins.__build_class__
    allowed["__name__"] = "semantic_function"
    return allowed


def load_function(source: str, primitives: dict):
    """Compile checked source and return its `semantic_function` callable."""
    check_source(source)
    namespace = {"__builtins__": _safe_builtins(), "math": math, **primitives}
    exec(compile(source, "<semantic_function>", "exec"), namespace)
    return namespace["semantic_function"]


@contextmanager
def time_limit(seconds: float):
    """Raise FunctionTimeout if the block runs longer than `seconds`.

    Uses SIGALRM, which only works in the main thread; elsewhere it is a no-op.
    Nested uses keep the outermost (earliest) deadline.
    """
    if threading.current_thread() is not threading.main_thread() or seconds <= 0:
        yield
        return
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    if remaining > 0:  # an outer limit is already running
        yield
        return

    def _handler(signum, frame):
        raise FunctionTimeout(f"cached code exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ----------------------------------------------------------------------------- standalone programs
# Tie-break solves (arc/labelfree.adjudicate) are whole programs that print their answer. They
# run in a separate Python process: a hard timeout that works from any thread, nothing shared
# with this process, and the same static checks plus restricted builtins inside the child.

PROGRAM_FORBIDDEN = FORBIDDEN_NAMES | {"getattr"}

_BOOT = """
import builtins, sys
ALLOWED = %r
FORBIDDEN = %r
src = sys.stdin.read()
safe = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith("_") and k not in FORBIDDEN}
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError("import of '%%s' is not allowed" %% name)
    return real_import(name, globals, locals, fromlist, level)
safe["__import__"] = guarded_import
safe["__build_class__"] = builtins.__build_class__
safe["__name__"] = "__main__"
exec(compile(src, "<program>", "exec"), {"__builtins__": safe, "__name__": "__main__"})
"""


@dataclass
class ProgramRun:
    ok: bool
    stdout: str
    error: str = ""


def run_program(source: str, seconds: float = 10, max_output: int = 4000) -> ProgramRun:
    """Run a standalone program that prints its answer. ok=False on unsafe code, a timeout,
    an exception or a non-zero exit; stdout is what it printed (capped)."""
    import subprocess
    import sys
    try:
        check_source(source, require_function=False, forbidden=PROGRAM_FORBIDDEN)
    except (UnsafeCodeError, SyntaxError) as e:
        return ProgramRun(False, "", f"{type(e).__name__}: {e}")
    boot = _BOOT % (sorted(ALLOWED_MODULES), sorted(PROGRAM_FORBIDDEN))
    try:
        proc = subprocess.run([sys.executable, "-I", "-S", "-c", boot], input=source, capture_output=True,
                              text=True, timeout=seconds, env={"PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return ProgramRun(False, "", f"timed out after {seconds:g}s")
    out = proc.stdout[-max_output:]
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["exit code %d" % proc.returncode])[-1]
        return ProgramRun(False, out, last[:300])
    return ProgramRun(True, out)
