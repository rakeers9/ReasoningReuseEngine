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


def check_source(source: str) -> ast.Module:
    """Parse `source` and reject anything outside the allowed surface."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    raise UnsafeCodeError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise UnsafeCodeError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(f"access to '{node.attr}' is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "semantic_function" for n in tree.body):
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
