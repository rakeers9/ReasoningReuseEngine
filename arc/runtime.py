"""The ARC runtime: the Context and the runtime primitives (paper Section 2.1).

A semantic function is `f(ctx) -> dict[str, Any]`. It never sees variable or
function names from its caller; it reaches everything through these primitives:

    load(ctx, query, d_type=None)        read a value by description (embeddings)
    load(result, query, d_type, ctx=ctx) read from a callee's output dict
    do(ctx, query, payloads)             call another function by description
    narrate(ctx, message)                record a human-readable step
    lm(ctx, prompt)                      ask the small target model T
    extract(ctx, text, prompt, d_type)   verbatim value extraction with T
    convert(ctx, value, type)            programmatic type coercion, then T
    smart_convert(ctx, text, type)       T rewrites to a Python literal
    invoke_tool(ctx, name, args)         call a registered external tool

DEVIATION (not in the paper): two unit helpers from arc/units.py are also exposed, so cached
code never types conversion factors by hand. They take no ctx and raise units.UnitError,
which turns the invocation into a fallback like any other error in cached code:

    convert_unit(value, from_unit, to_unit, substance=None)   exact unit conversion
    same_quantity(unit_a, unit_b, substance=None)             can unit_a convert to unit_b?

They are always present (flag-free): a function cached with them must keep running whatever
the current settings, and code that doesn't call them is unaffected.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config, embed, llm, units


class CacheInvocationError(Exception):
    """Anything that should make a cache invocation fall back to the full LLM."""


class LoadError(CacheInvocationError):
    pass


class DispatchError(CacheInvocationError):
    pass


class ConversionError(CacheInvocationError):
    pass


# ----------------------------------------------------------------------------
# Memory model: a stack of layers, each a tiny vector store
# ----------------------------------------------------------------------------

@dataclass
class Layer:
    """One scope. Keys are natural-language descriptions; search is by meaning."""
    values: dict[str, Any] = field(default_factory=dict)

    def put(self, description: str, value: Any) -> None:
        self.values[description] = value

    def search(self, query: str) -> tuple[str | None, float]:
        keys = list(self.values)
        i, sim = embed.best_match(query, keys)
        return (keys[i], sim) if i >= 0 else (None, -1.0)


@dataclass
class Lookup:
    """Debug record of one load()/do() resolution, so humans can audit matches."""
    kind: str
    query: str
    matched: str | None
    similarity: float
    depth: int


class Context:
    """Stack of memory layers plus the narration trace for one invocation."""

    MAX_DEPTH = 16

    def __init__(self, cache=None, tools: dict[str, Callable] | None = None):
        self.cache = cache                  # function registry used by do()
        self.tools = tools or {}
        self.layers: list[Layer] = [Layer()]
        self.trace: list[tuple[int, str]] = []
        self.lookups: list[Lookup] = []

    @property
    def depth(self) -> int:
        return len(self.layers) - 1

    def push(self, payloads: dict[str, Any] | None = None) -> Layer:
        if len(self.layers) > self.MAX_DEPTH:
            raise DispatchError("call depth limit exceeded")
        layer = Layer(dict(payloads or {}))
        self.layers.append(layer)
        return layer

    def pop(self) -> Layer:
        return self.layers.pop()

    def set(self, description: str, value: Any) -> None:
        """Put a value into the current (top) layer."""
        self.layers[-1].put(description, value)

    def narration(self) -> str:
        return "\n".join("    " * d + msg for d, msg in self.trace)


class Result(dict):
    """A callee's output dictionary, as returned by do()."""


# ----------------------------------------------------------------------------
# Type coercion
# ----------------------------------------------------------------------------

_TYPE_NAMES = {str: "str", int: "int", float: "float", bool: "bool", list: "list",
               tuple: "tuple", set: "set", dict: "dict", frozenset: "frozenset"}
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _programmatic(value: Any, t: type) -> Any:
    if t is None or t is Any:
        return value
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1", "false", "no", "0"}:
            return value.strip().lower() in {"true", "yes", "1"}
        raise ConversionError(f"cannot convert {value!r} to bool")
    if isinstance(value, t) and not (t in (int, float) and isinstance(value, bool)):
        return value
    if t is str:
        return str(value)
    if t is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and _NUMBER.match(value.strip().replace(",", "")):
            return float(value.strip().replace(",", ""))
        raise ConversionError(f"cannot convert {value!r} to float")
    if t is int:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value.strip())
        raise ConversionError(f"cannot convert {value!r} to int")
    if t in (list, tuple, set, frozenset, dict):
        if isinstance(value, (list, tuple, set, frozenset)) and t is not dict:
            return t(value)
        if isinstance(value, str):
            try:
                literal = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError) as e:
                raise ConversionError(f"cannot parse {value!r} as {t.__name__}") from e
            if isinstance(literal, (list, tuple, set, frozenset, dict)):
                return _programmatic(literal, t)
        raise ConversionError(f"cannot convert {value!r} to {t.__name__}")
    try:
        return t(value)
    except Exception as e:
        raise ConversionError(f"cannot convert {value!r} to {t}") from e


def convert(ctx: Context | None, value: Any, type_: type) -> Any:
    """Multi-tier coercion: plain Python first, then the target model."""
    try:
        return _programmatic(value, type_)
    except ConversionError:
        return smart_convert(ctx, value, type_)


def smart_convert(ctx: Context | None, text: Any, type_: type) -> Any:
    """Ask T to rewrite a value as a Python literal, validated by ast.literal_eval()."""
    name = _TYPE_NAMES.get(type_, getattr(type_, "__name__", str(type_)))
    result = llm.call(
        "converter",
        f"Rewrite this value as a single Python literal of type `{name}`.\n"
        f"Value: {text!r}\n\nReply with only the Python literal and nothing else.",
        model=config.TARGET_MODEL, max_tokens=512,
    )
    raw = result.text.strip().strip("`").strip()
    if raw.startswith("python"):
        raw = raw[len("python"):].strip()
    try:
        literal = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        raise ConversionError(f"target model returned an invalid literal: {raw!r}") from e
    return _programmatic(literal, type_)


# ----------------------------------------------------------------------------
# Runtime primitives
# ----------------------------------------------------------------------------

def load(source: Context | dict, query: str, d_type: type | None = None,
         ctx: Context | None = None) -> Any:
    """Retrieve a value by description via vector similarity search.

    On a Context: searches layers top-down and returns the best match from the
    most local scope that clears the threshold. On a dict (a callee's result):
    searches that dict's keys.
    """
    if isinstance(source, Context):
        ctx = source
        for layer in reversed(ctx.layers):
            key, sim = layer.search(query)
            if key is not None and sim >= config.LOAD_THRESHOLD:
                ctx.lookups.append(Lookup("load", query, key, sim, ctx.depth))
                value = layer.values[key]
                break
        else:
            available = [k for layer in ctx.layers for k in layer.values]
            raise LoadError(f"nothing in context matches {query!r}; available: {available}")
    elif isinstance(source, dict):
        keys = list(source)
        i, sim = embed.best_match(query, keys)
        if i < 0 or sim < config.LOAD_THRESHOLD:
            raise LoadError(f"no output matches {query!r}; outputs: {keys}")
        if ctx is not None:
            ctx.lookups.append(Lookup("load-result", query, keys[i], sim, ctx.depth))
        value = source[keys[i]]
    else:
        raise LoadError(f"load() needs a Context or a result dict, got {type(source).__name__}")
    return convert(ctx, value, d_type) if d_type is not None else value


def do(ctx: Context, query: str, payloads: dict[str, Any] | None = None) -> Result:
    """Discover and invoke another semantic function by natural-language description.

    Pushes an isolated layer pre-populated with the payloads, runs the matched
    function, stores its outputs into that layer, pops it, and returns the
    output dictionary.
    """
    if ctx.cache is None:
        raise DispatchError("this context has no function registry")
    record, sim = ctx.cache.find_function(query)
    if record is None or sim < config.DO_THRESHOLD:
        raise DispatchError(f"no function matches {query!r} (best similarity {sim:.2f})")
    ctx.lookups.append(Lookup("do", query, record.description, sim, ctx.depth))
    layer = ctx.push(payloads)
    try:
        output = record.fn(ctx)
        if not isinstance(output, dict):
            raise DispatchError(f"{record.description!r} returned {type(output).__name__}, not dict")
        for k, v in output.items():
            layer.put(k, v)
    finally:
        ctx.pop()
    return Result(output)


def narrate(ctx: Context, message: Any) -> None:
    """Record a human-readable reasoning step in the narration trace."""
    ctx.trace.append((ctx.depth, str(message)))


def lm(ctx: Context, prompt: str) -> str:
    """Query the target model for an elementary language task."""
    return llm.call("target", prompt, model=config.TARGET_MODEL, max_tokens=1024).text.strip()


def extract(ctx: Context, text: Any, prompt: str, d_type: type | None = None) -> Any:
    """Use the target model to extract a value verbatim from a piece of text."""
    result = llm.call(
        "target",
        f"Text: {text}\n\nExtract: {prompt}\n\n"
        "Copy the value exactly as it appears in the text. Reply with only the value.",
        model=config.TARGET_MODEL, max_tokens=256,
    )
    value = result.text.strip().strip("`").strip()
    return convert(ctx, value, d_type) if d_type is not None else value


def invoke_tool(ctx: Context, name: str, args: dict | None = None) -> Any:
    """Call a registered external tool (e.g. a terminal or web search API)."""
    if name not in ctx.tools:
        raise DispatchError(f"no tool named {name!r} is registered")
    return ctx.tools[name](**(args or {}))


PAPER_PRIMITIVES = {
    "load": load, "do": do, "narrate": narrate, "lm": lm, "extract": extract,
    "convert": convert, "smart_convert": smart_convert, "invoke_tool": invoke_tool,
}

# DEVIATION: unit helpers (see module docstring). UnitError is deliberately not exposed, so
# cached code can't catch a unit failure and carry on with a guessed value.
UNIT_PRIMITIVES = {"convert_unit": units.convert, "same_quantity": units.same_quantity}

PRIMITIVES = {**PAPER_PRIMITIVES, **UNIT_PRIMITIVES}
