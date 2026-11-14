"""Step 6: compilation by static resolution (paper Section 2.2.3, Figures 7 and 13).

Once a cache has matured, its runtime lookups can be resolved ahead of time:

    do(ctx, "<desc>", payloads={...})    ->  direct call  callee(ctx, arg=value, ...)
    load(ctx, "<desc>", d_type=T)        ->  a named parameter (wrapped in convert() if typed)
    load(result, "<desc>", d_type=T)     ->  result["<resolved output key>"]

lm(), extract() and convert() cannot be resolved statically because they need
a model at invocation time, so they survive compilation unchanged. narrate()
calls are preserved. The output is one self-contained Python module that needs
no embedding model and no vector store.
"""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass, field

from . import config, embed
from .cache import Cache, EntryPoint, FunctionRecord


class CompileError(RuntimeError):
    pass


def _ident(text: str, taken: set[str], max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in {"the", "a", "an"}] or ["value"]
    name = "_".join(words[:max_words])
    if name[0].isdigit() or keyword.iskeyword(name):
        name = "v_" + name
    base, i = name, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name


def _literal_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _kw(call: ast.Call, name: str, position: int | None = None) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    if position is not None and len(call.args) > position:
        return call.args[position]
    return None


@dataclass
class _FnInfo:
    record: FunctionRecord
    tree: ast.FunctionDef
    name: str
    params: dict[str, str] = field(default_factory=dict)       # load query -> arg name
    return_keys: list[str] = field(default_factory=list)


class Compiler:
    def __init__(self, cache: Cache, entry: EntryPoint):
        self.cache = cache
        self.entry = entry
        self.infos: dict[str, _FnInfo] = {}
        self._names: set[str] = {"run", "ctx", "Context"}

    # -- pass 1: collect functions, parameters, outputs ----------------------

    def _resolve_callee(self, call: ast.Call) -> FunctionRecord:
        query = _literal_str(call.args[1]) if len(call.args) > 1 else _literal_str(_kw(call, "query"))
        if query is None:
            raise CompileError("do() with a non-literal description cannot be resolved statically")
        record, sim = self.cache.find_function(query)
        if record is None or sim < config.DO_THRESHOLD:
            raise CompileError(f"do({query!r}) does not resolve to a registered function")
        return record

    def _collect(self, record: FunctionRecord) -> None:
        if record.id in self.infos:
            return
        module = ast.parse(record.source)
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "semantic_function")
        info = _FnInfo(record, fn, _ident(record.description, self._names))
        self.infos[record.id] = info
        taken: set[str] = {"ctx"}
        for node in ast.walk(fn):
            if _is_call(node, "load") and node.args and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id == "ctx" and _kw(node, "ctx") is None:
                query = _literal_str(node.args[1]) if len(node.args) > 1 else None
                if query is None:
                    raise CompileError(f"{record.id}: load() with a non-literal description")
                if query not in info.params:
                    info.params[query] = _ident(query, taken)
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                for k in node.value.keys:
                    key = _literal_str(k) if k is not None else None
                    if key and key not in info.return_keys:
                        info.return_keys.append(key)
            elif _is_call(node, "do"):
                self._collect(self._resolve_callee(node))

    # -- pass 2: rewrite each function ---------------------------------------

    def _match(self, query: str, candidates: list[str], what: str) -> str:
        i, sim = embed.best_match(query, candidates)
        if i < 0 or sim < config.LOAD_THRESHOLD:
            raise CompileError(f"{what}: {query!r} matches none of {candidates}")
        return candidates[i]

    def _rewrite(self, info: _FnInfo) -> ast.FunctionDef:
        compiler = self
        result_vars: dict[str, str] = {}   # local variable -> callee function id

        class Rewriter(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):
                if _is_call(node.value, "do") and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    result_vars[node.targets[0].id] = compiler._resolve_callee(node.value).id
                self.generic_visit(node)
                return node

            def visit_Call(self, node: ast.Call):
                self.generic_visit(node)
                if _is_call(node, "do"):
                    return compiler._compile_do(node)
                if _is_call(node, "load"):
                    return compiler._compile_load(info, node, result_vars)
                return node

        tree = Rewriter().visit(ast.parse(ast.unparse(info.tree)).body[0])
        tree.name = info.name
        tree.args = ast.arguments(posonlyargs=[], args=[ast.arg("ctx")] + [ast.arg(a) for a in info.params.values()],
                                  kwonlyargs=[], kw_defaults=[], defaults=[])
        return tree

    def _compile_do(self, call: ast.Call) -> ast.Call:
        callee = self.infos[self._resolve_callee(call).id]
        payloads = _kw(call, "payloads", 2)
        if payloads is None:
            payloads = ast.Dict(keys=[], values=[])
        if not isinstance(payloads, ast.Dict):
            raise CompileError("do() payloads must be a dict literal to compile")
        given = {_literal_str(k): v for k, v in zip(payloads.keys, payloads.values)}
        if None in given:
            raise CompileError("do() payload keys must be string literals to compile")
        keywords = []
        for query, arg in callee.params.items():
            key = self._match(query, list(given), f"{callee.name} input")
            keywords.append(ast.keyword(arg=arg, value=given[key]))
        return ast.Call(func=ast.Name(callee.name, ast.Load()), args=[ast.Name("ctx", ast.Load())],
                        keywords=keywords)

    def _compile_load(self, info: _FnInfo, call: ast.Call, result_vars: dict[str, str]) -> ast.AST:
        source, query = call.args[0], _literal_str(call.args[1]) if len(call.args) > 1 else None
        d_type = _kw(call, "d_type", 2)
        if query is None:
            raise CompileError(f"{info.record.id}: load() with a non-literal description")
        if isinstance(source, ast.Name) and source.id == "ctx" and _kw(call, "ctx") is None:
            value: ast.AST = ast.Name(info.params[query], ast.Load())
        else:
            if isinstance(source, ast.Name) and source.id in result_vars:
                callee = self.infos[result_vars[source.id]]
            elif isinstance(source, ast.Call) and isinstance(source.func, ast.Name) and source.func.id in \
                    {i.name for i in self.infos.values()}:
                callee = next(i for i in self.infos.values() if i.name == source.func.id)
            else:
                raise CompileError(f"{info.record.id}: cannot tell which function produced {ast.unparse(source)}")
            key = self._match(query, callee.return_keys, f"{callee.name} output")
            value = ast.Subscript(value=source, slice=ast.Constant(key), ctx=ast.Load())
        if d_type is None:
            return value
        return ast.Call(func=ast.Name("convert", ast.Load()),
                        args=[ast.Name("ctx", ast.Load()), value, d_type], keywords=[])

    # -- assemble the module -------------------------------------------------

    def compile(self) -> str:
        root = self.cache.functions[self.entry.root_id]
        self._collect(root)
        root_info = self.infos[root.id]
        # Map entry-point parameter descriptions to the root's arguments.
        entry_map = {}
        descriptions = [p.description for p in self.entry.params]
        for query, arg in root_info.params.items():
            entry_map[self._match(query, descriptions, "root input")] = arg

        # Emit callees before callers (reverse discovery order puts the root last).
        bodies = [ast.unparse(ast.fix_missing_locations(self._rewrite(i)))
                  for i in reversed(list(self.infos.values()))]
        header = (
            f'"""Compiled agentic reasoning cache: {self.entry.problem_id or self.cache.name}.\n\n'
            "Produced by arc.compiler through static resolution: every do() is a direct call, every\n"
            "load() is a parameter or a dictionary key. lm()/extract()/convert() remain because they\n"
            'need the target model at invocation time. No embedding model or vector store is used.\n"""\n\n'
            "from arc.runtime import Context, convert, extract, invoke_tool, lm, narrate, smart_convert\n"
        )
        footer = (
            f"\n\n# Entry-point parameter description -> root argument\nENTRY_PARAMETERS = {entry_map!r}\n\n\n"
            "def run(values: dict) -> tuple[dict, Context]:\n"
            '    """Run the compiled root on extracted parameter values (keyed by description)."""\n'
            "    ctx = Context()\n"
            f"    output = {root_info.name}(ctx, **{{ENTRY_PARAMETERS[k]: v for k, v in values.items() "
            "if k in ENTRY_PARAMETERS})\n"
            "    return output, ctx\n"
        )
        return header + "\n\n" + "\n\n\n".join(bodies) + footer


def compile_cache(cache: Cache, entry: EntryPoint) -> str:
    return Compiler(cache, entry).compile()
