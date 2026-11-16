"""The cache: a function registry plus an entry-point registry (paper Sec. 2).

"An agentic reasoning cache is a set of semantic functions combined with a
many-to-one map of problem instances to the natural language descriptions of
the semantic functions that can solve the respective instances."

On disk a cache is a folder:
    registry.json        descriptions, entry points, parameter specs
    functions/<id>.py    one source file per semantic function
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import embed, sandbox
from .runtime import PRIMITIVES


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n].strip("_")
    h = hashlib.sha1(text.encode()).hexdigest()[:6]
    return f"{s}_{h}"


@dataclass
class FunctionRecord:
    id: str
    description: str
    source: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)   # extra descriptions it answers to
    origin: str = ""
    fn: Callable | None = field(default=None, repr=False, compare=False)

    def names(self) -> list[str]:
        return [self.description, *self.aliases]

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("fn")
        d.pop("source")
        return d


@dataclass
class ParamSpec:
    description: str
    format_examples: list[str] = field(default_factory=list)
    original_value: str | None = None
    validity_range: str | None = None


@dataclass
class EntryPoint:
    text: str
    kind: str                     # "template" or "original"
    root_id: str
    params: list[ParamSpec]
    problem_id: str = ""


class Cache:
    def __init__(self, name: str = "cache"):
        self.name = name
        self.functions: dict[str, FunctionRecord] = {}
        self.entry_points: list[EntryPoint] = []
        # Gate cards (DEVIATION, see arc/gatecard.py): problem_id -> card dict.
        self.cards: dict[str, dict] = {}
        # Spec structures (DEVIATION, see arc/spec.py): problem_id -> spec dict.
        self.specs: dict[str, dict] = {}
        # Conventions (label-free builds, arc/labelfree.py): problem_id -> one-line rules for
        # every rounding or convention choice the entry's code makes that changes the answer.
        self.conventions: dict[str, list[str]] = {}
        # Output specs (label-free builds): problem_id -> {output name: {"kind": money | count |
        # measure | text, "unit": str, "decimals": int | None}}. The presenter rounds and
        # converts at the end from this; the cached code returns full precision.
        self.outputs: dict[str, dict[str, dict]] = {}

    # -- functions -----------------------------------------------------------

    def add_function(self, description: str, source: str, inputs=None, outputs=None,
                     origin: str = "", replace_id: str | None = None) -> FunctionRecord:
        fid = replace_id or _slug(description)
        record = FunctionRecord(fid, description, source, list(inputs or []),
                                list(outputs or []), [], origin)
        record.fn = sandbox.load_function(source, PRIMITIVES)
        if replace_id and replace_id in self.functions:
            record.aliases = self.functions[replace_id].aliases
        self.functions[fid] = record
        return record

    def add_alias(self, function_id: str, description: str) -> None:
        rec = self.functions[function_id]
        if description not in rec.names():
            rec.aliases.append(description)

    def _name_table(self) -> list[tuple[str, str]]:
        return [(fid, name) for fid, rec in self.functions.items() for name in rec.names()]

    def find_function(self, query: str) -> tuple[FunctionRecord | None, float]:
        table = self._name_table()
        i, sim = embed.best_match(query, [name for _, name in table])
        return (self.functions[table[i][0]], sim) if i >= 0 else (None, -1.0)

    def search_functions(self, query: str, k: int = 3) -> list[tuple[FunctionRecord, float]]:
        table = self._name_table()
        seen, out = set(), []
        for i, sim in embed.ranked(query, [name for _, name in table]):
            fid = table[i][0]
            if fid not in seen:
                seen.add(fid)
                out.append((self.functions[fid], sim))
            if len(out) == k:
                break
        return out

    # -- entry points --------------------------------------------------------

    def add_entry_point(self, ep: EntryPoint) -> None:
        self.entry_points = [e for e in self.entry_points if e.text != ep.text]
        self.entry_points.append(ep)

    def find_entry_point(self, request: str) -> tuple[EntryPoint | None, float]:
        i, sim = embed.best_match(request, [e.text for e in self.entry_points])
        return (self.entry_points[i], sim) if i >= 0 else (None, -1.0)

    # -- persistence ---------------------------------------------------------

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        if (folder / "functions").exists():
            shutil.rmtree(folder / "functions")
        (folder / "functions").mkdir(parents=True)
        for rec in self.functions.values():
            (folder / "functions" / f"{rec.id}.py").write_text(rec.source)
        data = {
            "name": self.name,
            "functions": [rec.to_json() for rec in self.functions.values()],
            "entry_points": [asdict(e) for e in self.entry_points],
        }
        (folder / "registry.json").write_text(json.dumps(data, indent=2))
        if self.cards:
            (folder / "cards.json").write_text(json.dumps(self.cards, indent=2, ensure_ascii=False))
        if self.specs:
            (folder / "specs.json").write_text(json.dumps(self.specs, indent=2, ensure_ascii=False))
        if self.conventions:
            (folder / "conventions.json").write_text(json.dumps(self.conventions, indent=2, ensure_ascii=False))
        if self.outputs:
            (folder / "outputs.json").write_text(json.dumps(self.outputs, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, folder: Path) -> "Cache":
        folder = Path(folder)
        data = json.loads((folder / "registry.json").read_text())
        cache = cls(data.get("name", folder.name))
        for f in data["functions"]:
            source = (folder / "functions" / f"{f['id']}.py").read_text()
            rec = FunctionRecord(f["id"], f["description"], source, f.get("inputs", []),
                                 f.get("outputs", []), f.get("aliases", []), f.get("origin", ""))
            rec.fn = sandbox.load_function(source, PRIMITIVES)
            cache.functions[rec.id] = rec
        for e in data["entry_points"]:
            params = [ParamSpec(**p) for p in e["params"]]
            cache.entry_points.append(EntryPoint(e["text"], e["kind"], e["root_id"], params,
                                                 e.get("problem_id", "")))
        if (folder / "cards.json").exists():
            cache.cards = json.loads((folder / "cards.json").read_text())
        if (folder / "specs.json").exists():
            cache.specs = json.loads((folder / "specs.json").read_text())
        if (folder / "conventions.json").exists():
            cache.conventions = json.loads((folder / "conventions.json").read_text())
        if (folder / "outputs.json").exists():
            cache.outputs = json.loads((folder / "outputs.json").read_text())
        return cache

    def merge(self, other: "Cache") -> None:
        """Fold a per-sample cache into this master cache (paper Sec. 3.4)."""
        for rec in other.functions.values():
            self.functions[rec.id] = rec
        for ep in other.entry_points:
            self.add_entry_point(ep)
        self.cards.update(other.cards)
        self.specs.update(other.specs)
        self.conventions.update(other.conventions)
        self.outputs.update(other.outputs)

    def summary(self) -> dict[str, Any]:
        return {"functions": len(self.functions), "entry_points": len(self.entry_points)}
