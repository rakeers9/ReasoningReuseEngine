"""Keeping each entry's hit guards (arc/outchecks.py, arc/reference.py) present and current.

Every guard records a fingerprint of the code it was validated against: the entry's root
function plus every function its runs actually called (do() looks functions up by description
at run time, so the set comes from real runs, not from reading the source). A rebuild that
replaces any of that code makes the guard stale:

  - at serve time a stale guard is ignored, like a missing one (it was checked against other
    code, and a disagreement would only send questions to the model for no reason);
  - ensure() rewrites missing and stale guards. Chatbot.learn() calls it after every build, so a
    new entry gets its guards before its first hit.

An entry whose guard can't be validated is still served (owner decision 2026-11-19: it was
already checked at build time, and serving it unguarded is how the cache behaved before guards
existed). Each such entry is reported once per process on stderr, and serve traces carry
"unguarded" with the reason. ensure() doesn't retry a guard that already failed on the same code,
so a hard entry costs one attempt per code version, not one per learn() call.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from . import config

KINDS = {"checks": ("outchecks", "checks.json"), "references": ("reference", "references.json")}

# Mutation tests (eval/outcheck_mutants.py) break the code on purpose and ask whether the
# guard catches it, so they turn the freshness check off. Nothing else should.
CHECK_FRESHNESS = True

_WARNED: set[tuple[str, str]] = set()


def _entry_point(cache, entry_id: str):
    return next((e for e in cache.entry_points if e.problem_id == entry_id and e.kind == "original"), None) or \
        next((e for e in cache.entry_points if e.problem_id == entry_id), None)


def used_functions(cache, entry_id: str) -> list[str]:
    """FREE. Descriptions of the functions the entry's runs call, from running it on its own
    format-example inputs (arc/outchecks.format_combos) and recording every do() lookup."""
    from .outchecks import format_combos
    from .serve import dispatch_root
    ep = _entry_point(cache, entry_id)
    if ep is None:
        return []
    used = {cache.functions[ep.root_id].description} if ep.root_id in cache.functions else set()
    for ex in format_combos(cache, entry_id):
        try:
            _, ctx = dispatch_root(cache, ep.root_id, ex["inputs"])
        except Exception:   # noqa: BLE001
            continue
        used.update(lk.matched for lk in ctx.lookups if lk.kind == "do" and lk.matched)
    return sorted(used)


def fingerprint(cache, entry_id: str, functions: list[str]) -> str:
    """Hash of the entry's root function and the named functions' current source. A function
    that no longer exists hashes as missing, so removing it also changes the fingerprint."""
    ep = _entry_point(cache, entry_id)
    by_desc = {f.description: f.source for f in cache.functions.values()}
    h = hashlib.sha256()
    h.update((ep.root_id if ep else "<no entry>").encode())
    h.update((cache.functions[ep.root_id].source if ep and ep.root_id in cache.functions else "").encode())
    for d in sorted(functions):
        h.update(b"\0" + d.encode() + b"\0" + by_desc.get(d, "<missing>").encode())
    return h.hexdigest()[:16]


def stamp(cache, entry_id: str, rec: dict) -> dict:
    """Record what code a just-written guard was validated against."""
    fns = used_functions(cache, entry_id)
    rec["code"] = {"functions": fns, "hash": fingerprint(cache, entry_id, fns)}
    return rec


def current(cache, entry_id: str, rec: dict) -> bool:
    if not CHECK_FRESHNESS:
        return True
    code = rec.get("code")
    if not code:                     # written before fingerprints existed: can't tell, keep using it
        return True
    return fingerprint(cache, entry_id, code.get("functions", [])) == code.get("hash")


def usable(cache, entry_id: str, kind: str) -> tuple[dict | None, str]:
    """(the guard record, "") if it can run on this entry's current code, else (None, why)."""
    rec = (getattr(cache, kind, None) or {}).get(entry_id)
    if not rec or not rec.get("source"):
        why = "no guard written"
    elif rec.get("status") != "ok":
        why = f"guard failed validation ({rec.get('status')})"
    elif not current(cache, entry_id, rec):
        why = "guard is stale (entry code changed since it was written)"
    else:
        return rec, ""
    warn(entry_id, kind, why)
    return None, why


def warn(entry_id: str, kind: str, why: str) -> None:
    if (entry_id, kind) not in _WARNED:
        _WARNED.add((entry_id, kind))
        print(f"[guards] {entry_id}: served without {kind}: {why}", file=sys.stderr)


def needs_writing(cache, entry_id: str, kind: str) -> str:
    """Why this guard should be (re)written, or "" if it shouldn't. A guard that failed
    validation on the same code isn't retried."""
    rec = (getattr(cache, kind, None) or {}).get(entry_id)
    if not rec:
        return "missing"
    code = rec.get("code") or {}
    same_code = code and fingerprint(cache, entry_id, code.get("functions", [])) == code.get("hash")
    if not same_code and code:
        return "stale"
    return ""


def ensure(cache, folder: Path | None = None, tag: str = "guards") -> list[dict]:
    """PAID (about $0.03 per guard written). Write every missing or stale guard for every entry
    in the cache, then save checks.json / references.json into `folder`. Returns one row per
    guard written: {"entry", "kind", "why", "status"}."""
    from . import outchecks, reference
    writers = {"checks": outchecks.write, "references": reference.write}
    rows = []
    for entry_id in sorted({e.problem_id for e in cache.entry_points}):
        for kind in KINDS:
            why = needs_writing(cache, entry_id, kind)
            if not why:
                continue
            try:
                rec = writers[kind](cache, entry_id, tag=f"{tag}:{kind}")
            except Exception as e:   # noqa: BLE001 - an outage leaves the entry unguarded, never broken
                print(f"[guards] {entry_id}: could not write {kind}: {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
                rows.append({"entry": entry_id, "kind": kind, "why": why, "status": "error"})
                continue
            stamp(cache, entry_id, rec)
            rows.append({"entry": entry_id, "kind": kind, "why": why, "status": rec.get("status")})
            if rec.get("status") != "ok":
                print(f"[guards] {entry_id}: {kind} failed validation ({rec.get('status')}); "
                      "the entry is served without it", file=sys.stderr)
    if folder is not None and rows:
        folder = Path(folder)
        for kind, (_, fname) in KINDS.items():
            data = getattr(cache, kind, None) or {}
            if data:
                (folder / fname).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def report(cache) -> dict[str, dict[str, str]]:
    """{entry: {kind: "ok" | why not}} for every entry: what each hit is protected by."""
    out = {}
    for entry_id in sorted({e.problem_id for e in cache.entry_points}):
        out[entry_id] = {}
        for kind in KINDS:
            rec = (getattr(cache, kind, None) or {}).get(entry_id)
            if not rec or not rec.get("source"):
                out[entry_id][kind] = "missing"
            elif rec.get("status") != "ok":
                out[entry_id][kind] = f"failed ({rec.get('status')})"
            elif not current(cache, entry_id, rec):
                out[entry_id][kind] = "stale"
            else:
                out[entry_id][kind] = "ok"
    return out
