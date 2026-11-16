"""Cache-worthiness: which task families in the backlog are worth building an entry for.

DEVIATION from the paper, which builds an entry for every problem it is given. In
production a build costs real money and most questions never come back, so we only
build for families that recur. The rule is ski-rental break-even (Karlin et al. 1988):

    build a family once  (recent saving the cache would have made)  >=  build cost
    and it has been seen at least FAMILY_MIN_SIGHTINGS times

Nothing in the rule is a fixed question count. Every input is measured:
  saving per question = what the customer's production answer cost (their model, their
                        tokens, arc/pricing.py) - what a cache answer costs us
  build cost          = what our builds actually cost, divided by the share that pass the
                        certificate (a rejected build is money spent with no entry)
  serve rate          = share of the family's questions the gate actually serves
So an Opus customer reaches break-even after a couple of questions and a Haiku customer
after dozens, from the same code. Whatever happens next, this rule never spends more
than twice what the best choice in hindsight would have (the ski-rental guarantee).

"Recent" means exponentially decayed: a question FAMILY_HALF_LIFE_DAYS old counts half.

Families are found by meaning: each question, with its numbers masked, is embedded, and
the families whose centroid is within FAMILY_SIMILARITY are candidates.
  FAMILY_CHECK "embed": join the closest candidate. Near-misses (same wording, different
                        task) embed as close as true variants, so they join too.
  FAMILY_CHECK "clef" (default): Clef must also say the question fits the candidate's
                        anchor, its most typical confirmed question (P >= FAMILY_CLEF_THRESHOLD),
                        or the question starts its own family. One Clef call per new
                        question, covering the top FAMILY_CLEF_TOP_K candidates. If Clef
                        fails, the question joins the closest candidate UNCONFIRMED: it is
                        counted but never built from.
Builds only use confirmed members.

State is kept in families.json and updated incrementally from the backlog.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import backlog, config, pricing

DAY = 86400.0
_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NEGATIVE = {"bad", "down", "negative", "thumbs_down", "wrong", "-1", "0"}


def mask_numbers(text: str) -> str:
    """'10.0 mol at 27°C' -> '● mol at ●°C': families are about the task, not the values."""
    return _NUMBER.sub("●", text)


def is_negative(feedback) -> bool:
    if feedback is None:
        return False
    return str(feedback).strip().lower() in _NEGATIVE


@dataclass
class Family:
    id: str
    label: str                                  # first question seen, shortened
    status: str = "watching"                    # watching | built | rejected
    built_entry: str | None = None
    note: str = ""
    size: int = 0                               # questions assigned so far
    ref_time: float = 0.0                       # decayed sums below are as of this time
    sightings: float = 0.0                      # decayed count of source-model answers
    savings: float = 0.0                        # decayed $ the cache would have saved
    hits: float = 0.0                           # decayed count of cache answers (built only)
    priced: int = 0
    unpriced: int = 0
    cache_hits_total: int = 0
    complaints_total: int = 0                   # negative feedback on cache answers
    serve_rate: float | None = None             # measured in shadow mode; None = default
    first_seen: float = 0.0
    last_seen: float = 0.0
    models: dict = field(default_factory=dict)
    unpriced_models: dict = field(default_factory=dict)
    recent_costs: list = field(default_factory=list)   # last 200 priced answers, for the median
    examples: list = field(default_factory=list)       # up to 3 record ids
    member_ids: list = field(default_factory=list)     # every record id assigned here, in arrival order
    last_source_id: str | None = None                  # newest record the source model answered
    last_source_time: float = 0.0
    tried_record: str | None = None                    # record a build was last tried from (mismatch)
    record_notes: dict = field(default_factory=dict)   # record id -> note, e.g. a failed caching try
    anchor_id: str | None = None                       # most typical confirmed question (Clef compares to it)
    anchor_text: str = ""
    anchor_vec: list = field(default_factory=list)
    confirmed_ids: list = field(default_factory=list)  # members Clef (or the embed rule) confirmed
    recent_texts: list = field(default_factory=list)   # latest confirmed questions (extra comparisons)
    build_checks: list = field(default_factory=list)   # last build-ready checks: record id, P, passed
    failed_builds: int = 0                             # builds that failed their own tests (cap: FAMILY_MAX_FAILED_BUILDS)
    centroid: list = field(default_factory=list)
    members: int = 0                                   # vectors averaged into the centroid

    # ---- decayed sums ----
    def _add(self, attr: str, amount: float, t: float, half_life: float) -> None:
        if t >= self.ref_time:
            k = 0.5 ** ((t - self.ref_time) / half_life)
            self.sightings *= k
            self.savings *= k
            self.hits *= k
            self.ref_time = t
            setattr(self, attr, getattr(self, attr) + amount)
        else:   # an older record arriving late counts for less
            setattr(self, attr, getattr(self, attr) + amount * 0.5 ** ((self.ref_time - t) / half_life))

    def decayed(self, attr: str, now: float, half_life: float) -> float:
        return getattr(self, attr) * 0.5 ** (max(0.0, now - self.ref_time) / half_life)


@dataclass
class Readiness:
    family: str
    label: str
    status: str
    verdict: str            # build | wait | built | stale | complaints | rejected
    meter: float            # recent saving x serve rate, $
    threshold: float        # effective build cost, $
    sightings: float
    size: int
    median_answer_cost: float | None
    saving_per_question: float | None
    questions_to_go: float | None
    hits: float
    unpriced: int
    unpriced_models: dict
    models: dict
    why: str


class FamilyTracker:
    """Assigns backlog records to families and keeps each family's break-even meter."""

    def __init__(self, path: Path | None = None, embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.path = path or config.WORK_DIR / "families.json"
        self._embed_fn = embed_fn
        self.families: dict[str, Family] = {}
        self.ingested_lines = 0        # backlog lines already read
        self.seen_ids: set[str] = set()
        self.latest = 0.0              # newest record timestamp seen
        self.skipped = 0               # records marked not reusable (chit-chat)
        self.assignments: dict[str, dict] = {}   # record id -> how it was placed (for review)
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.families = {f["id"]: Family(**f) for f in raw["families"]}
            self.ingested_lines = raw.get("ingested_lines", 0)
            self.seen_ids = set(raw.get("seen_ids", []))
            self.latest = raw.get("latest", 0.0)
            self.skipped = raw.get("skipped", 0)
            self.assignments = raw.get("assignments", {})

    # ---- persistence ----
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "settings": {k: getattr(config, k) for k in ("FAMILY_SIMILARITY", "FAMILY_HALF_LIFE_DAYS",
                                                         "FAMILY_MIN_SIGHTINGS", "CACHE_ANSWER_COST")},
            "ingested_lines": self.ingested_lines, "latest": self.latest, "skipped": self.skipped,
            "assignments": self.assignments,
            "seen_ids": sorted(self.seen_ids),
            "families": [asdict(f) for f in self.families.values()]}, indent=1))

    # ---- assignment ----
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is None:
            from . import embed
            self._embed_fn = embed.embedder().encode
        return np.asarray(self._embed_fn(texts), dtype=np.float32)

    def _new_family(self, label: str, fid: str | None = None) -> Family:
        fid = fid or f"f{len(self.families) + 1:04d}"
        while fid in self.families:
            fid += "_"
        fam = Family(fid, label[:120])
        self.families[fid] = fam
        return fam

    def assign(self, record: dict, vector: np.ndarray | None = None) -> Family:
        """The family a record belongs to, creating one if nothing is close enough."""
        if record.get("served_by"):
            for f in self.families.values():
                if f.built_entry == record["served_by"]:
                    return f
            fam = self._new_family(record["question"], f"entry:{record['served_by']}")
            fam.status, fam.built_entry = "built", record["served_by"]
            return fam
        if record.get("family"):
            return self.families.get(record["family"]) or self._new_family(record["question"], record["family"])
        v = self._embed([mask_numbers(record["question"])])[0] if vector is None else vector
        ranked = sorted(((f, float(np.dot(np.asarray(f.centroid, dtype=np.float32), v)))
                         for f in self.families.values() if f.centroid), key=lambda x: -x[1])
        cut = config.FAMILY_CLEF_SIMILARITY if config.FAMILY_CHECK == "clef" else config.FAMILY_SIMILARITY
        cands = [(f, sim) for f, sim in ranked if sim >= cut]
        how = {"candidates": {f.id: round(sim, 4) for f, sim in cands[:config.FAMILY_CLEF_TOP_K]}}
        best, confirmed = None, True
        if cands and config.FAMILY_CHECK == "clef":
            top = cands[:config.FAMILY_CLEF_TOP_K]
            try:
                from . import clef
                anchors = {}
                for f, _ in top:   # the anchor first, then the most recent other confirmed questions
                    texts = [f.anchor_text or f.label] + [t for t in reversed(f.recent_texts)
                                                          if t != (f.anchor_text or f.label)]
                    for k, t in enumerate(texts[:max(1, config.FAMILY_CLEF_ANCHORS)]):
                        anchors[f"{f.id}#{k}"] = t
                probs = clef.fits(record["question"], anchors, tag=f"family:{record['id']}")
                per_family = {}
                for key, p in probs.items():
                    fid = key.split("#")[0]
                    per_family[fid] = max(per_family.get(fid, 0.0), p)
                how["clef"] = {k: round(p, 4) for k, p in per_family.items()}
                f, p = max(((f, per_family.get(f.id, 0.0)) for f, _ in top), key=lambda x: x[1])
                if p >= config.FAMILY_CLEF_THRESHOLD:
                    best, how["joined_by"] = f, "clef"
                else:
                    how["joined_by"] = "new: clef said no"
            except Exception as e:   # Clef down: count it with the closest family, never build from it
                best, confirmed = cands[0][0], False
                how.update(joined_by="closest, unconfirmed (clef error)", error=str(e)[:200])
        elif cands:
            best, how["joined_by"] = cands[0][0], "embedding"
        else:
            how["joined_by"] = "new: nothing close"
        if best is None:
            best = self._new_family(record["question"])
        if confirmed:
            self._add_confirmed(best, record, v)
        how["family"] = best.id
        self.assignments[record["id"]] = how
        return best

    def _add_confirmed(self, fam: Family, record: dict, v: np.ndarray) -> None:
        """A confirmed member moves the centroid, and becomes the anchor if it is closer to the
        new centroid than the current anchor (greedy: older members' vectors are not kept, so
        they are not re-scored)."""
        c = np.asarray(fam.centroid, dtype=np.float32) if fam.centroid else np.zeros_like(v)
        c = (c * fam.members + v) / (fam.members + 1)
        c = c / (np.linalg.norm(c) or 1.0)
        fam.centroid = [round(float(x), 5) for x in c]
        fam.members += 1
        fam.confirmed_ids.append(record["id"])
        fam.recent_texts = (fam.recent_texts + [record["question"]])[-max(1, config.FAMILY_CLEF_ANCHORS):]
        old = np.asarray(fam.anchor_vec, dtype=np.float32) if fam.anchor_vec else None
        if old is None or float(np.dot(v, c)) > float(np.dot(old, c)):
            fam.anchor_id, fam.anchor_text = record["id"], record["question"]
            fam.anchor_vec = [round(float(x), 5) for x in v]

    # ---- metering ----
    def observe(self, record: dict, vector: np.ndarray | None = None) -> Family | None:
        """Count one backlog record. Returns its family, or None if the record was already seen."""
        if record["id"] in self.seen_ids:
            return None
        self.seen_ids.add(record["id"])
        if record.get("reusable") is False:   # chit-chat etc.: never a family
            self.skipped += 1
            return None
        fam = self.assign(record, vector)
        t = float(record["timestamp"])
        half_life = config.FAMILY_HALF_LIFE_DAYS * DAY
        self.latest = max(self.latest, t)
        fam.size += 1
        fam.first_seen = min(fam.first_seen or t, t)
        fam.last_seen = max(fam.last_seen, t)
        model = record.get("model") or "unknown"
        fam.models[model] = fam.models.get(model, 0) + 1
        if len(fam.examples) < 3:
            fam.examples.append(record["id"])
        fam.member_ids.append(record["id"])
        if record.get("served_by"):   # the cache answered: usage and complaints, for pruning
            fam.cache_hits_total += 1
            fam._add("hits", 1.0, t, half_life)
            if is_negative(record.get("feedback")):
                fam.complaints_total += 1
            return fam
        fam._add("sightings", 1.0, t, half_life)
        if fam.last_source_id is None or t >= fam.last_source_time:
            fam.last_source_id, fam.last_source_time = record["id"], t
        cost = pricing.answer_cost(record)
        if cost is None:
            fam.unpriced += 1
            fam.unpriced_models[model] = fam.unpriced_models.get(model, 0) + 1
        else:
            fam.priced += 1
            fam.recent_costs = (fam.recent_costs + [round(cost, 8)])[-200:]
            fam._add("savings", max(0.0, cost - config.CACHE_ANSWER_COST), t, half_life)
        return fam

    def ingest(self, path: Path | None = None, batch: int = 64) -> int:
        """Read new backlog lines since the last ingest. Returns how many records were added."""
        path = path or backlog.default_path()
        if not path.exists():
            return 0
        lines = path.read_text().splitlines()
        new = lines[self.ingested_lines:]
        records = []
        for line in new:
            try:
                records.append(backlog.normalise(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
        records = [r for r in records if r["id"] not in self.seen_ids]
        need = [r for r in records if not r.get("served_by") and not r.get("family") and r.get("reusable") is not False]
        vectors: dict[str, np.ndarray] = {}
        for i in range(0, len(need), batch):   # embed in batches, assign in order
            chunk = need[i:i + batch]
            for r, v in zip(chunk, self._embed([mask_numbers(r["question"]) for r in chunk])):
                vectors[r["id"]] = v
        before = len(self.seen_ids)
        for r in records:
            self.observe(r, vectors.get(r["id"]))
        added = len(self.seen_ids) - before
        self.ingested_lines = len(lines)
        return added

    def members_of(self, family_id: str, path: Path | None = None, served: bool | None = None,
                   confirmed: bool = False) -> list[dict]:
        """Backlog records in a family, in backlog order. served=False: only source-model
        answers (what a build/certificate needs); True: only cache answers; None: all.
        confirmed=True: only members Clef (or the embed rule) confirmed."""
        fam = self.families[family_id]
        ids = set(fam.confirmed_ids if confirmed else fam.member_ids)
        out = [r for r in backlog.read(path) if r["id"] in ids]
        if served is not None:
            out = [r for r in out if bool(r.get("served_by")) is served]
        return out

    # ---- lifecycle ----
    def mark_built(self, family_id: str, entry_id: str) -> None:
        f = self.families[family_id]
        f.status, f.built_entry, f.note = "built", entry_id, f"built as {entry_id}"
        f.tried_record = None

    def mark_rejected(self, family_id: str, reason: str) -> None:
        f = self.families[family_id]
        f.status, f.note = "rejected", reason[:200]

    def set_serve_rate(self, family_id: str, rate: float) -> None:
        self.families[family_id].serve_rate = max(0.0, min(1.0, rate))

    # ---- decisions ----
    def readiness(self, now: float | None = None, build_cost: float | None = None) -> list[Readiness]:
        if now is None:   # evaluate as of the newest record, so replaying an old log works
            now = self.latest if self.seen_ids else time.time()
        half_life = config.FAMILY_HALF_LIFE_DAYS * DAY
        threshold = build_cost if build_cost is not None else effective_build_cost()["effective"]
        rows = []
        for f in self.families.values():
            serve = f.serve_rate if f.serve_rate is not None else config.SERVE_RATE_DEFAULT
            meter = f.decayed("savings", now, half_life) * serve
            sightings = f.decayed("sightings", now, half_life)
            hits = f.decayed("hits", now, half_life)
            med = statistics.median(f.recent_costs) if f.recent_costs else None
            per_q = max(0.0, med - config.CACHE_ANSWER_COST) * serve if med is not None else None
            to_go = (max(0.0, threshold - meter) / per_q) if per_q else None
            if f.status == "rejected":
                verdict, why = "rejected", f.note or "build or certificate failed"
            elif f.tried_record is not None and f.tried_record == f.last_source_id:
                verdict, why = "wait", (f"{f.note or MISMATCH_NOTE}; waiting for a newer question "
                                        "to try again")
            elif f.status == "built":
                if f.cache_hits_total >= 5 and f.complaints_total / f.cache_hits_total > 0.2:
                    verdict, why = "complaints", f"{f.complaints_total}/{f.cache_hits_total} cache answers got negative feedback"
                elif hits < 1.0 and now - f.last_seen > half_life:
                    verdict, why = "stale", f"no cache use in {(now - f.last_seen) / DAY:.0f} days"
                else:
                    verdict, why = "built", f"{hits:.1f} recent cache answers"
            elif meter >= threshold and sightings >= 0.99 * config.FAMILY_MIN_SIGHTINGS:   # 3 asked minutes apart = 2.9999
                verdict, why = "build", f"recent saving ${meter:.4f} >= build cost ${threshold:.4f}"
            elif meter >= threshold:
                verdict, why = "wait", (f"pays already, but seen only {sightings:.1f} times "
                                        f"(need {config.FAMILY_MIN_SIGHTINGS:g} to show it recurs)")
            elif per_q is None:
                verdict, why = "wait", ("no priced answers yet: add token counts, cost, or a price for "
                                        + ", ".join(f.unpriced_models)) if f.unpriced_models else "no priced answers yet"
            elif per_q == 0:
                verdict, why = "wait", (f"answers here cost ${med:.5f}, no more than a cache answer "
                                        f"(${config.CACHE_ANSWER_COST:.5f}): caching saves nothing")
            else:
                verdict, why = "wait", f"${meter:.4f} of ${threshold:.4f}; about {math.ceil(to_go)} more questions"
            rows.append(Readiness(f.id, f.label, f.status, verdict, round(meter, 6), round(threshold, 6),
                                  round(sightings, 2), f.size, med, per_q,
                                  None if to_go is None else round(to_go, 1), round(hits, 2), f.unpriced,
                                  dict(f.unpriced_models), dict(f.models), why))
        order = {"build": 0, "complaints": 1, "stale": 2, "wait": 3, "built": 4, "rejected": 5}
        rows.sort(key=lambda r: (order[r.verdict], -(r.meter / r.threshold if r.threshold else 0), -r.size))
        return rows


# ---- acting on the report ----

def build_ready(tracker: FamilyTracker, build_fn: Callable | None = None, max_builds: int | None = None,
                path: Path | None = None, can_start: Callable[[], bool] | None = None) -> list[tuple[str, object]]:
    """Build every family whose verdict is "build", best first. PAID when build_fn is real.

    build_fn(family_id, records) gets the family's source-model records and returns an
    object with: accepted, entry_id, reason, cost (full spend, rejected attempts too),
    parts (cost by step), and optionally serve_rate and status. Default:
    arc.labelfree.build_family.
    status "need_more_data": built (or not yet built) but too few family questions to
    certify it; the family stays watching and is retried on newer records. Its spend is
    logged as pending: it counts toward cost per cached entry but not as an attempt.
    Each result is applied (mark_built / mark_rejected, record_build_cost, serve rate) and
    saved immediately, so a crash part-way loses nothing already paid for.
    can_start(): asked before EACH build (e.g. "does the budget left still cover a worst-case
    build?"); False stops the batch. Checking per build, against what earlier builds really
    cost, lets cheap builds go ahead instead of reserving the worst case for all of them."""
    if build_fn is None:
        from .labelfree import build_family as build_fn   # label-free build path (separate module)
    done = []
    for row in [r for r in tracker.readiness() if r.verdict == "build"][:max_builds]:
        if can_start is not None and not can_start():
            break
        records = tracker.members_of(row.family, path, served=False, confirmed=True)
        if config.FAMILY_BUILD_CHECK == "clef" and records:
            blocked = _build_check(tracker, row.family, records)
            if blocked is not None:
                done.append((row.family, blocked))
                continue
        try:
            result = build_fn(row.family, records)
        except Exception as e:   # one broken family must not stop the batch
            # A crash is a failure of the build machinery (a cut-off model reply, a network
            # error), not evidence the family is wrong: wait for a newer question and try again,
            # never reject the family for good.
            fam = tracker.families[row.family]
            newest = max(records, key=lambda r: r.get("timestamp") or 0)["id"] if records else fam.last_source_id
            why = f"build error: {type(e).__name__}: {str(e)[:120]}"
            fam.tried_record, fam.note = newest, f"{why}; waiting for a newer question"[:200]
            if newest:
                fam.record_notes[newest] = why[:160]
            tracker.save()
            done.append((row.family, None))
            continue
        status = getattr(result, "status", None) or ("accepted" if result.accepted else "rejected")
        fam = tracker.families[row.family]
        if status == "need_more_data":
            fam.note = f"pending: {result.reason or 'needs more family questions'}"[:200]
        elif status == "mismatch":   # stays watching; tried again once a newer question arrives
            tried = getattr(result, "record_id", None) or fam.last_source_id
            why = (result.reason or MISMATCH_NOTE)[:160]   # e.g. a failed fit check says so
            if getattr(result, "failure", None) == "code":   # the built code was wrong: counts toward the cap
                fam.failed_builds += 1
            if fam.failed_builds >= config.FAMILY_MAX_FAILED_BUILDS:
                tracker.mark_rejected(row.family, f"{why} ({fam.failed_builds} failed builds)")
            else:
                fam.tried_record = tried
                fam.note = f"{why} (record {tried})"
                if tried:
                    fam.record_notes[tried] = why
        elif status == "accepted" and result.accepted and result.entry_id:
            tracker.mark_built(row.family, result.entry_id)
        else:   # a build that failed: wait and retry; only CODE failures count toward the cap
            failure = getattr(result, "failure", None) or "code"   # unknown "rejected": treat as code
            if failure == "code":
                fam.failed_builds += 1
            why = (result.reason or "not accepted")[:160]
            if failure == "code" and fam.failed_builds >= config.FAMILY_MAX_FAILED_BUILDS:
                tracker.mark_rejected(row.family, f"{why} ({fam.failed_builds} failed builds)")
            else:
                tried = getattr(result, "record_id", None) or fam.last_source_id
                fam.tried_record = tried
                fam.note = f"{why}; build {fam.failed_builds} of {config.FAMILY_MAX_FAILED_BUILDS} failed, " \
                           "waiting for a newer question"[:200]
                if tried:
                    fam.record_notes[tried] = why
        if getattr(result, "serve_rate", None) is not None:
            tracker.set_serve_rate(row.family, result.serve_rate)
        if status != "need_more_data" or (result.cost or 0) > 0:
            record_build_cost(result.entry_id or row.family, result.cost or 0.0, status == "accepted",
                              getattr(result, "parts", None), pending=status == "need_more_data")
        tracker.save()
        done.append((row.family, result))
    return done


MISMATCH_NOTE = "caching tried but answer didn't match"


def _most_typical(tracker: FamilyTracker, fam: Family, records: list[dict]) -> dict:
    """The record closest to the family's centroid (numbers masked), i.e. its usual wording.
    The latest member can be an unusual wording (terse notes), which says little about whether
    a question belongs; the most typical one is the fair comparison. Local embeddings, no cost."""
    if len(records) == 1 or not fam.centroid:
        return records[-1]
    vecs = tracker._embed([mask_numbers(r["question"]) for r in records])
    c = np.asarray(fam.centroid, dtype=np.float32)
    return records[int(np.argmax(vecs @ c))]


def _build_check(tracker: FamilyTracker, family_id: str, records: list[dict]):
    """None if the newest question may be built from; otherwise a "not_ready" result, and
    the family waits for a newer question (same rule as a mismatch)."""
    from types import SimpleNamespace
    from . import clef, llm
    fam = tracker.families[family_id]
    newest = max(records, key=lambda r: r.get("timestamp") or 0)
    others = [r for r in records if r["id"] != newest["id"]]
    if fam.anchor_id and fam.anchor_id != newest["id"] and fam.anchor_text:
        reference, ref_name = fam.anchor_text, "its family's anchor"
    elif others:   # the newest question is the anchor: compare it with the family's most typical
        reference, ref_name = _most_typical(tracker, fam, others)["question"], "the family's most typical other question"
    else:
        reference = None
    before = len(llm.LEDGER.records)
    try:
        if reference is None:
            raise llm.LLMError("no other confirmed question to compare with")
        p = clef.build_check(newest["question"], reference, tag=f"{family_id}:build_check")
        why = None if p >= config.CLEF_THRESHOLD else \
            f"newest question not build-ready (P {p:.2f} vs {ref_name}, need {config.CLEF_THRESHOLD:g})"
    except llm.LLMError as e:
        p, why = None, f"build check could not run ({str(e)[:80]}); waiting"
    cost = llm.LEDGER.cost(since=before)
    fam.build_checks = (fam.build_checks + [{"record_id": newest["id"], "p": p, "passed": why is None}])[-20:]
    if why is None:
        return None
    fam.tried_record, fam.note = newest["id"], f"{why}; waiting for a newer question"[:200]
    fam.record_notes[newest["id"]] = why
    if cost:
        record_build_cost(family_id, cost, False, {"build_check": cost}, pending=True)
    tracker.save()
    return SimpleNamespace(status="not_ready", accepted=False, entry_id=None, reason=why, cost=cost,
                           parts={"build_check": cost}, serve_rate=None, record_id=newest["id"])


# ---- build cost, measured ----

def _build_log() -> Path:
    return config.WORK_DIR / "build_costs.jsonl"


def record_build_cost(problem_id: str, cost: float, accepted: bool, parts: dict | None = None,
                      pending: bool = False) -> None:
    """Log what one entry cost to add (explainer + build + certificate) and whether it was kept.
    pending=True: money spent on an entry not yet decided (counted in spend, not as an attempt)."""
    path = _build_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"problem": problem_id, "timestamp": time.time(), "cost": round(cost, 6),
                            "accepted": bool(accepted), "pending": bool(pending), "parts": parts or {}}) + "\n")


def _is_build_attempt(row: dict) -> bool:
    """A logged attempt counts only if something was BUILT. A re-solve mismatch (parts has no
    "build" step) is about the chatbot's answer, not the cost of building: its spend still
    counts, but it is not an attempt. Rows without a parts breakdown count as attempts."""
    parts = row.get("parts") or {}
    return not parts or "build" in parts


def effective_build_cost(min_builds: int | None = None) -> dict:
    """$ per entry that actually gets cached: all build spending / builds accepted.

    A rejected build is money spent with no entry to show for it, so the price of one
    cached entry is the attempt cost divided by the accept rate. Until BUILD_COST_MIN_BUILDS
    real build attempts are logged, the configured estimate is used: a handful of early
    failures (chat run 2026-11-15: 1 accepted of 7) otherwise sets the bar for every family."""
    min_builds = config.BUILD_COST_MIN_BUILDS if min_builds is None else min_builds
    path = _build_log()
    every = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    rows = [r for r in every if not r.get("pending") and _is_build_attempt(r)]   # decided build attempts
    if len(rows) >= min_builds and any(r["accepted"] for r in rows):
        per_attempt = sum(r["cost"] for r in every) / len(rows)   # pending spend is still spend
        rate = sum(r["accepted"] for r in rows) / len(rows)
        source = f"measured over {len(rows)} builds"
    else:
        per_attempt, rate = config.BUILD_COST_DEFAULT, config.BUILD_ACCEPT_RATE_DEFAULT
        source = f"defaults ({len(rows)} builds logged, need {min_builds})"
    return {"per_attempt": per_attempt, "accept_rate": rate, "effective": per_attempt / rate, "source": source}


def format_report(rows: list[Readiness], cost: dict, limit: int = 30) -> str:
    out = [f"Build cost per cached entry: ${cost['effective']:.4f} "
           f"(${cost['per_attempt']:.4f} per attempt / {cost['accept_rate']:.0%} accepted; {cost['source']})",
           f"Counts decay with a {config.FAMILY_HALF_LIFE_DAYS:g}-day half-life; minimum "
           f"{config.FAMILY_MIN_SIGHTINGS:g} sightings; a cache answer costs ${config.CACHE_ANSWER_COST:.4f}.",
           ""]
    counts = Counter(r.verdict for r in rows)
    out.append("  ".join(f"{k}: {counts[k]}" for k in ("build", "wait", "built", "stale", "complaints", "rejected")
                         if counts[k]))
    out.append("")
    out.append(f"{'family':10s} {'verdict':10s} {'seen':>6s} {'$/answer':>9s} {'meter':>9s} {'need':>9s}  label / why")
    for r in rows[:limit]:
        cost_s = f"{r.median_answer_cost:.5f}" if r.median_answer_cost is not None else "?"
        out.append(f"{r.family[:10]:10s} {r.verdict:10s} {r.sightings:6.1f} {cost_s:>9s} "
                   f"{r.meter:9.4f} {r.threshold:9.4f}  {r.label[:60]}")
        out.append(f"{'':48s}  -> {r.why}")
    if len(rows) > limit:
        out.append(f"... {len(rows) - limit} more families")
    return "\n".join(out)
