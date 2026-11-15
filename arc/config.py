"""Central settings for the ARC replication.

Every value the paper leaves unspecified ("configurable") lives here, so it is
easy to see which choices are ours rather than the paper's.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHES_DIR = ROOT / "caches"
RUNS_DIR = ROOT / "runs"
PROBLEMS_DIR = ROOT / "problems"


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from the project's .env file (no extra dependency)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# --- Profile: which models fill the big roles, and where outputs go ---
# "haiku" (default): every role uses Haiku 5.5, ~40x cheaper, for fast iteration.
# "paper": Opus 5.5 for source/builder/judges/generator, as in the paper.
# Each profile writes to its own folders so runs never mix caches:
#   paper -> runs/<problem>/ and caches/<name>/ (the original layout)
#   haiku -> runs/haiku/<problem>/ and caches/haiku/<name>/
# PROJECT RULE (2026-11-14): Haiku only, in every role. llm.call refuses any Opus model
# unless ARC_ALLOW_OPUS=1 is set, so the "paper" profile only runs with that opt-in.
PROFILE = os.environ.get("ARC_PROFILE", "haiku")
PROFILE_BIG_MODEL = {"paper": "claude-opus-5-5", "haiku": "claude-haiku-5-5"}


def namespace_for(profile: str) -> str:
    return os.environ.get("ARC_NAMESPACE", "" if profile == "paper" else profile)


NAMESPACE = namespace_for(PROFILE)
WORK_DIR = RUNS_DIR / NAMESPACE if NAMESPACE else RUNS_DIR      # per-problem outputs
CACHE_ROOT = CACHES_DIR / NAMESPACE if NAMESPACE else CACHES_DIR  # caches written by this profile


def problem_dir(problem_id: str) -> Path:
    return WORK_DIR / problem_id


def cache_dir(name: str) -> Path:
    """This profile's cache folder. Only the paper's hand-typed caches (paper_*) are shared
    across profiles, so one profile never silently runs another profile's built cache."""
    own = CACHE_ROOT / name
    if name.startswith("paper_") and not (own / "registry.json").exists() and \
            (CACHES_DIR / name / "registry.json").exists():
        return CACHES_DIR / name
    return own


# --- Models (paper: Opus 4.5 for S/B/J, Haiku 4.5 then Llama 3.2 1B for T) ---
_BIG = PROFILE_BIG_MODEL.get(PROFILE, "claude-haiku-5-5")
SOURCE_MODEL = os.environ.get("ARC_SOURCE_MODEL", _BIG)        # S: solves the original problem
BUILDER_MODEL = os.environ.get("ARC_BUILDER_MODEL", _BIG)      # B: turns the transcript into a cache entry
JUDGE_MODEL = os.environ.get("ARC_JUDGE_MODEL", _BIG)          # J: verifies outputs during build and eval
GENERATOR_MODEL = os.environ.get("ARC_GENERATOR_MODEL", _BIG)  # G: writes perturbation variants (paper: G = S)
TARGET_MODEL = os.environ.get("ARC_TARGET_MODEL", "claude-haiku-5-5")    # T: the small model used at invocation time
# E: writes the worked solution at build time from (question, production answer) only, so the
# production model never has to show reasoning (DEVIATION: the paper builds from S's own trace).
EXPLAINER_MODEL = os.environ.get("ARC_EXPLAINER_MODEL", SOURCE_MODEL)

# Effort replaces the paper's fixed 4096-token thinking budget (budgets are
# rejected by current Opus models).
SOURCE_EFFORT = os.environ.get("ARC_SOURCE_EFFORT", "high")
BUILDER_EFFORT = os.environ.get("ARC_BUILDER_EFFORT", "high")
JUDGE_EFFORT = os.environ.get("ARC_JUDGE_EFFORT", "medium")
GENERATOR_EFFORT = os.environ.get("ARC_GENERATOR_EFFORT", "high")

# --- Embeddings (paper: EmbeddingGemma 300M, 768-dim, FAISS flat index) ---
EMBED_MODEL = os.environ.get("ARC_EMBED_MODEL", "google/embeddinggemma-300m")
EMBED_FALLBACK = "sentence-transformers/all-mpnet-base-v2"

# --- Similarity thresholds (paper: "configurable", no values given) ---
# Measured with EmbeddingGemma (STS prompt): true paraphrases of a description
# score >= 0.84 while unrelated descriptions reach ~0.72; correct function calls
# score ~0.94 vs <= 0.70 for wrong ones; real question variants score 0.84-0.89
# vs <= 0.68 for unrelated questions. Near-miss questions (same wording, different
# reasoning needed) score 0.91-0.95, so no threshold separates them from hits.
LOAD_THRESHOLD = float(os.environ.get("ARC_LOAD_THRESHOLD", 0.80))
DO_THRESHOLD = float(os.environ.get("ARC_DO_THRESHOLD", 0.80))
ENTRY_THRESHOLD = float(os.environ.get("ARC_ENTRY_THRESHOLD", 0.80))

# --- Shorthand in extracted values (DEVIATION; arc/normalize.py) ---
# "$150k" / "€2.0 million" are expanded to full digits after IS2, before cached code parses
# them. Only unambiguous forms; anything else is untouched. Off = paper behaviour exactly.
EXPAND_SHORTHAND = os.environ.get("ARC_EXPAND_SHORTHAND", "1") == "1"
# IS2 output form (DEVIATION). "copy" = paper: values copied as written. "number_unit": Haiku
# returns number (plain full digits) + unit + text + evidence per value; code checks the number
# against the evidence and rebuilds the value (arc/extraction.py). Shorthand expansion above is
# the check for "copy" mode and the cross-check for "number_unit".
EXTRACT_FORM = os.environ.get("ARC_EXTRACT_FORM", "number_unit")

# --- Turn-away gate (DEVIATION from the paper; off = paper behaviour) ---
# The paper serves any request whose embedding clears ENTRY_THRESHOLD. Near-misses
# (same wording, different reasoning) clear it too, and get a confident wrong answer.
#   off  paper behaviour
#   v1   the Haiku extraction call may leave slots empty, lists facts no slot captures,
#        and says whether the procedure fits (serve.extract_and_check)
#   v2   extract-and-verify: the Haiku call returns structure (what is asked, method,
#        where every given goes, evidence spans) and code checks it (arc/verify.py).
#        Needs gate cards (arc/gatecard.py).
GATE = {"on": "v1"}.get(os.environ.get("ARC_GATE", "off"), os.environ.get("ARC_GATE", "off"))
GATE_CHECKS = os.environ.get("ARC_GATE_CHECKS", "all")   # v2: comma list of checks that can reject
#   clef  Haiku writes the request's value-free structure; the top CLEF_TOP_K entries by
#         structure cosine go to Clef 27B (Cloudflare Workers AI), which reads the RAW
#         question against each candidate's original question and returns P(fits). Serve the
#         best candidate if P >= CLEF_THRESHOLD, else hand the question to the model.
#         Measured on dev (eval/decision, prompt v1, top-3): AUC 0.997; at 0.8 about 92% recall
#         and 0.4% false hits. See arc/clef.py.
CLEF_MODEL = os.environ.get("ARC_CLEF_MODEL", "clef")
CLEF_THRESHOLD = float(os.environ.get("ARC_CLEF_THRESHOLD", 0.8))
CLEF_TOP_K = int(os.environ.get("ARC_CLEF_TOP_K", 3))
# Seed questions per candidate entry: the original plus the most recent verified family
# questions. P for the entry = its best seed. One wording made Clef reject true variants in
# other wordings (stopped chat run: depreciation 0-1 of 5 fit); same fix as family grouping.
CLEF_SEEDS = int(os.environ.get("ARC_CLEF_SEEDS", 3))
# How the top-k candidates are found: "raw" = embed the raw question (no model call; what the
# Clef dev numbers used), "structure" = one Haiku call per question writes its structure first.
CLEF_RETRIEVAL = os.environ.get("ARC_CLEF_RETRIEVAL", "raw")
# Measured on dev, top-3, at P >= 0.8, live pick-the-best rule (2026-11-15): v1 served 110/119
# true matches right, 2/214 wrong; v2 113/119 right, the same 2 wrong; v3 (v2 + "read every
# word" rule) 113/119 right, 0 wrong. v3 is the default (user decision); confirm on test split.
CLEF_PROMPT = os.environ.get("ARC_CLEF_PROMPT", "v3")
CLEF_PRICE_PER_MTOK = float(os.environ.get("ARC_CLEF_PRICE_PER_MTOK", 0.24))   # Cloudflare, input tokens
# Which texts entry search embeds: "paper" = stored original question + template;
# "cards" = the builder-written rewordings and intent line from each gate card;
# "spec" = the value-free structure text from arc/spec.py (needs ARC_GATE=spec).
INDEX = os.environ.get("ARC_INDEX", "paper")
#   spec  Haiku rewrites the request into a fixed structure (values removed) and it is
#         compared field by field with the entry's structure (arc/spec.py). Pair with
#         ARC_INDEX=spec to retrieve on the structure text too.
SPEC_DECIDER = os.environ.get("ARC_SPEC_DECIDER", "nli")              # spec: embed | nli | registered name
SPEC_THRESHOLD = float(os.environ.get("ARC_SPEC_THRESHOLD", 0.5))     # spec: every field must score >= this

# --- Label-free builds from the backlog (DEVIATION; arc/labelfree.py) ---
# An entry is accepted only with zero confirmed-wrong, zero unresolved and zero violated
# metamorphic relations, and at least LABELFREE_MIN_CHECKED verified answers (0 of 10 ->
# error rate <= 26% at 95%; 0 of 29 -> <= 10%). Fewer -> "need_more_data", re-certified later.
LABELFREE_MIN_CHECKED = int(os.environ.get("ARC_LABELFREE_MIN_CHECKED", 10))
LABELFREE_MAX_CHECKED = int(os.environ.get("ARC_LABELFREE_MAX_CHECKED", 40))   # cost cap per entry
LABELFREE_BUILD_TRIES = int(os.environ.get("ARC_LABELFREE_BUILD_TRIES", 3))     # records tried as the build question
LABELFREE_RELATIONS = int(os.environ.get("ARC_LABELFREE_RELATIONS", 6))         # metamorphic checks per entry
LABELFREE_FAIL_FAST = os.environ.get("ARC_LABELFREE_FAIL_FAST", "1") == "1"     # stop at the first failure
LABELFREE_AUDIT_RATE = float(os.environ.get("ARC_LABELFREE_AUDIT_RATE", 0.02))  # share of hits re-checked after launch
LABELFREE_MAX_COST = float(os.environ.get("ARC_LABELFREE_MAX_COST", 0.15))     # hard $ cap per build_family call
# Family index (DEVIATION inside label-free builds): certify every held-out family question
# against the entry directly (gate and code still run; IS1's cosine is recorded, not applied),
# and on acceptance add the VERIFIED questions as extra entry points (kind "family") and card
# index_texts, at most LABELFREE_INDEX_MAX per entry. Off = index by the build question only.
# Build flow (set 2026-11-15). off (default): the simple flow. Re-solve the family's newest
# record with the customer's own production model, compare final answers, build on a match,
# one retry, else status "mismatch". on: the certified flow (held-out checks, adjudication,
# metamorphic relations, family index) below.
LABELFREE_CERTIFY = os.environ.get("ARC_LABELFREE_CERTIFY", "0") == "1"
LABELFREE_RESOLVE_TRIES = int(os.environ.get("ARC_LABELFREE_RESOLVE_TRIES", 2))
# Simple flow, pre-publish check (user decisions 2026-11-15): run the N newest OTHER family
# questions through the new entry. At least LABELFREE_FIT_MIN must fit the gate (0.8: 4 of 5;
# fewer questions round up: 3 of 3), the code's answers must equal production's under the
# strict comparison (disagreements adjudicated by blind solves), and no metamorphic relation
# may be violated. Catches an entry built from a near-miss and code that misreads an input
# ("$150k" read as 150). 0 = off. Skipped only when the family has no other questions.
LABELFREE_FIT_CHECKS = int(os.environ.get("ARC_LABELFREE_FIT_CHECKS", 5))
LABELFREE_FIT_MIN = float(os.environ.get("ARC_LABELFREE_FIT_MIN", 0.8))
# Tie-breaks (owner decision 2026-11-15): when production and the cache disagree, each blind
# solve is a short Python program written from the question (plus the entry's stated
# conventions) and run in a separate process, so the arithmetic is exact. "blind" = the old
# hand-arithmetic solve. A program that fails or prints nothing sides with no one.
LABELFREE_ADJUDICATOR = os.environ.get("ARC_LABELFREE_ADJUDICATOR", "program")
LABELFREE_PROGRAM_SECONDS = float(os.environ.get("ARC_LABELFREE_PROGRAM_SECONDS", 10))
# Conventions (owner decision 2026-11-15, option A): each entry states every rounding or
# convention choice its code makes that changes the answer; a build-time check runs a program
# under those lines on the build question and requires it to equal the code's answer.
LABELFREE_CONVENTION_CHECK = os.environ.get("ARC_LABELFREE_CONVENTION_CHECK", "1") == "1"
LABELFREE_FAMILY_INDEX = os.environ.get("ARC_LABELFREE_FAMILY_INDEX", "1") == "1"
LABELFREE_INDEX_MAX = int(os.environ.get("ARC_LABELFREE_INDEX_MAX", 20))

# --- Builder settings (paper gives no numbers for these) ---
MAX_FUNCTION_ATTEMPTS = int(os.environ.get("ARC_MAX_FUNCTION_ATTEMPTS", 4))       # write/test/fix rounds per function
MAX_ROOT_REPAIRS = int(os.environ.get("ARC_MAX_ROOT_REPAIRS", 2))            # extra rounds if the end-to-end test fails
EXTRACTION_ROUNDS = int(os.environ.get("ARC_EXTRACTION_ROUNDS", 3))           # N in "repeated N times"
GRANULARITY = os.environ.get("ARC_GRANULARITY", "medium")          # fine | medium | coarse
FUNCTION_TIMEOUT_SECONDS = int(os.environ.get("ARC_FUNCTION_TIMEOUT_SECONDS", 60))   # wall-clock limit for running cached code

# --- Evaluation settings ---
VARIANTS_PER_MODE = int(os.environ.get("ARC_VARIANTS_PER_MODE", 5))
JUDGE_COMMITTEE_SIZE = int(os.environ.get("ARC_JUDGE_COMMITTEE_SIZE", 2))        # paper: 2 judges, unanimity required

# --- Cache-worthiness: when a task family is worth building (DEVIATION: the paper caches
# every problem it sees). Ski-rental break-even: build once a family's recent questions
# have cost, in answers the cache could have given, as much as building one entry.
# Every dollar figure is measured; these are only the starting values before any
# measurements exist. See arc/families.py and arc/pricing.py.
# Join a family at cosine >= this (numbers masked). Tuned on seed+dev (eval/family_corpus.py):
# 0.75 keeps 96% of a task's questions in one family with no two tasks merged; 0.70 starts
# merging, 0.85+ splits tasks apart. ~89% of near-misses still join their seed's family.
FAMILY_SIMILARITY = float(os.environ.get("ARC_FAMILY_SIMILARITY", 0.75))
# Joining a family (DEVIATION). "embed": nearest centroid at FAMILY_SIMILARITY, nothing else.
# "clef" (user decision 2026-11-15): the nearest FAMILY_CLEF_TOP_K families above that cosine
# are candidates, and Clef must say the question fits the family's anchor (its most typical
# confirmed question) at P >= CLEF_THRESHOLD, or the question starts its own family. Keeps
# near-misses (same topic, different task) out of a family's count and out of its builds.
FAMILY_CHECK = os.environ.get("ARC_FAMILY_CHECK", "clef")
FAMILY_CLEF_TOP_K = int(os.environ.get("ARC_FAMILY_CLEF_TOP_K", 3))
# Clef cutoff for JOINING a family. Separate from CLEF_THRESHOLD (serving), because joining is
# lower stakes: the build fit check and the serving gate still apply.
# Dev, 238 questions, 17 true families (eval/family_corpus.py --check clef, 2026-11-15):
#   embed only, cosine 0.75 ................... recall 0.96, purity 1.00, near-misses in 0.89
#   clef 0.8, 1 anchor, cosine 0.75 ........... recall 0.71, purity 1.00, near-misses in 0.04
#   clef 0.5, 1 anchor, cosine 0.75 ........... recall 0.85, purity 1.00, near-misses in 0.11
#   clef 0.5, 3 anchors, cosine 0.75 .......... recall 0.88, purity 1.00, near-misses in 0.09
#   clef 0.5, 3 anchors, cosine 0.65 (default)  recall 0.91, purity 1.00, near-misses in 0.09
# Clef cost about $0.00044 per new question at these settings.
FAMILY_CLEF_THRESHOLD = float(os.environ.get("ARC_FAMILY_CLEF_THRESHOLD", 0.5))
FAMILY_CLEF_SIMILARITY = float(os.environ.get("ARC_FAMILY_CLEF_SIMILARITY", 0.65))  # candidate cosine, clef mode
# How many of a family's confirmed questions a new question is compared with: the anchor plus
# the most recent others. Joins if ANY fits, so one unusual anchor wording does not split a task.
FAMILY_CLEF_ANCHORS = int(os.environ.get("ARC_FAMILY_CLEF_ANCHORS", 3))
FAMILY_HALF_LIFE_DAYS = float(os.environ.get("ARC_FAMILY_HALF_LIFE_DAYS", 30))   # older questions count half
FAMILY_MIN_SIGHTINGS = float(os.environ.get("ARC_FAMILY_MIN_SIGHTINGS", 3))  # proof it recurs, whatever the money says
CACHE_ANSWER_COST = float(os.environ.get("ARC_CACHE_ANSWER_COST", 0.0001))   # measured: one Haiku extraction call
BUILD_COST_DEFAULT = float(os.environ.get("ARC_BUILD_COST_DEFAULT", 0.034))  # measured: explainer + build + certificate
BUILD_ACCEPT_RATE_DEFAULT = float(os.environ.get("ARC_BUILD_ACCEPT_RATE_DEFAULT", 10 / 17))  # measured: 17 seeds
SERVE_RATE_DEFAULT = float(os.environ.get("ARC_SERVE_RATE_DEFAULT", 1.0))    # share of a family the gate serves

# --- Retail prices, $ per million tokens: (input, output, cache write, cache read) ---
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}
DEFAULT_PRICE = PRICES["claude-opus-5-5"]

# Every setting above can be overridden per run with an ARC_<NAME> env var
# (the experiment UI uses this). snapshot() records the values a run used.
SETTING_NAMES = [
    "PROFILE", "SOURCE_MODEL", "EXPLAINER_MODEL", "BUILDER_MODEL", "JUDGE_MODEL", "GENERATOR_MODEL", "TARGET_MODEL",
    "SOURCE_EFFORT", "BUILDER_EFFORT", "JUDGE_EFFORT", "GENERATOR_EFFORT", "EMBED_MODEL",
    "EXPAND_SHORTHAND", "EXTRACT_FORM", "LOAD_THRESHOLD", "DO_THRESHOLD", "ENTRY_THRESHOLD", "GATE", "GATE_CHECKS", "INDEX", "SPEC_DECIDER", "SPEC_THRESHOLD", "CLEF_MODEL", "CLEF_THRESHOLD", "CLEF_TOP_K", "CLEF_SEEDS", "CLEF_RETRIEVAL", "CLEF_PROMPT", "LABELFREE_MIN_CHECKED", "LABELFREE_MAX_CHECKED", "LABELFREE_BUILD_TRIES", "LABELFREE_RELATIONS", "LABELFREE_FAIL_FAST", "LABELFREE_AUDIT_RATE", "LABELFREE_MAX_COST", "LABELFREE_FAMILY_INDEX", "LABELFREE_INDEX_MAX", "LABELFREE_CERTIFY", "LABELFREE_RESOLVE_TRIES", "LABELFREE_FIT_CHECKS", "LABELFREE_FIT_MIN", "LABELFREE_ADJUDICATOR", "LABELFREE_PROGRAM_SECONDS", "LABELFREE_CONVENTION_CHECK",
    "MAX_FUNCTION_ATTEMPTS", "MAX_ROOT_REPAIRS", "EXTRACTION_ROUNDS", "GRANULARITY",
    "FUNCTION_TIMEOUT_SECONDS", "VARIANTS_PER_MODE", "JUDGE_COMMITTEE_SIZE",
    "FAMILY_SIMILARITY", "FAMILY_CHECK", "FAMILY_CLEF_TOP_K", "FAMILY_CLEF_THRESHOLD", "FAMILY_CLEF_SIMILARITY", "FAMILY_CLEF_ANCHORS", "FAMILY_HALF_LIFE_DAYS", "FAMILY_MIN_SIGHTINGS", "CACHE_ANSWER_COST",
    "BUILD_COST_DEFAULT", "BUILD_ACCEPT_RATE_DEFAULT", "SERVE_RATE_DEFAULT",
]


def snapshot() -> dict:
    values = {name: globals()[name] for name in SETTING_NAMES}
    values["NAMESPACE"] = NAMESPACE or "(root)"
    overrides = {name: os.environ[f"ARC_{name}"] for name in SETTING_NAMES if f"ARC_{name}" in os.environ}
    return {"values": values, "overrides": overrides}
