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
# Which texts entry search embeds: "paper" = stored original question + template;
# "cards" = the builder-written rewordings and intent line from each gate card.
INDEX = os.environ.get("ARC_INDEX", "paper")

# --- Builder settings (paper gives no numbers for these) ---
MAX_FUNCTION_ATTEMPTS = int(os.environ.get("ARC_MAX_FUNCTION_ATTEMPTS", 4))       # write/test/fix rounds per function
MAX_ROOT_REPAIRS = int(os.environ.get("ARC_MAX_ROOT_REPAIRS", 2))            # extra rounds if the end-to-end test fails
EXTRACTION_ROUNDS = int(os.environ.get("ARC_EXTRACTION_ROUNDS", 3))           # N in "repeated N times"
GRANULARITY = os.environ.get("ARC_GRANULARITY", "medium")          # fine | medium | coarse
FUNCTION_TIMEOUT_SECONDS = int(os.environ.get("ARC_FUNCTION_TIMEOUT_SECONDS", 60))   # wall-clock limit for running cached code

# --- Evaluation settings ---
VARIANTS_PER_MODE = int(os.environ.get("ARC_VARIANTS_PER_MODE", 5))
JUDGE_COMMITTEE_SIZE = int(os.environ.get("ARC_JUDGE_COMMITTEE_SIZE", 2))        # paper: 2 judges, unanimity required

# --- Retail prices, $ per million tokens: (input, output, cache write, cache read) ---
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}
DEFAULT_PRICE = PRICES["claude-opus-5-5"]

# Every setting above can be overridden per run with an ARC_<NAME> env var
# (the experiment UI uses this). snapshot() records the values a run used.
SETTING_NAMES = [
    "PROFILE", "SOURCE_MODEL", "BUILDER_MODEL", "JUDGE_MODEL", "GENERATOR_MODEL", "TARGET_MODEL",
    "SOURCE_EFFORT", "BUILDER_EFFORT", "JUDGE_EFFORT", "GENERATOR_EFFORT", "EMBED_MODEL",
    "LOAD_THRESHOLD", "DO_THRESHOLD", "ENTRY_THRESHOLD", "GATE", "GATE_CHECKS", "INDEX",
    "MAX_FUNCTION_ATTEMPTS", "MAX_ROOT_REPAIRS", "EXTRACTION_ROUNDS", "GRANULARITY",
    "FUNCTION_TIMEOUT_SECONDS", "VARIANTS_PER_MODE", "JUDGE_COMMITTEE_SIZE",
]


def snapshot() -> dict:
    values = {name: globals()[name] for name in SETTING_NAMES}
    values["NAMESPACE"] = NAMESPACE or "(root)"
    overrides = {name: os.environ[f"ARC_{name}"] for name in SETTING_NAMES if f"ARC_{name}" in os.environ}
    return {"values": values, "overrides": overrides}
