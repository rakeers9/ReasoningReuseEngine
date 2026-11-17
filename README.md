# Reasoning Reuse Engine

Repeated questions to a language model are often the same calculation with different numbers: a gas pressure, a loan payment, a buffer pH, a court deadline. This project turns one good worked answer to such a question into a small cached program. Later questions of the same kind are answered by running that program instead of calling the big model again.

The code started as a replication of the agentic reasoning cache (ARC) from the CPU-ARC paper. It then adds what production use needs and the paper doesn't cover:

- building cache entries without a golden answer key
- deciding which questions recur often enough to be worth caching
- turning away near-miss questions that look the same but need different reasoning
- exact unit handling and number formatting

Every change from the paper is marked `DEVIATION` in the module docstrings.

## How it works

A **cache** is a set of *semantic functions* (small Python functions written by a builder model) plus *entry points* that map a kind of question to the function that solves it. On disk, each cache is a folder holding `registry.json` and `functions/<id>.py`.

**Building an entry** (paper Section 2.2.1, `arc/builder.py`)
1. A source model solves the question and writes out its full working.
2. A builder model reads that transcript and goes through five phases: isolate inputs and outputs, decompose the problem, recognise the parameters, plan a function tree, then implement it. Each function is written, statically checked, run, judged and fixed.
3. The root function is tested end to end, and the entry is merged into the `master` cache.

**Serving a question** (paper Section 2.2.2, `arc/serve.py`)
1. **IS1 entry search:** the question is embedded and compared with the stored entry points.
2. **IS2 parameter extraction:** a small target model pulls out each input value.
3. **IS3 dispatch:** the cached root function runs in a fresh context.
4. **IS4 output:** the named results and a narration trace come back.

On a miss or an error, the question falls through to the full model.

**Compiling** (paper Section 2.2.3, `arc/compiler.py`): a mature entry's lookups by description are resolved ahead of time into one plain Python module. The compiled module needs no embedding model or vector store.

### What this project adds

| Area | What it does | Module |
| --- | --- | --- |
| Backlog | Logs every production question and answer, whether the model or the cache answered it | `arc/backlog.py` |
| Cache-worthiness | Groups backlog questions into task families and builds an entry only when a ski-rental break-even rule says the savings cover the build cost | `arc/families.py`, `arc/pricing.py` |
| Label-free builds | Builds and checks an entry from the backlog alone: the production model re-solves the question, other family questions are checked before publishing, metamorphic checks run, and a builder cascade escalates from Haiku to Sonnet to Opus on code failures | `arc/labelfree.py`, `arc/metamorphic.py`, `arc/certify.py` |
| Explainer | Writes a worked solution when production only gives a question and a final answer | `arc/explain.py` |
| Turn-away gates | Reject near-miss questions instead of serving a confident wrong answer. Options: `v1`, `v2` (extract-and-verify), `spec` (structure comparison), `clef` (a small yes/no decision model on Cloudflare Workers AI) | `arc/serve.py`, `arc/verify.py`, `arc/spec.py`, `arc/clef.py`, `arc/gatecard.py` |
| Exact numbers | Exact unit conversion, a scan for hand-typed conversion factors, a strict number comparison, and one shared final formatter for rounding and units | `arc/units.py`, `arc/factorscan.py`, `arc/checks.py`, `arc/present.py` |
| Chatbot loop | A chatbot wrapped in the cache. It starts empty, logs to the backlog, and builds entries between turns as families reach break-even | `arc/chat.py` |
| Sandbox | Allow-listed imports, blocked file, process and reflection builtins, and a timeout around generated code | `arc/sandbox.py` |

## Repository layout

```
arc/        the pipeline: runtime, builder, serving, gates, label-free builds, CLI
eval/       experiments and corpora (each script's docstring says how to run it and whether it is paid)
problems/   problem files: gas_law and thermo from the paper, plus 17 corpus seeds in problems/corpus/
caches/     built caches: one folder per cache, each with registry.json and functions/
tests/      offline unit tests (pytest)
web/        local experiment UI: a standard-library HTTP server and a single page
NOTES.md    open questions and next steps
```

Run outputs go to `runs/`, which is git-ignored.

## Setup

You need Python 3.12 or later. There is no requirements file yet, so install the packages directly:

```bash
python -m venv .venv
```

Activate the environment. On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Then install the packages:

```bash
pip install anthropic pydantic numpy sentence-transformers pytest
```

`sentence-transformers` pulls in PyTorch and is only needed for real runs. The tests run without it. Some `eval/` scripts need extras, such as `scipy` for `eval/corpus/build_corpus.py`, `huggingface_hub`, and `transformers`. Each script imports what it needs.

Create a `.env` file in the project root. It is read automatically and git-ignored.

```
ANTHROPIC_API_KEY=...
# Only for the Clef gate and Clef family grouping (Cloudflare Workers AI):
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
```

The default embedding model, `google/embeddinggemma-300m`, is gated on Hugging Face. Log in and accept its license to use it. Without access, the code falls back to `sentence-transformers/all-mpnet-base-v2`.

## Usage

### Command line

```bash
python -m arc paper   gas_law            # Step 1: run the paper's own published functions (checks the runtime)
python -m arc ask     master "..."       # Step 2: serve a question (cache first, LLM fallback)
python -m arc solve   gas_law            # Step 3: source model writes a worked solution
python -m arc build   gas_law            # Step 4: builder turns the transcript into a cache entry
python -m arc eval    gas_law            # Step 5: correctness, perturbations, costs
python -m arc compile gas_law            # Step 6: static compilation to plain Python
python -m arc all     gas_law            # Steps 3-6 in order
python -m arc families                   # which backlog task families are worth building
```

Useful flags:

- `build --no-golden`: build without the answer key.
- `build --verify agreement`: merge the entry only if the agreement certificate passes.
- `eval` / `compile --cache <name>`: use a cache other than `master`.
- `compile --extract`: extract parameters with the target model instead of using the original values.

Every run prints its cost broken down by model role. It also writes `runs/experiments/<run_id>/`, which holds the settings snapshot, every model call, and copies of the reports it changed.

### Web UI

```bash
python web/server.py          # http://127.0.0.1:8765 (binds to localhost only)
```

From the UI you can start pipeline commands, watch their output live, read every model call, and compare experiments. The server uses only the standard library. It starts runs with `.venv/bin/python`, so it expects the virtual environment above at `.venv`.

### Chatbot loop

```python
from arc.chat import Chatbot

bot = Chatbot()                       # cache "master"; logs to runs/<namespace>/backlog.jsonl
reply = bot.chat("What is the monthly payment on a $300,000 loan at 6% over 30 years?")
print(reply.answer, reply.from_cache, reply.cost)
bot.learn()                           # PAID: builds every family that has reached break-even
```

The chatbot only handles single-turn questions. Every message must stand on its own.

## Configuration

Every setting lives in `arc/config.py`, with a comment saying why it has its current value. Any setting can be overridden for one run with an `ARC_<NAME>` environment variable. A few to know:

- **`ARC_PROFILE`**
  - `haiku` (default): every big role uses Haiku. This is about 40× cheaper and writes to `runs/haiku/` and `caches/haiku/`.
  - `paper`: Opus fills the source, builder, judge and generator roles, as in the paper. Opus calls are refused unless `ARC_ALLOW_OPUS=1` is set. The exceptions are the builder cascade's last step and re-solves with the customer's own production model.
- **`ARC_GATE`**: `off` (paper behaviour), `v1`, `v2`, `spec` or `clef`. With `off`, near-miss questions can get confident wrong answers.
- **`ARC_LABELFREE_CERTIFY`**: `0` uses the simple label-free flow (the default). `1` uses the full certified flow.
- **`ARC_FAMILY_CHECK`**: `clef` (default) confirms family membership with Clef, which calls Cloudflare. `embed` keeps family grouping fully local.
- **Model overrides**: `ARC_SOURCE_MODEL`, `ARC_BUILDER_MODEL`, `ARC_JUDGE_MODEL`, `ARC_TARGET_MODEL`, and so on.

## Evaluation

The `eval/` folder holds the experiments behind the numbers recorded in `arc/config.py`. Each script's docstring gives its exact command and says whether it is **FREE** (local only) or **PAID** (calls the API). Run each script from the project root.

| Script | Question it answers |
| --- | --- |
| `eval/run_corpus.py` | Hit, miss and wrong-answer rates on the 383-row cache-eval corpus, per gate setting |
| `eval/family_corpus.py` | Does family grouping recover the 17 real task families? (free) |
| `eval/decision/run_decision.py` | Can a decision model tell whether an entry applies to a question? |
| `eval/labelfree/run_eval.py` | Does label-free certification accept good entries and reject bad ones? |
| `eval/chat_loop.py` | End to end: a chatbot that starts empty and learns from about 105 messages (`--dry-run` is free) |
| `eval/builder_compare.py` | Haiku vs Sonnet as the builder |
| `eval/cert_experiment.py`, `cert_mutants.py`, `cert_truth.py` | Does the agreement certificate catch broken transcripts and broken code? |

Corpus answers are computed in code (`eval/corpus/build_corpus.py`, `eval/labelfree/families.py`), never by a model. Experiments tune on the `dev` split. The `test` split is held out.

## Tests

```bash
python -m pytest
```

The tests are free and offline. Any real Anthropic or Cloudflare call fails the test, and model calls are stubbed. Tests that need a local model download are skipped unless you pass `--models`.

## Status

This is a research codebase under active development. `NOTES.md` lists the open questions. The main known gap: if the production model is wrong the same way every time and users never notice, every source of evidence agrees on the wrong answer. That case matters most in medicine, law and finance, where an occasional audit of cached entries by a stronger model or a person is still needed.
