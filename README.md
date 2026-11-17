# Reasoning Reuse Engine

Language models re-think everything from scratch. Ask one for a mortgage payment and it works through the whole formula. Ask again tomorrow with a different loan amount and it does all of that thinking again, and you pay for it again. Do that ten thousand times and you've paid for ten thousand rounds of the same reasoning.

We think a model should only have to figure out a kind of problem once.

The Reasoning Reuse Engine watches the questions a model answers, notices the ones that keep coming back, and turns the model's reasoning for them into real code. After that, those questions are answered by running the code. The only model involved is a tiny one that reads the numbers out of the question. Each answer costs about a hundredth of a cent, uses exact arithmetic, and comes with a step-by-step trace you can read. Anything the code can't handle still goes to the model, exactly as before.

The goal is a model that gets cheaper, faster and more reliable the more people use it.

## What it looks like

Someone asks:

> Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. Predict the pressure exerted by the ethane from the perfect gas.

The model works it out (50.7 atm). A builder model then reads that working and writes it as a handful of small Python functions: convert the temperature to kelvin, convert the volume to cubic metres, apply p = nRT/V, convert pascals to atmospheres, and cross-check with a second value of R. You can see the result in `caches/master/compiled_gas_law.py`.

Next time someone asks about 12 mol of methane in 3 L at 50 °C, nobody re-derives the gas law. A small model pulls out the numbers, the cached code runs, and the answer (about 106 atm) comes back with every step narrated.

## Why this is hard

Caching an answer is easy. Caching reasoning so that it only fires when it's actually right is the whole project. These are the problems we've spent most of our time on.

**Questions that look the same but aren't.** "What's the pressure of this gas?" and "What volume does this gas need to reach this pressure?" use almost the same words but need different math. Embedding similarity, the usual way to match questions, scores these near-misses at 0.91–0.95, which is *higher* than real rewordings of the same question (0.84–0.89). No threshold separates them. So every cache hit goes through a gate that asks whether this entry really solves this question. Our current gate uses a small yes/no model called Clef. On our dev set it served 117 of the 119 questions it should have. It never served any of the 214 near-misses and unrelated questions it should have left alone. We still need to confirm that on the held-out test split.

**No answer key.** The research we started from builds every cache entry with the correct answer in hand. In real life nobody hands you one. We build from the log of real questions and answers instead: re-solve with the production model and check they agree, run the new entry on other questions from the same family before publishing it, have a different model write acceptance tests without ever seeing the builder's code (so the code and its tests don't share blind spots), and run rewrite checks no model is involved in ("25 m/s" and "90 km/h" are the same speed, so they must give the same answer).

**Knowing what's worth caching.** Most questions never come back, and every build costs money. We group questions into task families and only build an entry once a family has come up often enough that caching it would have paid for itself. It's the ski-rental rule: keep renting until the rent you've paid equals the price of buying. A build on Haiku costs about $0.04. If Haiku can't write working code, the build moves up to Sonnet, then Opus.

**Numbers have to be exactly right.** Some of the bugs that taught us this: an entry read "$150k" as 150 and answered a $921.13 monthly payment with "0.92 dollars". Another typed its own litre-to-cm³ factor and came out 1000× off. Now the builder works under strict rules for inputs, units and precision. Rounding happens once, in one tested formatter, instead of in generated code. Every new entry also has to pass the rewrite checks above before it's published. We also built an exact unit-conversion library and a scan for hand-typed factors (`arc/units.py`, `arc/factorscan.py`). Both are off by default right now, so the builder writes its own conversions.

## Where we want to take it

- A chatbot that starts with an empty cache and learns from real use. The loop already works end to end in `arc/chat.py`.
- Compile mature entries down to plain Python with no model and no vector store. That works for single entries today. Next is handling updates to an entry after it's been compiled.
- Let models call cached functions as tools, even for questions that aren't an exact match. Once the reasoning is written down as code, it'd be a waste not to use it everywhere.
- Spot-check high-stakes topics. If a model is wrong the same way every time and nobody notices, every check above agrees on the wrong answer. That's most likely in medicine, law and finance, where users can't check the answer themselves, so those entries need an occasional audit by a stronger model or a person.

`NOTES.md` has the running list.

## Where it came from

This started as a replication of the agentic reasoning cache (ARC) from the CPU-ARC paper. The paper's six steps are all still here: running its published functions, serving, solving, building, evaluating and compiling. Everything we changed or added on top is marked `DEVIATION` in the module docstrings, with the reason why.

## Running it

You need Python 3.12 or later.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows (PowerShell), activate with `.venv\Scripts\Activate.ps1` instead.

```bash
pip install anthropic pydantic numpy sentence-transformers pytest
```

Put your keys in a `.env` file in the project root. The Cloudflare keys are only needed for the Clef gate and for grouping questions into families with Clef.

```
ANTHROPIC_API_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
```

The default embedding model, `google/embeddinggemma-300m`, is gated on Hugging Face. Log in and accept its license, or the code falls back to `all-mpnet-base-v2`.

Then:

```bash
python -m arc paper gas_law                  # run the paper's own functions (free, checks the runtime)
python -m arc all gas_law                    # solve, build, evaluate and compile one problem
python -m arc ask master "your question"     # ask the cache; falls back to the model on a miss
python -m arc families                       # which question families are worth building
python web/server.py                         # local UI for runs and every model call, on port 8765
```

By default every role runs on Haiku, which keeps experiments cheap. Every setting is in `arc/config.py`, along with the measurement behind its value, and can be overridden with an `ARC_<NAME>` environment variable. Opus is blocked unless you set `ARC_ALLOW_OPUS=1`.

## Repo map

```
arc/        the engine: runtime, builder, serving, gates, label-free builds, CLI
eval/       the experiments behind our numbers (each script says whether it costs money)
problems/   problems to build from: the paper's two, plus 17 seeds across science, medicine, finance and law
caches/     built caches, one folder of plain Python functions per cache
tests/      offline tests (python -m pytest, no API calls)
web/        the experiment UI
```
