"""The candidate decision models and how to reach each one.

Every model answers the same thing: given an incoming question (the "state") and one
yes/no ("noul") question per candidate cache entry, return P(yes) for each.

kinds
    cloudflare   hosted Workers AI (needs CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN in .env)
    llama        a GGUF file served by Homebrew llama.cpp's llama-server, /v1/systemone
    jevk5        a GGUF file on llama-server, read via answer-letter log-probs (JevK5's own client)
    reranker     sentence-transformers CrossEncoder (Qwen3-Reranker)
    hf_decision  transformers AutoModel(trust_remote_code=True).system_one (runs repo code)
    gliner       the gliner2 pip package
    embed        baseline: EmbeddingGemma cosine, the paper's matcher (already on disk)

`remote_code` marks models that run Python downloaded with the model; they only run with
--allow-remote-code. `files` lists what download.py fetches, with sizes from the HF API.
"""

def LAYA_FILES(prefix: str) -> list[str]:
    """The files one Laya checkpoint needs (what laya.Agent itself fetches); never the repo's .py."""
    return [prefix + f for f in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]


MODELS = {
    # ---- baseline (free, already downloaded) -------------------------------------
    "embed-gemma": dict(kind="embed", note="paper matcher: cosine(question, seed question)"),

    # ---- hosted -----------------------------------------------------------------
    "clef": dict(kind="cloudflare", cf_model="clef", price_per_mtok=0.24,
                 note="Cloudflare, 27B, Apache-2.0 weights (too big for this Mac)"),
    "clef-flash": dict(kind="cloudflare", cf_model="clef-flash", price_per_mtok=0.038,
                       note="Cloudflare, 9B, Apache-2.0 weights"),

    # ---- local, llama.cpp, data files only (no downloaded code) ------------------
    "jevos-1b": dict(kind="llama", repo="feder-cr/jev", file="jevos-v4-q4_k_m.gguf", mb=619,
                     license="MIT", note="MiniCPM5 1B, yes/no only"),
    "kev-0.8b": dict(kind="llama", repo="ggml-org/Kev-0.8B-GGUF", file="Kev-0.8B-Q8_0.gguf", mb=812,
                     license="Apache-2.0", note="Qwen3.5 0.8B"),
    "standardone-3b": dict(kind="llama", repo="StandardThinking/StandardOne-3B-GGUF",
                           file="StandardOne-3B-Q4_K_M.gguf", mb=2147, license="Apache-2.0",
                           note="Ministral 3B"),
    "jevk5-2b": dict(kind="jevk5", repo="alibiserikbay/JevK5-GGUF", file="jevk5-2b-v0.2-Q8_0.gguf",
                     mb=2012, license="Apache-2.0", temperature=1.42,
                     note="Qwen3.5 2B; the card's pick for CPU"),

    "kev-4b": dict(kind="llama", repo="ggml-org/Kev-4B-GGUF", file="Kev-4B-Q4_K_M.gguf", mb=3033,
                   license="Apache-2.0", note="Qwen3.5 4B; strongest small decision model reported"),
    "julia-1": dict(kind="llama", repo="ggml-org/Julia-1-GGUF", file="Julia-1-Q8_0.gguf", mb=168,
                    license="Apache-2.0", note="mmBERT-small 144M encoder"),

    # ---- local, sentence-transformers (no downloaded code) -----------------------
    "qwen3-reranker-0.6b": dict(kind="reranker", repo="Qwen/Qwen3-Reranker-0.6B", mb=1207,
                                license="Apache-2.0", instruct=True, note="reranker, not a decision model"),
    "langcache-reranker-v2": dict(kind="reranker", repo="redis/langcache-reranker-v2-bce", mb=299,
                                  license="Apache-2.0",
                                  note="ModernBERT reranker Redis tuned for semantic-cache same-intent pairs"),
    "nli-deberta-v3-large": dict(kind="nli", repo="cross-encoder/nli-deberta-v3-large", mb=1740,
                                 license="Apache-2.0", note="NLI cross-encoder, P(entailment)"),

    # ---- local, runs code shipped with the model ---------------------------------
    "decision2-kai-0.6b": dict(kind="hf_decision", repo="vllm-sr/Decision-2.0-Kai-0.6B", mb=1522,
                               license="Apache-2.0", remote_code=True, note="vLLM Semantic Router"),
    # Laya (pip package `laya`, one HF repo with three checkpoints). remote_code: runs the
    # laya package's code. max_len/head_max_len widened so our gate prompts are not cut
    # (Laya's default head budget is 192 tokens; see backends.Laya).
    "laya": dict(kind="laya", repo="convaiinnovations/laya", mb=846, license="Apache-2.0", remote_code=True,
                 max_len=1024, head_max_len=768, allow=LAYA_FILES(""),
                 note="ModernBERT-large 421M, English"),
    "laya-typed": dict(kind="laya", repo="convaiinnovations/laya", subfolder="typed-decisions", mb=846,
                       license="Apache-2.0", remote_code=True, max_len=1024, head_max_len=768,
                       allow=LAYA_FILES("typed-decisions/"),
                       note="ModernBERT-large 421M, fine-tuned for typed decisions"),
    "laya-multilingual": dict(kind="laya", repo="convaiinnovations/laya", subfolder="multilingual", mb=678,
                              license="Apache-2.0", remote_code=True, max_len=1024, head_max_len=768,
                              allow=LAYA_FILES("multilingual/"), note="mmBERT-base 322M"),
    "gliner2.5-decide":dict(kind="gliner", repo="fastino/GLiNER2.5-Decide", mb=1955,
                             license="Apache-2.0", remote_code=True,
                             note="DeBERTa encoder; needs `pip install gliner2`"),
}
