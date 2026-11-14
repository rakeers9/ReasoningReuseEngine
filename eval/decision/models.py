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

MODELS = {
    # ---- baseline (free, already downloaded) -------------------------------------
    "embed-gemma": dict(kind="embed", note="paper matcher: cosine(question, seed question)"),

    # ---- hosted -----------------------------------------------------------------
    "clef": dict(kind="cloudflare", cf_model="clef", price_per_mtok=0.24,
                 note="Cloudflare, 27B, Apache-2.0 weights (too big for this Mac)"),
    "clef-flash": dict(kind="cloudflare", cf_model="clef-flash", price_per_mtok=0.09,
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

    # ---- local, sentence-transformers (no downloaded code) -----------------------
    "qwen3-reranker-0.6b": dict(kind="reranker", repo="Qwen/Qwen3-Reranker-0.6B", mb=1207,
                                license="Apache-2.0", note="reranker, not a decision model"),

    # ---- local, runs code shipped with the model ---------------------------------
    "decision2-kai-0.6b": dict(kind="hf_decision", repo="vllm-sr/Decision-2.0-Kai-0.6B", mb=1522,
                               license="Apache-2.0", remote_code=True, note="vLLM Semantic Router"),
    "gliner2.5-decide": dict(kind="gliner", repo="fastino/GLiNER2.5-Decide", mb=1955,
                             license="Apache-2.0", remote_code=True,
                             note="DeBERTa encoder; needs `pip install gliner2`"),
}
