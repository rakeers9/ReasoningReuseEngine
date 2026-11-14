"""Backends: each turns (state, {qid: instructions}) into {qid: P(yes)}.

    b = make_backend("clef-flash")
    with b:                                  # starts/stops a local server if one is needed
        probs, usage = b.decide(state, {"perfect_gas_pressure": "...", ...})

`usage` is {"input_tokens": int} when the backend reports it, else {}.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from models import MODELS

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 120, tries: int = 5) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def _p_yes(ans) -> float:
    """Read P(yes) out of a SystemOne noul answer, whatever shape the server uses."""
    if isinstance(ans, (int, float)):
        return float(ans)
    if isinstance(ans, dict):
        for k in ("noul", "probability", "p", "value", "yes", "true"):
            if isinstance(ans.get(k), (int, float)):
                return float(ans[k])
        probs = ans.get("probabilities")
        if isinstance(probs, dict):
            for k in ("true", "yes", "True", "Yes"):
                if k in probs:
                    return float(probs[k])
    raise ValueError(f"cannot read P(yes) from {json.dumps(ans)[:300]}")


def noul_questions(questions: dict[str, str]) -> dict:
    return {qid: {"type": "noul", "instructions": text} for qid, text in questions.items()}


class Backend:
    name = "?"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def decide(self, state: str, questions: dict[str, str]) -> tuple[dict[str, float], dict]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- hosted

class Cloudflare(Backend):
    def __init__(self, name: str, spec: dict):
        load_dotenv()
        acct, tok = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
        if not acct or not tok:
            raise SystemExit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env first.")
        self.name, self.model = name, spec["cf_model"]
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/cloudflare/{self.model}"
        self.headers = {"Authorization": f"Bearer {tok}"}

    def decide(self, state, questions):
        out = _post(self.url, {"model": self.model, "state": state, "questions": noul_questions(questions)},
                    self.headers)
        if "result" in out and isinstance(out["result"], dict):
            if out.get("success") is False:
                raise RuntimeError(f"Cloudflare error: {out.get('errors')}")
            out = out["result"]
        ans = out["answers"]
        return {q: _p_yes(ans[q]) for q in questions}, out.get("usage") or {}


# ----------------------------------------------------------------------------- llama.cpp

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gguf_path(spec: dict) -> Path:
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(spec["repo"], spec["file"], local_files_only=True))
    except Exception:
        raise SystemExit(f"{spec['repo']}/{spec['file']} is not downloaded. Run eval/decision/download.py first.")


class LlamaServer(Backend):
    """Starts Homebrew's llama-server on the GGUF file, CPU only by default (-ngl 0)."""

    def __init__(self, name: str, spec: dict, gpu: bool = False, threads: int = 4):
        self.name, self.spec = name, spec
        self.model_path = gguf_path(spec)
        self.gpu, self.threads = gpu, threads
        self.proc = None
        self.base = None

    def __enter__(self):
        exe = shutil.which("llama-server")
        if not exe:
            raise SystemExit("llama-server not found. Install it with: brew install llama.cpp")
        port = _free_port()
        self.base = f"http://127.0.0.1:{port}"
        log = open(ROOT / "eval" / "decision" / f".llama-{self.name}.log", "w")
        cmd = [exe, "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(port),
               "-c", "8192", "-np", "1", "-t", str(self.threads), "-ngl", "99" if self.gpu else "0",
               "--jinja", "--no-webui"]
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(240):
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited; see eval/decision/.llama-{self.name}.log")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        return self
            except Exception:
                pass
            time.sleep(0.5)
        raise SystemExit("llama-server did not become ready in 120 s")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return False

    def decide(self, state, questions):
        out = _post(self.base + "/v1/systemone",
                    {"model": self.name, "state": state, "questions": noul_questions(questions)})
        return {q: _p_yes(out["answers"][q]) for q in questions}, out.get("usage") or {}


class JevK5(LlamaServer):
    """JevK5's documented standalone client: one pass per question, P from answer-letter logprobs."""

    SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
              "Respond with only its uppercase letter, with no explanation or reasoning.")

    def decide(self, state, questions):
        T = self.spec["temperature"]
        probs, tokens = {}, 0
        for qid, text in questions.items():
            user = json.dumps({"evidence": state, "criterion": text,
                               "options": [{"letter": "A", "description": "true: The proposition is true."},
                                           {"letter": "B", "description": "false: The proposition is false."}]},
                              ensure_ascii=False)
            prompt = (f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            toks = _post(self.base + "/tokenize", {"content": prompt, "add_special": False,
                                                   "parse_special": True})["tokens"]
            tokens += len(toks)
            top = _post(self.base + "/completion", {"prompt": toks, "n_predict": 1, "n_probs": 40,
                                                    "temperature": 0, "cache_prompt": False}
                        )["completion_probabilities"][0]["top_logprobs"]
            seen = {t["token"]: t["logprob"] for t in top}
            floor = min(seen.values()) - 2.0
            z = [seen.get("A", floor), seen.get("B", floor)]
            w = [math.exp((v - max(z)) / T) for v in z]
            probs[qid] = w[0] / sum(w)
        return probs, {"input_tokens": tokens}


# ----------------------------------------------------------------------------- python models

class Reranker(Backend):
    INSTRUCT = ("Given an incoming question (query) and the example question a cached solver was built "
                "from (document), judge whether the solver answers the incoming question exactly: same "
                "result asked for, same method, all needed inputs, no extra conditions")

    def __init__(self, name, spec):
        from sentence_transformers import CrossEncoder
        import torch
        self.name, self._sig = name, torch.nn.Sigmoid()
        self.model = CrossEncoder(spec["repo"], device="cpu", local_files_only=True,
                                  model_kwargs={"dtype": torch.float32},  # bf16 is slow on CPU
                                  prompts={"gate": self.INSTRUCT}, default_prompt_name="gate")

    def decide(self, state, questions):
        # questions here carry the raw seed text (see run_decision.pair_text)
        ids = list(questions)
        scores = self.model.predict([(state, questions[q]) for q in ids], activation_fn=self._sig)
        return {q: float(s) for q, s in zip(ids, scores)}, {}


class HFDecision(Backend):
    def __init__(self, name, spec):
        from transformers import AutoModel
        self.name = name
        self.model = AutoModel.from_pretrained(spec["repo"], trust_remote_code=True, local_files_only=True)

    def decide(self, state, questions):
        out = self.model.system_one(state=state, questions=noul_questions(questions))
        ans = out["answers"]
        return {q: _p_yes(ans[q]) for q in questions}, out.get("usage") or {}


class Gliner(Backend):
    def __init__(self, name, spec):
        from gliner2 import AutoExtractor
        self.name = name
        self.model = AutoExtractor.from_pretrained(spec["repo"])

    def decide(self, state, questions):
        probs = {}
        for qid, text in questions.items():
            schema = {qid: {"labels": ["yes", "no"], "prompt": text}}
            try:
                out = self.model.classify_text(state, schema, include_confidence=True)
            except TypeError:
                out = self.model.classify_text(state, schema)
            a = out[qid]
            if isinstance(a, dict) and "confidence" in a:
                p = float(a["confidence"])
                probs[qid] = p if a.get("label") == "yes" else 1 - p
            else:
                probs[qid] = 1.0 if (a == "yes" or (isinstance(a, dict) and a.get("label") == "yes")) else 0.0
        return probs, {}


class Embed(Backend):
    """The paper's matcher, as a 'decision': cosine(question, seed question)."""

    def __init__(self, name, spec):
        import sys
        sys.path.insert(0, str(ROOT))
        from arc.embed import embedder
        self.name, self.e = name, embedder()

    def decide(self, state, questions):
        ids = list(questions)
        sims = self.e.encode([questions[q] for q in ids]) @ self.e.one(state)
        return {q: float(s) for q, s in zip(ids, sims)}, {}


RAW_SEED = {"reranker", "embed"}   # these compare against the bare seed text, not the noul prompt


def make_backend(name: str, gpu: bool = False, allow_remote_code: bool = False) -> Backend:
    spec = MODELS[name]
    if spec.get("remote_code") and not allow_remote_code:
        raise SystemExit(f"{name} runs code shipped with the model; pass --allow-remote-code to use it.")
    kind = spec["kind"]
    if kind == "cloudflare":
        return Cloudflare(name, spec)
    if kind == "llama":
        return LlamaServer(name, spec, gpu=gpu)
    if kind == "jevk5":
        return JevK5(name, spec, gpu=gpu)
    if kind == "reranker":
        return Reranker(name, spec)
    if kind == "hf_decision":
        return HFDecision(name, spec)
    if kind == "gliner":
        return Gliner(name, spec)
    if kind == "embed":
        return Embed(name, spec)
    raise ValueError(kind)
