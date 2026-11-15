"""Download the local candidate models into the Hugging Face cache (~/.cache/huggingface).

    .venv/bin/python eval/decision/download.py --list
    .venv/bin/python eval/decision/download.py jevos-1b kev-0.8b ...
    .venv/bin/python eval/decision/download.py --all-safe      # every model that runs no downloaded code

GGUF models fetch one file. Python models fetch the repo's weights and config files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODELS  # noqa: E402

LOCAL = [n for n, s in MODELS.items() if s.get("repo")]


def fetch(name: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    s = MODELS[name]
    print(f"-> {name}: {s['repo']} {s.get('file', '')} (~{s['mb']} MB)", flush=True)
    if s.get("file"):
        p = hf_hub_download(s["repo"], s["file"])
    elif s.get("allow"):
        p = snapshot_download(s["repo"], allow_patterns=s["allow"])
    else:
        p = snapshot_download(s["repo"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                          "*.py", "*.jinja", "backbone/*", "tokenizer*"])
    print(f"   saved {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all-safe", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.names or a.all_safe):
        total = 0
        for n in LOCAL:
            s = MODELS[n]
            total += s["mb"]
            code = "runs model code" if s.get("remote_code") else "data only"
            print(f"{n:<22} {s['mb']:>6} MB  {s['license']:<11} {code:<16} {s['repo']} {s.get('file', '')}")
        print(f"{'total':<22} {total:>6} MB")
        return
    names = [n for n in LOCAL if not MODELS[n].get("remote_code")] if a.all_safe else a.names
    for n in names:
        if n not in LOCAL:
            raise SystemExit(f"unknown local model {n}; see --list")
        fetch(n)


if __name__ == "__main__":
    main()
