"""Checks that use no language model, shown next to LLM judge verdicts.

number_check: does the final number in a reference answer appear in another
answer (within a relative tolerance)? It is a heuristic, not a grader: it can
miss answers written in other units and it can match a stray number. Its job
is to flag verdicts worth a human look, never to replace the judges.
"""

from __future__ import annotations

import re

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SCI = re.compile(r"(\d(?:[\d.]*))\s*(?:×|x|\*|·)\s*10\s*\^?\s*\(?\s*([-−+]?\d+)\)?")
_NUM = re.compile(r"(?<![\w.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-−+]?\d+)?(?!\.?\d)")


def numbers(text: str) -> list[float]:
    """Every number in the text, with 1.2×10^6, 1.2 × 10⁶ and 1.2e6 all read as 1.2e6."""
    t = (text or "").translate(_SUPERSCRIPT).replace("−", "-")
    t = _SCI.sub(lambda m: f"{m.group(1)}e{m.group(2)}", t)
    out = []
    for m in _NUM.finditer(t):
        s = m.group(0).replace(",", "")
        if not re.search(r"\d", s):
            continue
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def final_line(reference: str) -> str:
    """The part of a reference that states the answer (the first paragraph)."""
    return (reference or "").strip().split("\n\n")[0]


def number_check(answer: str, reference: str, rel_tol: float = 0.01) -> dict:
    """Compare the numbers in a reference's final answer with the numbers in an answer."""
    ref = [x for x in numbers(final_line(reference)) if x != 0]
    ans = numbers(answer)
    base = {"tolerance": rel_tol, "reference_numbers": ref[:8], "answer_numbers": ans[:16]}
    if not ref or not ans:
        return {**base, "status": "unclear", "detail": "no numbers to compare"}
    best = None
    # Factor 1 first; powers of 1000 catch the same value with another SI prefix (kPa vs Pa).
    for factor in (1, 1e3, 1e-3, 1e6, 1e-6):
        for r in ref:
            for a in ans:
                diff = abs(a - r * factor) / abs(r * factor)
                if best is None or diff < best[2] - 1e-12:
                    best = (r, a, diff, factor)
        if best and best[2] <= rel_tol:
            break
    r, a, diff, factor = best
    status = "match" if diff <= rel_tol else "mismatch"
    scale = "" if factor == 1 else f" after scaling by {factor:g} (likely a unit prefix)"
    return {**base, "status": status, "closest_reference": r, "closest_answer": a,
            "relative_difference": diff, "scale": factor,
            "detail": f"reference {r:g} vs answer {a:g}{scale}: {diff * 100:.2f}% apart"}


def prompt_sections(prompt: str) -> list[dict]:
    """Split a '## Heading' style prompt into labeled sections (used to show judge evidence)."""
    sections, title, buf = [], "", []
    for line in (prompt or "").splitlines():
        if line.startswith("## "):
            if title or any(x.strip() for x in buf):
                sections.append({"title": title, "text": "\n".join(buf).strip()})
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if title or any(x.strip() for x in buf):
        sections.append({"title": title, "text": "\n".join(buf).strip()})
    return sections
