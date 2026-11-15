"""Expand number shorthand in extracted values before cached code sees them.

DEVIATION from the paper (config.EXPAND_SHORTHAND, default on). IS2 copies each value as
written, so a question can hand the cached code "$150k" or "€2.0 million". Builder-written
parsers often read only the leading digits, and "$150k" became 150 (the chat-loop run served
"0.92 dollars" for a $921.13 payment). Expanding the shorthand here, in plain code, fixes
every entry at once and can't misread anything.

Only unambiguous forms are expanded:
  words, always            "2.5 million people" -> "2,500,000 people", "3 thousand"
  single letters, only     "$150k" -> "$150,000", "€2.0M" -> "€2,000,000", "£1.5bn"
  after a currency sign    (never "15 m" metres, "0.1 M" molar, "300 K" kelvin, "5 km")
A value with no shorthand is returned unchanged, character for character.
"""

from __future__ import annotations

import re

_CURRENCY = r"[$£€¥₹]|USD|EUR|GBP|US\$"
_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_LETTERS = {"k": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12, "tn": 1e12}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_WORD_RE = re.compile(rf"(?P<cur>(?:{_CURRENCY})\s?)?(?P<num>{_NUM})\s*(?P<suf>{'|'.join(_WORDS)})\b", re.I)
_LETTER_RE = re.compile(rf"(?P<cur>(?:{_CURRENCY})\s?)(?P<num>{_NUM})\s?(?P<suf>{'|'.join(sorted(_LETTERS, key=len, reverse=True))})(?![A-Za-z])",
                        re.I)


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.10f}".rstrip("0").rstrip(".")


def _expand(m: re.Match, factor: float) -> str:
    value = float(m.group("num").replace(",", "")) * factor
    return f"{m.group('cur') or ''}{_fmt(value)}"


def expand_shorthand(value: str) -> str:
    """'$150k' -> '$150,000'; '€2.0 million' -> '€2,000,000'; anything else unchanged."""
    if not isinstance(value, str) or not value:
        return value
    out = _WORD_RE.sub(lambda m: _expand(m, _WORDS[m.group("suf").lower()]), value)
    return _LETTER_RE.sub(lambda m: _expand(m, _LETTERS[m.group("suf").lower()]), out)


def expand_values(values: dict) -> dict:
    return {k: expand_shorthand(v) for k, v in values.items()}
