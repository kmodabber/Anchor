"""Extract a 0..100 probability from a raw model response.

Handles: "65", "65%", "0.65", "around 65 percent", trailing junk, multiple
numbers (takes the first plausible one, preferring percentage-shaped).
Returns None on failure — caller handles retry + logging.
"""
from __future__ import annotations

import re
from typing import Optional

_PCT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b|pct\b)", re.I)
_FRAC = re.compile(r"(?<![\d])0?\.(\d{1,4})(?!\d)")
# Bare number, integer or decimal ("65", "65.5"). The lookahead (?!\.?\d) stops
# "12345" matching as "123", while a sentence-ending "60." still matches 60.
_NUM = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)(?!\.?\d)")


def parse_probability(text: str) -> Optional[float]:
    """Return a float in [0, 100] or None."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    # 1) Explicit percentage: "65%", "65 percent", "65.5 pct".
    m = _PCT.search(s)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 100:
            return v

    # 2) Decimal fraction like "0.65" → 65. Only if clearly 0.x form.
    m = _FRAC.search(s)
    if m:
        frac = float("0." + m.group(1))
        if 0 <= frac <= 1:
            return round(frac * 100, 4)

    # 3) First bare number in range. A decimal like "1.0" is skipped: it could
    #    mean 1% or a probability of 1.0 (100%), so we refuse to guess.
    for m in _NUM.finditer(s):
        v = float(m.group(1))
        if "." in m.group(1) and v <= 1:
            continue
        if 0 <= v <= 100:
            return v

    return None
