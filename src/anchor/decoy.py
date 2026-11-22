"""Decoy price access.

PLACEHOLDER until Kamyar's generator lands. The pipeline calls one
function here; swapping Kamyar's implementation in means replacing the
body of `decoy_for` only.
"""
from __future__ import annotations

import random
from typing import Optional


def decoy_for(
    market_id: str,
    real_price: int,
    *,
    seed: int,
    min_distance: int = 20,
) -> int:
    """Return an integer decoy price in [0, 100] at least ``min_distance``
    away from ``real_price``. Deterministic given ``seed``.

    PLACEHOLDER IMPLEMENTATION. Kamyar owns the real one.
    """
    rng = random.Random(f"{seed}|{market_id}")
    low_max = real_price - min_distance
    high_min = real_price + min_distance
    candidates: list[tuple[int, int]] = []
    if low_max >= 0:
        candidates.append((0, low_max))
    if high_min <= 100:
        candidates.append((high_min, 100))
    if not candidates:
        # Impossible gap — fall back to the farthest endpoint.
        return 0 if real_price >= 50 else 100
    lo, hi = rng.choice(candidates)
    return rng.randint(lo, hi)


def load_precomputed_decoys(path: Optional[str]) -> dict[str, int]:
    """Optional hook: if Kamyar ships a CSV/JSON of precomputed decoys,
    load it here. Returns {market_id: decoy_price}.
    """
    if not path:
        return {}
    import json
    import os
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}
