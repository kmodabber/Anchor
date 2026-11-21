"""Decoy price generator.

One decoy per market, drawn once with a fixed seed. The decoy is a whole
percentage in [low, high] that is at least `min_gap` points from the real
snapshot price. Each market's draw depends only on (decoy_seed, market_id),
so adding, removing or reordering markets never changes another market's decoy.
"""
from __future__ import annotations

import hashlib
import random


def _market_rng(seed: int, market_id: str) -> random.Random:
    # Python's built-in hash() is salted per process, so use sha256 instead.
    digest = hashlib.sha256(f"{seed}:{market_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def decoy_candidates(real_price: float, low: int = 5, high: int = 95,
                     min_gap: int = 20) -> list[int]:
    """All allowed decoy values (in percentage points) for a real price in [0, 1]."""
    if not 0.0 <= real_price <= 1.0:
        raise ValueError(f"real_price must be in [0, 1], got {real_price}")
    real_pts = real_price * 100
    return [k for k in range(low, high + 1) if abs(k - real_pts) >= min_gap]


def make_decoy(market_id: str, real_price: float, seed: int, low: int = 5,
               high: int = 95, min_gap: int = 20) -> float:
    """Return the decoy price for one market as a probability in [0, 1]."""
    candidates = decoy_candidates(real_price, low, high, min_gap)
    if not candidates:
        raise ValueError(f"no valid decoy for market {market_id} at price {real_price}")
    return _market_rng(seed, str(market_id)).choice(candidates) / 100


def make_decoys(markets: list[dict], seed: int, low: int = 5, high: int = 95,
                min_gap: int = 20) -> list[dict]:
    """markets: dicts with 'market_id' and 'snapshot_price' (0-1).
    Returns rows of market_id, snapshot_price, decoy_price, gap_points."""
    rows = []
    for m in markets:
        price = float(m["snapshot_price"])
        decoy = make_decoy(m["market_id"], price, seed, low, high, min_gap)
        rows.append({
            "market_id": m["market_id"],
            "snapshot_price": price,
            "decoy_price": decoy,
            "gap_points": round(abs(decoy - price) * 100, 2),
        })
    return rows
