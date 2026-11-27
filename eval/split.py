"""Seeded, category-stratified dev/test split.

Usage:
    python -m eval.split --markets data/markets.csv --out data/split.csv

Within each category, markets are sorted by market_id (so the input row order
doesn't matter), shuffled with a fixed seed, and the first round(30% x n) go to
dev. Running it again on the same markets.csv gives the same split.
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from . import config
from .io import load_markets


def make_split(markets: pd.DataFrame, dev_fraction: float = config.DEV_FRACTION,
               seed: int = config.SPLIT_SEED) -> pd.DataFrame:
    rows = []
    for cat, grp in sorted(markets.groupby("category"), key=lambda t: t[0]):
        ids = np.sort(grp["market_id"].astype(str).to_numpy())
        # Category-specific seed so adding one category doesn't reshuffle the others.
        cat_seed = int(hashlib.sha256(f"{seed}:{cat}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(cat_seed)
        ids = ids[rng.permutation(len(ids))]
        n_dev = int(round(dev_fraction * len(ids)))
        rows += [(i, "dev") for i in ids[:n_dev]] + [(i, "test") for i in ids[n_dev:]]
    return pd.DataFrame(rows, columns=["market_id", "split"]).sort_values("market_id").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    markets = load_markets(args.markets)
    split = make_split(markets)
    split.to_csv(args.out, index=False)
    summary = split.merge(markets[["market_id", "category"]], on="market_id") \
        .groupby(["category", "split"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print(f"\nTotal: {(split.split == 'dev').sum()} dev, {(split.split == 'test').sum()} test -> {args.out}")


if __name__ == "__main__":
    main()
