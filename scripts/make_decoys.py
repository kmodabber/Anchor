"""Write data/decoys.csv from data/markets.csv using the settings in config.yaml.

Run from the repo root:
    python -m scripts.make_decoys
"""
from __future__ import annotations

import csv
import sys

from common.config import REPO_ROOT, load_config
from pipeline.decoy import make_decoys


FIELDS = ["market_id", "snapshot_price", "decoy_price", "gap_points"]


def main(markets_path=None, out_path=None) -> int:
    """The path arguments exist so tests can point at temp files; normal runs
    take both paths from config.yaml."""
    cfg = load_config()
    d = cfg["decoy"]
    markets_path = markets_path or REPO_ROOT / cfg["data"]["markets_csv"]
    out_path = out_path or REPO_ROOT / d["output_csv"]

    if not markets_path.exists():
        print(f"{markets_path} not found yet (waiting on the data pull).", file=sys.stderr)
        return 1

    with open(markets_path, newline="", encoding="utf-8") as f:
        markets = list(csv.DictReader(f))

    if not markets:
        print(f"{markets_path} has no market rows; nothing to do.", file=sys.stderr)
        return 1

    try:
        rows = make_decoys(markets, seed=cfg["experiment"]["decoy_seed"],
                           low=d["low_points"], high=d["high_points"],
                           min_gap=d["min_gap_points"])
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    above = sum(r["decoy_price"] > r["snapshot_price"] for r in rows)
    print(f"wrote {len(rows)} decoys to {out_path} "
          f"({above} above the real price, {len(rows) - above} below)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
