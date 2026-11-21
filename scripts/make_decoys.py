"""Write data/decoys.csv from data/markets.csv using the settings in config.yaml.

Run from the repo root:
    python -m scripts.make_decoys
"""
from __future__ import annotations

import csv
import sys

from common.config import REPO_ROOT, load_config
from pipeline.decoy import make_decoys


def main() -> int:
    cfg = load_config()
    d = cfg["decoy"]
    markets_path = REPO_ROOT / cfg["data"]["markets_csv"]
    out_path = REPO_ROOT / d["output_csv"]

    if not markets_path.exists():
        print(f"{markets_path} not found yet (waiting on the data pull).", file=sys.stderr)
        return 1

    with open(markets_path, newline="", encoding="utf-8") as f:
        markets = list(csv.DictReader(f))

    rows = make_decoys(markets, seed=cfg["experiment"]["decoy_seed"],
                       low=d["low_points"], high=d["high_points"],
                       min_gap=d["min_gap_points"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    above = sum(r["decoy_price"] > r["snapshot_price"] for r in rows)
    print(f"wrote {len(rows)} decoys to {out_path} "
          f"({above} above the real price, {len(rows) - above} below)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
