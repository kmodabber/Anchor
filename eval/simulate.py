"""Simulated ANCHOR data with known anchoring weights, for testing the evaluation code.

Usage:
    python -m eval.simulate --out sim_data [--markets 300] [--seed 7]

Writes markets.csv, decoys.csv and responses.csv in the same formats the team's
real files use, plus truth.json with the weights each simulated model was given.
None of this is real data. Never mix these files with results.

Each simulated model has a private belief b for every market. Per run it answers
  blind:    b                    + noise
  informed: b + w_inf * (s - b)  + noise    (s = real snapshot price)
  decoy:    b + w_dec * (d - b)  + noise    (d = decoy price)
so w_inf and w_dec are the true anchoring weights the metrics should recover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {"politics": 0.30, "economics": 0.20, "sports": 0.20, "crypto": 0.15, "weather": 0.10, "other": 0.05}

# w_inf, w_dec, belief informativeness, per-run noise sd, parse failure rate
SIM_MODELS = {
    "llama3.2:3b": dict(w_inf=0.60, w_dec=0.60, info=0.25, noise=0.12, fail=0.06),  # pure parrot
    "llama3.1:8b": dict(w_inf=0.65, w_dec=0.35, info=0.45, noise=0.07, fail=0.02),
    "qwen2.5:3b": dict(w_inf=0.45, w_dec=0.40, info=0.30, noise=0.10, fail=0.05),
    "qwen2.5:7b": dict(w_inf=0.55, w_dec=0.20, info=0.50, noise=0.06, fail=0.01),
}
PARAPHRASE_SHIFT = {"w_inf": -0.05, "w_dec": +0.05}  # paraphrase narrows every gap a little


def simulate(n_markets=300, seed=7, runs=5, paraphrase_runs=3, test_ids=None):
    rng = np.random.default_rng(seed)
    ids = np.array([f"M{i:04d}" for i in range(n_markets)])
    cats = rng.choice(list(CATEGORIES), size=n_markets, p=list(CATEGORIES.values()))
    q = rng.beta(1.5, 1.5, n_markets)                     # true probability
    s = np.clip(q + rng.normal(0, 0.05, n_markets), 0.05, 0.95)
    y = rng.binomial(1, q)
    d = np.empty(n_markets)
    for i in range(n_markets):                            # decoy at least 0.20 away
        while True:
            v = rng.uniform(0.05, 0.95)
            if abs(v - s[i]) >= 0.20:
                d[i] = v
                break
    markets = pd.DataFrame({
        "market_id": ids, "source": rng.choice(["polymarket", "kalshi"], n_markets, p=[0.6, 0.4]),
        "category": cats, "question": [f"Simulated question {i}" for i in range(n_markets)],
        "created": "2025-06-01", "resolved": "2026-03-01", "snapshot_date": "2026-02-22",
        "snapshot_price": np.round(s * 100, 1), "outcome": np.where(y == 1, "YES", "NO"),
        "volume": rng.integers(10_000, 2_000_000, n_markets),
    })
    decoys = pd.DataFrame({"market_id": ids, "decoy_price": np.round(d * 100, 1)})

    test_set = set(ids if test_ids is None else test_ids)
    rows = []
    for model, p in SIM_MODELS.items():
        b = np.clip(0.5 + p["info"] * (q - 0.5) * 2 * 0.8 + rng.normal(0, 0.12, n_markets), 0.03, 0.97)
        for variant, k, shift in (("original", runs, {"w_inf": 0, "w_dec": 0}),
                                  ("paraphrase", paraphrase_runs, PARAPHRASE_SHIFT)):
            w_inf, w_dec = p["w_inf"] + shift["w_inf"], p["w_dec"] + shift["w_dec"]
            for i, mid in enumerate(ids):
                if variant == "paraphrase" and mid not in test_set:
                    continue
                for cond, target, w in (("blind", None, 0.0), ("informed", s[i], w_inf), ("decoy", d[i], w_dec)):
                    mean = b[i] if target is None else b[i] + w * (target - b[i])
                    for r in range(1, k + 1):
                        val = float(np.clip(mean + rng.normal(0, p["noise"]), 0.01, 0.99))
                        ok = rng.random() >= p["fail"]
                        rows.append((variant, model, mid, cond, r, round(val * 100) if ok else np.nan, int(ok),
                                     None if target is None else round(target * 100, 1)))
    responses = pd.DataFrame(rows, columns=["prompt_variant", "model", "market_id", "condition", "run",
                                            "forecast", "parse_ok", "shown_price"])
    truth = {m: {"w_inf": p["w_inf"], "w_dec": p["w_dec"], "gap": round(p["w_inf"] - p["w_dec"], 4)}
             for m, p in SIM_MODELS.items()}
    return markets, decoys, responses, truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markets", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .split import make_split
    markets, decoys, _, _ = simulate(args.markets, args.seed)
    split = make_split(markets.assign(snapshot_price=markets.snapshot_price / 100))
    test_ids = split.loc[split.split == "test", "market_id"].to_numpy()
    markets, decoys, responses, truth = simulate(args.markets, args.seed, test_ids=test_ids)
    markets.to_csv(out / "markets.csv", index=False)
    decoys.to_csv(out / "decoys.csv", index=False)
    responses.to_csv(out / "responses.csv", index=False)
    split.to_csv(out / "split.csv", index=False)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote simulated data to {out} ({len(markets)} markets, {len(responses)} response rows)")


if __name__ == "__main__":
    main()
