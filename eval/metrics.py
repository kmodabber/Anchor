"""Metrics and bootstrap intervals for ANCHOR.

All statistics are computed from one market-level table: for each prompt variant,
model and market, the mean forecast over valid runs in each condition.

Every statistic is written so it can be evaluated on many bootstrap resamples at
once: functions take arrays indexed by a (B, n) matrix of resampled market
positions and return one value per resample. The point estimate is the same
function evaluated on the identity resample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Statistics reported per model. Keys are stable; they become CSV column names.
STAT_LABELS = {
    "brier_blind": "Brier, blind",
    "brier_informed": "Brier, informed",
    "brier_decoy": "Brier, decoy",
    "brier_parrot": "Brier, parrot baseline (real price)",
    "ac_informed": "Anchoring coefficient, informed",
    "ac_decoy": "Anchoring coefficient, decoy",
    "ac_gap": "Anchoring coefficient gap (informed - decoy)",
    "slope_informed": "Anchoring slope, informed",
    "slope_decoy": "Anchoring slope, decoy",
    "slope_gap": "Anchoring slope gap (informed - decoy)",
    "cslope_informed": "Corrected anchoring slope, informed",
    "cslope_decoy": "Corrected anchoring slope, decoy",
    "cslope_gap": "Corrected anchoring slope gap (informed - decoy)",
    "brier_gain": "Brier improvement from real price (blind - informed)",
    "brier_vs_parrot": "Brier, informed - parrot",
}


# ---------------------------------------------------------------------------
# Market-level table
# ---------------------------------------------------------------------------

def market_table(responses: pd.DataFrame, markets: pd.DataFrame, decoys: pd.DataFrame,
                 min_valid_frac: float = config.MIN_VALID_RUN_FRAC) -> pd.DataFrame:
    """One row per (prompt_variant, model, market_id) with per-condition means.

    Columns: p_<cond>, k_<cond> (valid runs), n_<cond> (runs attempted),
    var_blind (sample variance of blind runs), shown_informed, shown_decoy,
    outcome, category, source, complete (all three conditions usable).
    """
    keys = ["prompt_variant", "model", "market_id"]
    r = responses[responses["market_id"].isin(markets["market_id"])]
    g = r.groupby(keys + ["condition"])["forecast"]
    agg = pd.DataFrame({
        "p": g.mean(),
        "k": g.count(),
        "n": g.size(),
        "var": g.var(ddof=1),
    }).reset_index()

    usable = agg["k"] >= np.ceil(min_valid_frac * agg["n"]).clip(lower=1)
    agg.loc[~usable, "p"] = np.nan

    wide = agg.pivot_table(index=keys, columns="condition", values=["p", "k", "n", "var"], dropna=False)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in config.CONDITIONS:
        for a in ("p", "k", "n"):
            if f"{a}_{c}" not in wide.columns:
                wide[f"{a}_{c}"] = np.nan
    wide["var_blind"] = wide.get("var_blind", np.nan)
    wide = wide[[*keys, *[f"{a}_{c}" for c in config.CONDITIONS for a in ("p", "k", "n")], "var_blind"]]

    m = markets[["market_id", "snapshot_price", "outcome", "category", "source"]].merge(
        decoys, on="market_id", how="left")
    wide = wide.merge(m, on="market_id", how="left")
    wide["shown_informed"] = wide["snapshot_price"]
    wide["shown_decoy"] = wide["decoy_price"]

    if "shown_price" in responses.columns:
        shown = (r[r["condition"] != "blind"]
                 .groupby(keys + ["condition"])["shown_price"].mean().unstack("condition"))
        shown.columns = [f"shown_{c}" for c in shown.columns]
        shown = shown.reset_index()
        wide = wide.merge(shown, on=keys, how="left", suffixes=("", "_resp"))
        for c in ("informed", "decoy"):
            col = f"shown_{c}_resp"
            if col in wide.columns:
                wide[f"shown_{c}"] = wide[col].fillna(wide[f"shown_{c}"])
                wide = wide.drop(columns=col)

    wide["complete"] = wide[[f"p_{c}" for c in config.CONDITIONS]].notna().all(axis=1) & \
        wide[["shown_informed", "shown_decoy"]].notna().all(axis=1)
    return wide


def model_arrays(mt: pd.DataFrame) -> dict[str, np.ndarray]:
    """Arrays for one model's complete markets, in a fixed order (by market_id)."""
    d = mt[mt["complete"]].sort_values("market_id")
    k = d["k_blind"].to_numpy(float)
    var = d["var_blind"].to_numpy(float)
    # Noise variance of the blind mean. Markets with one valid blind run have no
    # variance estimate; use the median of the others.
    sig2 = var / k
    if np.isnan(sig2).all():
        sig2 = np.zeros_like(sig2)
    else:
        sig2 = np.where(np.isnan(sig2), np.nanmedian(sig2), sig2)
    return {
        "market_id": d["market_id"].to_numpy(),
        "y": d["outcome"].to_numpy(float),
        "pb": d["p_blind"].to_numpy(float),
        "pi": d["p_informed"].to_numpy(float),
        "pd": d["p_decoy"].to_numpy(float),
        "si": d["shown_informed"].to_numpy(float),
        "sd": d["shown_decoy"].to_numpy(float),
        "sig2": sig2,
    }


# ---------------------------------------------------------------------------
# Vectorised statistics
# ---------------------------------------------------------------------------

def _masked_mean(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (np.where(m, v, 0.0)).sum(axis=1) / cnt, np.nan)


def _anchoring(pc, shown, pb, sig2, min_gap):
    """Return (mean coefficient with exclusion, slope, corrected slope) per resample row."""
    x = shown - pb            # gap between shown price and blind forecast
    yv = pc - pb              # how far the model moved
    elig = np.abs(x) >= min_gap
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(elig, yv / np.where(elig, x, 1.0), 0.0)
    ac = _masked_mean(ratio, elig)
    sxy = (x * yv).sum(axis=1)
    sxx = (x * x).sum(axis=1)
    ss = sig2.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = sxy / sxx
        denom = sxx - ss
        cslope = np.where(denom > 0.05 * sxx, (sxy - ss) / denom, np.nan)
    return ac, slope, cslope


def compute_stats(a: dict[str, np.ndarray], idx: np.ndarray | None = None,
                  min_gap: float = config.MIN_GAP) -> dict[str, np.ndarray]:
    """Evaluate every statistic in STAT_LABELS on each row of idx (B x n)."""
    n = len(a["y"])
    if idx is None:
        idx = np.arange(n)[None, :]
    g = {k: v[idx] for k, v in a.items() if k != "market_id"}
    y = g["y"]
    out: dict[str, np.ndarray] = {}
    out["brier_blind"] = ((g["pb"] - y) ** 2).mean(axis=1)
    out["brier_informed"] = ((g["pi"] - y) ** 2).mean(axis=1)
    out["brier_decoy"] = ((g["pd"] - y) ** 2).mean(axis=1)
    out["brier_parrot"] = ((g["si"] - y) ** 2).mean(axis=1)
    for cond, pc, s in (("informed", g["pi"], g["si"]), ("decoy", g["pd"], g["sd"])):
        ac, sl, cs = _anchoring(pc, s, g["pb"], g["sig2"], min_gap)
        out[f"ac_{cond}"], out[f"slope_{cond}"], out[f"cslope_{cond}"] = ac, sl, cs
    out["ac_gap"] = out["ac_informed"] - out["ac_decoy"]
    out["slope_gap"] = out["slope_informed"] - out["slope_decoy"]
    out["cslope_gap"] = out["cslope_informed"] - out["cslope_decoy"]
    out["brier_gain"] = out["brier_blind"] - out["brier_informed"]
    out["brier_vs_parrot"] = out["brier_informed"] - out["brier_parrot"]
    return out


def eligible_counts(a: dict[str, np.ndarray], min_gap: float = config.MIN_GAP) -> dict[str, int]:
    return {
        "n_markets": int(len(a["y"])),
        "n_ac_informed": int((np.abs(a["si"] - a["pb"]) >= min_gap).sum()),
        "n_ac_decoy": int((np.abs(a["sd"] - a["pb"]) >= min_gap).sum()),
    }


def medians(a: dict[str, np.ndarray], min_gap: float = config.MIN_GAP) -> dict[str, float]:
    """Median anchoring coefficients (point estimates only; a robustness check)."""
    res = {}
    for cond, pc, s in (("informed", a["pi"], a["si"]), ("decoy", a["pd"], a["sd"])):
        x = s - a["pb"]
        e = np.abs(x) >= min_gap
        res[f"ac_median_{cond}"] = float(np.median((pc - a["pb"])[e] / x[e])) if e.any() else np.nan
    return res


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def resample_index(n: int, n_boot: int = config.N_BOOT, seed: int = config.BOOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def ci(samples: np.ndarray, level: float = config.CI_LEVEL) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if len(s) < 0.9 * len(samples) or len(s) == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def summarise(a: dict[str, np.ndarray], n_boot=config.N_BOOT, seed=config.BOOT_SEED,
              min_gap=config.MIN_GAP) -> dict[str, float]:
    """Point estimate and 95% interval for every statistic, plus counts and medians."""
    n = len(a["y"])
    row: dict[str, float] = dict(eligible_counts(a, min_gap))
    if n == 0:
        return row
    point = compute_stats(a, None, min_gap)
    boot = compute_stats(a, resample_index(n, n_boot, seed), min_gap)
    for k in STAT_LABELS:
        lo, hi = ci(boot[k])
        row[k] = float(point[k][0])
        row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
    row.update(medians(a, min_gap))
    return row


def align(a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
    """Restrict two models' arrays to their common markets, in the same order."""
    common = np.intersect1d(a["market_id"], b["market_id"])
    ia = np.searchsorted(a["market_id"], common)
    ib = np.searchsorted(b["market_id"], common)
    return ({k: v[ia] for k, v in a.items()}, {k: v[ib] for k, v in b.items()})


def contrast(a: dict, b: dict, stats=("cslope_gap", "ac_gap", "brier_gain", "brier_informed"),
             n_boot=config.N_BOOT, seed=config.BOOT_SEED, min_gap=config.MIN_GAP) -> dict[str, float]:
    """Paired bootstrap of (model a - model b) on their common markets."""
    a2, b2 = align(a, b)
    n = len(a2["y"])
    row: dict[str, float] = {"n_markets": n}
    if n == 0:
        return row
    idx = resample_index(n, n_boot, seed)
    pa, pb_ = compute_stats(a2, None, min_gap), compute_stats(b2, None, min_gap)
    ba, bb = compute_stats(a2, idx, min_gap), compute_stats(b2, idx, min_gap)
    for k in stats:
        row[k] = float(pa[k][0] - pb_[k][0])
        row[f"{k}_lo"], row[f"{k}_hi"] = ci(ba[k] - bb[k])
    return row


# ---------------------------------------------------------------------------
# Verdicts (success criteria from PROTOCOL.md)
# ---------------------------------------------------------------------------

def _sign(lo: float, hi: float) -> int:
    """+1 if the interval is above zero, -1 if below, 0 if it contains zero or is undefined."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def verdict(row: dict, primary: str = "cslope") -> str:
    gap = _sign(row[f"{primary}_gap_lo"], row[f"{primary}_gap_hi"])
    gain = _sign(row["brier_gain_lo"], row["brier_gain_hi"])
    inf = _sign(row[f"{primary}_informed_lo"], row[f"{primary}_informed_hi"])
    dec = _sign(row[f"{primary}_decoy_lo"], row[f"{primary}_decoy_hi"])
    if gap > 0 and gain > 0:
        return "Reasoning: weighs the real price more than the decoy, and the real price improves accuracy"
    if gap > 0:
        return "Discriminates: weighs the real price more than the decoy, but no clear accuracy gain"
    if gap == 0 and dec > 0:
        return "Parroting: moves toward the decoy about as much as toward the real price"
    if inf <= 0 and dec <= 0:
        return "Ignores the price: no clear movement toward either price"
    if gap < 0:
        return "Inverted: weighs the decoy more than the real price"
    return "Inconclusive"


def parse_rates(responses: pd.DataFrame) -> pd.DataFrame:
    g = responses.groupby(["prompt_variant", "model", "condition"])["forecast"]
    out = pd.DataFrame({"calls": g.size(), "parse_failures": g.apply(lambda s: int(s.isna().sum()))})
    out["failure_rate"] = out["parse_failures"] / out["calls"]
    return out.reset_index()
