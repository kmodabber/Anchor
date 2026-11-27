"""Loading and validating the input files.

Expected inputs (see PROTOCOL.md, "Input files"):
  markets.csv   - Evan:   market_id, source, category, question, created, resolved,
                          snapshot_date, snapshot_price, outcome, volume
  decoys.csv    - Kamyar: market_id, decoy_price
  split.csv     - this package (split.py): market_id, split
  responses     - Nima:   either the pipeline's raw JSONL (results/raw/<phase>/<model>.jsonl[.gz],
                          one record per call, see docs/method_notes.md) or a CSV with
                          market_id, model, condition, run, forecast
                          optional: prompt_variant, shown_price, parse_ok
Prices and forecasts may be on a 0-1 or 0-100 scale; they are converted to 0-1.
"""
from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class DataError(ValueError):
    pass


def _require(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{name} is missing columns: {missing}")


def _to_unit(series: pd.Series, name: str) -> pd.Series:
    """Convert a probability column to 0-1. A column whose max is above 1 is read as percent."""
    s = pd.to_numeric(series, errors="coerce")
    if s.max(skipna=True) > 1.0:
        s = s / 100.0
    bad = s.notna() & ((s < 0) | (s > 1))
    if bad.any():
        raise DataError(f"{name} has {int(bad.sum())} values outside 0-1 (or 0-100)")
    return s


def _to_outcome(series: pd.Series) -> pd.Series:
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = series[out.isna()].unique()[:5]
        raise DataError(f"outcome has unreadable values, e.g. {list(bad)}")
    return out.astype(int)


def load_markets(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"market_id": str})
    _require(df, ["market_id", "category", "snapshot_price", "outcome"], "markets.csv")
    if df["market_id"].duplicated().any():
        raise DataError("markets.csv has duplicate market_id values")
    df["snapshot_price"] = _to_unit(df["snapshot_price"], "snapshot_price")
    df["outcome"] = _to_outcome(df["outcome"])
    if "source" not in df.columns:
        df["source"] = "unknown"
    df["category"] = df["category"].fillna("other").astype(str)
    return df


def load_decoys(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"market_id": str})
    _require(df, ["market_id", "decoy_price"], "decoys.csv")
    df["decoy_price"] = _to_unit(df["decoy_price"], "decoy_price")
    return df[["market_id", "decoy_price"]]


def load_split(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"market_id": str})
    _require(df, ["market_id", "split"], "split.csv")
    df["split"] = df["split"].str.strip().str.lower()
    if not df["split"].isin(["dev", "test"]).all():
        raise DataError("split.csv: split must be 'dev' or 'test'")
    return df[["market_id", "split"]]


def _is_jsonl(path: Path) -> bool:
    return path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _count_lines(path: Path) -> int:
    with _open_text(path) as f:
        return sum(1 for line in f if line.endswith("\n"))  # a half-written last line does not count


def _jsonl_files(directory: Path) -> list[Path]:
    """The per-model files directly inside a phase directory (results/raw/<phase>/).
    The pipeline writes <model>.jsonl (resume copy, git-ignored) and <model>.jsonl.gz
    (committed). Read the .gz when it exists, since that is the reproducible copy,
    and warn when the plain file has more lines (the .gz is stale)."""
    stems: dict[str, dict[str, Path]] = {}
    for p in sorted(directory.iterdir()):
        if p.name.endswith(".jsonl.gz"):
            stems.setdefault(p.name[: -len(".jsonl.gz")], {})["gz"] = p
        elif p.name.endswith(".jsonl"):
            stems.setdefault(p.name[: -len(".jsonl")], {})["plain"] = p
    if not stems:
        raise DataError(f"{directory} has no .jsonl or .jsonl.gz files. Point --responses at a "
                        "phase folder such as results/raw/main, not at results/raw itself.")
    files = []
    for stem, found in stems.items():
        if "gz" in found:
            if "plain" in found and _count_lines(found["plain"]) > _count_lines(found["gz"]):
                warnings.warn(f"{found['plain']} has more records than {found['gz'].name}; reading the "
                              ".gz. Re-run the pipeline command to refresh it if the run has finished.",
                              stacklevel=3)
            files.append(found["gz"])
        else:
            files.append(found["plain"])
    return files


def _read_jsonl(path: Path) -> pd.DataFrame:
    """One pipeline JSONL file -> the responses.csv columns (plus provenance columns).

    parsed_probability is 0-100 (None on parse failure), shown_price is a whole
    percent (None when blind), final_status is "ok" or "parse_failure".
    """
    records, broken = [], 0
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1  # half-written last line of an interrupted run
    if broken:
        warnings.warn(f"{path}: skipped {broken} unreadable line(s)", stacklevel=3)
    need = ["market_id", "model", "condition", "run", "parsed_probability", "final_status"]
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["market_id", "model", "condition", "run", "forecast", "parse_ok"])
    _require(df, need, str(path))
    out = pd.DataFrame({
        "market_id": df["market_id"].astype(str),
        "model": df["model"].astype(str),
        "condition": df["condition"].astype(str),
        "run": pd.to_numeric(df["run"]).astype(int),
        "forecast": pd.to_numeric(df["parsed_probability"], errors="coerce") / 100.0,
        "parse_ok": df["final_status"].eq("ok"),
    })
    if "shown_price" in df.columns:
        out["shown_price"] = pd.to_numeric(df["shown_price"], errors="coerce") / 100.0
    if "prompt_variant" in df.columns:
        out["prompt_variant"] = df["prompt_variant"].map(lambda v: config.VARIANT_ALIASES.get(v, v))
    for col in ("phase", "config_hash", "model_digest"):
        if col in df.columns:
            out[col] = df[col]
    out["source_file"] = path.name
    return out


def _read_responses_one(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return pd.concat([_read_jsonl(p) for p in _jsonl_files(path)], ignore_index=True)
    if _is_jsonl(path):
        return _read_jsonl(path)
    return pd.read_csv(path, dtype={"market_id": str, "model": str})


def load_responses(paths) -> pd.DataFrame:
    """Read model responses from one or more sources, each of which may be
      - a pipeline phase folder (results/raw/main): every <model>.jsonl[.gz] in it,
      - a single pipeline .jsonl or .jsonl.gz file, or
      - a responses CSV.
    Pass the Llama and Qwen folders/files together, or main and paraphrase together.
    Do not pass dev and main together: both use the v1 prompt, so their rows collide."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise DataError(f"responses: {p} does not exist")
    df = pd.concat([_read_responses_one(p) for p in paths], ignore_index=True)
    _require(df, ["market_id", "model", "condition", "run", "forecast"], "responses")
    if "phase" in df.columns and (df["phase"] == "parser_test").any():
        warnings.warn("responses include parser_test records; these are pipeline checks, "
                      "not experiment runs", stacklevel=2)
    df["condition"] = df["condition"].str.strip().str.lower()
    bad = ~df["condition"].isin(config.CONDITIONS)
    if bad.any():
        raise DataError(f"responses.csv: unknown conditions {df.loc[bad, 'condition'].unique()[:5]}")
    if "prompt_variant" not in df.columns:
        df["prompt_variant"] = config.ORIGINAL_VARIANT
    df["prompt_variant"] = df["prompt_variant"].fillna(config.ORIGINAL_VARIANT).astype(str)
    df["forecast"] = _to_unit(df["forecast"], "forecast")
    if "parse_ok" in df.columns:
        ok = df["parse_ok"].astype(str).str.lower().isin(["1", "true", "yes", "1.0"])
        df.loc[~ok, "forecast"] = np.nan
    if "shown_price" in df.columns:
        df["shown_price"] = _to_unit(df["shown_price"], "shown_price")
    key = ["prompt_variant", "model", "market_id", "condition", "run"]
    dup = df.duplicated(key)
    if dup.any():
        hint = ""
        if "phase" in df.columns and df["phase"].nunique() > 1:
            hint = f" (phases {sorted(df['phase'].dropna().unique())} were mixed; pass one of dev/main, not both)"
        raise DataError(f"responses have {int(dup.sum())} duplicate rows for {key}{hint}")
    return df


def check_inputs(markets: pd.DataFrame, decoys: pd.DataFrame, responses: pd.DataFrame) -> list[str]:
    """Return human-readable warnings about rule violations. Never modifies data."""
    notes: list[str] = []
    lo, hi = config.SNAPSHOT_RANGE
    out = markets[(markets["snapshot_price"] < lo) | (markets["snapshot_price"] > hi)]
    if len(out):
        notes.append(f"{len(out)} markets have a snapshot price outside {lo:.0%}-{hi:.0%}.")
    if "created" in markets.columns:
        created = pd.to_datetime(markets["created"], errors="coerce", utc=True)
        early = created < pd.Timestamp(config.MIN_CREATED_DATE, tz="UTC")
        if early.any():
            notes.append(f"{int(early.sum())} markets were created before {config.MIN_CREATED_DATE} (contamination risk).")
    m = markets.merge(decoys, on="market_id", how="left")
    if m["decoy_price"].isna().any():
        notes.append(f"{int(m['decoy_price'].isna().sum())} markets have no decoy price.")
    close = (m["decoy_price"] - m["snapshot_price"]).abs() < config.MIN_DECOY_DISTANCE - 1e-9
    if close.any():
        notes.append(f"{int(close.sum())} decoys are closer than {config.MIN_DECOY_DISTANCE:.0%} to the real price.")
    unknown = set(responses["market_id"]) - set(markets["market_id"])
    if unknown:
        notes.append(f"{len(unknown)} market_ids in responses.csv are not in markets.csv; they are ignored.")
    unknown_models = set(responses["model"]) - set(config.MODELS)
    if unknown_models:
        notes.append(f"Models not in config.MODELS (kept, but left out of planned contrasts): {sorted(unknown_models)}")
    if "config_hash" in responses.columns:
        for variant, g in responses.groupby("prompt_variant"):
            hashes = g["config_hash"].dropna().unique()
            if len(hashes) > 1:
                notes.append(f"Prompt variant '{variant}' mixes {len(hashes)} pipeline config hashes "
                             f"{sorted(hashes)}; both machines should run the same config.")
    if "model_digest" in responses.columns:
        for model, g in responses.groupby("model"):
            digests = g["model_digest"].dropna().unique()
            if len(digests) > 1:
                notes.append(f"{model} was run with {len(digests)} different model digests.")
    if "shown_price" in responses.columns:
        r = responses.merge(m[["market_id", "snapshot_price", "decoy_price"]], on="market_id", how="left")
        expect = np.where(r["condition"] == "informed", r["snapshot_price"],
                          np.where(r["condition"] == "decoy", r["decoy_price"], np.nan))
        mism = (r["condition"] != "blind") & r["shown_price"].notna() & (np.abs(r["shown_price"] - expect) > 0.006)
        if mism.any():
            notes.append(f"{int(mism.sum())} response rows show a price that differs from markets.csv/decoys.csv; "
                         "shown_price from responses.csv is used.")
    return notes


def warn_all(notes: list[str]) -> None:
    for n in notes:
        warnings.warn(n, stacklevel=2)
