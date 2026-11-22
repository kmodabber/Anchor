"""Load and validate config/config.yaml.

Usage:
    from common.config import load_config
    cfg = load_config()
    cfg["experiment"]["temperature"]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

REQUIRED_SECTIONS = ("experiment", "data", "decoy", "conditions", "models",
                     "parser", "evaluation", "paths")
SEED_KEYS = ("seed", "split_seed", "decoy_seed", "call_seed", "bootstrap_seed")


def _is_int(x) -> bool:
    # bool is a subclass of int in Python, so `seed: true` would otherwise pass
    return isinstance(x, int) and not isinstance(x, bool)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config missing sections: {missing}")

    exp = cfg["experiment"]
    if not 0 < exp["dev_fraction"] < 1:
        raise ValueError("experiment.dev_fraction must be between 0 and 1")
    if exp["temperature"] < 0:
        raise ValueError("experiment.temperature must be >= 0")
    for key in SEED_KEYS:
        if not _is_int(exp.get(key)):
            raise ValueError(f"experiment.{key} must be an integer")

    d = cfg["decoy"]
    if not 0 <= d["low_points"] < d["high_points"] <= 100:
        raise ValueError("decoy.low_points/high_points must satisfy 0 <= low < high <= 100")
    if (d["high_points"] - d["low_points"]) < 2 * d["min_gap_points"]:
        # guarantees at least one side always has room for a decoy
        raise ValueError("decoy range too narrow for min_gap_points")

    if set(cfg["conditions"]) != {"blind", "informed", "decoy"}:
        raise ValueError("conditions must be exactly blind, informed, decoy")

    names = [m["name"] for m in cfg["models"]]
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique")
    for m in cfg["models"]:
        for key in ("name", "tag", "family", "size", "machine"):
            if key not in m:
                raise ValueError(f"model entry {m} missing '{key}'")

    p = cfg["parser"]
    if not (_is_int(p["retries"]) and p["retries"] >= 0):
        raise ValueError("parser.retries must be an integer >= 0")
    if not 0 < p["max_failure_rate"] < 1:
        raise ValueError("parser.max_failure_rate must be between 0 and 1")

    ev = cfg["evaluation"]
    if ev["anchoring_min_gap_points"] < 0:
        raise ValueError("evaluation.anchoring_min_gap_points must be >= 0")
    if not (_is_int(ev["bootstrap_resamples"]) and ev["bootstrap_resamples"] > 0):
        raise ValueError("evaluation.bootstrap_resamples must be a positive integer")
    if not 0 < ev["ci_level"] < 1:
        raise ValueError("evaluation.ci_level must be between 0 and 1 (e.g. 0.95)")


def config_hash(cfg: dict) -> str:
    """Short hash of the config, logged with every raw response so we can prove
    both machines ran identical settings."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def call_seed(cfg: dict, market_id: str, run: int) -> int:
    """Ollama seed for one model call.

    Depends only on (experiment.call_seed, market_id, run). The condition is
    deliberately left out, so blind, informed and decoy for the same market and
    run share a seed and differ only in the price line. The "call:" prefix keeps
    this hash separate from the decoy hash in pipeline/decoy.py.
    Returns a non-negative 31-bit int, which every Ollama build accepts.
    """
    key = f"call:{cfg['experiment']['call_seed']}:{market_id}:{run}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def models_for_machine(cfg: dict, machine: str) -> list[dict]:
    return [m for m in cfg["models"] if m["machine"] == machine]
