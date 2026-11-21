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

REQUIRED_SECTIONS = ("experiment", "data", "decoy", "conditions", "models", "paths")


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


def config_hash(cfg: dict) -> str:
    """Short hash of the config, logged with every raw response so we can prove
    both machines ran identical settings."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def models_for_machine(cfg: dict, machine: str) -> list[dict]:
    return [m for m in cfg["models"] if m["machine"] == machine]
