"""Verify Ollama is reachable and the required models are present.

Prints the Ollama version and digest for each required tag. Exits non-zero
if anything is missing so teammates notice. Does NOT pull models.
Tags, host and the expected Ollama version come from config/config.yaml.

Run from the repo root:
    python -m scripts.check_ollama
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from common.config import load_config

CFG = load_config()
REQUIRED_TAGS = [m["tag"] for m in CFG["models"]]
OLLAMA_URL = CFG["ollama"]["host"].rstrip("/")
EXPECTED_VERSION = CFG["ollama"]["version"]  # null until the team pins one


def _get(path: str) -> dict:
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=5) as r:
        return json.loads(r.read())


def main() -> int:
    try:
        version = _get("/api/version").get("version", "unknown")
    except (urllib.error.URLError, OSError) as e:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_URL}: {e}")
        print("Is `ollama serve` running?")
        return 2

    print(f"Ollama version: {version}")
    if EXPECTED_VERSION is None:
        print("  (ollama.version is not pinned in config/config.yaml yet)")
    elif str(EXPECTED_VERSION) != version:
        print(f"ERROR: config/config.yaml pins Ollama {EXPECTED_VERSION}; this machine "
              f"has {version}. Both machines must match.")
        return 1

    try:
        tags = _get("/api/tags").get("models", [])
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: /api/tags failed: {e}")
        return 2

    by_name = {m["name"]: m for m in tags}
    missing = []
    print("\nInstalled required models:")
    for tag in REQUIRED_TAGS:
        if tag in by_name:
            digest = by_name[tag].get("digest", "?")
            size_gb = by_name[tag].get("size", 0) / 1e9
            print(f"  {tag:20s}  digest={digest[:19]}  size={size_gb:.2f} GB")
        else:
            missing.append(tag)

    if missing:
        print("\nMISSING tags (run `ollama pull <tag>`):")
        for tag in missing:
            print(f"  {tag}")
        return 1

    print("\nAll required models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
