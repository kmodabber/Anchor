"""Verify Ollama is reachable and the required models are present.

Prints the Ollama version and digest for each required tag. Exits non-zero
if anything is missing so CI / teammates notice. Does NOT pull models.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

REQUIRED_TAGS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "qwen2.5:3b",
    "qwen2.5:7b",
]

OLLAMA_URL = "http://localhost:11434"


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
