"""Model clients. Every call is a fresh request, no chat history.

Two implementations:
- ``OllamaClient`` talks to a local Ollama server via /api/generate.
- ``FakeClient`` is the dry-run model: no network, deterministic-ish,
  produces responses the parser can handle (plus some messy variants).
"""
from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelResponse:
    raw_text: str
    model_digest: str  # best-effort identifier for reproducibility


class ModelClient(Protocol):
    def generate(self, *, model: str, prompt: str, seed: int,
                 temperature: float) -> ModelResponse: ...

    def digest_for(self, model: str) -> str: ...


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout_s: int = 120):
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self._digest_cache: dict[str, str] = {}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.host + path, timeout=10) as r:
            return json.loads(r.read())

    def digest_for(self, model: str) -> str:
        if model in self._digest_cache:
            return self._digest_cache[model]
        try:
            tags = self._get("/api/tags").get("models", [])
            for m in tags:
                if m.get("name") == model:
                    d = m.get("digest", "unknown")
                    self._digest_cache[model] = d
                    return d
        except Exception:  # noqa: BLE001
            pass
        return "unknown"

    def generate(self, *, model: str, prompt: str, seed: int,
                 temperature: float) -> ModelResponse:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "seed": seed,
                "temperature": temperature,
            },
        }
        resp = self._post("/api/generate", body)
        return ModelResponse(
            raw_text=resp.get("response", ""),
            model_digest=self.digest_for(model),
        )


class FakeClient:
    """Deterministic-ish fake for dry runs. ~10% messy, ~5% unparsable."""

    def digest_for(self, model: str) -> str:
        return f"fake:{model}"

    def generate(self, *, model: str, prompt: str, seed: int,
                 temperature: float) -> ModelResponse:
        rng = random.Random(f"{seed}|{model}|{prompt}")
        base = rng.randint(10, 90)
        r = rng.random()
        if r < 0.05:
            text = "I don't want to guess."           # unparsable
        elif r < 0.15:
            text = f"around {base} percent probably"  # messy but parseable
        elif r < 0.25:
            text = f"{base/100:.2f}"                  # fraction form
        elif r < 0.35:
            text = f"My estimate is {base}%."
        else:
            text = str(base)
        return ModelResponse(raw_text=text, model_digest=self.digest_for(model))
