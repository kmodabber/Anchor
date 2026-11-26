"""Pre-run checks, so both GPU machines provably run the same setup.

Checks (all collected in one pass, so every problem shows at once):
- the git tree has no uncommitted changes, so the logged commit is the code that ran
- Ollama is reachable and its version matches ollama.version in the config
  (main and paraphrase runs require the version to be pinned)
- every model has a readable digest

The run command refuses to start a real run if any check fails, and appends a
manifest line (commit, config hash, versions, digests, hardware) to
results/manifests/ for the reproducibility appendix.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from common.config import REPO_ROOT, config_hash
from .model_client import ModelClient
from .runner import now_iso

PINNED_VERSION_PHASES = ("main", "paraphrase")


@dataclass
class PreflightResult:
    commit: str
    dirty_paths: list[str]
    config_hash: str
    ollama_version: str
    expected_version: str | None
    digests: dict[str, str]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def _git(*args: str) -> subprocess.CompletedProcess:
    # read-only git calls only
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def git_commit() -> str:
    out = _git("rev-parse", "HEAD")
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def parse_porcelain(text: str) -> list[str]:
    """Paths from `git status --porcelain` that count as uncommitted changes.
    Untracked files under results/ are outputs of earlier runs, not code, so
    they do not count."""
    paths = []
    for line in text.splitlines():
        status, path = line[:2], line[3:]
        if status == "??" and path.startswith("results/"):
            continue
        paths.append(path)
    return paths


def git_dirty_paths() -> list[str]:
    out = _git("status", "--porcelain", "--untracked-files=all")
    if out.returncode != 0:
        return ["(git status failed: is this a git clone of the repo?)"]
    return parse_porcelain(out.stdout)


def run_checks(cfg: dict, client: ModelClient, models: list[str], phase: str) -> PreflightResult:
    problems: list[str] = []

    dirty = git_dirty_paths()
    if dirty:
        shown = ", ".join(dirty[:5]) + (f" and {len(dirty) - 5} more" if len(dirty) > 5 else "")
        problems.append(f"uncommitted changes: {shown}. Commit or discard them, then "
                        "pull, so this run matches a commit on GitHub.")

    expected = cfg["ollama"]["version"]
    expected = None if expected is None else str(expected)
    version = client.version()
    if version == "unreachable":
        problems.append(f"cannot reach Ollama at {cfg['ollama']['host']}. Is it running?")
    elif expected is None:
        if phase in PINNED_VERSION_PHASES:
            problems.append(f"ollama.version is not pinned in config/config.yaml; the {phase} "
                            "phase requires both machines on one pinned version.")
    elif version != expected:
        problems.append(f"Ollama here is {version}, config pins {expected}.")

    digests = {m: client.digest_for(m) for m in models}
    missing = [m for m, d in digests.items() if d == "unknown"]
    if missing:
        problems.append(f"no digest for {missing}: run `ollama pull <tag>` for each.")

    return PreflightResult(
        commit=git_commit(), dirty_paths=dirty, config_hash=config_hash(cfg),
        ollama_version=version, expected_version=expected, digests=digests,
        problems=problems)


def gpu_info() -> str:
    """GPU name and driver from nvidia-smi, for the appendix. Best effort."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def write_manifest(path: Path, pf: PreflightResult, **run_info) -> None:
    """Append one JSON line describing this run invocation."""
    record = {
        "started": now_iso(),
        **run_info,
        "commit": pf.commit,
        "config_hash": pf.config_hash,
        "ollama_version": pf.ollama_version,
        "digests": pf.digests,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_info(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def format_report(pf: PreflightResult) -> str:
    pinned = pf.expected_version or "not pinned"
    lines = [
        f"  commit:       {pf.commit[:12]}{'  (uncommitted changes!)' if pf.dirty_paths else ''}",
        f"  config hash:  {pf.config_hash}",
        f"  ollama:       {pf.ollama_version}  (config: {pinned})",
    ]
    lines += [f"  {m:<13} {d}" for m, d in pf.digests.items()]
    return "\n".join(lines)
