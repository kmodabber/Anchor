"""The `run` command and the pre-run checks, without Ollama or a GPU."""
import gzip
import json

import pytest
import yaml

from common.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from pipeline import cli, preflight, prompts
from pipeline.markets import load_markets
from pipeline.model_client import FakeClient
from pipeline.runner import compress

QWEN = ["qwen2.5:3b", "qwen2.5:7b"]


@pytest.fixture
def clean_git(monkeypatch):
    """Pretend the tree is clean, whatever state the real checkout is in."""
    monkeypatch.setattr(preflight, "git_dirty_paths", lambda: [])


def _tmp_config(tmp_path, ollama_version=None):
    """Copy of config.yaml whose output paths point into tmp_path.
    (REPO_ROOT / an absolute path gives the absolute path, so the code needs no change.)"""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["paths"]["raw_responses"] = str(tmp_path / "raw")
    raw["paths"]["manifests"] = str(tmp_path / "manifests")
    raw["ollama"]["version"] = ollama_version
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- preflight ----------------------------------------------------------------

def test_porcelain_ignores_untracked_results_only():
    text = "?? results/raw/main/qwen2.5_3b.jsonl.gz\n M pipeline/cli.py\n?? scratch.py\n"
    assert preflight.parse_porcelain(text) == ["pipeline/cli.py", "scratch.py"]


@pytest.mark.parametrize("pinned, phase, has_version_problem", [
    (None, "main", True),         # main/paraphrase need a pinned version
    (None, "paraphrase", True),
    (None, "dev", False),
    ("0.1.0", "dev", True),       # pinned but different from the running server
    ("fake", "main", False),      # FakeClient reports version "fake"
])
def test_version_check(clean_git, pinned, phase, has_version_problem):
    cfg = load_config()
    cfg["ollama"]["version"] = pinned
    pf = preflight.run_checks(cfg, FakeClient(), QWEN, phase)
    assert any("ollama" in p.lower() for p in pf.problems) == has_version_problem


def test_dirty_tree_is_a_problem(monkeypatch):
    monkeypatch.setattr(preflight, "git_dirty_paths", lambda: ["pipeline/cli.py"])
    pf = preflight.run_checks(load_config(), FakeClient(), QWEN, "dev")
    assert not pf.ok and "uncommitted" in pf.problems[0]


def test_missing_digest_is_a_problem(clean_git):
    class NoDigest(FakeClient):
        def digest_for(self, model):
            return "unknown"
    pf = preflight.run_checks(load_config(), NoDigest(), QWEN, "dev")
    assert any("no digest" in p for p in pf.problems)


# --- compression --------------------------------------------------------------

def test_compress_round_trips_and_is_byte_identical(tmp_path):
    src = tmp_path / "m.jsonl"
    src.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    first = compress(src).read_bytes()
    assert gzip.decompress(first) == src.read_bytes()
    assert compress(src).read_bytes() == first  # no timestamp inside
    assert src.exists()  # the .jsonl stays as the resume file


# --- run command, rehearsal (--fixture --dry-run) -----------------------------

def test_rehearsal_main_phase(tmp_path, clean_git):
    cfg_path = _tmp_config(tmp_path)
    argv = ["--config", str(cfg_path), "run", "--phase", "main",
            "--machine", "talha-5070", "--fixture", "--dry-run"]
    assert cli.main(argv) == 0

    out = tmp_path / "raw" / "scratch" / "main"
    first = {}
    for tag in QWEN:
        gz = out / f"{tag.replace(':', '_')}.jsonl.gz"
        recs = _records(gz)
        assert len(recs) == 10 * 3 * 5  # fixture markets x conditions x main runs
        assert {r["phase"] for r in recs} == {"main"}
        assert {r["prompt_variant"] for r in recs} == {"v1"}
        assert {r["model"] for r in recs} == {tag}
        first[tag] = gz.read_bytes()

    assert cli.main(argv) == 0  # re-run: everything resumes as done
    for tag in QWEN:
        assert (out / f"{tag.replace(':', '_')}.jsonl.gz").read_bytes() == first[tag]
    assert not (tmp_path / "manifests").exists()  # rehearsals leave no manifest


def test_paraphrase_phase_waits_for_the_prompt(tmp_path, clean_git):
    argv = ["--config", str(_tmp_config(tmp_path)), "run", "--phase", "paraphrase",
            "--machine", "aryan-4060", "--fixture", "--dry-run"]
    assert cli.main(argv) == 2


def test_paraphrase_phase_uses_its_variant(tmp_path, clean_git, monkeypatch):
    monkeypatch.setitem(prompts.TEMPLATES, "paraphrase", prompts.TEMPLATES["v1"])
    argv = ["--config", str(_tmp_config(tmp_path)), "run", "--phase", "paraphrase",
            "--machine", "aryan-4060", "--fixture", "--dry-run"]
    assert cli.main(argv) == 0
    recs = _records(tmp_path / "raw" / "scratch" / "paraphrase" / "llama3.1_8b.jsonl.gz")
    assert {r["prompt_variant"] for r in recs} == {"paraphrase"}
    assert len(recs) == 10 * 3 * 3  # paraphrase runs = 3


def test_unknown_machine(tmp_path):
    argv = ["--config", str(_tmp_config(tmp_path)), "run", "--phase", "main",
            "--machine", "bob", "--fixture", "--dry-run"]
    assert cli.main(argv) == 2


# --- run command, real (non-rehearsal) path with a fake Ollama ----------------

@pytest.fixture
def real_path_with_fake_ollama(monkeypatch):
    """Exercise the strict path: real-run rules, but FakeClient and fixture markets."""
    monkeypatch.setattr(cli, "_client", lambda cfg, dry_run: FakeClient())
    fixture = load_markets(REPO_ROOT / "fixtures" / "dev_markets.csv")
    monkeypatch.setattr(cli, "_phase_markets", lambda cfg, phase, use_fixture: fixture)


def test_real_run_refuses_on_problems(tmp_path, real_path_with_fake_ollama, monkeypatch):
    monkeypatch.setattr(preflight, "git_dirty_paths", lambda: ["pipeline/cli.py"])
    argv = ["--config", str(_tmp_config(tmp_path, ollama_version="fake")),
            "run", "--phase", "main", "--machine", "talha-5070"]
    assert cli.main(argv) == 1
    assert not (tmp_path / "raw").exists()  # nothing ran
    assert not (tmp_path / "manifests").exists()


def test_real_run_refuses_unpinned_version(tmp_path, real_path_with_fake_ollama, clean_git):
    argv = ["--config", str(_tmp_config(tmp_path, ollama_version=None)),
            "run", "--phase", "main", "--machine", "talha-5070"]
    assert cli.main(argv) == 1


def test_real_run_writes_results_and_manifest(tmp_path, real_path_with_fake_ollama, clean_git):
    argv = ["--config", str(_tmp_config(tmp_path, ollama_version="fake")),
            "run", "--phase", "main", "--machine", "talha-5070"]
    assert cli.main(argv) == 0
    assert (tmp_path / "raw" / "main" / "qwen2.5_7b.jsonl.gz").exists()
    lines = (tmp_path / "manifests" / "main_talha-5070.jsonl").read_text().splitlines()
    m = json.loads(lines[0])
    assert m["phase"] == "main" and m["machine"] == "talha-5070"
    assert m["models"] == QWEN and m["runs"] == 5 and m["markets"] == 10
    assert set(m["digests"]) == set(QWEN) and m["commit"]
