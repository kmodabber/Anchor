"""CLI: parser-test command.

Runs the first N dev markets through all three conditions on each configured
model and prints a per-model summary (parse failure rate, retries, seconds
per call, failed-response examples).

Run from the repo root:
  python -m pipeline.cli parser-test --n 10                    # real Ollama, dev markets
  python -m pipeline.cli parser-test --n 10 --fixture --dry-run  # fake model, no network
  python -m pipeline.cli parser-test --models qwen2.5:3b qwen2.5:7b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from .markets import filter_by_split, load_markets, load_split
from .model_client import FakeClient, OllamaClient
from .runner import run_pipeline


def _select_markets(cfg: dict, use_fixture: bool):
    """Fixture markets, or the dev split of the real dataset. Never test markets."""
    data = cfg["data"]
    if use_fixture:
        return load_markets(REPO_ROOT / data["fixture_csv"])
    real = REPO_ROOT / data["markets_csv"]
    if not real.exists():
        raise SystemExit(f"{real} not found yet; use --fixture until the data lands.")
    dev = load_split(REPO_ROOT / data["split_csv"], "dev")
    if dev is None:
        raise SystemExit(f"{data['split_csv']} not found. The parser test runs on dev "
                         "markets only, so the test set stays untouched.")
    return filter_by_split(load_markets(real), dev)


def cmd_parser_test(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    markets = _select_markets(cfg, args.fixture)[: args.n]
    if not markets:
        print("No markets to run.", file=sys.stderr)
        return 2

    known = [m["tag"] for m in cfg["models"]]
    models = args.models or known
    unknown = [m for m in models if m not in known]
    if unknown:
        print(f"Not in config/config.yaml: {unknown}. Known tags: {known}", file=sys.stderr)
        return 2

    ollama = cfg["ollama"]
    client = FakeClient() if args.dry_run else OllamaClient(
        host=ollama["host"], timeout_s=int(ollama["request_timeout_s"]))

    results_dir = (REPO_ROOT / cfg["paths"]["raw_responses"]
                   / ("dry_run" if args.dry_run else "parser_test"))
    print(f"Markets: {len(markets)}  Models: {models}  Dry-run: {args.dry_run}")
    print(f"Writing JSONL to: {results_dir}")

    stats = run_pipeline(
        cfg=cfg,
        client=client,
        markets=markets,
        models=models,
        runs=1,  # parser-test: one run per cell is enough
        results_dir=results_dir,
    )

    max_rate = cfg["parser"]["max_failure_rate"]
    print("\n=== Per-model summary ===")
    overall_fail = overall_calls = 0
    for model, s in stats.items():
        avg = s.seconds / s.calls if s.calls else 0
        print(f"\n{model}")
        print(f"  calls:        {s.calls}")
        print(f"  retries:      {s.retries}")
        print(f"  parse fails:  {s.parse_failures}  ({s.failure_rate():.1%})")
        print(f"  sec/call:     {avg:.2f}")
        if s.failed_examples:
            print("  failed examples:")
            for ex in s.failed_examples:
                print(f"    - {ex!r}")
        if s.failure_rate() > max_rate:
            print(f"  WARNING: {model} exceeds the {max_rate:.0%} parse failure cutoff.")
        overall_fail += s.parse_failures
        overall_calls += s.calls

    print(f"\nOverall: {overall_fail}/{overall_calls} parse failures")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pipeline.cli")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("parser-test", help="run N dev markets across all conditions and models")
    pt.add_argument("--n", type=int, default=10)
    pt.add_argument("--dry-run", action="store_true", help="use FakeClient, no Ollama")
    pt.add_argument("--fixture", action="store_true", help="use fixtures/dev_markets.csv")
    pt.add_argument("--models", nargs="*", help="Ollama tags to run (default: all in config)")
    pt.set_defaults(func=cmd_parser_test)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
