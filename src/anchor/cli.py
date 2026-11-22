"""CLI: parser-test command.

Runs the first N dev markets through all three conditions on each of the
configured models and prints a per-model summary (parse failure rate,
retries, seconds per call, failed-response examples).

Usage:
  python -m anchor.cli parser-test --n 10                # real Ollama
  python -m anchor.cli parser-test --n 10 --dry-run      # FakeClient, no net
  python -m anchor.cli parser-test --n 10 --fixture      # use fixture CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .markets import filter_by_split, load_markets, load_split
from .model_client import FakeClient, OllamaClient
from .pipeline import run_pipeline


def _resolve_markets_path(cfg, use_fixture: bool) -> Path:
    if use_fixture:
        return cfg.path("fixtures_markets")
    real = cfg.path("markets_csv")
    if real.exists():
        return real
    # Fall back to fixture and warn.
    fx = cfg.path("fixtures_markets")
    print(f"[warn] {real} not found; using fixture {fx}", file=sys.stderr)
    return fx


def cmd_parser_test(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    markets_path = _resolve_markets_path(cfg, args.fixture)
    markets = load_markets(markets_path)

    split = None
    try:
        split = load_split(cfg.path("dev_split"))
    except Exception:  # noqa: BLE001
        split = None
    markets = filter_by_split(markets, split)[: args.n]

    if not markets:
        print("No markets to run.", file=sys.stderr)
        return 2

    client = FakeClient() if args.dry_run else OllamaClient(
        host=cfg.ollama.get("host", "http://localhost:11434"),
        timeout_s=int(cfg.ollama.get("request_timeout_s", 120)),
    )

    models = args.models or cfg.models
    results_dir = cfg.path("results_dir") / ("dry_run" if args.dry_run else "parser_test")

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

    print("\n=== Per-model summary ===")
    overall_fail = 0
    overall_calls = 0
    for model, s in stats.items():
        rate = s.failure_rate() * 100
        avg = s.seconds / s.calls if s.calls else 0
        print(f"\n{model}")
        print(f"  calls:        {s.calls}")
        print(f"  retries:      {s.retries}")
        print(f"  parse fails:  {s.parse_failures}  ({rate:.1f}%)")
        print(f"  sec/call:     {avg:.2f}")
        if s.failed_examples:
            print("  failed examples:")
            for ex in s.failed_examples:
                print(f"    - {ex!r}")
        if rate > 10:
            print(f"  WARNING: {model} exceeds 10% parse failure rate.")
        overall_fail += s.parse_failures
        overall_calls += s.calls

    print(f"\nOverall: {overall_fail}/{overall_calls} parse failures")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anchor")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("parser-test", help="run N dev markets across all conditions and models")
    pt.add_argument("--n", type=int, default=10)
    pt.add_argument("--dry-run", action="store_true", help="use FakeClient, no Ollama")
    pt.add_argument("--fixture", action="store_true", help="force use of fixture CSV")
    pt.add_argument("--models", nargs="*", help="override models from config")
    pt.set_defaults(func=cmd_parser_test)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
