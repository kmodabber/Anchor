"""Command-line entry points. Run from the repo root.

parser-test: first N dev markets, all three conditions, one run, every model.
  python -m pipeline.cli parser-test --n 10                      # real Ollama, dev markets
  python -m pipeline.cli parser-test --n 10 --fixture --dry-run  # fake model, no network
  python -m pipeline.cli parser-test --models qwen2.5:3b qwen2.5:7b

run: a full experiment phase for one machine's models, after pre-run checks.
  python -m pipeline.cli run --phase main --machine talha-5070
  python -m pipeline.cli run --phase paraphrase --machine aryan-4060
  python -m pipeline.cli run --phase main --machine talha-5070 --fixture --dry-run  # rehearsal

  Phases (run counts from experiment.runs in config.yaml):
    dev         dev split,   v1 prompt
    main        all markets, v1 prompt
    paraphrase  test split,  paraphrase prompt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common.config import DEFAULT_CONFIG, REPO_ROOT, load_config, models_for_machine
from .markets import Market, filter_by_split, load_markets, load_split
from .model_client import FakeClient, OllamaClient
from .preflight import format_report, run_checks, write_manifest
from .prompts import CONDITIONS, TEMPLATES
from .runner import RunStats, compress, results_file, run_pipeline

# phase -> (split it runs on, None = all markets; prompt variant)
PHASES = {
    "dev": ("dev", "v1"),
    "main": (None, "v1"),
    "paraphrase": ("test", "paraphrase"),
}


def _phase_markets(cfg: dict, phase: str, use_fixture: bool) -> list[Market]:
    """Markets for a phase. Fixture runs use all fixture markets (no split)."""
    data = cfg["data"]
    if use_fixture:
        return load_markets(REPO_ROOT / data["fixture_csv"])
    real = REPO_ROOT / data["markets_csv"]
    if not real.exists():
        raise SystemExit(f"{real} not found yet; use --fixture until the data lands.")
    markets = load_markets(real)
    split_name = PHASES[phase][0]
    if split_name is None:
        return markets
    ids = load_split(REPO_ROOT / data["split_csv"], split_name)
    if ids is None:
        raise SystemExit(f"{data['split_csv']} not found. The {phase} phase runs on "
                         f"{split_name} markets only, so it needs the split file.")
    return filter_by_split(markets, ids)


def _client(cfg: dict, dry_run: bool):
    ollama = cfg["ollama"]
    return FakeClient() if dry_run else OllamaClient(
        host=ollama["host"], timeout_s=int(ollama["request_timeout_s"]))


def _print_summary(stats: dict[str, RunStats], max_rate: float) -> None:
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
    print(f"\nOverall: {overall_fail}/{overall_calls} parse failures (this session)")


def _progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_parser_test(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    markets = _phase_markets(cfg, "dev", args.fixture)[: args.n]
    if not markets:
        print("No markets to run.", file=sys.stderr)
        return 2

    known = [m["tag"] for m in cfg["models"]]
    models = args.models or known
    unknown = [m for m in models if m not in known]
    if unknown:
        print(f"Not in config/config.yaml: {unknown}. Known tags: {known}", file=sys.stderr)
        return 2

    results_dir = (REPO_ROOT / cfg["paths"]["raw_responses"]
                   / ("dry_run" if args.dry_run else "parser_test"))
    print(f"Markets: {len(markets)}  Models: {models}  Dry-run: {args.dry_run}")
    print(f"Writing JSONL to: {results_dir}")

    stats = run_pipeline(
        cfg=cfg,
        client=_client(cfg, args.dry_run),
        markets=markets,
        models=models,
        runs=1,  # parser-test: one run per cell is enough
        results_dir=results_dir,
        phase="parser_test",
    )
    _print_summary(stats, cfg["parser"]["max_failure_rate"])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    phase, machine = args.phase, args.machine
    variant = PHASES[phase][1]
    scratch = args.dry_run or args.fixture  # rehearsal: warnings only, output not kept

    models = [m["tag"] for m in models_for_machine(cfg, machine)]
    if not models:
        machines = sorted({m["machine"] for m in cfg["models"]})
        print(f"No models for machine {machine!r}. Machines in config: {machines}",
              file=sys.stderr)
        return 2
    if variant not in TEMPLATES:
        print(f"The {phase} phase needs the {variant!r} prompt, which is not in "
              "pipeline/prompts.py yet (Talha's paraphrases).", file=sys.stderr)
        return 2

    markets = _phase_markets(cfg, phase, args.fixture)
    if not markets:
        print(f"No markets for the {phase} phase.", file=sys.stderr)
        return 2
    runs = cfg["experiment"]["runs"][phase]

    raw_root = REPO_ROOT / cfg["paths"]["raw_responses"]
    results_dir = raw_root / "scratch" / phase if scratch else raw_root / phase

    client = _client(cfg, args.dry_run)
    pf = run_checks(cfg, client, models, phase)
    print(f"Pre-run check: phase {phase}, machine {machine}"
          f"{'  [rehearsal: --fixture/--dry-run]' if scratch else ''}")
    print(format_report(pf))
    calls = len(markets) * len(CONDITIONS) * runs * len(models)
    print(f"  plan:         {len(markets)} markets x {len(CONDITIONS)} conditions x "
          f"{runs} runs x {len(models)} models = {calls} calls")
    print(f"  output:       {results_dir}")

    if pf.problems:
        sys.stdout.flush()  # keep the report above the warnings when output is piped
        label = "WARNING" if scratch else "ERROR"
        for p in pf.problems:
            print(f"{label}: {p}", file=sys.stderr)
        if not scratch:
            print("Not starting. Fix the above and re-run.", file=sys.stderr)
            return 1

    if not scratch:
        write_manifest(
            REPO_ROOT / cfg["paths"]["manifests"] / f"{phase}_{machine}.jsonl", pf,
            phase=phase, machine=machine, models=models, prompt_variant=variant,
            markets=len(markets), runs=runs)
        print("Paste the block above in the group chat so both machines can be compared.")

    print()
    stats = run_pipeline(
        cfg=cfg, client=client, markets=markets, models=models, runs=runs,
        results_dir=results_dir, phase=phase, prompt_variant=variant,
        progress=_progress)

    # Only reached when every cell is done (Ctrl-C or a crash skips this; re-run resumes).
    for model in models:
        gz = compress(results_file(results_dir, model))
        print(f"Compressed: {gz.relative_to(REPO_ROOT) if gz.is_relative_to(REPO_ROOT) else gz}")
    _print_summary(stats, cfg["parser"]["max_failure_rate"])
    if not scratch:
        print(f"\nDone. Commit the .jsonl.gz files and {cfg['paths']['manifests']}/.")
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

    r = sub.add_parser("run", help="run a full experiment phase for one machine's models")
    r.add_argument("--phase", required=True, choices=sorted(PHASES))
    r.add_argument("--machine", required=True, help="machine name from config.yaml, "
                   "e.g. talha-5070 or aryan-4060")
    r.add_argument("--dry-run", action="store_true",
                   help="rehearsal with FakeClient; output goes to results/raw/scratch/")
    r.add_argument("--fixture", action="store_true",
                   help="rehearsal on fixtures/dev_markets.csv; output goes to scratch")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
