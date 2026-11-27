"""Run the full ANCHOR evaluation.

Usage:
    python -m eval.evaluate --markets data/markets.csv --decoys data/decoys.csv \
        --responses results/raw/main --split data/split.csv \
        --subset test --out results/eval_test

--responses takes one or more pipeline phase folders (results/raw/main,
results/raw/paraphrase), single .jsonl/.jsonl.gz files, or a CSV.

--subset dev   first numbers while prompts are still being tuned
--subset test  the reported results (run once prompts are frozen)
--subset all   every market (for diagnostics only; never report as the main result)

Writes CSV tables, summary.md and figures/ into --out. See PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import io
from . import metrics as M


def fmt(row, key, digits=3):
    v, lo, hi = row.get(key), row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if v is None or not np.isfinite(v):
        return "n/a"
    if lo is None or not np.isfinite(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def label(model):
    return config.MODELS.get(model, {}).get("label", model)


def model_order(models):
    models = set(models)
    known = [m for m in config.MODELS if m in models]
    return known + sorted(models - set(known))


def run(markets, decoys, responses, split, subset="test", out=Path("results/eval"),
        n_boot=config.N_BOOT, figures=True, primary=config.PRIMARY):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    notes = io.check_inputs(markets, decoys, responses)

    if subset != "all":
        keep = set(split.loc[split["split"] == subset, "market_id"])
        markets = markets[markets["market_id"].isin(keep)]
        responses = responses[responses["market_id"].isin(keep)]
    if markets.empty:
        raise io.DataError(f"No markets in subset '{subset}'.")

    mt = M.market_table(responses, markets, decoys)
    mt.to_csv(out / "market_level.csv", index=False)
    parse = M.parse_rates(responses)
    parse.to_csv(out / "parse_rates.csv", index=False)

    # Per model, per prompt variant
    arrays, rows = {}, []
    for (variant, model), grp in mt.groupby(["prompt_variant", "model"]):
        a = M.model_arrays(grp)
        arrays[(variant, model)] = a
        r = {"prompt_variant": variant, "model": model, "label": label(model)}
        r.update(M.summarise(a, n_boot=n_boot))
        r["n_incomplete"] = int((~grp["complete"]).sum())
        r["verdict"] = M.verdict(r, primary) if r["n_markets"] else "No complete markets"
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "model_summary.csv", index=False)

    orig = {m: a for (v, m), a in arrays.items() if v == config.ORIGINAL_VARIANT}

    # Planned model contrasts (paired over common markets)
    crow = []
    for name, ma, mb in config.CONTRASTS:
        if ma in orig and mb in orig:
            r = {"contrast": name, "model_a": ma, "model_b": mb}
            r.update(M.contrast(orig[ma], orig[mb], stats=(f"{primary}_gap", "ac_gap", "brier_gain", "brier_informed"),
                                n_boot=n_boot))
            crow.append(r)
    contrasts = pd.DataFrame(crow)
    contrasts.to_csv(out / "model_comparisons.csv", index=False)

    # Robustness: each non-original prompt variant against the original
    rrow = []
    for (variant, model), a in arrays.items():
        if variant == config.ORIGINAL_VARIANT or model not in orig:
            continue
        s_new = summary[(summary.prompt_variant == variant) & (summary.model == model)].iloc[0]
        s_old = summary[(summary.prompt_variant == config.ORIGINAL_VARIANT) & (summary.model == model)].iloc[0]
        g = f"{primary}_gap"
        sign_old = M._sign(s_old[f"{g}_lo"], s_old[f"{g}_hi"])
        sign_new = M._sign(s_new[f"{g}_lo"], s_new[f"{g}_hi"])
        diff = M.contrast(a, orig[model], stats=(g,), n_boot=n_boot)
        rrow.append({
            "model": model, "label": label(model), "prompt_variant": variant,
            "gap_original": s_old[g], "gap_original_lo": s_old[f"{g}_lo"], "gap_original_hi": s_old[f"{g}_hi"],
            "gap_variant": s_new[g], "gap_variant_lo": s_new[f"{g}_lo"], "gap_variant_hi": s_new[f"{g}_hi"],
            "change": diff[g], "change_lo": diff[f"{g}_lo"], "change_hi": diff[f"{g}_hi"],
            "result_holds": bool(sign_old != 0 and sign_old == sign_new) or (sign_old == 0 and sign_new == 0),
        })
    robust = pd.DataFrame(rrow)
    robust.to_csv(out / "robustness.csv", index=False)

    # Secondary: breakdown by category and by platform (original prompt)
    breakdowns = {}
    base = mt[mt.prompt_variant == config.ORIGINAL_VARIANT]
    for col in ("category", "source"):
        brow = []
        for (model, grp_val), grp in base.groupby(["model", col]):
            a = M.model_arrays(grp)
            if len(a["y"]) < 2:
                continue
            s = M.summarise(a, n_boot=n_boot)
            brow.append({"model": model, "label": label(model), col: grp_val, "n_markets": s["n_markets"],
                         **{k: s[k] for k in s if k.startswith((f"{primary}_", "brier_gain", "brier_informed", "brier_blind"))},
                         "small_group": s["n_markets"] < config.MIN_GROUP_N})
        breakdowns[col] = pd.DataFrame(brow)
        breakdowns[col].to_csv(out / f"breakdown_{col}.csv", index=False)

    meta = {"subset": subset, "primary": primary, "n_boot": n_boot, "boot_seed": config.BOOT_SEED,
            "min_gap": config.MIN_GAP, "min_valid_run_frac": config.MIN_VALID_RUN_FRAC,
            "n_markets_in_subset": int(len(markets)),
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "data_checks": notes}
    (out / "run_info.json").write_text(json.dumps(meta, indent=2))

    write_markdown(out, summary, contrasts, robust, breakdowns, parse, meta)
    if figures:
        from .figures import make_figures
        make_figures(out, summary, contrasts, mt, primary)
    return summary, contrasts, robust


def write_markdown(out, summary, contrasts, robust, breakdowns, parse, meta):
    p = meta["primary"]
    names = {"cslope": "corrected slope", "ac": "anchoring coefficient", "slope": "slope"}
    L = [f"# ANCHOR evaluation results ({meta['subset']} set)", "",
         f"Generated {meta['generated']}. {meta['n_markets_in_subset']} markets in subset. "
         f"Intervals are 95% bootstrap percentile intervals ({meta['n_boot']} resamples over markets, seed {meta['boot_seed']}). "
         f"Primary anchoring measure: {names.get(p, p)}.", ""]
    if meta["data_checks"]:
        L += ["## Data check warnings", ""] + [f"- {n}" for n in meta["data_checks"]] + [""]

    s0 = summary[summary.prompt_variant == config.ORIGINAL_VARIANT].copy()
    s0 = s0.set_index("model").loc[model_order(s0.model)].reset_index()

    L += ["## Main test", "",
          f"| Model | Markets | Informed {names[p]} | Decoy {names[p]} | Gap (informed - decoy) | Brier gain (blind - informed) | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for _, r in s0.iterrows():
        L.append(f"| {r.label} | {int(r.n_markets)} | {fmt(r, f'{p}_informed')} | {fmt(r, f'{p}_decoy')} | "
                 f"{fmt(r, f'{p}_gap')} | {fmt(r, 'brier_gain')} | {r.verdict} |")
    L += ["", "## Brier scores (lower is better)", "",
          "| Model | Blind | Informed | Decoy | Parrot baseline | Informed - parrot |", "|---|---|---|---|---|---|"]
    for _, r in s0.iterrows():
        L.append(f"| {r.label} | {fmt(r, 'brier_blind')} | {fmt(r, 'brier_informed')} | {fmt(r, 'brier_decoy')} | "
                 f"{fmt(r, 'brier_parrot')} | {fmt(r, 'brier_vs_parrot')} |")
    L += ["", "## All anchoring measures", "",
          "Corrected slope is primary. The mean anchoring coefficient (proposal definition, markets with "
          f"|shown - blind| < {meta['min_gap']} excluded) and the uncorrected slope are secondary.", "",
          "| Model | Measure | Informed | Decoy | Gap | Median coefficient (inf / dec) |", "|---|---|---|---|---|---|"]
    for _, r in s0.iterrows():
        for key, nm in (("cslope", "Corrected slope"), ("slope", "Slope"), ("ac", "Mean coefficient")):
            med = f"{r.ac_median_informed:.3f} / {r.ac_median_decoy:.3f}" if key == "ac" else ""
            extra = f" (n = {int(r.n_ac_informed)} / {int(r.n_ac_decoy)})" if key == "ac" else ""
            L.append(f"| {r.label} | {nm}{extra} | {fmt(r, f'{key}_informed')} | {fmt(r, f'{key}_decoy')} | "
                     f"{fmt(r, f'{key}_gap')} | {med} |")
    if not contrasts.empty:
        L += ["", "## Model comparisons (paired over common markets)", "",
              f"| Comparison | Markets | Difference in {names[p]} gap | Difference in Brier gain | Difference in informed Brier |",
              "|---|---|---|---|---|"]
        for _, r in contrasts.iterrows():
            L.append(f"| {r.contrast} | {int(r.n_markets)} | {fmt(r, f'{p}_gap')} | {fmt(r, 'brier_gain')} | "
                     f"{fmt(r, 'brier_informed')} |")
    if not robust.empty:
        L += ["", "## Robustness to prompt wording", "",
              "| Model | Variant | Gap, original | Gap, variant | Change | Result holds |", "|---|---|---|---|---|---|"]
        for _, r in robust.iterrows():
            L.append(f"| {r.label} | {r.prompt_variant} | {fmt(r, 'gap_original')} | {fmt(r, 'gap_variant')} | "
                     f"{fmt(r, 'change')} | {'yes' if r.result_holds else 'no'} |")
    for col, df in breakdowns.items():
        if df.empty:
            continue
        L += ["", f"## Secondary: by {col}", "", f"Groups under {config.MIN_GROUP_N} markets are marked * and should not be interpreted.", "",
              f"| Model | {col.title()} | Markets | {names[p].capitalize()} gap | Brier gain |", "|---|---|---|---|---|"]
        for _, r in df.sort_values(["model", col]).iterrows():
            star = "*" if r.small_group else ""
            L.append(f"| {r.label} | {r[col]}{star} | {int(r.n_markets)} | {fmt(r, f'{p}_gap')} | {fmt(r, 'brier_gain')} |")
    L += ["", "## Parse failures", "", "| Variant | Model | Condition | Calls | Failures | Rate |", "|---|---|---|---|---|---|"]
    for _, r in parse.iterrows():
        flag = " (above 10%)" if r.failure_rate > 0.10 else ""
        L.append(f"| {r.prompt_variant} | {label(r.model)} | {r.condition} | {r.calls} | {r.parse_failures} | {r.failure_rate:.1%}{flag} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", required=True)
    ap.add_argument("--decoys", required=True)
    ap.add_argument("--responses", required=True, nargs="+",
                    help="pipeline phase folder(s) such as results/raw/main, .jsonl[.gz] file(s), or a CSV")
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=config.N_BOOT)
    ap.add_argument("--primary", choices=["cslope", "ac", "slope"], default=config.PRIMARY)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    summary, _, _ = run(io.load_markets(args.markets), io.load_decoys(args.decoys), io.load_responses(args.responses),
                        io.load_split(args.split), args.subset, args.out, args.n_boot, not args.no_figures, args.primary)
    print(f"Done. Open {Path(args.out) / 'summary.md'}")
    s0 = summary[summary.prompt_variant == config.ORIGINAL_VARIANT].set_index("model")
    for m in model_order(s0.index):
        print(f"  {s0.loc[m, 'label']}: {s0.loc[m, 'verdict']}")


if __name__ == "__main__":
    main()
