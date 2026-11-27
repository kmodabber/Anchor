"""Report figures (PNG, light background, sized for a two-column report page)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
INFORMED = "#2a78d6"   # categorical slot 1 (blue)
DECOY = "#eb6834"      # categorical slot 2 (orange)
NEUTRAL = "#8a8984"    # blind / reference marks
NAMES = {"cslope": "Corrected anchoring slope", "ac": "Mean anchoring coefficient", "slope": "Anchoring slope"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False,
})


def _order(df):
    known = [m for m in config.MODELS if m in set(df.model)]
    return df.set_index("model").loc[known + sorted(set(df.model) - set(known))].reset_index()


def _interval(ax, y, v, lo, hi, color, marker="o", label=None):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([v], [y], marker=marker, ms=8, color=color, mec=SURFACE, mew=2, ls="none", label=label, zorder=3)


def anchoring_plot(summary, primary, path):
    df = _order(summary[summary.prompt_variant == config.ORIGINAL_VARIANT])
    fig, ax = plt.subplots(figsize=(6.4, 0.75 * len(df) + 1.3))
    for i, r in enumerate(df.itertuples()):
        y = len(df) - 1 - i
        _interval(ax, y + 0.14, getattr(r, f"{primary}_informed"), getattr(r, f"{primary}_informed_lo"),
                  getattr(r, f"{primary}_informed_hi"), INFORMED, label="Real price (informed)" if i == 0 else None)
        _interval(ax, y - 0.14, getattr(r, f"{primary}_decoy"), getattr(r, f"{primary}_decoy_lo"),
                  getattr(r, f"{primary}_decoy_hi"), DECOY, marker="D", label="Random price (decoy)" if i == 0 else None)
    ax.set_yticks(range(len(df)), df.label[::-1])
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.axvline(1, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(0, len(df) - 0.45, " ignores price", color=INK2, fontsize=8, va="bottom")
    ax.text(1, len(df) - 0.45, " copies price", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel(f"{NAMES[primary]} (0 = ignores the shown price, 1 = copies it)")
    ax.set_title("How far each model moves toward the price it is shown", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(df) - 0.1)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def brier_plot(summary, path):
    df = _order(summary[summary.prompt_variant == config.ORIGINAL_VARIANT])
    fig, ax = plt.subplots(figsize=(6.4, 0.85 * len(df) + 1.3))
    for i, r in enumerate(df.itertuples()):
        y = len(df) - 1 - i
        for off, key, col, mk, nm in ((0.22, "brier_blind", NEUTRAL, "o", "Blind"),
                                      (0.0, "brier_informed", INFORMED, "o", "Informed"),
                                      (-0.22, "brier_decoy", DECOY, "D", "Decoy")):
            _interval(ax, y + off, getattr(r, key), getattr(r, f"{key}_lo"), getattr(r, f"{key}_hi"),
                      col, mk, label=nm if i == 0 else None)
        ax.plot([r.brier_parrot] * 2, [y - 0.36, y + 0.36], color=INK, lw=2,
                label="Parrot baseline (market price)" if i == 0 else None, zorder=1)
    ax.set_yticks(range(len(df)), df.label[::-1])
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_title("Forecast accuracy by condition", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(df) - 0.4)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def shift_plot(mt, summary, path):
    base = mt[(mt.prompt_variant == config.ORIGINAL_VARIANT) & mt.complete]
    models = [m for m in config.MODELS if m in set(base.model)] + sorted(set(base.model) - set(config.MODELS))
    ncol = 2
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4, 3.0 * nrow + 0.5), sharex=True, sharey=True, squeeze=False)
    s0 = summary[summary.prompt_variant == config.ORIGINAL_VARIANT].set_index("model")
    for ax, m in zip(axes.flat, models):
        d = base[base.model == m]
        line = np.array([-1, 1])
        ax.plot(line, line, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        for cond, col, mk in (("informed", INFORMED, "o"), ("decoy", DECOY, "D")):
            x = d[f"shown_{cond}"] - d["p_blind"]
            y = d[f"p_{cond}"] - d["p_blind"]
            ax.scatter(x, y, s=12, color=col, alpha=0.45, marker=mk, linewidths=0, zorder=2)
            ax.plot(line, s0.loc[m, f"cslope_{cond}"] * line, color=col, lw=2, zorder=3)
        ax.set_title(config.MODELS.get(m, {}).get("label", m), loc="left")
        ax.set_xlim(-0.95, 0.95)
        ax.set_ylim(-0.95, 0.95)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    fig.supxlabel("Shown price minus blind forecast", color=INK2, fontsize=9, y=0.055)
    fig.supylabel("Forecast with price minus blind forecast", color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], color=INFORMED, marker="o", lw=2, label="Real price (informed)"),
               plt.Line2D([], [], color=DECOY, marker="D", lw=2, label="Random price (decoy)"),
               plt.Line2D([], [], color=INK2, ls=(0, (3, 3)), lw=1, label="Full copying")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def contrast_plot(contrasts, primary, path):
    if contrasts.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(contrasts) + 1.2))
    for i, r in enumerate(contrasts.itertuples()):
        y = len(contrasts) - 1 - i
        _interval(ax, y, getattr(r, f"{primary}_gap"), getattr(r, f"{primary}_gap_lo"),
                  getattr(r, f"{primary}_gap_hi"), INFORMED)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(contrasts)), contrasts.contrast[::-1])
    ax.set_xlabel("Difference in informed-minus-decoy gap (positive = first model separates real from random more)")
    ax.set_title("Does size or family change how models treat real vs. random prices?", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(out, summary, contrasts, mt, primary=config.PRIMARY):
    fdir = Path(out) / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    anchoring_plot(summary, primary, fdir / "anchoring.png")
    brier_plot(summary, fdir / "brier.png")
    shift_plot(mt, summary, fdir / "shift_vs_gap.png")
    contrast_plot(contrasts, primary, fdir / "model_comparisons.png")
