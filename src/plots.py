"""Figures for the paper (PDF for LaTeX, PNG for quick viewing)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CRISES
from strategies import RULE_LABELS

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 150,
})


def _save(fig, name, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _shade_crises(ax):
    for a, b in CRISES.values():
        ax.axvspan(pd.Period(a, "M").start_time, pd.Period(b, "M").end_time,
                   color=GREY, alpha=0.15, lw=0)


def fig_premium(p, outdir):
    x = p.index.to_timestamp(how="end")
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0]
    _shade_crises(ax)
    ax.plot(x, 100 * p["iv"], color=BLUE, lw=1.2, label="Implied vol at the start of the month (30-day ATM)")
    ax.plot(x, 100 * np.sqrt(p["rv_next"]), color=ORANGE, lw=1.0, label="Realised vol over the month")
    ax.set_yscale("log")
    ax.set_yticks([20, 40, 80, 160, 300])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Annualised vol (%, log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    _shade_crises(ax)
    v = 100 * p["vrp_vol"]
    ax.bar(x, v.clip(lower=-60), width=25, color=np.where(v >= 0, BLUE, ORANGE), lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-60, 50)
    ax.set_ylabel("Implied minus realised\n(vol points, floored at -60)")
    fig.align_ylabels(axes)
    _save(fig, "fig1_premium", outdir)


def fig_cumulative(rets, outdir):
    rules = [c for c in rets.columns if c != "unmanaged"]
    x = rets.index.to_timestamp(how="end")
    base = 100 * rets["unmanaged"].cumsum()
    fig, axes = plt.subplots(4, 2, figsize=(6.5, 7.6), sharex=True, sharey=True)
    for ax, rule in zip(axes.ravel(), rules):
        _shade_crises(ax)
        ax.plot(x, base, color=GREY, lw=1.0, label="Unmanaged")
        ax.plot(x, 100 * rets[rule].cumsum(), color=BLUE, lw=1.2, label=RULE_LABELS[rule])
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(RULE_LABELS[rule], fontsize=9, loc="left", color=INK)
    handles = axes.ravel()[0].get_lines()[:2]
    last = axes.ravel()[-1]
    last.axis("off")
    last.legend(handles, ["Unmanaged (same in every panel)", "Managed by the rule in the title"],
                loc="center", fontsize=8)
    last.text(0.5, 0.18, "Shaded: crisis episodes in Table 4.\nAll strategies scaled to 10% annual vol.",
              ha="center", va="center", fontsize=7.5, color=INK2, transform=last.transAxes)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative P&L (%)")
    _save(fig, "fig2_cumulative", outdir)


def fig_weights(w, outdir):
    rules = [c for c in w.columns if c != "unmanaged"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    x = w.index.to_timestamp(how="start")
    data = w[rules].T.values
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=2, interpolation="nearest",
                   extent=[matplotlib.dates.date2num(x[0]), matplotlib.dates.date2num(x[-1]) + 30,
                           len(rules) - 0.5, -0.5])
    ax.xaxis_date()
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([RULE_LABELS[r] for r in rules], fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Weight", fontsize=8)
    _save(fig, "fig3_weights", outdir)
