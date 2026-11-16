"""Cross-asset summary of results already produced by run_all.py.

Usage (after run_all.py):
    python src/summarise.py                    # every commodity in results/
    python src/summarise.py --commodity CL GC C S
    python src/summarise.py --strike atm       # results computed with the ATM strike
Outputs go to results/summary/STRIKE/ (CSV, LaTeX tables, figures).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from commodities import INSTRUMENTS
from plots import BLUE, GREY, INK, INK2, ORANGE
from run_all import latex_table


STRIKE = C.STRIKE


def _read(code, name):
    return pd.read_csv(C.results_dir(code, STRIKE) / "tables" / f"{name}.csv", index_col=0)


def _panel(code):
    return pd.read_csv(C.processed_dir(code, STRIKE) / "monthly_panel.csv", index_col=0)


def vol_quintiles(code, n=5):
    """Mechanism sorts: months sorted on past-month realised variance (proxy
    for the conditional variance v_t); premium x = K - RV in variance units."""
    p = _panel(code)
    x = p["iv"] ** 2 - p["rv_next"]
    q = pd.qcut(p["rv_past"], n, labels=False) + 1
    g = pd.DataFrame({"x": x, "v": p["rv_past"], "q": q}).groupby("q")
    out = pd.DataFrame({"Mean past variance": 100 * g["v"].mean(),
                        "Mean premium": 100 * g["x"].mean(),
                        "Std premium": 100 * g["x"].std()})
    out["Sharpe (ann.)"] = np.sqrt(12) * out["Mean premium"] / out["Std premium"]
    out["Mean / variance"] = 100 * out["Mean premium"] / out["Std premium"] ** 2
    out.index.name = "Quintile"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commodity", nargs="+", metavar="CODE")
    ap.add_argument("--strike", choices=["smile", "atm"], default=C.STRIKE)
    args = ap.parse_args(argv)
    global STRIKE
    STRIKE = args.strike
    base = C.ROOT / "results" / STRIKE
    done = [d.name for d in base.iterdir() if (d / "tables").exists()] if base.exists() else []
    codes = [c.upper() for c in args.commodity] if args.commodity else \
        [c for c in INSTRUMENTS if c in done]           # order as in commodities.py
    if not codes:
        raise SystemExit("No results found: run src/run_all.py first.")
    out = C.ROOT / "results" / "summary" / STRIKE
    out.mkdir(parents=True, exist_ok=True)
    names = {c: SHORT.get(c, INSTRUMENTS[c]["name"]) for c in codes}

    # 1. premium and unmanaged strategy
    rows = {}
    for c in codes:
        t1, t2 = _read(c, "tab1_premium"), _read(c, "tab2_performance")
        vol = t1[t1.index.str.startswith("Vol premium")].iloc[0]
        var = t1[t1.index.str.startswith("Variance premium")].iloc[0]
        um = t2["Unmanaged"]
        pan = _panel(c)
        rows[names[c]] = {
            "Asset class": INSTRUMENTS[c].get("asset", ""),
            "Smile - ATM (pts)": 100 * (pan["iv_smile"] - pan["iv_atm"]).mean(),
            "Mean IV (%)": t1[t1.index.str.startswith("Implied vol")].iloc[0]["Mean"],
            "Mean RV (%)": t1[t1.index.str.startswith("Realised vol")].iloc[0]["Mean"],
            "Vol premium (pts)": vol["Mean"], "t": vol["t (NW)"],
            "Premium (% of IV)": 100 * pan["vrp_vol"].mean() / pan["iv"].mean(),
            "Var premium": var["Mean"], "t ": var["t (NW)"],
            "Months > 0 (%)": 100 * vol["Share $>0$"],
            "Sharpe": um["Sharpe"], "Skewness": um["Skewness"],
            "MaxDD (%)": um["Max drawdown (%)"], "CVaR99 (%)": um["CVaR 99 (%)"],
        }
    s1 = pd.DataFrame(rows).T
    s1.to_csv(out / "summary_premium.csv")
    latex_table(s1, out / "summary_premium.tex", index_name="Instrument")

    # 2. Sharpe ratio of every rule, and bootstrap p-value of the difference
    sr = pd.DataFrame({names[c]: _read(c, "tab2_performance").loc["Sharpe"] for c in codes})
    pv = pd.DataFrame({names[c]: _read(c, "tab3_spanning")["p-value"].astype(float) for c in codes})
    sr.to_csv(out / "summary_sharpe.csv")
    pv.to_csv(out / "summary_sharpe_pvalues.csv")
    latex_table(sr, out / "summary_sharpe.tex", index_name="Strategy")

    # 2b. costs proportional to the implied vol level (a fixed number of vol
    #     points is not comparable between a 1.4% 2-year note and 50% gas)
    rel = {}
    for c in codes:
        pan = _panel(c)
        rel[names[c]] = {f"{x:g}% of IV": np.sqrt(12) * (r := pan["ret_short_var"] - x / 100 * pan["iv"]).mean() / r.std()
                         for x in RELATIVE_COSTS}
    rel = pd.DataFrame(rel).T
    rel.to_csv(out / "summary_costs_relative.csv")
    latex_table(rel, out / "summary_costs_relative.tex", index_name="Instrument")

    # 3. strike sensitivity and crises
    pd.DataFrame({names[c]: _read(c, "tab8_strike").loc["Unmanaged"] for c in codes}).T \
        .to_csv(out / "summary_strike.csv")
    pd.DataFrame({names[c]: _read(c, "tab4_crises")["Unmanaged"] for c in codes}) \
        .to_csv(out / "summary_crises_unmanaged.csv")

    # 4. mechanism sorts
    quint = {c: vol_quintiles(c) for c in codes}
    pd.concat(quint, names=["Commodity"]).to_csv(out / "summary_quintiles.csv")

    fig_sharpe(sr, pv, out)
    fig_quintiles(quint, names, out)
    print(s1.round(2).to_string())
    print(sr.round(2).to_string())
    print(f"Summary written to {out}")


RELATIVE_COSTS = [0, 2.5, 5, 10]   # bid-ask cost per trade, % of implied vol

SHORT = {"CL": "Crude oil", "NG": "Natural gas", "GC": "Gold", "SI": "Silver",
         "HG": "Copper", "C": "Corn", "S": "Soybeans", "W": "Wheat",
         "TU": "UST 2y", "FV": "UST 5y", "TY": "UST 10y", "US": "UST 30y", "RX": "Bund",
         "HYG": "HYG", "LQD": "LQD", "CDXIG": "CDX IG", "SPX": "S&P 500"}
SECTORS = {}
for _c, _s in INSTRUMENTS.items():
    SECTORS.setdefault(_s.get("asset", "Other"), []).append(_c)
LINE_COLORS = [BLUE, ORANGE, "#1baf7a", "#eda100", "#8a5cd1"]


def fig_sharpe(sr, pv, out):
    rules = list(sr.index)
    n = sr.shape[1]
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.9 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y = np.arange(len(rules))[::-1]
    for ax, col in zip(axes, sr.columns):
        base = sr.loc["Unmanaged", col]
        ax.axvline(base, color=GREY, lw=1, ls="--")
        ax.axvline(0, color=INK2, lw=0.6)
        for yi, r in zip(y, rules):
            v = sr.loc[r, col]
            if r == "Unmanaged":
                ax.plot(v, yi, "s", color=INK, ms=6)
                continue
            sig = pv.loc[r, col] < 0.05 if r in pv.index else False
            c = BLUE if v >= base else ORANGE
            ax.plot([base, v], [yi, yi], color=c, lw=1.2)
            ax.plot(v, yi, "o", color=c, ms=6 if sig else 4, mfc=None if sig else "white")
        ax.set_title(col, fontsize=9, loc="left", color=INK)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(rules, fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[(nrow - 1) * ncol:n]:
        ax.set_xlabel("Sharpe ratio", fontsize=8)
    fig.suptitle("Sharpe ratio of each rule against the unmanaged strategy (dashed line). "
                 "Blue: better, orange: worse;\nfilled dot: difference significant at 5% "
                 "(block bootstrap), hollow: not significant. All at 10% annual volatility.",
                 fontsize=8, x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_summary_sharpe.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_summary_sharpe.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_quintiles(quint, names, out):
    groups = {k: [c for c in v if c in quint] for k, v in SECTORS.items()}
    groups = {k: v for k, v in groups.items() if v}
    other = [c for c in quint if not any(c in v for v in groups.values())]
    if other:
        groups["Other"] = other
    fig, axes = plt.subplots(2, len(groups), figsize=(max(7.0, 2.3 * len(groups)), 4.6),
                             sharex=True)
    axes = np.array(axes).reshape(2, len(groups))
    for j, (sector, codes) in enumerate(groups.items()):
        for i, (col, lab) in enumerate([("Sharpe (ann.)", "Conditional Sharpe ratio"),
                                        ("Mean / variance", "Mean / variance\n(optimal weight)")]):
            ax = axes[i, j]
            ax.axhline(0, color=INK2, lw=0.6)
            for k, c in enumerate(codes):
                q = quint[c][col]
                ax.plot(q.index, q.values, "-o", color=LINE_COLORS[k % len(LINE_COLORS)], lw=1.6, ms=4,
                        label=SHORT.get(c, c))
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 0:
                ax.set_title(sector, fontsize=9, loc="left", color=INK)
                ax.legend(fontsize=7.5, loc="best", frameon=True, framealpha=0.9,
                          edgecolor="none")
            else:
                ax.set_xlabel("Past-variance quintile (1 = calmest)", fontsize=8)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_summary_quintiles.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_summary_quintiles.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
