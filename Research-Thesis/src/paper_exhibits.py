"""Tables, figures and numbers for the paper, from the outputs of run_all.py.

Usage (after `run_all.py --data FILE` with both strikes and `summarise.py`):
    python src/paper_exhibits.py
Writes paper/generated/: tab_*.tex, fig_*.pdf and numbers.tex (one LaTeX
macro per number quoted in the text, so the text always matches the data).
These files are derived from licensed data and are not tracked by git.
"""
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import config as C
from commodities import INSTRUMENTS, is_spot
from evaluation import bootstrap_sharpe_diff, newey_west_mean, sharpe

OUT = C.ROOT / "paper" / "generated"
STRIKE = "smile"

NAME = {"CL": "Crude oil", "NG": "Natural gas", "GC": "Gold", "SI": "Silver", "HG": "Copper",
        "C": "Corn", "W": "Wheat", "S": "Soybeans", "TU": "US 2-year", "FV": "US 5-year",
        "TY": "US 10-year", "US": "US 30-year", "RX": "Bund", "HYG": "HYG", "LQD": "LQD",
        "CDXIG": "CDX IG", "SPX": "S\\&P 500"}
PLAIN = {k: v.replace("\\&", "&") for k, v in NAME.items()}
MACRO = {"CL": "Oil", "NG": "Gas", "GC": "Gold", "SI": "Silver", "HG": "Copper", "C": "Corn",
         "W": "Wheat", "S": "Soy", "TU": "Twoy", "FV": "Fivey", "TY": "Teny", "US": "Thirtyy",
         "RX": "Bund", "HYG": "Hyg", "LQD": "Lqd", "CDXIG": "Cdx", "SPX": "Spx"}
CLASSES = ["Energy", "Metals", "Grains", "Rates", "Credit"]
CLASS_COL = {"Energy": "#d9622b", "Metals": "#8f8d87", "Grains": "#1a9e6e", "Rates": "#2a78d6",
             "Credit": "#7a52c7", "Equity": "#0b0b0b"}
RULES = ["Volatility scaling", "Realised-vol threshold", "Implied-vol jump", "Implied-vol trend",
         "Term structure", "Premium filter", "Backwardation"]
RULE_KEY = {"Volatility scaling": "vol_scaling", "Realised-vol threshold": "rv_threshold",
            "Implied-vol jump": "iv_jump", "Implied-vol trend": "iv_trend",
            "Term structure": "term_structure", "Premium filter": "premium_filter",
            "Backwardation": "backwardation"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.axisbelow": True,
})
MACROS = {}


# ------------------------------------------------------------------ readers
def tab(code, name, strike=STRIKE):
    return pd.read_csv(C.results_dir(code, strike) / "tables" / f"{name}.csv", index_col=0)


def panel(code, strike=STRIKE):
    p = pd.read_csv(C.processed_dir(code, strike) / "monthly_panel.csv", index_col=0)
    p.index = pd.PeriodIndex(p.index, freq="M")
    return p


def rets(code, strike=STRIKE):
    r = pd.read_csv(C.processed_dir(code, strike) / "returns_volmatched.csv", index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M")
    return r


def asset(code):
    return INSTRUMENTS[code].get("asset", "Other")


def macro(name, value, fmt="{:.2f}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value
    return MACROS[name]


def num(x, fmt="{:.2f}"):
    if pd.isna(x):
        return "--"
    s = fmt.format(x)
    return f"${s}$" if s.startswith("-") else s


def write_table(name, header, rows, colspec, caption=None, groups=None, resize=True, notes=None):
    """booktabs tabular; rows = list of lists of strings; groups = {row index: label}."""
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"]
    for i, row in enumerate(rows):
        if groups and i in groups:
            if i > 0:
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(header), groups[i]))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if resize:
        body = "\\resizebox{\\linewidth}{!}{%\n" + body + "}"
    (OUT / f"{name}.tex").write_text(body + "\n")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def grouped(codes):
    """Rows ordered by asset class, with the class label at the first row."""
    order, groups = [], {}
    for cls in CLASSES + ["Equity"]:
        cs = [c for c in codes if asset(c) == cls]
        if cs:
            groups[len(order)] = cls if cls != "Equity" else "Benchmark"
            order += cs
    return order, groups


# ------------------------------------------------------------------ tables
def table_sample(codes):
    rows = []
    order, groups = grouped(codes)
    for c in order:
        p = panel(c)
        s = INSTRUMENTS[c]
        if c == "CDXIG":
            strike = "VIXIG (bp)"
        elif c == "SPX":
            strike = "VIX"
        else:
            strike = "Smile" if s.get("smile", True) else "ATM"
        units = " (bp)" if c == "CDXIG" else ""
        rows.append([NAME[c], s["exchange"], "Rolled futures" if not is_spot(c) else
                     ("Index" if c in ("CDXIG", "SPX") else "ETF"),
                     p.index[0].strftime("%b %Y"), f"{len(p)}", strike,
                     num(100 * p["iv"].mean(), "{:.1f}") + units,
                     num(100 * np.sqrt(p["rv_next"]).mean(), "{:.1f}") + units])
    write_table("tab_sample", ["Instrument", "Exchange", "Position", "First month", "Months",
                               "Strike", "Mean IV", "Mean RV"], rows, "lllllrrr", groups=groups)


def table_premium(codes):
    order, groups = grouped(codes)
    rows = []
    for c in order:
        t1, t2 = tab(c, "tab1_premium"), tab(c, "tab2_performance")
        vol = t1[t1.index.str.startswith("Vol premium")].iloc[0]
        var = t1[t1.index.str.startswith("Variance premium")].iloc[0]
        p = panel(c)
        um = t2["Unmanaged"]
        rel = 100 * p["vrp_vol"].mean() / p["iv"].mean()
        rows.append([NAME[c], num(vol["Mean"]), num(vol["t (NW)"]), num(rel, "{:.1f}"),
                     num(var["t (NW)"]), num(100 * vol["Share $>0$"], "{:.0f}"),
                     num(um["Sharpe"]), num(um["Skewness"], "{:.1f}"),
                     num(um["Max drawdown (%)"], "{:.0f}"), num(um["CVaR 99 (%)"], "{:.1f}")])
        m = MACRO[c]
        macro(f"VP{m}", vol["Mean"])
        macro(f"VPt{m}", vol["t (NW)"], "{:.1f}")
        macro(f"Rel{m}", rel, "{:.1f}")
        macro(f"SR{m}", um["Sharpe"])
        macro(f"Skew{m}", um["Skewness"], "{:.1f}")
        macro(f"MDD{m}", um["Max drawdown (%)"], "{:.0f}")
        macro(f"Pos{m}", 100 * vol["Share $>0$"], "{:.0f}")
        macro(f"IV{m}", 100 * p["iv"].mean(), "{:.1f}")
        macro(f"RV{m}", 100 * np.sqrt(p["rv_next"]).mean(), "{:.1f}")
        macro(f"N{m}", len(p), "{:d}")
    write_table("tab_premium", ["", "Vol premium", "$t$", "\\% of IV", "$t$ (variance)",
                                "Months $>0$ (\\%)", "Sharpe", "Skewness", "Max DD (\\%)",
                                "CVaR 99 (\\%)"], rows, "lrrrrrrrrr", groups=groups)


def relative_cost_sharpe(c, x):
    p = panel(c)
    r = p["ret_short_var"] - x / 100 * p["iv"]
    return sharpe(r)


def table_robust(codes):
    order, groups = grouped(codes)
    rows = []
    for c in order:
        sm = tab(c, "tab2_performance").loc["Sharpe", "Unmanaged"]
        atm = tab(c, "tab2_performance", "atm").loc["Sharpe", "Unmanaged"]
        t6 = tab(c, "tab6_subperiods").loc["Unmanaged"]
        t7 = tab(c, "tab7_robustness")["Unmanaged"]
        k0 = t7.get("Roll at expiry (generic F1)", np.nan)
        k10 = t7.get("Roll 10 days before expiry", np.nan)
        costs = [relative_cost_sharpe(c, x) for x in (2.5, 5, 10)]
        rows.append([NAME[c], num(sm), num(atm) if INSTRUMENTS[c].get("smile", True) else "--",
                     num(t6["SR in-sample"]), num(t6["SR out-of-sample"]),
                     num(t7["Excl. Mar-Apr 2020"]), num(k0), num(k10)] + [num(v) for v in costs])
        m = MACRO[c]
        macro(f"SRatm{m}", atm)
        macro(f"SRa{m}", t6["SR in-sample"])
        macro(f"SRb{m}", t6["SR out-of-sample"])
        macro(f"SRexc{m}", t7["Excl. Mar-Apr 2020"])
        macro(f"SRcostA{m}", costs[0])
        macro(f"SRcostB{m}", costs[1])
        macro(f"SRcostC{m}", costs[2])
    write_table("tab_robust", ["", "Baseline", "ATM strike", "2007--16", "2017--26",
                               "Excl.\\ Mar--Apr 2020", "Roll $k=0$", "Roll $k=10$",
                               "Cost 2.5\\%", "Cost 5\\%", "Cost 10\\%"], rows,
                "lrrrrrrrrrr", groups=groups)


def table_crises(codes):
    order, groups = grouped(codes)
    names = list(C.CRISES)
    rows = []
    for c in order:
        t4 = tab(c, "tab4_crises")["Unmanaged"]
        p = panel(c)
        vals = []
        for k in names:
            a, b = C.CRISES[k]
            covered = p.index[0] <= pd.Period(a, "M")
            vals.append(num(t4[k], "{:.1f}") if covered else "--")
        rows.append([NAME[c]] + vals)
        if p.index[0] <= pd.Period(C.CRISES["COVID 2020"][0], "M"):
            macro(f"Covid{MACRO[c]}", -t4["COVID 2020"], "{:.0f}")
    short = [k.replace(" 20", " '").replace("Commodity crash '14-16", "Commodities '14--16")
             for k in names]
    write_table("tab_crises", [""] + short, rows, "l" + "r" * len(names), groups=groups)


def timing_tables(codes):
    """Change in Sharpe ratio by rule, bootstrap p-values, multiple testing."""
    d = pd.DataFrame(index=RULES, columns=codes, dtype=float)
    pv = d.copy()
    for c in codes:
        sr = tab(c, "tab2_performance").loc["Sharpe"]
        d[c] = (sr[RULES] - sr["Unmanaged"]).values
        pv[c] = tab(c, "tab3_spanning").loc[RULES, "p-value"].astype(float).values
    # rules that need data an instrument does not have are not applicable:
    # backwardation needs a futures curve, the term-structure rule a second
    # implied-vol maturity (missing for VIXIG and VIX)
    for c in codes:
        if is_spot(c):
            d.loc["Backwardation", c] = np.nan
            pv.loc["Backwardation", c] = np.nan
        if panel(c)["iv2m"].isna().all():
            d.loc["Term structure", c] = np.nan
            pv.loc["Term structure", c] = np.nan
    flat = pv.stack().dropna()
    n = len(flat)
    sig = (flat < 0.05)
    up = sum(1 for (r, c), p in flat.items() if p < 0.05 and d.loc[r, c] > 0)
    down = int(sig.sum()) - up
    binom = stats.binomtest(int(sig.sum()), n, 0.05, alternative="greater").pvalue
    # Holm step-down over all tests
    srt = np.sort(flat.values)
    holm_rej = 0
    for i, p in enumerate(srt):
        if p <= 0.05 / (n - i):
            holm_rej += 1
        else:
            break
    macro("NTests", n, "{:d}")
    macro("NSig", int(sig.sum()), "{:d}")
    macro("NSigUp", up, "{:d}")
    macro("NSigDown", down, "{:d}")
    macro("NExpected", 0.05 * n, "{:.0f}")
    macro("BinomP", binom, "{:.2f}")
    macro("HolmRej", holm_rej, "{:d}")
    macro("MinP", srt[0], "{:.3f}")
    for r in RULES:
        dd = d.loc[r].dropna()
        macro(f"Up{RULE_KEY[r].replace('_', '')}", int((dd > 0).sum()), "{:d}")
        macro(f"N{RULE_KEY[r].replace('_', '')}", len(dd), "{:d}")
    srs = tab("CDXIG", "tab2_performance").loc["Sharpe"]
    macro("SRCdxIVJ", srs["Implied-vol jump"])
    macro("SRSpxVS", tab("SPX", "tab2_performance").loc["Sharpe", "Volatility scaling"])
    macro("PSpxVS", pv.loc["Volatility scaling", "SPX"])
    # summary by rule
    rows = []
    for r in RULES:
        dd, pp = d.loc[r].dropna(), pv.loc[r].dropna()
        rows.append([r, f"{len(dd)}", f"{int((dd > 0).sum())}", num(dd.mean()), num(dd.median()),
                     f"{int(((pp < 0.05) & (dd > 0)).sum())}", f"{int(((pp < 0.05) & (dd < 0)).sum())}"])
    write_table("tab_timing", ["Rule", "Instruments", "$\\Delta SR>0$", "Mean $\\Delta SR$",
                               "Median $\\Delta SR$", "Better ($p<0.05$)", "Worse ($p<0.05$)"],
                rows, "lrrrrrr")
    return d, pv


def fig_timing(d, pv, codes):
    order, _ = grouped(codes)
    d, pv = d[order], pv[order]
    cmap = LinearSegmentedColormap.from_list("div", ["#c8483f", "#f2f1ee", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    vals = d.values.astype(float)
    ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(len(RULES)):
        for j in range(len(order)):
            v, p = vals[i, j], pv.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.5, color=INK2)
                continue
            star = "*" if p < 0.05 else ""
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=6.2,
                    fontweight="bold" if star else "normal", color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PLAIN[c] for c in order], rotation=50, ha="left", fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(RULES)))
    ax.set_yticklabels(RULES, fontsize=7.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    prev = None
    for j, c in enumerate(order):
        if prev is not None and asset(c) != prev:
            ax.axvline(j - 0.5, color="white", lw=2.5)
        prev = asset(c)
    save(fig, "fig_timing")


def portfolios(codes):
    """Equal-weight portfolios of the vol-matched strategies (instruments that
    exist in a month), rescaled to 10%; by class and all non-equity."""
    core = [c for c in codes if asset(c) in CLASSES]
    R = {c: rets(c) for c in codes}

    def port(cs, rule="unmanaged"):
        return pd.DataFrame({c: R[c][rule] for c in cs}).mean(axis=1)

    def scaled(x):
        x = x.dropna()
        return x * 0.10 / (x.std() * np.sqrt(12))

    def row(label, x):
        x = scaled(x)
        cum = x.cumsum()
        a, b = x.loc[:"2016-12"], x.loc["2017-01":]
        return [label, num(sharpe(x)), num(sharpe(a)), num(sharpe(b)), num(x.skew(), "{:.1f}"),
                num(100 * (cum.cummax() - cum).max(), "{:.0f}"),
                num(-100 * x[x <= x.quantile(0.01)].mean(), "{:.1f}")]

    rows, groups = [], {0: "Unmanaged, by asset class"}
    cls_port = {}
    for cls in CLASSES:
        cs = [c for c in core if asset(c) == cls]
        cls_port[cls] = port(cs)
        rows.append(row(cls, cls_port[cls]))
    allp = port(core)
    groups[len(rows)] = f"All {len(core)} instruments, equal weight"
    rows.append(row("Unmanaged", allp))
    macro("SRAll", sharpe(allp))
    macro("SRAllA", sharpe(allp.loc[:"2016-12"]))
    macro("SRAllB", sharpe(allp.loc["2017-01":]))
    macro("SkewAll", scaled(allp).skew(), "{:.1f}")
    out = {}
    for rule in ["Volatility scaling", "Implied-vol jump"]:
        k = RULE_KEY[rule]
        x = port(core, k)
        rows.append(row(rule, x))
        dsr, p, _, _ = bootstrap_sharpe_diff(x, allp)
        out[rule] = (sharpe(allp), sharpe(x), p)
    ex = [c for c in core if c != "CDXIG"]
    groups[len(rows)] = "Excluding CDX IG"
    exu = port(ex)
    rows.append(row("Unmanaged", exu))
    exj = port(ex, "iv_jump")
    rows.append(row("Implied-vol jump", exj))
    _, p_ex, _, _ = bootstrap_sharpe_diff(exj, exu)
    groups[len(rows)] = "Benchmark"
    spx = rets("SPX")["unmanaged"]
    rows.append(row("S\\&P 500 (VIX strike)", spx))
    write_table("tab_portfolio", ["Portfolio", "Sharpe", "2007--16", "2017--26", "Skewness",
                                  "Max DD (\\%)", "CVaR 99 (\\%)"], rows, "lrrrrrr", groups=groups,
                resize=False)
    macro("SRAllVS", out["Volatility scaling"][1])
    macro("PAllVS", out["Volatility scaling"][2])
    macro("SRAllIVJ", out["Implied-vol jump"][1])
    macro("PAllIVJ", out["Implied-vol jump"][2])
    macro("SRExCdx", sharpe(exu))
    macro("SRExCdxIVJ", sharpe(exj))
    macro("PExCdxIVJ", p_ex)
    keep = ~allp.index.isin(pd.PeriodIndex(["2020-03", "2020-04"], freq="M"))
    macro("SRAllExCovid", sharpe(allp[keep]))
    macro("SRAllIVJExCovid", sharpe(port(core, "iv_jump")[keep]))
    # correlations
    um = pd.DataFrame({c: R[c]["unmanaged"] for c in core})
    cm = um.corr().values
    macro("AvgCorr", cm[np.triu_indices(len(core), 1)].mean())
    cc = pd.DataFrame(cls_port).assign(Equity=spx).corr()
    rows = [[k] + [num(cc.loc[k, j]) for j in cc.columns] for k in cc.index]
    write_table("tab_corr", [""] + list(cc.columns), rows, "l" + "r" * len(cc.columns),
                resize=False)
    macro("CorrRatesCredit", cc.loc["Rates", "Credit"])
    macro("CorrGrainsRates", cc.loc["Grains", "Rates"])
    macro("CorrGrainsCredit", cc.loc["Grains", "Credit"])
    macro("CorrRatesEq", cc.loc["Rates", "Equity"])
    macro("CorrCreditEq", cc.loc["Credit", "Equity"])
    # figure: cumulative P&L by class
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    lines = {"Energy": cls_port["Energy"], "Metals": cls_port["Metals"],
             "Grains": cls_port["Grains"], "Rates": cls_port["Rates"],
             "Credit ETFs": port(["HYG", "LQD"]), "CDX IG": R["CDXIG"]["unmanaged"]}
    colors = {"Energy": CLASS_COL["Energy"], "Metals": CLASS_COL["Metals"],
              "Grains": CLASS_COL["Grains"], "Rates": CLASS_COL["Rates"],
              "Credit ETFs": "#b9a5e3", "CDX IG": CLASS_COL["Credit"]}
    for k, ser in lines.items():
        s = scaled(ser)
        x = s.index.to_timestamp(how="end")
        ax.plot(x, 100 * s.cumsum(), color=colors[k], lw=1.5, label=k)
    s = scaled(spx)
    ax.plot(s.index.to_timestamp(how="end"), 100 * s.cumsum(), color=INK, lw=1.0, ls="--",
            label="S&P 500 (VIX)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("Cumulative P&L (% of capital)")
    ax.legend(ncol=4, loc="upper left", fontsize=7.5, bbox_to_anchor=(0, 1.2))
    save(fig, "fig_cumulative")


def mechanism(codes):
    """How the premium scales with the variance level (Proposition 1).

    Under (A1)-(A2), x/v = lambda * v^(b-1) + e with homoskedastic e, where
    x = K - RV and v = past-month realised variance (the proxy for the
    conditional variance). The slope of x/v on ln v, with Newey-West errors,
    is zero when b = 1 (constant conditional Sharpe ratio, the case in which
    inverse-variance scaling is optimal), positive when b > 1 and negative
    when b < 1. Conditional Sharpe ratios by past-variance quintile give the
    same information without the functional form."""
    import statsmodels.api as sm
    rows, quint = [], {}
    order, groups = grouped(codes)
    for c in order:
        p = panel(c)
        x, v = p["iv"] ** 2 - p["rv_next"], p["rv_past"]
        ok = np.isfinite(x) & np.isfinite(v) & (v > 0)
        y, lv = (x / v)[ok], np.log(v[ok])
        res = sm.OLS(y.values, sm.add_constant(lv.values)).fit(
            cov_type="HAC", cov_kwds={"maxlags": C.NEWEY_WEST_LAGS})
        slope, tval = res.params[1], res.tvalues[1]
        q = pd.qcut(p["rv_past"], 5, labels=False) + 1
        g = x.groupby(q)
        csr = np.sqrt(12) * g.mean() / g.std()
        quint[c] = csr
        verdict = "$b>1$" if tval > 1.96 else ("$b<1$" if tval < -1.96 else "$b=1$ not rejected")
        rows.append([NAME[c], num(slope), num(tval, "{:.1f}"), verdict,
                     num(csr.loc[1]), num(csr.loc[3]), num(csr.loc[5])])
        macro(f"Slope{MACRO[c]}", slope)
        macro(f"SlopeT{MACRO[c]}", tval, "{:.1f}")
        macro(f"QoneSR{MACRO[c]}", csr.loc[1])
        macro(f"QfiveSR{MACRO[c]}", csr.loc[5])
    write_table("tab_mechanism", ["", "Slope of $x/v$ on $\\ln v$", "$t$ (NW)", "Implied elasticity",
                                  "SR, Q1 (calm)", "SR, Q3", "SR, Q5 (volatile)"], rows,
                "lrrlrrr", groups=groups)
    # figure: conditional Sharpe by quintile, averaged within class
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 2.0), sharey=True)
    for ax, cls in zip(axes, CLASSES + ["Equity"]):
        cs = [c for c in order if asset(c) == cls]
        for c in cs:
            ax.plot(range(1, 6), quint[c].values, color=CLASS_COL[cls], lw=0.8, alpha=0.45)
        ax.plot(range(1, 6), pd.DataFrame({c: quint[c] for c in cs}).mean(axis=1).values,
                color=CLASS_COL[cls], lw=2.0, marker="o", ms=3)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(cls if cls != "Equity" else "S&P 500", fontsize=8, loc="left")
        ax.set_xticks([1, 3, 5])
    axes[0].set_ylabel("Conditional Sharpe (ann.)")
    fig.supxlabel("Quintile of past-month realised variance (1 = calmest)", fontsize=7.5, y=0.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_quintiles")


def fig_premium(codes):
    order, _ = grouped([c for c in codes if c != "SPX"])
    rel, se, srs, col = [], [], [], []
    for c in order:
        p = panel(c)
        t1 = tab(c, "tab1_premium")
        vol = t1[t1.index.str.startswith("Vol premium")].iloc[0]
        r = 100 * p["vrp_vol"].mean() / p["iv"].mean()
        rel.append(r)
        se.append(r / vol["t (NW)"])
        srs.append(tab(c, "tab2_performance").loc["Sharpe", "Unmanaged"])
        col.append(CLASS_COL[asset(c)])
    spx_sr = tab("SPX", "tab2_performance").loc["Sharpe", "Unmanaged"]
    y = np.arange(len(order))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=True,
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    a1.barh(y, rel, color=col, height=0.62)
    a1.errorbar(rel, y, xerr=1.96 * np.array(se), fmt="none", ecolor=INK2, elinewidth=0.7,
                capsize=1.8)
    a1.axvline(0, color=INK2, lw=0.7)
    a1.set_yticks(y)
    a1.set_yticklabels([PLAIN[c] for c in order])
    a1.set_xlabel("Implied minus realised vol, % of implied vol (95% CI)")
    a2.barh(y, srs, color=col, height=0.62)
    a2.axvline(0, color=INK2, lw=0.7)
    a2.axvline(spx_sr, color=INK, lw=0.9, ls="--")
    a2.text(spx_sr + 0.02, y[0] + 0.6, f"S&P 500: {spx_sr:.2f}", fontsize=7, color=INK)
    a2.set_xlabel("Sharpe ratio of short one-month variance")
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
        prev = None
        for yi, c in zip(y, order):
            if prev is not None and asset(c) != prev:
                ax.axhline(yi + 0.5, color=GRID, lw=0.8)
            prev = asset(c)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COL[k]) for k in CLASSES]
    fig.legend(handles, CLASSES, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_premium")


def fig_seasonal():
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    w = 0.27
    months = np.arange(1, 13)
    for k, (c, col) in enumerate([("C", "#1a9e6e"), ("S", "#7cc9a8"), ("W", "#b9b6ae")]):
        p = panel(c)
        m = 100 * p.groupby(p.index.month)["vrp_vol"].mean()
        ax.bar(months + (k - 1) * w, m.reindex(months).values, width=w, color=col,
               label=PLAIN[c])
    ax.axhline(0, color=INK2, lw=0.7)
    for mth in (3, 6):
        ax.axvspan(mth - 0.5, mth + 0.5, color="#eda100", alpha=0.12, lw=0)
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec"])
    ax.set_ylabel("Vol premium (vol points)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, fontsize=7.5, loc="upper right")
    save(fig, "fig_seasonal")
    for c in ("C", "S"):
        p = panel(c)
        m = 100 * p.groupby(p.index.month)["vrp_vol"].mean()
        macro(f"Mar{MACRO[c]}", m.loc[3], "{:.1f}")
        macro(f"Jun{MACRO[c]}", m.loc[6], "{:.1f}")
        macro(f"Win{MACRO[c]}", m.loc[[11, 12, 1, 2]].mean(), "{:.1f}")


def table_calendar(codes):
    rows = []
    for c in [c for c in codes if not is_spot(c)]:
        a = tab(c, "tabA1_calendar")["ratio_to_all_days"]
        rows.append([NAME[c], INSTRUMENTS[c]["cycle"]] + [num(a.loc[o], "{:.1f}") for o in
                                                            (-1, 0, 1, 2)])
    write_table("tab_calendar", ["Contract", "Months", "Offset $-1$", "$0$", "$+1$", "$+2$"],
                rows, "llrrrr", resize=False)


def cdx_detail():
    p = panel("CDXIG")
    r = p["ret_short_var"]
    worst = r.idxmin()
    typical = r.median()
    macro("CdxWorstMonth", worst.strftime("%B %Y"))
    macro("CdxWorstMultiple", -r.min() / r.mean(), "{:.0f}")
    by_year = 100 * p["vrp_vol"].groupby(p.index.year).mean()
    macro("CdxYearMin", by_year.min(), "{:.0f}")
    macro("CdxYearMax", by_year.max(), "{:.0f}")
    macro("CdxYearsPos", int((by_year > 0).sum()), "{:d}")
    macro("CdxYears", len(by_year), "{:d}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = C.ROOT / "results" / STRIKE
    done = [d.name for d in base.iterdir() if (d / "tables").exists()]
    codes = [c for c in INSTRUMENTS if c in done]
    table_sample(codes)
    table_premium(codes)
    table_robust(codes)
    table_crises(codes)
    d, pv = timing_tables(codes)
    fig_timing(d, pv, codes)
    portfolios(codes)
    mechanism(codes)
    fig_premium(codes)
    fig_seasonal()
    table_calendar(codes)
    cdx_detail()
    macro("NInstr", len([c for c in codes if c != "SPX"]), "{:d}")
    (OUT / "numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n")
    print(f"{len(MACROS)} numbers, tables and figures written to {OUT}")


if __name__ == "__main__":
    main()
