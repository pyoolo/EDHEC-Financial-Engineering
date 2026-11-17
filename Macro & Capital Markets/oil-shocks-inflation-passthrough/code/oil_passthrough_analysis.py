"""
Oil shocks, inflation pass-through and market inflation expectations
Macro & Capital Markets project — full analysis

Usage:  python oil_passthrough_analysis.py Bloomberg_download_oil_inflation.xlsx [output_folder]
Needs:  pandas numpy statsmodels matplotlib openpyxl

A  Pass-through of Brent to YoY inflation (local projections), US vs EA, headline vs core, pre/post 2020
B  Sensitivity of 5y and 5y5y inflation expectations to oil (weekly), by period + rolling
C  Consistency: realized pass-through vs what the 5y breakeven / swap prices
D  Case study: 2022 (Ukraine) vs 2026 (Iran / Hormuz)
E  Out-of-sample test: model estimated to Feb 2026, forecast of inflation after the March 2026 shock
F  Global-demand vs oil-specific (supply) shocks: sign + narrative restrictions on weekly (Brent, S&P 500)
G  State dependence (high vs low inflation regime) and asymmetry (oil up vs down)
H  Robustness checks
All effects are reported for a +10% move in Brent.
"""
import sys, os, warnings
import numpy as np, pandas as pd, openpyxl
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

FILE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "output"
os.makedirs(OUT, exist_ok=True)
SHOCK, P, H = 10, 6, 24           # shock size (%), lags of controls, horizons (months)
RED, BLUE, GREY, LRED, LBLUE = "#c0392b", "#1f4e79", "#8a8a8a", "#e8a39b", "#9db4cf"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
TABLES, NOTES = {}, []

# =============================================================== 0. DATA
def read_sheet(wb, name):
    """Bloomberg BDH layout: blocks of 3 columns (date, value, blank); row 1 = label, row 2 = ticker."""
    ws = wb[name]; rows = list(ws.iter_rows(values_only=True)); out = {}
    for c in range(0, ws.max_column, 3):
        if rows[0][c] is None: continue
        pts = [(pd.Timestamp(r[c]), r[c + 1]) for r in rows[3:]
               if r[c] is not None and isinstance(r[c + 1], (int, float))]
        out[rows[0][c]] = pd.Series(dict(pts)).sort_index()
    return out

wb = openpyxl.load_workbook(FILE, data_only=True)
mon, day = read_sheet(wb, "MENSILE"), read_sheet(wb, "GIORNALIERO")

def to_month(s):
    s = s.copy(); s.index = s.index.to_period("M"); return s[~s.index.duplicated(keep="last")]

M = pd.DataFrame({
    "brent": to_month(mon["Brent (front month)"]), "wti": to_month(mon["WTI (front month)"]),
    "bcom": to_month(mon["Bloomberg Commodity Index"]),
    "us_head": to_month(mon["US CPI headline YoY"]), "us_core": to_month(mon["US CPI core YoY"]),
    "ea_head": to_month(mon["Euro HICP headline YoY"]), "ea_core": to_month(mon["Euro HICP core YoY"]),
    "us_lvl": to_month(mon["US CPI livello indice (NSA)"]), "ea_lvl": to_month(mon["Euro HICP livello indice"]),
})
D = pd.DataFrame({
    "brent": day["Brent (front month)"], "spx": day["S&P 500 "],
    "us5": day["US 5y breakeven"], "us5y5y": day["US 5y5y inflation swap fwd"],
    "ea5": day["Euro 5y inflation swap"], "ea5y5y": day["Euro 5y5y inflation swap fwd"],
})

# Oct 2025 US CPI was never published (federal shutdown): fill that one month by linear interpolation
for c in ["us_head", "us_core"]:
    M.loc[:"2026-09", c] = M.loc[:"2026-09", c].interpolate(limit=1, limit_area="inside")
NOTES.append("US CPI for Oct 2025 was never published (government shutdown); filled by linear interpolation.")

# Monthly AVERAGE Brent matches how CPI is measured (average prices over the month).
# 2004+: true average of daily closes. Before 2004: geometric mean of adjacent month-end prices (corr. 0.87 with true).
avg = D.brent.resample("ME").mean(); avg.index = avg.index.to_period("M")
proxy = np.exp((np.log(M.brent) + np.log(M.brent.shift(1))) / 2)
M["brent_avg"] = avg.combine_first(proxy)
M["oil"] = 100 * np.log(M.brent_avg).diff()           # baseline shock: monthly-average Brent, log %
M["oil_eom"] = 100 * np.log(M.brent).diff()           # robustness: month-end Brent
M["wti_eom"] = 100 * np.log(M.wti).diff()             # robustness: WTI (positive throughout month-ends)
M["bcom_eom"] = 100 * np.log(M.bcom).diff()           # robustness: broad commodities
NOTES.append("Baseline oil shock = monthly log change in AVERAGE Brent (daily average from 2004; mean of adjacent month-ends before).")

series = {"us_head": "US headline CPI", "us_core": "US core CPI",
          "ea_head": "Euro area headline HICP", "ea_core": "Euro area core HICP"}
regime_of = {"us_head": "us_head", "us_core": "us_head", "ea_head": "ea_head", "ea_core": "ea_head"}

# =============================================================== helpers
def lp(df, y, shocks, start=None, end=None, extra=None, p=P, h_max=H, level=False, exclude=None):
    """
    Local projections:  y_{t+h} - y_{t-1} = a + sum_k b_{k,h} shock_{k,t} + p lags of each shock / extra / dy + e
    y = YoY inflation in pp, or 100*log index (level=True, adds month dummies). Newey-West s.e. (h+1 lags).
    Returns long DataFrame: h, shock, b, se, n   with b scaled to a +10% shock.
    """
    d = pd.DataFrame(index=df.index)
    d["Y"] = 100 * np.log(df[y]) if level else df[y]
    d["dY"] = d.Y.diff()
    X = []
    for k, s in {**shocks, **(extra or {})}.items():
        X.append(s.rename(k)); X += [s.shift(j).rename(f"{k}_l{j}") for j in range(1, p + 1)]
    X += [d.dY.shift(j).rename(f"dy_l{j}") for j in range(1, p + 1)]
    X = pd.concat(X, axis=1)
    if level:
        X = X.join(pd.DataFrame(pd.get_dummies(d.index.month, prefix="m", drop_first=True, dtype=float).values,
                                index=d.index, columns=[f"m{i}" for i in range(2, 13)]))
    out = []
    for h in range(h_max + 1):
        lhs = (d.Y.shift(-h) - d.Y.shift(1)).rename("lhs")
        z = pd.concat([lhs, X], axis=1).loc[start:end].dropna()
        if exclude is not None: z = z[~z.index.isin(exclude)]
        f = sm.OLS(z.lhs, sm.add_constant(z.drop(columns="lhs"))).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        for k in shocks: out.append((h, k, SHOCK * f.params[k], SHOCK * f.bse[k], int(f.nobs)))
    return pd.DataFrame(out, columns=["h", "shock", "b", "se", "n"])

def one(r):  # single-shock LP -> indexed by h
    return r.set_index("h")

def ols_hac(y, X, lags=4):
    z = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(z.iloc[:, 0], sm.add_constant(z.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

def savefig(fig, name):
    rect = (0, 0.09, 1, 1) if fig.legends else (0, 0, 1, 1)
    fig.tight_layout(rect=rect); fig.savefig(f"{OUT}/{name}", dpi=180)
    # vector copy for the paper, without the overall title (the paper caption replaces it)
    if fig._suptitle is not None: fig._suptitle.set_text("")
    if len(fig.axes) == 1: fig.axes[0].set_title("")
    scale = 1.25 if fig.get_figwidth() <= 11.5 else 1.45          # larger text: figures are shrunk to page width
    for t in fig.findobj(matplotlib.text.Text): t.set_fontsize(t.get_fontsize() * scale)
    fig.tight_layout(rect=(0, 0.11, 1, 1) if fig.legends else (0, 0, 1, 1))
    fig.savefig(f"{OUT}/{name.replace('.png', '.pdf')}"); plt.close(fig)

# =============================================================== A. PASS-THROUGH
samples = {"Full sample": (None, None), "Pre-2020": (None, "2019-12"), "2020-2026": ("2020-01", None)}
LP = {(s, k): one(lp(M, s, {"oil": M.oil}, *v)) for s in series for k, v in samples.items()}

rows = []
for (s, k), r in LP.items():
    pk = r.b.abs().idxmax()
    rows.append({"series": series[s], "sample": k, "h=0": r.b[0], "h=3": r.b[3], "h=6": r.b[6],
                 "h=12": r.b[12], "h=18": r.b[18], "h=24": r.b[24], "peak": r.b[pk], "peak month": pk,
                 "s.e. h=6": r.se[6], "s.e. h=12": r.se[12], "obs": r.n[0]})
TABLES["A_passthrough"] = pd.DataFrame(rows)

col_s = {"Full sample": BLUE, "Pre-2020": GREY, "2020-2026": RED}
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, s in zip(axes.flat, series):
    for k in samples:
        r = LP[(s, k)]
        ax.plot(r.index, r.b, color=col_s[k], lw=2.2 if k == "Full sample" else 1.6, label=k)
        if k == "Full sample":
            ax.fill_between(r.index, r.b - 1.645 * r.se, r.b + 1.645 * r.se, color=BLUE, alpha=.13, lw=0)
    ax.axhline(0, color="k", lw=.6); ax.set_title(series[s]); ax.set_xticks(range(0, H + 1, 3)); ax.grid(alpha=.25)
for ax in axes[1]: ax.set_xlabel("Months after the oil move")
for ax in axes[:, 0]: ax.set_ylabel("pp of YoY inflation")
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle(f"Response of YoY inflation to a +{SHOCK}% rise in Brent (local projections; 90% band = full sample)")
savefig(fig, "fig01_passthrough_by_sample.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (hh, cc, t) in zip(axes, [("us_head", "us_core", "United States"), ("ea_head", "ea_core", "Euro area")]):
    for s, col, lab in [(hh, RED, "Headline"), (cc, BLUE, "Core")]:
        r = LP[(s, "Full sample")]
        ax.plot(r.index, r.b, color=col, lw=2, label=lab)
        ax.fill_between(r.index, r.b - 1.645 * r.se, r.b + 1.645 * r.se, color=col, alpha=.13, lw=0)
    ax.axhline(0, color="k", lw=.6); ax.set_title(t); ax.set_xlabel("Months after the oil move")
    ax.set_xticks(range(0, H + 1, 3)); ax.grid(alpha=.25)
axes[0].set_ylabel("pp of YoY inflation"); axes[0].legend(frameon=False)
fig.suptitle(f"Headline vs core: response to a +{SHOCK}% rise in Brent (1990-2026, 90% bands)")
savefig(fig, "fig02_headline_vs_core.png")

# =============================================================== B. MARKET REACTION (weekly)
W = D.ffill(limit=3).resample("W-FRI").last()
W["oil"] = 100 * np.log(W.brent).diff()
W["mkt"] = 100 * np.log(W.spx).diff()
mkt = {"us5": "US 5y breakeven", "us5y5y": "US 5y5y swap", "ea5": "EA 5y swap", "ea5y5y": "EA 5y5y swap"}
for k in mkt: W["d_" + k] = 100 * W[k].diff()          # basis points

periods = {"2004-2026": ("2004", "2026"), "2004-2008": ("2004", "2008"), "2009-2019": ("2009", "2019"),
           "2020": ("2020", "2020"), "2021-2023": ("2021", "2023"), "2024-2026": ("2024", "2026"),
           "Iran shock (Mar-Nov 2026)": ("2026-02-27", "2026-11-30")}

def beta(k, s, e, control=True):
    z = W.loc[s:e]
    f = ols_hac(z["d_" + k], z[["oil", "mkt"]] if control else z[["oil"]])
    return SHOCK * f.params["oil"], SHOCK * f.bse["oil"], f.rsquared, int(f.nobs)

rows = []
for p_, (s, e) in periods.items():
    for k, lab in mkt.items():
        b, se, r2, n = beta(k, s, e); b0, _, _, _ = beta(k, s, e, control=False)
        rows.append({"period": p_, "series": lab, "bp per +10% Brent": b, "s.e.": se, "t": b / se,
                     "bp per +10% (no S&P control)": b0, "R2": r2, "weeks": n})
TB = pd.DataFrame(rows); TABLES["B_market_betas"] = TB

def rolling_beta(k, win=104):
    z = W[["d_" + k, "oil", "mkt"]].dropna(); out = {}
    for i in range(win, len(z) + 1):
        zz = z.iloc[i - win:i]
        cf = np.linalg.lstsq(np.c_[np.ones(win), zz.oil, zz.mkt], zz["d_" + k], rcond=None)[0]
        out[zz.index[-1]] = SHOCK * cf[1]
    return pd.Series(out)

RB = {k: rolling_beta(k) for k in mkt}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (a, b, t) in zip(axes, [("us5", "us5y5y", "United States"), ("ea5", "ea5y5y", "Euro area")]):
    ax.plot(RB[a], color=RED, lw=1.6, label="5-year"); ax.plot(RB[b], color=BLUE, lw=1.6, label="5y5y forward")
    ax.axhline(0, color="k", lw=.6); ax.set_title(t); ax.grid(alpha=.25)
axes[0].set_ylabel(f"bp per +{SHOCK}% Brent (2-year rolling)"); axes[0].legend(frameon=False)
fig.suptitle("How much do inflation expectations move with oil? Weekly changes, controlling for the S&P 500")
savefig(fig, "fig03_rolling_market_betas.png")

per = [p_ for p_ in periods if p_ != "2004-2026"]
fig, ax = plt.subplots(figsize=(12, 4.2)); x = np.arange(len(per)); wd = .2
for i, (k, lab) in enumerate(mkt.items()):
    v = TB[TB.series == lab].set_index("period").loc[per]
    ax.bar(x + (i - 1.5) * wd, v["bp per +10% Brent"], wd, yerr=1.645 * v["s.e."], capsize=2, label=lab,
           color=[RED, LRED, BLUE, LBLUE][i])
ax.set_xticks(x); ax.set_xticklabels([p_.replace(" (", "\n(") for p_ in per]); ax.axhline(0, color="k", lw=.6)
ax.grid(axis="y", alpha=.25); ax.set_ylabel(f"bp per +{SHOCK}% Brent"); ax.legend(frameon=False, ncol=4, fontsize=9)
ax.set_title("Sensitivity of market inflation expectations to oil, by period (weekly, S&P 500 control, 90% CI)")
savefig(fig, "fig04_market_betas_by_period.png")

# =============================================================== C. CONSISTENCY CHECK
LVL = {r: one(lp(M, f"{r}_lvl", {"oil": M.oil}, level=True)) for r in ["us", "ea"]}
rows = []
for reg, r, k in [("US", "us", "us5"), ("Euro area", "ea", "ea5")]:
    L = LVL[r]; b = TB[(TB.period == "2004-2026") & (TB.series == mkt[k])]["bp per +10% Brent"].iloc[0]
    rows.append({"region": reg, "price-level effect after 12m (%)": L.b[12], "after 24m (%)": L.b[24],
                 "s.e. 24m": L.se[24], "implied rise in 5y average inflation (bp)": 100 * L.b[24] / 5,
                 "observed 5y market move (bp)": b, "market / implied": b / (100 * L.b[24] / 5)})
TABLES["C_consistency"] = pd.DataFrame(rows)
NOTES.append("Consistency check: implied 5y effect = price-level effect after 24 months / 5 (assumes no further effect "
             "in years 3-5); indicative only.")

# =============================================================== D. 2022 vs 2026
Dd = D.ffill(limit=3).dropna(subset=["brent"])
events = {"2022 (Ukraine)": "2022-02-23", "2026 (Iran / Hormuz)": "2026-02-27"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9)); rows = []
for lab, t0 in events.items():
    seg = Dd.loc[t0:].iloc[:121]; xx = np.arange(len(seg)); ls = "-" if "2026" in lab else "--"
    axes[0].plot(xx, 100 * (seg.brent / seg.brent.iloc[0] - 1), ls, color="k", label=lab)
    for ax, (a, b, t) in zip(axes[1:], [("us5", "us5y5y", "US"), ("ea5", "ea5y5y", "Euro area")]):
        ax.plot(xx, 100 * (seg[a] - seg[a].iloc[0]), ls, color=RED, label=f"5y, {lab}")
        ax.plot(xx, 100 * (seg[b] - seg[b].iloc[0]), ls, color=BLUE, label=f"5y5y, {lab}")
        ax.set_title(f"{t}: change since shock (bp)")
    pk = seg.brent.idxmax()
    rows.append({"episode": lab, "start": t0, "Brent peak move (%)": 100 * (seg.brent.max() / seg.brent.iloc[0] - 1),
                 "peak date": pk.date(), **{f"{mkt[k]} at Brent peak (bp)": 100 * (seg.loc[pk, k] - seg[k].iloc[0]) for k in mkt},
                 **{f"{mkt[k]} max move in 120 days (bp)": 100 * (seg[k] - seg[k].iloc[0]).max() for k in mkt}})
axes[0].set_title("Brent: change since shock (%)")
for ax in axes: ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25); ax.set_xlabel("Trading days after shock")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color="k", ls="--"), Line2D([], [], color="k"), Line2D([], [], color=RED), Line2D([], [], color=BLUE)],
           ["2022 (Ukraine)", "2026 (Iran / Hormuz)", "5-year", "5y5y forward"], loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Two oil supply shocks, two different market reactions (dashed = 2022, solid = 2026)")
savefig(fig, "fig05_2022_vs_2026.png")
TABLES["D_2022_vs_2026"] = pd.DataFrame(rows)

# =============================================================== E. OUT-OF-SAMPLE 2026
# Distributed-lag (ARDL) model of the monthly change in YoY inflation:
#   dpi_t = c + sum_{j=0..13} b_j oil_{t-j} + sum_{j=1..13} g_j dpi_{t-j} + e_t
# (13 lags of oil capture the base effect after 12 months). Estimated up to Feb 2026, then simulated
# forward with the ACTUAL Brent path from Mar 2026, and with Brent frozen at its Feb 2026 level (counterfactual).
LO, LPI = 13, 13
T0 = pd.Period("2026-02", "M")

def ardl_fit(y, end, start=None, oil="oil"):
    d = pd.DataFrame({"dpi": M[y].diff()})
    for j in range(LO + 1): d[f"o{j}"] = M[oil].shift(j)
    for j in range(1, LPI + 1): d[f"p{j}"] = d.dpi.shift(j)
    d = d.loc[start:end].dropna()
    return sm.OLS(d.dpi, sm.add_constant(d.drop(columns="dpi"))).fit()

def ardl_sim(f, y, t1, zero_oil=False, shocks=None, oil="oil"):
    o = M[oil].copy(); dpi = M[y].diff().copy(); idx = pd.period_range(T0 + 1, t1, freq="M")
    if zero_oil: o.loc[idx] = 0.0
    p_ = f.params; lvl = M.loc[T0, y]; out = {}
    for i, t in enumerate(idx):
        x = p_["const"] + sum(p_[f"o{j}"] * o.loc[t - j] for j in range(LO + 1)) \
            + sum(p_[f"p{j}"] * dpi.loc[t - j] for j in range(1, LPI + 1))
        if shocks is not None: x += shocks[i]
        dpi.loc[t] = x; lvl += x; out[t] = lvl
    return pd.Series(out)

def ardl_band(f, y, t1, n=1000, seed=1):
    rng = np.random.default_rng(seed); res = f.resid.values; k = len(pd.period_range(T0 + 1, t1, freq="M"))
    paths = np.array([ardl_sim(f, y, t1, shocks=rng.choice(res, k)).values for _ in range(n)])
    return np.percentile(paths, 5, 0), np.percentile(paths, 95, 0)

last = {s: M[s].last_valid_index() for s in series}
rows = []; fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, s in zip(axes.flat, series):
    t1 = last[s]
    f_old = ardl_fit(s, "2019-12"); f_all = ardl_fit(s, "2026-02")
    fc_old, fc_all = ardl_sim(f_old, s, t1), ardl_sim(f_all, s, t1)
    cf = ardl_sim(f_old, s, t1, zero_oil=True); cf_all = ardl_sim(f_all, s, t1, zero_oil=True)
    lo_, hi_ = ardl_band(f_old, s, t1)
    act = M.loc[T0 + 1:t1, s]
    hist = M.loc["2025-06":t1, s]
    ax.plot(hist.index.to_timestamp(), hist.values, color="k", lw=2.2, label="Actual")
    tt = fc_old.index.to_timestamp()
    ax.fill_between(tt, lo_, hi_, color=BLUE, alpha=.13, lw=0)
    ax.plot(tt, fc_old.values, color=BLUE, lw=1.8, label="Forecast: model estimated 1990-2019")
    ax.plot(tt, fc_all.values, "--", color=RED, lw=1.6, label="Forecast: model estimated to Feb 2026")
    ax.plot(tt, cf.values, ":", color=GREY, lw=2, label="Counterfactual: Brent frozen at Feb 2026")
    ax.axvline(pd.Timestamp("2026-03-01"), color="k", lw=.6, ls=":")
    ax.set_title(series[s]); ax.grid(alpha=.25); ax.set_ylabel("YoY inflation (%)")
    ax.tick_params(axis="x", labelrotation=30)
    for lab, fc, ff in [("1990-2019 model", fc_old, f_old), ("model to Feb 2026", fc_all, f_all)]:
        e = act - fc.loc[act.index]
        rows.append({"series": series[s], "model": lab, "R2 in-sample": ff.rsquared,
                     "forecast end": str(t1), "actual at end": act.iloc[-1], "forecast at end": fc.iloc[-1],
                     "no-shock counterfactual at end": (cf if ff is f_old else cf_all).iloc[-1],
                     "oil contribution at end (pp)": fc.iloc[-1] - (cf if ff is f_old else cf_all).iloc[-1],
                     "peak oil contribution (pp)": (fc - (cf if ff is f_old else cf_all)).max(),
                     "mean error (actual-fcst)": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "RMSE no-shock counterfactual": np.sqrt(((act - (cf if ff is f_old else cf_all).loc[act.index]) ** 2).mean()),
                     "share of months inside 90% band": float(((act.values >= lo_[:len(act)]) & (act.values <= hi_[:len(act)])).mean()) if ff is f_old else np.nan})
h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("Out-of-sample test: inflation after the March 2026 oil shock, predicted with pre-shock models\n"
             "(simulated with the actual Brent path; 90% band for the 1990-2019 model)")
savefig(fig, "fig06_out_of_sample_2026.png")
TABLES["E_out_of_sample_2026"] = pd.DataFrame(rows)

# =============================================================== F. DEMAND vs OIL-SPECIFIC SHOCKS
# Weekly reduced-form (Brent, S&P 500) returns u = A e. Sign restrictions:
#   global demand shock: oil up, stocks up     oil-specific (supply/precautionary) shock: oil up, stocks down
# plus narrative restrictions (Antolin-Diaz & Rubio-Ramirez 2018) on 4 well-known weeks. The admissible
# rotations form an interval; baseline = its midpoint, robustness = its two ends.
U = W[["oil", "mkt"]].dropna(); U = U - U.mean()
Pc = np.linalg.cholesky(np.cov(U.T))

def impact(th):
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); A = Pc @ Q
    if np.any(A[0] == 0): return None
    A = A * np.sign(A[0])                                  # both shocks raise oil on impact
    return A if (A[1, 0] > 0 and A[1, 1] < 0) else None    # col 0 demand (stocks up), col 1 oil-specific (stocks down)

def components(th):
    A = impact(th); e = np.linalg.solve(A, U.T.values)
    return pd.DataFrame({"oil_demand": A[0, 0] * e[0], "oil_specific": A[0, 1] * e[1]}, index=U.index), A

NARR = {"2008-10-10": ("demand", -1),     # post-Lehman collapse
        "2020-03-13": ("specific", -1),   # Saudi-Russia price war, 9 Mar 2020
        "2022-03-04": ("specific", +1),   # Russia invades Ukraine
        "2026-03-06": ("specific", +1)}   # Iran war / Hormuz disruption

def narrative_ok(th):
    c, _ = components(th)
    for dte, (k, sg) in NARR.items():
        main = c.loc[dte, "oil_" + k]; other = c.loc[dte, "oil_" + ("specific" if k == "demand" else "demand")]
        if not (np.sign(main) == sg and abs(main) > abs(other)): return False
    return True

grid = np.linspace(0, np.pi, 3000, endpoint=False)
sign_ok = np.array([th for th in grid if impact(th) is not None])
narr_ok = np.array([th for th in sign_ok if narrative_ok(th)])
TH = {"baseline (midpoint)": narr_ok.mean(), "bound: least oil-specific": narr_ok.min(),
      "bound: most oil-specific": narr_ok.max()}
C, A = components(TH["baseline (midpoint)"])
Wf = W.join(C)
share = C.oil_specific.var() / (C.oil_specific.var() + C.oil_demand.var())
NOTES.append(f"Shock identification: sign restrictions alone admit oil-specific variance shares from 0% to 100%; "
             f"with the 4 narrative restrictions the range narrows to {100*components(narr_ok.min())[0].oil_specific.var()/(components(narr_ok.min())[0].var().sum()):.0f}-"
             f"{100*components(narr_ok.max())[0].oil_specific.var()/(components(narr_ok.max())[0].var().sum()):.0f}%; baseline {100*share:.0f}%.")

# F1: market reaction by shock type
rows = []
for p_, (s, e) in {"2004-2026": ("2004", "2026"), "2004-2019": ("2004", "2019"), "2020-2026": ("2020", "2026")}.items():
    for k, lab in mkt.items():
        f = ols_hac(Wf.loc[s:e, "d_" + k], Wf.loc[s:e, ["oil_demand", "oil_specific"]])
        rows.append({"period": p_, "series": lab,
                     "demand-driven: bp per +10%": SHOCK * f.params.oil_demand, "s.e. demand": SHOCK * f.bse.oil_demand,
                     "oil-specific: bp per +10%": SHOCK * f.params.oil_specific, "s.e. oil-specific": SHOCK * f.bse.oil_specific,
                     "R2": f.rsquared, "weeks": int(f.nobs)})
TF1 = pd.DataFrame(rows); TABLES["F1_market_by_shock_type"] = TF1

# F2: inflation pass-through by shock type (monthly sums of weekly components, 2005-2026)
Cm = C.groupby(C.index.to_period("M")).sum()
Mf = M.join(Cm); Mf["oil_wk"] = Mf.oil_demand + Mf.oil_specific
LPF = {s: lp(Mf, s, {"oil_demand": Mf.oil_demand, "oil_specific": Mf.oil_specific}, start="2005-01") for s in series}
LPF_tot = {s: one(lp(Mf, s, {"oil_wk": Mf.oil_wk}, start="2005-01")) for s in series}
rows = []
for s in series:
    r = LPF[s]
    for k, lab in [("oil_demand", "demand-driven"), ("oil_specific", "oil-specific")]:
        rr = r[r.shock == k].set_index("h")
        rows.append({"series": series[s], "shock": lab, **{f"h={h}": rr.b[h] for h in [0, 3, 6, 12, 18, 24]},
                     "s.e. h=6": rr.se[6], "s.e. h=12": rr.se[12]})
    rows.append({"series": series[s], "shock": "all oil moves (same sample)", **{f"h={h}": LPF_tot[s].b[h] for h in [0, 3, 6, 12, 18, 24]},
                 "s.e. h=6": LPF_tot[s].se[6], "s.e. h=12": LPF_tot[s].se[12]})
TABLES["F2_inflation_by_shock_type"] = pd.DataFrame(rows)

# F3: robustness of the market result across admissible rotations
rows = []
for nm, th in TH.items():
    c, _ = components(th); ww = W.join(c)
    for k, lab in mkt.items():
        f = ols_hac(ww["d_" + k], ww[["oil_demand", "oil_specific"]])
        rows.append({"identification": nm, "oil-specific variance share": c.oil_specific.var() / c.var().sum(),
                     "series": lab, "demand-driven: bp per +10%": SHOCK * f.params.oil_demand,
                     "oil-specific: bp per +10%": SHOCK * f.params.oil_specific, "s.e. oil-specific": SHOCK * f.bse.oil_specific})
TABLES["F3_identification_robustness"] = pd.DataFrame(rows)

# charts F
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(U.oil.cumsum(), color="k", lw=1.3, label="Brent (cumulative log change excl. average trend, %)")
ax.plot(C.oil_demand.cumsum(), color=BLUE, lw=1.4, label="Global-demand component")
ax.plot(C.oil_specific.cumsum(), color=RED, lw=1.4, label="Oil-specific (supply) component")
ax.set_xlim(C.index[0], C.index[-1] + pd.Timedelta(days=60))
for dte, txt, ha in [("2008-07-04", "2008 peak", "right"), ("2014-11-28", "OPEC Nov 2014", "right"), ("2020-03-13", "Covid", "right"),
                     ("2022-03-04", "Ukraine", "left"), ("2026-03-06", "Iran", "right")]:
    ax.axvline(pd.Timestamp(dte), color=GREY, lw=.6, ls=":")
    ax.text(pd.Timestamp(dte), ax.get_ylim()[1] * .95, txt + " ", fontsize=8, color=GREY, ha=ha, va="top")
ax.axhline(0, color="k", lw=.5); ax.grid(alpha=.25); h_, l_ = ax.get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=9)
ax.set_title("What drove oil? Historical decomposition of weekly Brent returns (sign + narrative restrictions)")
savefig(fig, "fig07_oil_shock_decomposition.png")

fig, ax = plt.subplots(figsize=(10, 4)); v = TF1[TF1.period == "2004-2026"].set_index("series").loc[list(mkt.values())]
x = np.arange(4); wd = .36
ax.bar(x - wd / 2, v["demand-driven: bp per +10%"], wd, yerr=1.645 * v["s.e. demand"], capsize=3, color=BLUE, label="Oil rise driven by global demand")
ax.bar(x + wd / 2, v["oil-specific: bp per +10%"], wd, yerr=1.645 * v["s.e. oil-specific"], capsize=3, color=RED, label="Oil-specific (supply) rise")
ax.set_xticks(x); ax.set_xticklabels(v.index); ax.axhline(0, color="k", lw=.6); ax.grid(axis="y", alpha=.25)
ax.set_ylabel(f"bp per +{SHOCK}% Brent"); ax.legend(frameon=False)
ax.set_title("Why the oil move matters: response of inflation expectations by type of shock (weekly 2004-2026, 90% CI)")
savefig(fig, "fig08_market_by_shock_type.png")

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, s in zip(axes.flat, series):
    r = LPF[s]
    for k, col, lab in [("oil_demand", BLUE, "Global-demand-driven"), ("oil_specific", RED, "Oil-specific (supply)")]:
        rr = r[r.shock == k].set_index("h")
        ax.plot(rr.index, rr.b, color=col, lw=2, label=lab)
        ax.fill_between(rr.index, rr.b - 1.645 * rr.se, rr.b + 1.645 * rr.se, color=col, alpha=.12, lw=0)
    ax.axhline(0, color="k", lw=.6); ax.set_title(series[s]); ax.set_xticks(range(0, H + 1, 3)); ax.grid(alpha=.25)
for ax in axes[1]: ax.set_xlabel("Months after the oil move")
for ax in axes[:, 0]: ax.set_ylabel("pp of YoY inflation")
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle(f"Inflation response to a +{SHOCK}% Brent move, by type of shock (2005-2026, 90% bands)")
savefig(fig, "fig09_inflation_by_shock_type.png")

# =============================================================== G. STATE DEPENDENCE & ASYMMETRY
THR = 3.0
rows = []; LPG = {}
for s in series:
    S = (M[regime_of[s]].shift(1) > THR).astype(float).where(M[regime_of[s]].shift(1).notna())
    r = lp(M, s, {"low": M.oil * (1 - S), "high": M.oil * S}, extra={"S": S})
    a = lp(M, s, {"rise": M.oil.clip(lower=0), "fall": M.oil.clip(upper=0)})
    LPG[s] = (r, a)
    for d_, k, lab in [(r, "low", f"inflation <= {THR:g}% (last month)"), (r, "high", f"inflation > {THR:g}% (last month)"),
                       (a, "rise", "oil rise (+10%)"), (a, "fall", "oil fall (effect of -10%, sign flipped)")]:
        rr = d_[d_.shock == k].set_index("h")
        rows.append({"series": series[s], "case": lab, **{f"h={h}": rr.b[h] for h in [0, 3, 6, 12, 18, 24]},
                     "s.e. h=6": rr.se[6], "s.e. h=12": rr.se[12],
                     "months in regime": int(S.sum()) if k == "high" else (int((1 - S).sum()) if k == "low" else np.nan)})
TABLES["G1_inflation_regime_asymmetry"] = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for j, s in enumerate(series):
    r, a = LPG[s]
    for ax, d_, ks in [(axes[0, j], r, [("low", GREY, f"Inflation ≤ {THR:g}%"), ("high", RED, f"Inflation > {THR:g}%")]),
                       (axes[1, j], a, [("rise", RED, "Oil rises"), ("fall", BLUE, "Oil falls (sign flipped)")])]:
        for k, col, lab in ks:
            rr = d_[d_.shock == k].set_index("h")
            ax.plot(rr.index, rr.b, color=col, lw=2, label=lab)
            ax.fill_between(rr.index, rr.b - 1.645 * rr.se, rr.b + 1.645 * rr.se, color=col, alpha=.12, lw=0)
        ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25); ax.set_xticks(range(0, H + 1, 6))
    axes[0, j].set_title(series[s]); axes[1, j].set_xlabel("Months after the oil move")
axes[0, 0].set_ylabel("State dependence\npp of YoY inflation"); axes[1, 0].set_ylabel("Asymmetry\npp of YoY inflation")
axes[0, 0].legend(frameon=False, fontsize=8); axes[1, 0].legend(frameon=False, fontsize=8)
fig.suptitle(f"Does pass-through depend on the inflation regime (top) or on the direction of the oil move (bottom)? (+{SHOCK}% Brent, 90% bands)")
savefig(fig, "fig10_regime_and_asymmetry.png")

# G2: market counterpart (weekly). Regime = headline YoY of two months earlier (what investors knew).
def wk_regime(col):
    s = (M[col] > THR).astype(float).shift(2); s.index = s.index.to_timestamp(how="end").normalize()
    return s.reindex(W.index, method="ffill")
W["S_us"], W["S_ea"] = wk_regime("us_head"), wk_regime("ea_head")
rows = []
for k, lab in mkt.items():
    S = W["S_us" if k.startswith("us") else "S_ea"]
    X = pd.DataFrame({"oil_low": W.oil * (1 - S), "oil_high": W.oil * S, "mkt": W.mkt, "S": S})
    f = ols_hac(W["d_" + k], X)
    g = ols_hac(W["d_" + k], pd.DataFrame({"up": W.oil.clip(lower=0), "down": W.oil.clip(upper=0), "mkt": W.mkt}))
    rows.append({"series": lab, "low-inflation regime": SHOCK * f.params.oil_low, "s.e. low": SHOCK * f.bse.oil_low,
                 "high-inflation regime": SHOCK * f.params.oil_high, "s.e. high": SHOCK * f.bse.oil_high,
                 "weeks high": int(S.sum()), "oil up": SHOCK * g.params.up, "s.e. up": SHOCK * g.bse.up,
                 "oil down": SHOCK * g.params.down, "s.e. down": SHOCK * g.bse.down})
TABLES["G2_market_regime_asymmetry"] = pd.DataFrame(rows)

# =============================================================== H. ROBUSTNESS
excl = pd.period_range("2008-07", "2009-06", freq="M").union(pd.period_range("2020-02", "2020-12", freq="M"))
specs = {"Baseline (avg Brent, 6 lags)": dict(shocks={"oil": M.oil}),
         "Month-end Brent": dict(shocks={"oil": M.oil_eom}),
         "Month-end WTI": dict(shocks={"oil": M.wti_eom}),
         "Broad commodities (BCOM)": dict(shocks={"oil": M.bcom_eom}),
         "3 lags of controls": dict(shocks={"oil": M.oil}, p=3),
         "12 lags of controls": dict(shocks={"oil": M.oil}, p=12),
         "Excl. 2008-09 crisis and 2020": dict(shocks={"oil": M.oil}, exclude=excl),
         "Sample 2004-2026": dict(shocks={"oil": M.oil}, start="2004-01")}
rows = []
for nm, kw in specs.items():
    for s in series:
        r = one(lp(M, s, **kw))
        rows.append({"specification": nm, "series": series[s], "h=6": r.b[6], "s.e. h=6": r.se[6], "h=12": r.b[12], "s.e. h=12": r.se[12]})
# distributed-lag (ARDL) dynamic multiplier = the 'sum of lag coefficients' method of the original plan
for s in series:
    f = ardl_fit(s, "2026-02"); p_ = f.params; dpi = np.zeros(H + 1 + LPI)
    for h in range(H + 1):
        dpi[LPI + h] = (p_[f"o{h}"] * SHOCK if h <= LO else 0) + sum(p_[f"p{j}"] * dpi[LPI + h - j] for j in range(1, LPI + 1))
    cum = np.cumsum(dpi[LPI:])
    rows.append({"specification": "Distributed-lag model (ARDL), 1990-Feb 2026", "series": series[s], "h=6": cum[6], "s.e. h=6": np.nan,
                 "h=12": cum[12], "s.e. h=12": np.nan})
TH_ = pd.DataFrame(rows)
TABLES["H1_robustness_passthrough"] = TH_.pivot_table(index="specification", columns="series", values=["h=6", "h=12"], sort=False)
TABLES["H1_robustness_long"] = TH_

# market: daily (with 1-day lead/lag to handle different closing times) and monthly vs weekly
Dl = D.ffill(limit=3).dropna(subset=["brent"])
Dl["oil"] = 100 * np.log(Dl.brent).diff(); Dl["mkt"] = 100 * np.log(Dl.spx).diff()
Mo = D.ffill(limit=3).resample("ME").last(); Mo["oil"] = 100 * np.log(Mo.brent).diff(); Mo["mkt"] = 100 * np.log(Mo.spx).diff()
rows = []
for k, lab in mkt.items():
    dk = 100 * Dl[k].diff()
    fd = ols_hac(dk, pd.DataFrame({"o_m1": Dl.oil.shift(1), "o0": Dl.oil, "o_p1": Dl.oil.shift(-1), "mkt": Dl.mkt}), lags=5)
    sum_b = SHOCK * (fd.params.o_m1 + fd.params.o0 + fd.params.o_p1)
    sum_se = SHOCK * np.sqrt(np.ones(3) @ fd.cov_params().loc[["o_m1", "o0", "o_p1"], ["o_m1", "o0", "o_p1"]].values @ np.ones(3))
    fd0 = ols_hac(dk, Dl[["oil", "mkt"]], lags=5)
    fm = ols_hac(100 * Mo[k].diff(), Mo[["oil", "mkt"]], lags=3)
    b_w = TB[(TB.period == "2004-2026") & (TB.series == lab)].iloc[0]
    rows.append({"series": lab, "daily (same day)": SHOCK * fd0.params.oil, "daily (lead+same+lag)": sum_b, "s.e. daily sum": sum_se,
                 "weekly (baseline)": b_w["bp per +10% Brent"], "s.e. weekly": b_w["s.e."],
                 "monthly": SHOCK * fm.params.oil, "s.e. monthly": SHOCK * fm.bse.oil})
TABLES["H2_robustness_market_frequency"] = pd.DataFrame(rows)

# =============================================================== OUTPUT
with pd.ExcelWriter(f"{OUT}/results_tables.xlsx") as xw:
    pd.DataFrame({"README": [
        "All effects are for a +10% move in Brent.",
        "Inflation responses: percentage points of YoY inflation, h months after the oil move (local projections, Newey-West s.e.).",
        "Market responses: basis points, weekly changes, controlling for the S&P 500 unless stated.",
        "90% confidence interval = estimate +/- 1.645 x s.e.", ""] + NOTES}).to_excel(xw, sheet_name="README", index=False)
    for nm, t in TABLES.items():
        t.round(3).to_excel(xw, sheet_name=nm[:31], index=not isinstance(t.index, pd.RangeIndex))

pd.set_option("display.width", 250, "display.max_columns", 30)
for nm, t in TABLES.items():
    if nm != "H1_robustness_long": print(f"\n=== {nm}\n", t.round(2).to_string())
print("\nNOTES:", *NOTES, sep="\n- ")
