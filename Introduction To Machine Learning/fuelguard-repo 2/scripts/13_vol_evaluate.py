"""FuelGuard 2.0 - step 3. Forecast evaluation, diesel hedging backtest, VaR coverage, figures, tables."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.evaluate import newey_west_var  # noqa: E402
from fuelguard.panel import MARKETS, NAMES  # noqa: E402
from fuelguard.report import (AQUA, BLUE, GREEN, GRID, INK, INK2, ORANGE, RED, VIOLET, YELLOW,  # noqa: E402
                              style, to_latex)

style()
F, T, R = C.FIGURES, C.TABLES, C.RESULTS
P = pd.read_csv(R / "vol_oos_predictions.csv.gz", parse_dates=["date"])
D = pd.read_csv(C.DATA_PROCESSED / "vol_dataset.csv.gz", parse_dates=["date", "label_end_date"])
P = P.merge(D[["date", "market", "liv"]], on=["date", "market"], how="left")
BENCH = ["Random walk", "HAR", "HAR-IV"]
ML = ["Lasso", "Random forest", "Gradient boosting", "Neural network", "ML ensemble"]
FG = "FuelGuard (HAR-IV + ML)"
MODELS = BENCH + ML + [FG]
COL = {"Random walk": INK2, "HAR": INK2, "HAR-IV": INK, "Lasso": BLUE, "Random forest": ORANGE,
       "Gradient boosting": AQUA, "Neural network": VIOLET, "ML ensemble": RED, "Implied vol": YELLOW,
       "FuelGuard (HAR-IV + ML)": GREEN}
key = {}


def qlike(y, f):
    """QLIKE on variances (Patton 2011), robust to noise in the realised-variance proxy."""
    r = np.exp(2 * (y - f))
    return r - np.log(r) - 1


def dm_pooled(df, a, b, loss="mse"):
    """DM test of model a vs b on the cross-sectional average loss per date (Newey-West, 21 lags)."""
    la = (df["y"] - df[a]) ** 2 if loss == "mse" else qlike(df["y"], df[a])
    lb = (df["y"] - df[b]) ** 2 if loss == "mse" else qlike(df["y"], df[b])
    d = (la - lb).groupby(df["date"]).mean().to_numpy()
    t = d.mean() / np.sqrt(newey_west_var(d, 21) / len(d))
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def table(df, ref="HAR-IV"):
    rows = {}
    for m in MODELS:
        e = df["y"] - df[m]
        row = {"MSE": np.mean(e ** 2), "QLIKE": qlike(df["y"], df[m]).mean(),
               "R2 vs RW (%)": (1 - np.sum(e ** 2) / np.sum((df["y"] - df["Random walk"]) ** 2)) * 100,
               "R2 vs HAR-IV (%)": (1 - np.sum(e ** 2) / np.sum((df["y"] - df[ref]) ** 2)) * 100,
               "R2 (%)": (1 - np.sum(e ** 2) / np.sum((df["y"] - df["y"].mean()) ** 2)) * 100}
        if m != ref:
            row["DM vs HAR-IV"], row["p-value"] = dm_pooled(df, m, ref)
        rows[m] = row
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ 1. forecast accuracy
P = P.dropna(subset=MODELS)
pooled = table(P)
ho = P[P["market"] == "HO"]
diesel = table(ho)
energy = table(P[P["market"].isin(["HO", "CO", "CL", "NG"])])
for name, t in [("vol_pooled", pooled), ("vol_diesel", diesel), ("vol_energy", energy)]:
    t.to_csv(T / f"{name}.csv")
    # a pooled R2 mostly measures the cross-section of volatility levels (bonds 1%, gas 44%): not shown
    cols_tex = (["R2 (%)"] if name != "vol_pooled" else []) + ["R2 vs RW (%)", "R2 vs HAR-IV (%)", "MSE", "QLIKE", "DM vs HAR-IV", "p-value"]
    to_latex(t[cols_tex],
             T / f"{name}.tex", index_name="Model",
             fmt={"R2 (%)": "{:.1f}", "R2 vs RW (%)": "{:.1f}", "R2 vs HAR-IV (%)": "{:.1f}", "MSE": "{:.4f}",
                  "QLIKE": "{:.4f}", "DM vs HAR-IV": "{:.2f}", "p-value": "{:.3f}"}, bold_rows=(FG,))
key["pooled"] = pooled.round(4).to_dict(orient="index")
key["diesel"] = diesel.round(4).to_dict(orient="index")
key["energy"] = energy.round(4).to_dict(orient="index")
key["n_oos"] = int(len(P)); key["n_oos_diesel"] = int(len(ho))

# per market: ML ensemble vs HAR-IV
pm = []
for mk, g in P.groupby("market"):
    e_ml, e_b = (g["y"] - g[FG]) ** 2, (g["y"] - g["HAR-IV"]) ** 2
    d = (e_ml - e_b).to_numpy()
    t = d.mean() / np.sqrt(newey_west_var(d, 21) / len(d))
    best = min(ML, key=lambda m: np.mean((g["y"] - g[m]) ** 2))
    pm.append({"market": mk, "name": NAMES[mk], "class": MARKETS[mk][3],
               "R2 HAR-IV (%)": (1 - e_b.sum() / ((g["y"] - g["y"].mean()) ** 2).sum()) * 100,
               "R2 FuelGuard (%)": (1 - e_ml.sum() / ((g["y"] - g["y"].mean()) ** 2).sum()) * 100,
               "Gain vs HAR-IV (%)": (1 - e_ml.sum() / e_b.sum()) * 100, "DM t": t,
               "Best ML": best})
pm = pd.DataFrame(pm).set_index("market").sort_values(["class", "Gain vs HAR-IV (%)"])
pm.to_csv(T / "vol_by_market.csv")
to_latex(pm.drop(columns=["class"]).set_index("name"), T / "vol_by_market.tex", index_name="Market",
         fmt={"R2 HAR-IV (%)": "{:.1f}", "R2 FuelGuard (%)": "{:.1f}", "Gain vs HAR-IV (%)": "{:.1f}", "DM t": "{:.2f}"})
key["markets_better"] = int((pm["Gain vs HAR-IV (%)"] > 0).sum())
key["markets_sig_better"] = int((pm["DM t"] < -1.96).sum())
key["markets_sig_worse"] = int((pm["DM t"] > 1.96).sum())
key["n_markets"] = int(len(pm))

# by year (pooled and diesel)
def gain_by_year(df, m):
    return df.groupby("fold_year").apply(lambda g: (1 - ((g["y"] - g[m]) ** 2).sum() /
                                                    ((g["y"] - g["HAR-IV"]) ** 2).sum()) * 100)
by_year = pd.DataFrame({f"{m} (all)": gain_by_year(P, m) for m in [FG, "ML ensemble"]})
by_year["FuelGuard (diesel)"] = gain_by_year(ho, FG)
by_year.index = by_year.index.astype(int)
by_year.to_csv(T / "vol_gain_by_year.csv")
key["years_fg_better_all"] = int((by_year[f"{FG} (all)"] > 0).sum())
key["years_ens_better_all"] = int((by_year["ML ensemble (all)"] > 0).sum())
key["years_fg_better_diesel"] = int((by_year["FuelGuard (diesel)"] > 0).sum())

# hyper-parameters and importance
ch = pd.read_csv(T / "vol_chosen_hyperparameters.csv")
hp = ch.groupby("model")["params"].agg(lambda s: s.value_counts().index[0] + f" ({s.value_counts().iloc[0]}/{len(s)})")
hp = hp.str.replace(r"[{}']", "", regex=True).str.replace("lasso__", "").str.replace("mlpregressor__", "")
hp.to_frame("Most frequent choice (years)").to_csv(T / "vol_hyperparameters.csv")
to_latex(hp.to_frame("Most frequent choice (years)"), T / "vol_hyperparameters.tex", index_name="Model", col_align="ll")
imp = pd.read_csv(R / "vol_group_importance_by_year.csv", index_col=0)
imp_mean = imp.mean(axis=1).sort_values(ascending=False)
imp_mean.to_csv(T / "vol_group_importance.csv")
key["importance"] = imp_mean.round(2).to_dict()

# ------------------------------------------------------------------ 2. diesel hedging backtest
panel = pd.read_csv(C.DATA_PROCESSED / "panel_daily.csv", index_col=0, parse_dates=True)
hd = ho.set_index("date")
hd["Implied vol"] = hd["liv"]
dates = pd.DatetimeIndex(hd.index.to_series().groupby(hd.index.to_period("M")).min().values)
dates = dates[dates.isin(panel.index) & panel.reindex(dates)["premium"].notna().to_numpy()]
prem = panel.loc[dates, "premium"]
lock, mkt = panel.loc[dates, "lock_price"], panel.loc[dates, "market_cost"]
FORE = ["Random walk", "HAR", "HAR-IV", "Implied vol"] + ML + [FG]
ho_all = D[D["market"] == "HO"].set_index("date")["y"].dropna()
H = pd.DataFrame(index=dates)
H["Never lock (market)"], H["Always lock"], H["Always 50%"] = 0.0, 1.0, 0.5
for f in FORE:
    s = hd.loc[dates, f]
    med = pd.Series([ho_all[ho_all.index < pd.Timestamp(d.year, 1, 1)].median() for d in dates], index=dates)
    q75 = pd.Series([ho_all[ho_all.index < pd.Timestamp(d.year, 1, 1)].quantile(0.75) for d in dates], index=dates)
    H[f] = np.where(s > q75, 1.0, np.where(s > med, 0.5, 0.0))
cost = H.mul(lock * (1 + C.HEDGE_COST), axis=0) + (1 - H).mul(mkt, axis=0)
over = cost.div(lock, axis=0) - 1
k = int(np.ceil(0.05 * len(dates)))
bt = pd.DataFrame({
    "Mean %": over.mean() * 100, "St.dev %": over.std() * 100,
    "CVaR95 %": over.apply(lambda s: s.sort_values(ascending=False).iloc[:k].mean()) * 100,
    "Worst %": over.max() * 100, "Months >5%": (over > 0.05).sum(),
    "Hedge ratio": H.mean(),
    "CVaR95 EUR/month": (cost.sub(lock, axis=0)).apply(lambda s: s.sort_values(ascending=False).iloc[:k].mean()) * C.MONTHLY_LITRES,
    "Saving EUR/yr": cost.rsub(cost["Never lock (market)"], axis=0).mean() * C.MONTHLY_LITRES * 12})
bt.to_csv(T / "vol_backtest.csv")
to_latex(bt, T / "vol_backtest.tex", index_name="Strategy",
         fmt={"Mean %": "{:.2f}", "St.dev %": "{:.2f}", "CVaR95 %": "{:.1f}", "Worst %": "{:.1f}",
              "Months >5%": "{:.0f}", "Hedge ratio": "{:.2f}", "CVaR95 EUR/month": "{:,.0f}", "Saving EUR/yr": "{:,.0f}"},
         bold_rows=(FG,))
key["backtest"] = bt.round(3).to_dict(orient="index"); key["backtest_months"] = int(len(dates))

# ------------------------------------------------------------------ 3. "how bad can it get": 95% cost quantile
# Premium = average of the next 21 days vs the price at t+1. For a random walk with daily variance s^2,
# Var(average) = s^2 (n+1)(2n+1)/(6n) with n=21, i.e. about 7.5 trading days of variance.
n = C.HORIZON
scale = np.sqrt((n + 1) * (2 * n + 1) / (6 * n) / 252)
z = stats.norm.ppf(0.95)
cov = []
days = hd.index[hd.index.isin(panel.index)]
for f in FORE:
    for label, idx in [("monthly", dates), ("daily", days)]:
        sig = np.exp(hd.loc[idx, f])
        var95 = z * sig * scale
        pr = panel.loc[idx, "premium"]
        ok = pr.notna()
        hit = (pr[ok] > var95[ok]).astype(int)
        x, N = hit.sum(), len(hit)
        ph = x / N
        lr = -2 * ((N - x) * np.log(0.95) + x * np.log(0.05) - (N - x) * np.log(max(1 - ph, 1e-12)) - x * np.log(max(ph, 1e-12)))
        cov.append({"forecast": f, "sample": label, "N": N, "breaches": x, "rate %": ph * 100,
                    "Kupiec p": 1 - stats.chi2.cdf(lr, 1) if label == "monthly" else np.nan,
                    "avg 95% bound %": var95[ok].mean() * 100})
cov = pd.DataFrame(cov)
cov.to_csv(T / "vol_var_coverage.csv", index=False)
cm = cov[cov["sample"] == "monthly"].set_index("forecast")[["N", "breaches", "rate %", "Kupiec p", "avg 95% bound %"]]
cm["daily rate %"] = cov[cov["sample"] == "daily"].set_index("forecast")["rate %"]
to_latex(cm, T / "vol_var_coverage.tex", index_name="Volatility forecast",
         fmt={"N": "{:.0f}", "breaches": "{:.0f}", "rate %": "{:.1f}", "Kupiec p": "{:.2f}",
              "avg 95% bound %": "{:.1f}", "daily rate %": "{:.1f}"})
key["var_coverage"] = cm.round(3).to_dict(orient="index")

# market list for the appendix
mk = pd.read_csv(T / "vol_markets.csv", index_col=0)
mk["ticker"] = [MARKETS[m][2] for m in mk.index]
mk["class"] = [MARKETS[m][3] for m in mk.index]
mk = mk.sort_values(["class", "name"])
mk_t = mk.set_index("name")[["ticker", "class", "days", "vol_median", "vol_p95", "iv_median"]].copy()
for c in ["vol_median", "vol_p95", "iv_median"]:
    mk_t[c] = mk_t[c] * 100
mk_t.columns = ["Bloomberg ticker", "Class", "Days", "Median vol %", "95th pct vol %", "Median IV %"]
to_latex(mk_t, T / "vol_markets.tex", index_name="Market", col_align="lllrrrr",
         fmt={"Days": "{:,.0f}", "Median vol %": "{:.1f}", "95th pct vol %": "{:.1f}", "Median IV %": "{:.1f}"})
dz = pd.read_csv(T / "vol_validation_design.csv")
dz["model"] = dz["model"].fillna("")
to_latex(dz.set_index("design"), T / "vol_validation_design.tex", index_name="Design", col_align="llrr",
         fmt={"MSE": "{:.4f}", "vs HAR-IV (%)": "{:+.1f}"})

# ------------------------------------------------------------------ figures
# A. Diesel: realised vs forecast
fig, ax = plt.subplots(figsize=(7.2, 2.9))
r = np.exp(ho.set_index("date")["y"]) * 100
ax.plot(r.index, r, color=GRID, lw=2.2, label="Realised next 21 days")
ax.plot(r.index, np.exp(ho.set_index("date")["HAR-IV"]) * 100, color=INK2, lw=0.9, label="HAR-IV forecast")
ax.plot(r.index, np.exp(ho.set_index("date")[FG]) * 100, color=GREEN, lw=0.9, label="FuelGuard forecast")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("Diesel: next-month volatility, realised and forecast out of sample")
ax.legend(loc="upper left", ncol=3)
fig.savefig(F / "v2_fig_diesel_forecast.pdf"); plt.close(fig)

# B. Pooled R2 vs HAR-IV by model + per-market gains
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
g = pooled["R2 vs HAR-IV (%)"].drop("HAR-IV")
ax[0].barh(range(len(g)), g.values, color=[COL[m] for m in g.index], height=0.65)
ax[0].set_yticks(range(len(g))); ax[0].set_yticklabels(g.index, fontsize=7.5)
ax[0].axvline(0, color=INK2, lw=0.8)
ax[0].set_xlabel("MSE reduction vs HAR-IV (%)"); ax[0].set_title("All 22 markets")
pg = pm.sort_values("Gain vs HAR-IV (%)")
cols = [GREEN if t < -1.96 else (RED if t > 1.96 else GRID) for t in pg["DM t"]]
ax[1].barh(range(len(pg)), pg["Gain vs HAR-IV (%)"], color=cols, height=0.7)
ax[1].set_yticks(range(len(pg))); ax[1].set_yticklabels(pg["name"], fontsize=6.5)
ax[1].axvline(0, color=INK2, lw=0.8)
ax[1].set_xlabel("FuelGuard: MSE reduction vs HAR-IV (%)"); ax[1].set_title("By market")
from matplotlib.patches import Patch  # noqa: E402
ax[1].legend(handles=[Patch(color=GREEN, label="significantly better (5%)"), Patch(color=GRID, label="not significant"),
                      Patch(color=RED, label="significantly worse (5%)")], loc="lower right", fontsize=6.5)
fig.tight_layout(); fig.savefig(F / "v2_fig_accuracy.pdf"); plt.close(fig)

# C. Gain by year
fig, ax = plt.subplots(figsize=(7.2, 2.5))
for c, colr in [(f"{FG} (all)", GREEN), ("ML ensemble (all)", RED), ("FuelGuard (diesel)", INK2)]:
    ax.plot(by_year.index, by_year[c], marker="o", ms=4, color=colr, label=c, ls="--" if "diesel" in c else "-")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(by_year.index); ax.tick_params(axis="x", labelsize=7.5)
ax.set_ylabel("MSE reduction vs HAR-IV (%)"); ax.set_title("Out-of-sample gain over HAR-IV, by test year")
ax.legend(loc="lower left", ncol=3, fontsize=7.5)
fig.savefig(F / "v2_fig_by_year.pdf"); plt.close(fig)

# D. Group importance
fig, ax = plt.subplots(figsize=(7.2, 2.6))
gi = imp_mean[::-1]
ax.barh(range(len(gi)), gi.values, color=BLUE, height=0.65)
ax.set_yticks(range(len(gi))); ax.set_yticklabels(gi.index, fontsize=7.5)
ax.set_xlabel("Increase in test MSE when the group is shuffled (%)")
ax.set_title("What the gradient-boosting model uses (mean over 2012-2026)")
fig.savefig(F / "v2_fig_importance.pdf"); plt.close(fig)

# E. Backtest: mean cost and tail risk per strategy (two aligned panels)
order = ["Never lock (market)", "Always 50%", "Always lock", "Random walk", "HAR", "HAR-IV", "Implied vol",
         "Lasso", "Random forest", "Gradient boosting", "Neural network", "ML ensemble", FG][::-1]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for j, (c, lab) in enumerate([("Mean %", "Average overrun (%)"), ("CVaR95 %", "CVaR 95% of the overrun (%)")]):
    vals = bt.loc[order, c]
    colors = [COL[s] if s in ML + [FG] else (YELLOW if s == "Implied vol" else INK2) for s in order]
    ax[j].hlines(range(len(order)), 0, vals, color=GRID, lw=1.2)
    ax[j].scatter(vals, range(len(order)), color=colors, s=36, zorder=3, edgecolor="white", lw=1)
    for yy, v in enumerate(vals):
        ax[j].text(v, yy, f"  {v:.2f}" if c == "Mean %" else f"  {v:.1f}", va="center", fontsize=6.5, color=INK2)
    ax[j].set_xlabel(lab); ax[j].grid(axis="y", visible=False)
    ax[j].axhline(len(order) - 3.5, color=INK2, lw=0.6, ls=":")
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=7)
ax[0].set_xlim(0, 1.25); ax[1].set_xlim(0, 20)
ax[0].set_title("Cost (lower is cheaper)"); ax[1].set_title("Tail risk (lower is safer)")
fig.suptitle(f"Diesel hedging, {len(dates)} monthly decisions {dates.min().year}-{dates.max().year}, hedge cost {C.HEDGE_COST:.0%}",
             x=0.01, ha="left", fontsize=9.5, fontweight="bold")
fig.tight_layout(); fig.savefig(F / "v2_fig_backtest.pdf"); plt.close(fig)

with open(R / "vol_key_numbers.json", "w") as f:
    json.dump(key, f, indent=1, default=str)
pd.set_option("display.width", 220)
print(pooled.round(4).to_string()); print(); print(diesel.round(4).to_string()); print(); print(energy.round(3).to_string())
print(); print(pm.round(2).to_string()); print(); print(by_year.round(1).to_string()); print(); print(imp_mean.round(2).to_string())
print(); print(bt.round(2).to_string()); print(); print(cm.round(3).to_string()); print(); print(hp.to_string())
