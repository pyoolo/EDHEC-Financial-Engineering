"""Step 3. Statistical evaluation, hedging backtest, figures and LaTeX tables."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.backtest import costs, hedge_ratios, monthly_decision_dates, summary  # noqa: E402
from fuelguard.data import SERIES  # noqa: E402
from fuelguard.evaluate import (auc_by_year, block_bootstrap_ci, calibration_bins,  # noqa: E402
                                classification_table, regression_table)
from fuelguard.features import GROUPS  # noqa: E402
from fuelguard.report import (AQUA, BLUE, FULL_MODELS, GRID, INK, INK2, MODEL_COLORS,  # noqa: E402
                              ORANGE, RED, VIOLET, YELLOW, style, to_latex)

style()
F, T = C.FIGURES, C.TABLES
F.mkdir(parents=True, exist_ok=True)
T.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(C.DATA_PROCESSED / "model_dataset.csv", index_col=0, parse_dates=True)
preds = pd.read_csv(C.RESULTS / "oos_predictions.csv", index_col=0, parse_dates=True)
imp = pd.read_csv(C.RESULTS / "permutation_importance_by_year.csv", index_col=0)
chosen = pd.read_csv(T / "chosen_hyperparameters.csv")
y_clf, y_reg = data["_spike"], data["_premium"]
key = {}

# ------------------------------------------------------------------ statistics
ct = classification_table(preds, y_clf)
rt = regression_table(preds, y_reg)
aby = auc_by_year(preds, y_clf)
ct.to_csv(T / "classification_oos.csv"); rt.to_csv(T / "regression_oos.csv"); aby.to_csv(T / "auc_by_year.csv")
to_latex(ct, T / "classification_oos.tex", index_name="Model",
         fmt={"Share flagged": "{:.2f}", "Precision": "{:.2f}", "Recall": "{:.2f}"})
to_latex(rt, T / "regression_oos.tex", index_name="Model",
         fmt={"RMSE (%)": "{:.2f}", "MAE (%)": "{:.2f}", "OOS R2 vs futures (%)": "{:.1f}",
              "Hit rate (sign)": "{:.2f}", "DM stat": "{:.2f}", "DM p-value": "{:.2f}"})
ci = block_bootstrap_ci(preds, y_clf)
ci.to_csv(T / "classification_bootstrap_ci.csv")
to_latex(ci, T / "classification_bootstrap_ci.tex", index_name="Model", default="{:.3f}")
key["bootstrap"] = ci.round(3).to_dict(orient="index")
best_clf = ct.loc[FULL_MODELS, "AUC"].idxmax()   # best of the 57-feature models
# model-free benchmark: rank days by implied volatility (sign fixed a priori: more vol, more spikes)
from sklearn.metrics import roc_auc_score  # noqa: E402
iv_auc = roc_auc_score(y_clf.loc[preds.index], data.loc[preds.index, "_HO_iv"])
key["iv_auc_model_free"] = round(float(iv_auc), 3)
ct.loc["Implied vol rank (no model)", "AUC"] = iv_auc
ct.to_csv(T / "classification_oos.csv")
to_latex(ct, T / "classification_oos.tex", index_name="Model",
         fmt={"Share flagged": "{:.2f}", "Precision": "{:.2f}", "Recall": "{:.2f}"})
key["best_clf"] = best_clf
key["auc"] = ct["AUC"].round(3).to_dict()
key["brier_skill"] = (ct["Brier skill"] * 100).round(1).to_dict()
key["years_auc_above_05"] = {m: int((aby[m] > 0.5).sum()) for m in MODEL_COLORS}
iv_y = {}
for yr_, idx in preds.groupby("fold_year").groups.items():
    yy = y_clf.loc[idx]
    iv_y[int(yr_)] = roc_auc_score(yy, data.loc[idx, "_HO_iv"]) if yy.nunique() > 1 else np.nan
aby["Implied vol rank (no model)"] = pd.Series(iv_y)
aby.to_csv(T / "auc_by_year.csv")
key["n_years"] = int(aby.shape[0])
key["oos_days"] = int(len(preds))
key["oos_spike_rate"] = round(float(y_clf.loc[preds.index].mean()), 3)
key["reg_r2"] = rt["OOS R2 vs futures (%)"].round(2).to_dict()
key["reg_dm_p"] = rt["DM p-value"].round(3).to_dict()

# ------------------------------------------------------------------ backtest
dates = monthly_decision_dates(preds.index)
# median implied vol known at the start of each test year (expanding, training data only)
vol_med = pd.Series(index=dates, dtype=float)
for d in dates:
    vol_med[d] = data.loc[data.index < pd.Timestamp(d.year, 1, 1), "_HO_iv"].median()
H = hedge_ratios(preds, data, vol_med, dates)
cost = costs(H, data)
bt = summary(cost, data, H)
bt.to_csv(T / "backtest_summary.csv")
H.to_csv(C.RESULTS / "hedge_ratios_monthly.csv"); cost.to_csv(C.RESULTS / "cost_monthly.csv")
short = {"Avg overrun (%)": "Mean %", "Overrun st.dev (%)": "St.dev %", "CVaR95 overrun (%)": "CVaR95 %",
         "Worst month (%)": "Worst %", "Months >5% overrun": "Months >5%", "Avg hedge ratio": "Hedge ratio",
         "CVaR95 extra cost (EUR/month)": "CVaR95 EUR/month", "Saving vs market (EUR/yr)": "Saving EUR/yr"}
to_latex(bt.drop(columns=["Avg cost (EUR/l)"]).rename(columns=short), T / "backtest_summary.tex",
         index_name="Strategy",
         fmt={"Mean %": "{:.2f}", "St.dev %": "{:.2f}", "CVaR95 %": "{:.1f}", "Worst %": "{:.1f}",
              "Months >5%": "{:.0f}", "Hedge ratio": "{:.2f}", "Saving EUR/yr": "{:,.0f}",
              "CVaR95 EUR/month": "{:,.0f}"})
sens = []
for hc in C.HEDGE_COST_GRID:
    s = summary(costs(H, data, hc), data, H)
    for strat in s.index:
        sens.append({"hedge cost": f"{hc:.0%}", "strategy": strat,
                     "avg": s.loc[strat, "Avg overrun (%)"], "cvar": s.loc[strat, "CVaR95 overrun (%)"]})
sens = pd.DataFrame(sens)
sens.to_csv(T / "backtest_sensitivity.csv", index=False)
piv = sens.pivot(index="strategy", columns="hedge cost", values="avg").loc[bt.index]
piv.columns = [f"Avg overrun, cost {c}" for c in piv.columns]
piv["CVaR95, cost 1%"] = sens[sens["hedge cost"] == "1%"].set_index("strategy").loc[bt.index, "cvar"]
to_latex(piv, T / "backtest_sensitivity.tex", index_name="Strategy (avg overrun, %)", default="{:.2f}")
key["backtest_months"] = int(len(dates))
key["backtest"] = bt.round(3).to_dict(orient="index")
# by regime: years where the market ended above vs below the lock price on average
yr = (data.loc[dates, "_premium"].groupby(dates.year).mean())
up_years = yr[yr > 0].index
over = cost.div(data.loc[dates, "_lock_price"], axis=0) - 1
reg = pd.DataFrame({"Rising-price years": over[dates.year.isin(up_years)].mean() * 100,
                    "Falling-price years": over[~dates.year.isin(up_years)].mean() * 100})
reg.to_csv(T / "backtest_by_regime.csv")
key["regime_up_years"] = [int(y) for y in up_years]

# ------------------------------------------------------------------ descriptive table (compact)
desc = pd.read_csv(T / "descriptive_raw.csv", index_col=0)
desc_tex = desc[["count", "mean", "std", "min", "max"]].copy()
desc_tex.index = [f"{SERIES[i][0]}" if i in SERIES else i for i in desc_tex.index]
desc_tex.insert(0, "Series", [d[:58] for d in desc["description"]])
to_latex(desc_tex.reset_index(drop=False).set_index("index"), T / "descriptive_raw.tex",
         index_name="Ticker", fmt={"count": "{:,.0f}", "mean": "{:.2f}", "std": "{:.2f}",
                                   "min": "{:.2f}", "max": "{:.2f}"}, col_align="llrrrrr")

# hyper-parameter choices
hp = (chosen.groupby(["task", "model"])["params"].agg(lambda s: s.value_counts().index[0] +
                                                     f" ({s.value_counts().iloc[0]}/{len(s)} years)"))
hp.to_csv(T / "hyperparameters_mode.csv")
hpt = hp.reset_index()
hpt["task"] = hpt["task"].map({"clf": "Spike (classif.)", "reg": "Premium (regr.)"})
hpt["params"] = hpt["params"].str.replace(r"[{}']", "", regex=True).str.replace("logisticregression__", "").str.replace("ridge__", "")
to_latex(hpt.set_index("task"), T / "hyperparameters_mode.tex", index_name="Task", col_align="lll")

# ------------------------------------------------------------------ figures
# 1. Diesel price and the target
p = data["_diesel_eur"]
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
ax[0].plot(p.index, p, color=BLUE, lw=1.1)
ax[0].set_ylabel("EUR / litre")
ax[0].set_title("Wholesale diesel price (NY Harbor ULSD futures, converted to EUR/litre)")
prem = data["_premium"] * 100
ax[1].fill_between(prem.index, 0, prem.where(prem > 5), color=RED, lw=0, label="Spike (> 5%)")
ax[1].fill_between(prem.index, 0, prem.where(prem <= 5), color=GRID, lw=0, label="No spike")
ax[1].plot(prem.index, prem, color=INK2, lw=0.4)
ax[1].axhline(5, color=RED, lw=0.8, ls="--")
ax[1].set_ylabel("%")
ax[1].set_title("Target: next-month market cost vs price lockable today")
ax[1].legend(loc="upper left", ncol=2)
for a in ax:
    a.axvspan(pd.Timestamp(C.FIRST_TEST_YEAR, 1, 1), p.index[-1], color="#f3f2ee", zorder=-1, lw=0)
ax[0].text(pd.Timestamp(C.FIRST_TEST_YEAR, 3, 1), p.max() * 0.95, "out-of-sample (walk-forward)",
           color=INK2, fontsize=7.5, va="top")
fig.savefig(F / "fig1_price_target.pdf"); plt.close(fig)

# 2. Walk-forward scheme
fig, ax = plt.subplots(figsize=(7.2, 2.3))
years = sorted(chosen["year"].unique())
show = years[:3] + years[-2:]
for i, yv in enumerate(show):
    row = chosen[(chosen["year"] == yv)].iloc[0]
    ts, te_ = pd.Timestamp(row["train_start"]), pd.Timestamp(row["train_end"])
    yy = len(show) - i
    ax.barh(yy, (te_ - ts).days, left=ts, height=0.55, color=BLUE, label="Training (tuned by purged CV)" if i == 0 else None)
    ax.barh(yy, (pd.Timestamp(yv, 1, 1) - te_).days, left=te_, height=0.55, color=YELLOW,
            label="Purge + embargo gap" if i == 0 else None)
    ax.barh(yy, 365, left=pd.Timestamp(yv, 1, 1), height=0.55, color=ORANGE, label="Test year" if i == 0 else None)
    ax.text(pd.Timestamp(yv + 1, 2, 1), yy, f"test {yv}", va="center", fontsize=7.5, color=INK2)
ax.text(pd.Timestamp(2018, 1, 1), 2.5 + 0.0, "...", fontsize=14, color=INK2, ha="center", va="center")
ax.set_yticks([]); ax.grid(axis="y", visible=False)
ax.set_title("Expanding-window walk-forward evaluation (one refit per year)")
ax.legend(loc="lower left", ncol=3, fontsize=7.5, bbox_to_anchor=(0, -0.42))
fig.savefig(F / "fig2_walk_forward.pdf"); plt.close(fig)

# 3. AUC by year
fig, ax = plt.subplots(figsize=(7.2, 2.6))
for m, c in MODEL_COLORS.items():
    ax.plot(aby.index, aby[m], marker="o", ms=4, color=c, label=m)
ax.axhline(0.5, color=INK2, lw=1, ls="--")
ax.text(aby.index.min() - 0.3, 0.51, "no skill", color=INK2, fontsize=7.5)
ax.set_ylim(0, 1.0); ax.set_ylabel("ROC AUC")
ax.set_xticks(aby.index); ax.tick_params(axis="x", labelsize=7.5)
ax.set_title("Out-of-sample discrimination of spikes, by test year")
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=7.5)
fig.savefig(F / "fig3_auc_by_year.pdf"); plt.close(fig)

# 4. Calibration (reliability diagram)
fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.plot([0, 0.45], [0, 0.45], color=INK2, lw=1, ls="--")
for m, c in MODEL_COLORS.items():
    cb = calibration_bins(preds[f"clf|{m}"], y_clf)
    ax.plot(cb["predicted"], cb["observed"], marker="o", ms=4, color=c, label=m)
ax.set_xlabel("Predicted spike probability"); ax.set_ylabel("Observed spike frequency")
ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.45)
ax.set_title("Calibration (octiles)")
ax.legend(loc="upper left", fontsize=7)
fig.savefig(F / "fig4_calibration.pdf"); plt.close(fig)

# 5. Permutation importance (best model, averaged over test years)
cols = [c for c in imp.columns if c.endswith("|" + best_clf)]
mi = imp[cols].mean(axis=1).sort_values(ascending=False)
grp = {f: g for g, fs in GROUPS.items() for f in fs}
top = mi.head(15)[::-1]
group_imp = mi.groupby(mi.index.map(grp)).sum().sort_values(ascending=False)
group_imp.to_csv(T / "importance_by_group.csv"); mi.to_csv(T / "importance_features.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
ax[0].barh(range(len(top)), top.values * 100, color=BLUE, height=0.7)
ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.index, fontsize=7)
ax[0].set_xlabel("Drop in AUC when shuffled (x100)")
ax[0].set_title(f"Top 15 features ({best_clf})")
g = group_imp[::-1]
ax[1].barh(range(len(g)), g.values * 100, color=BLUE, height=0.7)
ax[1].set_yticks(range(len(g))); ax[1].set_yticklabels(g.index, fontsize=7)
ax[1].set_xlabel("Summed AUC drop (x100)")
ax[1].set_title("By feature group")
fig.tight_layout(); fig.savefig(F / "fig5_importance.pdf"); plt.close(fig)
key["top_features"] = [str(x) for x in mi.head(8).index]
key["group_importance"] = (group_imp * 100).round(2).to_dict()

# 6. Risk-cost trade-off of hedging strategies
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for s, r in bt.iterrows():
    is_ml = s.startswith("ML")
    ax.scatter(r["Avg overrun (%)"], r["CVaR95 overrun (%)"], s=42, color=AQUA if is_ml else INK2,
               edgecolor="white", lw=1.2, zorder=3)
    ax.annotate(s.replace("ML: regression", "ML: Regression").replace("ML: ", ""), (r["Avg overrun (%)"], r["CVaR95 overrun (%)"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
ax.scatter([], [], color=AQUA, label="Machine-learning rules"); ax.scatter([], [], color=INK2, label="Simple rules")
ax.set_xlim(0.45, 1.08)
ax.set_xlabel("Average cost overrun vs price lockable at decision (%)  <- cheaper")
ax.set_ylabel("CVaR 95%: average of worst 5% months (%)  <- safer")
ax.set_title(f"Monthly hedging decisions, {dates.min().year}-{dates.max().year}, hedge cost {C.HEDGE_COST:.0%}")
ax.legend(loc="lower left")
fig.savefig(F / "fig6_tradeoff.pdf"); plt.close(fig)

# 7. Cumulative saving vs market, in EUR for the reference haulier
cum = (cost.rsub(cost["Never lock (market)"], axis=0) * C.MONTHLY_LITRES).cumsum() / 1000
fig, ax = plt.subplots(figsize=(7.2, 2.7))
for s, c, ls in [("Always lock", INK2, "-"), ("Always 50%", INK2, "--"), ("Volatility rule", INK2, ":"),
                 (f"ML: {best_clf}", MODEL_COLORS[best_clf], "-"),
                 ("ML: Logistic (implied vol only)", MODEL_COLORS["Logistic (implied vol only)"], "-")]:
    ax.plot(cum.index, cum[s], color=c, ls=ls, label=s, lw=1.6 if s.startswith("ML") else 1.1)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylabel("EUR thousands")
ax.set_title(f"Cumulative saving vs buying at market price ({C.MONTHLY_LITRES:,} litres/month)")
ax.legend(loc="upper left", ncol=2)
fig.savefig(F / "fig7_cumulative.pdf"); plt.close(fig)


# 8. Diagnostic: tail risk at equal hedging budget (rank months by each signal, hedge the top q)
# Ranks use the whole test period, so this is NOT an implementable strategy: it compares how well
# each signal orders months, the economic analogue of the AUC.
dd = data.loc[dates]
over_mkt = dd["_market_cost"] / dd["_lock_price"] - 1
signals = {f"ML: {best_clf}": preds.loc[dates, f"clf|{best_clf}"], "Implied volatility": dd["_HO_iv"],
           "Momentum (21d return)": dd["diesel_eur_ret21"]}
budget = []
kk = max(1, int(np.ceil(0.05 * len(dates))))
for q in np.arange(0.1, 0.91, 0.1):
    for name, sig in signals.items():
        h = (sig.rank(pct=True) > 1 - q).astype(float)
        o = h * C.HEDGE_COST + (1 - h) * over_mkt
        budget.append({"budget": q, "signal": name, "avg": o.mean() * 100,
                       "cvar": o.sort_values(ascending=False).iloc[:kk].mean() * 100})
    o = q * C.HEDGE_COST + (1 - q) * over_mkt     # same budget spread evenly (constant ratio)
    budget.append({"budget": q, "signal": "Constant ratio", "avg": o.mean() * 100,
                   "cvar": o.sort_values(ascending=False).iloc[:kk].mean() * 100})
budget = pd.DataFrame(budget)
budget.to_csv(T / "hedge_budget_diagnostic.csv", index=False)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
for name, c, ls in [(f"ML: {best_clf}", MODEL_COLORS[best_clf], "-"),
                    ("Implied volatility", MODEL_COLORS["Logistic (implied vol only)"], "-"),
                    ("Momentum (21d return)", INK2, ":"), ("Constant ratio", INK2, "--")]:
    b = budget[budget["signal"] == name]
    lab = "Implied volatility (= vol-only logistic)" if name == "Implied volatility" else name
    ax.plot(b["budget"] * 100, b["cvar"], marker="o", ms=4, color=c, label=lab, ls=ls)
ax.set_xlabel("Share of months hedged (%)"); ax.set_ylabel("CVaR 95% overrun (%)")
ax.set_title("Tail risk at equal hedging budget: which signal picks the right months?")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
fig.savefig(F / "fig8_budget.pdf"); plt.close(fig)
key["budget"] = budget.round(2).to_dict(orient="records")

# ------------------------------------------------------------------ key numbers (unlabelled tail)
with open(C.RESULTS / "key_numbers.json", "w") as f:
    json.dump(key, f, indent=1, default=str)
print(ci.round(3).to_string()); print(); print(ct.round(3).to_string()); print(); print(rt.round(3).to_string()); print()
print(bt.round(3).to_string()); print(); print(reg.round(2).to_string()); print(); print(aby.round(2).to_string())
print(); print(group_imp.round(4).to_string()); print(); print(hp.to_string())
