"""Economic evaluation: monthly hedging decisions of a small haulier.

On the first trading day of each month the firm chooses a hedge ratio h in {0, 0.5, 1}:
cost per litre = h * lockable price * (1 + hedge cost) + (1 - h) * market cost.
Rules are fixed ex ante; the only inputs are model outputs and quantities that are
known in the training window (base rate, median implied volatility).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def monthly_decision_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).min().values)


def hedge_ratios(preds: pd.DataFrame, data: pd.DataFrame, train_vol_median: pd.Series,
                 dates: pd.DatetimeIndex) -> pd.DataFrame:
    p = preds.loc[dates]
    d = data.loc[dates]
    base = p["clf|Climatology"]
    H = pd.DataFrame(index=dates)
    H["Never lock (market)"] = 0.0
    H["Always lock"] = 1.0
    H["Always 50%"] = 0.5
    H["Momentum rule"] = (d["diesel_eur_ret21"] > 0).astype(float)
    H["Volatility rule"] = (d["_HO_iv"] > train_vol_median.loc[dates]).astype(float)
    for m in ["Logistic (implied vol only)", "Logistic (L2)", "Random forest", "Gradient boosting"]:
        q = p[f"clf|{m}"]
        H[f"ML: {m}"] = np.where(q > 2 * base, 1.0, np.where(q > base, 0.5, 0.0))
    H["ML: regression (GB)"] = (p["reg|Gradient boosting"] > C.HEDGE_COST).astype(float)
    return H


def costs(H: pd.DataFrame, data: pd.DataFrame, hedge_cost: float = C.HEDGE_COST) -> pd.DataFrame:
    d = data.loc[H.index]
    lock = d["_lock_price"] * (1 + hedge_cost)
    mkt = d["_market_cost"]
    return H.mul(lock, axis=0) + (1 - H).mul(mkt, axis=0)


def summary(cost: pd.DataFrame, data: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    """Cost relative to the lockable price isolates the effect of the decision from
    the level of diesel prices (which moved from 0.3 to 1.1 EUR/l)."""
    lock = data.loc[cost.index, "_lock_price"]
    over = cost.div(lock, axis=0) - 1          # overrun vs price available at decision time
    mkt = cost["Never lock (market)"]
    n = len(cost)
    k = max(1, int(np.ceil(0.05 * n)))
    rows = {}
    for c in cost.columns:
        o = over[c].sort_values(ascending=False)
        rows[c] = {
            "Avg cost (EUR/l)": cost[c].mean(),
            "Avg overrun (%)": over[c].mean() * 100,
            "Overrun st.dev (%)": over[c].std() * 100,
            "CVaR95 overrun (%)": o.iloc[:k].mean() * 100,
            "Worst month (%)": o.iloc[0] * 100,
            "Months >5% overrun": int((over[c] > C.SPIKE_THRESHOLD).sum()),
            "Avg hedge ratio": H[c].mean(),
            "CVaR95 extra cost (EUR/month)": (cost[c] - lock).sort_values(ascending=False).iloc[:k].mean()
            * C.MONTHLY_LITRES,
            "Saving vs market (EUR/yr)": (mkt - cost[c]).mean() * C.MONTHLY_LITRES * 12,
        }
    return pd.DataFrame(rows).T
