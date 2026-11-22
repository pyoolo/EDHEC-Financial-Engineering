"""Prediction targets.

Decision at the close of day t, executed at the settlement of day t+1.
The hedged period is the next HORIZON trading days, t+2 ... t+1+HORIZON.

* Market cost (no hedge): the average daily diesel price over the period,
  i.e. the firm buys at the prevailing price each day.
* Lockable price: a *strip* of futures settled on t+1. Each day of the period is
  priced with the futures contract that is front month on that day: the 1st-month
  contract for days in the same calendar month as t+1, the 2nd-month contract for
  days after the next roll (NYMEX ULSD rolls on the first trading day of the month).
  Comparing the same contracts avoids a spurious roll yield.

Targets
* ``premium``  = market cost / lockable price - 1  (regression)
* ``spike``    = 1 if premium > SPIKE_THRESHOLD     (classification, main task)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def add_targets(df: pd.DataFrame, horizon: int = C.HORIZON, lag: int = C.EXECUTION_LAG,
                threshold: float = C.SPIKE_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    price = df["diesel_eur"].to_numpy()
    fx = df["EURUSD"].to_numpy()
    ho1, ho2 = df["HO1"].to_numpy(), df["HO2"].to_numpy()
    month = df.index.to_period("M").to_numpy()
    k = 1 / 100 / C.LITRES_PER_GALLON

    market = np.full(n, np.nan)
    lock = np.full(n, np.nan)
    end_idx = np.full(n, -1)
    for t in range(n):
        e = t + lag            # execution day
        a, b = e + 1, e + horizon   # hedged period [a, b]
        if b >= n:
            break
        market[t] = price[a:b + 1].mean()
        same = (month[a:b + 1] == month[e]).mean()   # share of days still on the current front contract
        lock[t] = (same * ho1[e] + (1 - same) * ho2[e]) * k / fx[e]
        end_idx[t] = b
    df["market_cost"] = market
    df["lock_price"] = lock
    df["label_end"] = end_idx          # position of the last day needed to know the label
    df["premium"] = df["market_cost"] / df["lock_price"] - 1
    df["spike"] = (df["premium"] > threshold).astype(float)
    df.loc[df["premium"].isna(), "spike"] = np.nan
    return df
