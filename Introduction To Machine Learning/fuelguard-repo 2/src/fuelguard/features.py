"""Feature engineering.

Every feature at date t uses only information published up to the close of t
(rolling windows, lags, differences: all backward-looking). ``tests/test_leakage.py``
checks this by changing future values and verifying past features do not move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RETURN_SERIES = ["diesel_eur", "CO1", "CL2", "NG1", "HG1", "GC1", "SPX",
                 "SP_ENERGY", "STOXX_OILGAS", "TY1", "EURUSD"]
RETURN_WINDOWS = [5, 21, 63]

GROUPS = {
    "Momentum (returns)": [f"{s}_ret{k}" for s in RETURN_SERIES for k in RETURN_WINDOWS],
    "Futures curve": ["HO_slope", "CO_slope", "NG_slope", "CL_slope"],
    "Refining margin": ["crack", "crack_chg21"],
    "Volatility": ["HO_iv", "CO_iv", "HO_rv21", "HO_rv63", "iv_minus_rv", "iv_chg21",
                   "VIX", "VIX_term"],
    "Inflation expectations": ["US_BE5", "EU_INF5", "BE5_chg21", "EUINF_chg21"],
    "Price level": ["diesel_vs_ma252", "diesel_z252"],
    "Past outcomes": ["past_premium", "past_spike_rate12m"],
    "Seasonality": ["month_sin", "month_cos"],
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for s in RETURN_SERIES:
        for k in RETURN_WINDOWS:
            X[f"{s}_ret{k}"] = df[s].pct_change(k, fill_method=None)
    # Futures curve: >0 contango (next month dearer), <0 backwardation
    X["HO_slope"] = df["HO2"] / df["HO1"] - 1
    X["CO_slope"] = df["CO2"] / df["CO1"] - 1
    X["NG_slope"] = df["NG2"] / df["NG1"] - 1
    X["CL_slope"] = (df["CL2"] - df["CL1"]) / df["CL2"]   # CL1 was negative in April 2020
    # Diesel crack spread in USD/bbl (42 gallons per barrel)
    crack = df["HO1"] * 0.42 - df["CO1"]
    X["crack"] = crack
    X["crack_chg21"] = crack.diff(21)
    # Volatility
    logret = np.log(df["HO1"]).diff()
    X["HO_iv"] = df["HO_iv"]
    X["CO_iv"] = df["CO_iv"]
    X["HO_rv21"] = logret.rolling(21).std() * np.sqrt(252) * 100
    X["HO_rv63"] = logret.rolling(63).std() * np.sqrt(252) * 100
    X["iv_minus_rv"] = X["HO_iv"] - X["HO_rv21"]
    X["iv_chg21"] = X["HO_iv"].diff(21)
    X["VIX"] = df["VIX"]
    X["VIX_term"] = df["VIX3M"] / df["VIX"]
    # Inflation expectations
    X["US_BE5"] = df["US_BE5"]
    X["EU_INF5"] = df["EU_INF5"]
    X["BE5_chg21"] = df["US_BE5"].diff(21)
    X["EUINF_chg21"] = df["EU_INF5"].diff(21)
    # Price level relative to its own history
    p = df["diesel_eur"]
    X["diesel_vs_ma252"] = p / p.rolling(252).mean() - 1
    X["diesel_z252"] = (p - p.rolling(252).mean()) / p.rolling(252).std()
    # Past realised outcomes: the label of date s is known at label_end(s) <= s + PURGE - 1,
    # so lagging by PURGE days only uses labels that are already observed at t.
    X["past_premium"] = df["premium"].shift(C.PURGE)
    X["past_spike_rate12m"] = df["spike"].shift(C.PURGE).rolling(252, min_periods=126).mean()
    # Seasonality (heating season matters for distillates)
    m = df.index.month
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)
    cols = [c for g in GROUPS.values() for c in g]
    assert set(cols) == set(X.columns), set(X.columns) ^ set(cols)
    return X[cols]


def clean_implied_vol(df: pd.DataFrame, floor: float = 5.0) -> pd.DataFrame:
    """Implied volatilities below 5% are data errors (e.g. 1.7% on 2007-09-12): set
    to missing, then carry the last valid value forward."""
    df = df.copy()
    for c in ["HO_iv", "CO_iv"]:
        df.loc[df[c] < floor, c] = np.nan
        df[c] = df[c].ffill(limit=5)
    return df
