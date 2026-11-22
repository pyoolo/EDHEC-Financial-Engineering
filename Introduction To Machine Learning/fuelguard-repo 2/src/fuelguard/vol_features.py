"""FuelGuard 2.0: volatility target and features for a panel of futures markets.

Volatility is measured with the Garman-Klass (1980) range estimator, which uses the open, high, low
and settlement of the *same* contract on the *same* day. It is therefore not affected by the jumps
of generic futures series at contract rolls, and it is far less noisy than squared daily returns.

Target: log annualised volatility over the next HORIZON trading days, t+1 ... t+HORIZON.
All features at t use information up to the close of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .panel import MARKETS

H = C.HORIZON
LN2 = 2 * np.log(2) - 1
ASSET_CLASSES = sorted({v[3] for v in MARKETS.values()})

GROUPS = {
    "HAR (past realised vol)": ["lrv_1", "lrv_5", "lrv_22", "lrv_63"],
    "Implied vol level": ["liv", "liv_chg5", "liv_chg22"],
    "Variance risk premium": ["vrp"],
    "Implied vol smile": ["skew", "put_wing", "call_wing"],
    "Implied vol term structure": ["iv_term"],
    "Returns & asymmetry": ["ret_oc22", "bad_share22", "jump_share22"],
    "Volume & curve": ["lvolume", "curve"],
    "Common risk factors": ["lvix", "vix_term", "lv2x", "vix_chg5"],
    "Market & calendar": ["lrv_longrun", "month_sin", "month_cos"] + [f"cls_{c}" for c in ASSET_CLASSES],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def _clean_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Implied vols that are <1/3 or >3x their own past 63-day median are data errors."""
    cols = ["iv", "iv_m1", "iv_m2", "iv90", "iv95", "iv975", "iv1025", "iv105", "iv110"]
    ref = df["iv"].rolling(63, min_periods=20).median().shift(1)
    for c in cols:
        bad = (df[c] <= 0) | (df[c] < ref / 3) | (df[c] > ref * 3)
        df.loc[bad, c] = np.nan
        df[c] = df[c].ffill(limit=5)
    return df


def daily_gk_variance(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["settle"]
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    hl = np.log(h / l).clip(upper=0.4)            # cap absurd ranges (data errors)
    co = np.log(c / o).abs()
    co = np.minimum(co, hl)                        # settlement can sit slightly outside the day's range
    v = 0.5 * hl ** 2 - LN2 * co ** 2
    return v.where(ok & (v >= 0))


def _lvol(var_mean: pd.Series) -> pd.Series:
    return np.log(np.sqrt(252 * var_mean.clip(lower=1e-10)))


def market_frame(m: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """Features and target for one market (rows = that market's trading days)."""
    m = m.sort_values("date").set_index("date").copy()
    m.loc[(m[["open", "high", "low", "settle"]] <= 0).any(axis=1), ["open", "high", "low", "settle"]] = np.nan
    m = _clean_iv(m)
    v = daily_gk_variance(m)
    out = pd.DataFrame(index=m.index)
    # ---- target: next H days (needs at least 15 valid days)
    fwd_mean = v[::-1].rolling(H, min_periods=15).mean()[::-1].shift(-1)
    out["y"] = _lvol(fwd_mean)
    out.loc[fwd_mean.isna(), "y"] = np.nan
    pos = np.arange(len(m))
    end = np.minimum(pos + H, len(m) - 1)
    out["label_end_date"] = m.index[end]
    out.loc[pos + H >= len(m), "label_end_date"] = pd.NaT
    # ---- HAR components
    out["lrv_22"] = _lvol(v.rolling(22, min_periods=15).mean())
    # a single day can have a zero range (limit days, no trading): floor 1- and 5-day terms
    out["lrv_1"] = np.maximum(_lvol(v.rolling(1).mean()), out["lrv_22"] - 2.5)
    out["lrv_5"] = np.maximum(_lvol(v.rolling(5, min_periods=3).mean()), out["lrv_22"] - 2.5)
    out["lrv_63"] = _lvol(v.rolling(63, min_periods=40).mean())
    out["lrv_longrun"] = out["lrv_22"].expanding(min_periods=252).mean()
    # ---- implied volatility (annualised %, -> log of decimal)
    liv = np.log(m["iv"] / 100)
    out["liv"] = liv
    out["liv_chg5"] = liv.diff(5)
    out["liv_chg22"] = liv.diff(22)
    out["vrp"] = liv - out["lrv_22"]
    out["skew"] = (m["iv90"] - m["iv110"]) / m["iv"]
    out["put_wing"] = (m["iv90"] - m["iv"]) / m["iv"]
    out["call_wing"] = (m["iv110"] - m["iv"]) / m["iv"]
    out["iv_term"] = np.log(m["iv_m2"] / m["iv_m1"])
    # ---- intraday returns (same contract, roll-free) and asymmetry
    r = np.log(m["settle"] / m["open"]).where((m["settle"] > 0) & (m["open"] > 0))
    out["ret_oc22"] = r.rolling(22, min_periods=15).sum()
    neg = (r.clip(upper=0) ** 2).rolling(22, min_periods=15).sum()
    tot = (r ** 2).rolling(22, min_periods=15).sum()
    out["bad_share22"] = neg / tot.replace(0, np.nan)
    big = (v > 4 * v.rolling(63, min_periods=40).median().shift(1)).astype(float)
    out["jump_share22"] = big.rolling(22, min_periods=15).mean()
    # ---- volume and futures curve
    vol = m["volume"].where(m["volume"] > 0)
    out["lvolume"] = np.log(vol / vol.rolling(63, min_periods=40).median())
    out["curve"] = np.log(m["settle2"] / m["settle"]).where((m["settle2"] > 0) & (m["settle"] > 0))
    # ---- common risk factors (aligned forward-only)
    cm = common.reindex(out.index.union(common.index)).ffill(limit=5).reindex(out.index)
    out["lvix"] = np.log(cm["VIX"])
    out["vix_term"] = np.log(cm["VIX3M"] / cm["VIX"])
    out["lv2x"] = np.log(cm["V2X"])
    out["vix_chg5"] = out["lvix"].diff(5)
    # ---- calendar and market type
    mo = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * mo / 12)
    out["month_cos"] = np.cos(2 * np.pi * mo / 12)
    return out


def build_vol_dataset(panel: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    np.seterr(divide="ignore", invalid="ignore")
    frames = []
    for code, g in panel.groupby("market"):
        f = market_frame(g, common)
        f.insert(0, "market", code)
        cls = MARKETS[code][3]
        for c in ASSET_CLASSES:
            f[f"cls_{c}"] = float(cls == c)
        frames.append(f.reset_index())
    d = pd.concat(frames, ignore_index=True)
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.sort_values(["date", "market"]).reset_index(drop=True)
