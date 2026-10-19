"""Timing rules: each maps the decision-date signals to a weight w_m.

Managed return in month m+1 = w_m * short-variance return - costs.
Rules that need a history (percentiles, medians) use an expanding window
of past decision dates only and keep w = 1 until MIN_HISTORY_MONTHS exist.
"""
import numpy as np
import pandas as pd

from config import (IV_JUMP_THRESHOLD, MIN_HISTORY_MONTHS, PERCENTILE,
                    REDUCED_WEIGHT, VOL_SCALING_CAP)

RULE_LABELS = {
    "unmanaged": "Unmanaged",
    "vol_scaling": "Volatility scaling",
    "rv_threshold": "Realised-vol threshold",
    "iv_jump": "Implied-vol jump",
    "iv_trend": "Implied-vol trend",
    "term_structure": "Term structure",
    "premium_filter": "Premium filter",
    "backwardation": "Backwardation",
}


RULE_SHORT = {
    "unmanaged": "UM", "vol_scaling": "VS", "rv_threshold": "RVT", "iv_jump": "IVJ",
    "iv_trend": "IVT", "term_structure": "TS", "premium_filter": "PF", "backwardation": "BW",
}


def expanding_quantile(x, q, min_periods):
    """Quantile of x over all observations up to and including each date."""
    return x.expanding(min_periods=min_periods).quantile(q)


def rule_weights(p, min_hist=MIN_HISTORY_MONTHS, pct=PERCENTILE):
    w = pd.DataFrame(index=p.index)
    w["unmanaged"] = 1.0

    # 1. Moreira-Muir: w = c / past variance, c = expanding median of past
    #    variance (real time), capped
    c = p["rv_past"].expanding(min_periods=min_hist).median()
    w["vol_scaling"] = (c / p["rv_past"]).clip(upper=VOL_SCALING_CAP).fillna(1.0)

    # 2. realised vol above its expanding 80th percentile -> halve
    hi = expanding_quantile(p["rv_past"], pct, min_hist)
    w["rv_threshold"] = np.where(p["rv_past"] > hi, REDUCED_WEIGHT, 1.0)

    # 3. implied vol up more than 20% over the last 5 trading days -> out
    w["iv_jump"] = np.where(p["iv_chg5"] > np.log(1 + IV_JUMP_THRESHOLD), 0.0, 1.0)

    # 4. implied vol above its 3-month moving average -> halve
    w["iv_trend"] = np.where(p["iv_over_ma"] > 1.0, REDUCED_WEIGHT, 1.0)

    # 5. front-month IV unusually high relative to second-month IV -> halve
    #    (the curve is inverted half of the time, so "unusual" = above its
    #    expanding 80th percentile)
    ratio = p["iv1m"] / p["iv2m"]
    hi = expanding_quantile(ratio, pct, min_hist)
    w["term_structure"] = np.where(ratio > hi, REDUCED_WEIGHT, 1.0)

    # 6. implied vol below trailing realised vol -> do not sell
    w["premium_filter"] = np.where(p["iv"] < np.sqrt(p["rv_past"]), 0.0, 1.0)

    # 7. commodity-specific: steep backwardation (scarcity) -> halve
    hi = expanding_quantile(p["slope"], pct, min_hist)
    w["backwardation"] = np.where(p["slope"] > hi, REDUCED_WEIGHT, 1.0)
    return w


def managed_returns(p, w, cost_vol_points=0.0):
    """Monthly returns per unit of vega notional, net of a bid-ask cost.

    Selling variance at a strike `cost_vol_points` below mid costs about
    that many vol points per unit of vega notional, i.e. cost/100 per month.
    """
    gross = w.mul(p["ret_short_var"], axis=0)
    return gross - w.abs() * cost_vol_points / 100
