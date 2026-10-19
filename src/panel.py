"""Monthly decision panel.

At the last trading day of month m (the decision date) the investor sees
implied volatility, the futures curve and past realised volatility, sells
one month of variance, and is paid realised variance over month m+1.
All signals in a row use information up to the decision date only; the
`rv_next` column is the outcome and is never used as a signal.
"""
import numpy as np
import pandas as pd

from config import ANNUALISATION, IV_FIELD, IV_JUMP_WINDOW, IV_TREND_WINDOW


def realised_variance(ret, start, end):
    """Annualised realised variance of daily log returns in (start, end]."""
    r = ret.loc[(ret.index > start) & (ret.index <= end)]
    return ANNUALISATION * np.mean(r.values ** 2), len(r)


def build_panel(daily, front):
    d = daily.copy()
    for c in ["iv30", "iv1m", "iv2m"]:
        d[c] = d[c].ffill(limit=3)          # bridge isolated missing quotes
    d["iv_ma"] = d[IV_FIELD].rolling(IV_TREND_WINDOW, min_periods=IV_TREND_WINDOW // 2).mean()
    d["iv_chg5"] = np.log(d[IV_FIELD] / d[IV_FIELD].shift(IV_JUMP_WINDOW))

    decision = d.groupby(d.index.to_period("M")).tail(1).index   # month-end trading days
    last = decision[-1]
    if (last + pd.offsets.BMonthEnd(0) - last).days > 3:
        decision = decision[:-1]            # final month incomplete in the data
    ret = front["ret"]
    rows = []
    for i in range(1, len(decision) - 1):
        t_prev, t, t_next = decision[i - 1], decision[i], decision[i + 1]
        rv_past, _ = realised_variance(ret, t_prev, t)
        rv_next, n_next = realised_variance(ret, t, t_next)
        x = d.loc[t]
        rows.append({
            "decision_date": t,
            "month": t.to_period("M") + 1,          # realisation month
            "iv": x[IV_FIELD] / 100,
            "iv1m": x["iv1m"] / 100,
            "iv2m": x["iv2m"] / 100,
            "iv_over_ma": x[IV_FIELD] / x["iv_ma"],
            "iv_chg5": x["iv_chg5"],
            "slope": np.log(x["f1"] / x["f2"]) if x["f1"] > 0 else np.nan,  # >0 = backwardation
            "rv_past": rv_past,
            "rv_next": rv_next,
            "n_days": n_next,
        })
    p = pd.DataFrame(rows).set_index("month")
    p = p.dropna(subset=["iv", "rv_next"])
    # short variance-swap P&L per unit of vega notional (decimal return)
    p["ret_short_var"] = (p["iv"] ** 2 - p["rv_next"]) / (2 * p["iv"])
    p["vrp_var"] = p["iv"] ** 2 - p["rv_next"]                  # variance units
    p["vrp_vol"] = p["iv"] - np.sqrt(p["rv_next"])              # vol units
    return p
