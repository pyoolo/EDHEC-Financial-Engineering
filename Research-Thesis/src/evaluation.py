"""Performance metrics, spanning regressions, Sharpe-ratio bootstrap."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import BOOTSTRAP_BLOCK, BOOTSTRAP_REPS, NEWEY_WEST_LAGS, RANDOM_SEED

TARGET_VOL = 0.10   # every strategy is scaled to 10% annualised volatility


def vol_match(r, target=TARGET_VOL, ref=None):
    """Scale returns to a target annual volatility.

    `ref` is the sample on which the scale is estimated (defaults to r
    itself, i.e. ex-post scaling as in Moreira and Muir, 2017).
    """
    ref = r if ref is None else ref
    return r * target / (ref.std() * np.sqrt(12))


def max_drawdown(r):
    """Maximum drawdown of cumulative (additive) P&L, in return units.

    Returns are P&L per unit of capital on an unfunded swap position, so
    P&L is summed rather than compounded.
    """
    cum = r.cumsum()
    return (cum.cummax() - cum).max()


def metrics(r):
    r = r.dropna()
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    mdd = max_drawdown(r)
    q95, q99 = r.quantile(0.05), r.quantile(0.01)
    return pd.Series({
        "Mean (% p.a.)": 100 * 12 * r.mean(),
        "Vol (% p.a.)": 100 * np.sqrt(12) * r.std(),
        "Sharpe": np.sqrt(12) * r.mean() / r.std(),
        "Sortino": np.sqrt(12) * r.mean() / down if down > 0 else np.nan,
        "Skewness": stats.skew(r),
        "Excess kurtosis": stats.kurtosis(r),
        "Max drawdown (%)": 100 * mdd,
        "Calmar": 12 * r.mean() / mdd if mdd > 0 else np.nan,
        "VaR 95 (%)": -100 * q95,
        "CVaR 95 (%)": -100 * r[r <= q95].mean(),
        "VaR 99 (%)": -100 * q99,
        "CVaR 99 (%)": -100 * r[r <= q99].mean(),
        "Worst month (%)": 100 * r.min(),
        "Months": len(r),
    })


def newey_west_mean(r, lags=NEWEY_WEST_LAGS):
    """Mean with a Newey-West t-statistic."""
    res = sm.OLS(r.values, np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res.params[0], res.tvalues[0]


def spanning(managed, base, lags=NEWEY_WEST_LAGS):
    """managed = alpha + beta * base + e, Newey-West standard errors."""
    df = pd.concat([managed, base], axis=1).dropna()
    X = sm.add_constant(df.iloc[:, 1].values)
    res = sm.OLS(df.iloc[:, 0].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    resid_sd = np.std(res.resid, ddof=2)
    return pd.Series({
        "Alpha (% p.a.)": 100 * 12 * res.params[0],
        "t(alpha)": res.tvalues[0],
        "Beta": res.params[1],
        "t(beta)": res.tvalues[1],
        "Appraisal ratio": np.sqrt(12) * res.params[0] / resid_sd,
        "R2": res.rsquared,
    })


def sharpe(r):
    return np.sqrt(12) * r.mean() / r.std()


def bootstrap_sharpe_diff(a, b, reps=BOOTSTRAP_REPS, block=BOOTSTRAP_BLOCK, seed=RANDOM_SEED):
    """Circular block bootstrap of Sharpe(a) - Sharpe(b) on paired months.

    Two-sided p-value of H0: no difference, from the bootstrap distribution
    re-centred at zero (in the spirit of Ledoit and Wolf, 2008).
    """
    df = pd.concat([a, b], axis=1).dropna().values
    n = len(df)
    rng = np.random.default_rng(seed)
    obs = sharpe(pd.Series(df[:, 0])) - sharpe(pd.Series(df[:, 1]))
    diffs = np.empty(reps)
    n_blocks = int(np.ceil(n / block))
    for k in range(reps):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        x = df[idx]
        diffs[k] = (np.sqrt(12) * x[:, 0].mean() / x[:, 0].std(ddof=1)
                    - np.sqrt(12) * x[:, 1].mean() / x[:, 1].std(ddof=1))
    p = np.mean(np.abs(diffs - diffs.mean()) >= abs(obs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, p, lo, hi
