"""Implied variance from the 30-day smile (variance-swap strike).

The fair variance-swap strike is the model-free implied variance
(Britten-Jones and Neuberger, 2000; Demeterfi et al., 1999). With the
futures price normalised to 1, zero rates and maturity T,

    K = (2 / T) * [ int_0^1 P(k) / k^2 dk + int_1^inf C(k) / k^2 dk ],

where P and C are out-of-the-money put and call prices at moneyness k.
Option prices come from Black (1976) with the implied volatility
interpolated linearly in moneyness between the smile points available
(e.g. 90%, 100%, 110%) and held flat outside them (Jiang and Tian, 2005).

With only a few points close to the money this is a *smile-adjusted*
approximation: it captures the slope and curvature of the smile near the
money but not the far tails, so it is still a lower bound when the tails
are steep. With a flat smile it equals ATM variance exactly.
"""
import numpy as np
from scipy.stats import norm

T_30D = 30 / 365
_X = np.linspace(-4.0, 4.0, 4001)       # log-moneyness grid, k = exp(x)
_K = np.exp(_X)


def _black_otm(k, vol, T):
    """Undiscounted out-of-the-money Black price, forward = 1."""
    s = vol * np.sqrt(T)
    d1 = (-np.log(k) + 0.5 * s ** 2) / s
    d2 = d1 - s
    call = norm.cdf(d1) - k * norm.cdf(d2)
    put = k * norm.cdf(-d2) - norm.cdf(-d1)
    return np.where(k >= 1.0, call, put)


def smile_variance(moneyness, vols, T=T_30D):
    """Annualised implied variance from smile points.

    moneyness : 1-D array of strikes / forward (e.g. [0.9, 1.0, 1.1])
    vols      : array (n_dates, n_points) of implied vols in decimals
    Rows with any missing point fall back to the points available; rows
    with no ATM vol return NaN.
    """
    m = np.asarray(moneyness, float)
    order = np.argsort(m)
    m = m[order]
    v = np.atleast_2d(np.asarray(vols, float))[:, order]
    out = np.full(v.shape[0], np.nan)
    atm = int(np.argmin(np.abs(m - 1.0)))
    for i, row in enumerate(v):
        ok = np.isfinite(row) & (row > 0)
        if not ok[atm]:
            continue
        vk = np.interp(_K, m[ok], row[ok])            # flat beyond the end points
        q = _black_otm(_K, vk, T)
        integrand = q / _K                            # Q(k)/k^2 dk = Q(k)/k dx
        out[i] = 2.0 / T * np.trapezoid(integrand, _X) if hasattr(np, "trapezoid") \
            else 2.0 / T * np.trapz(integrand, _X)
    return out
