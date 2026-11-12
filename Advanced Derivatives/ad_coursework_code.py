"""
Advanced Derivatives -- Coursework 2026-27 (Dr J. Kerkhof)
EDHEC M.Sc. in Financial Engineering -- Group X

One single code file for Questions 1-4.  It is split into cells with "# %%"
markers, so it can be run cell by cell in VS Code / Jupyter, or in one go with

    python ad_coursework_code.py

Every figure is saved to ./figures and every table to ./tables, from where the
LaTeX report picks them up.  All random numbers come from fixed seeds, so the
numbers in the report are exactly reproducible.

The lines "# [name_start]" / "# [name_end]" are markers used by the LaTeX
report to print the relevant pieces of this file (they do nothing in Python).
"""

# %% ==========================================================================
# Set-up
# =============================================================================
import os
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# [imports_start]
# Question 1(a): call the FinancePy *model* functions directly instead of
# building EquityVanillaOption objects -- this is roughly 60x faster.
# Older FinancePy releases call them bs_value / bs_delta / bs_gamma; the
# current release (1.1.x) calls them value / delta / gamma.  We support both.
try:
    from financepy.models.black_scholes_analytic import (
        bs_value, bs_delta, bs_gamma)
except ImportError:
    from financepy.models.black_scholes_analytic import (
        value as bs_value, delta as bs_delta, gamma as bs_gamma)
from financepy.utils.global_types import OptionTypes

PUT = OptionTypes.EUROPEAN_PUT.value     # integer flag the model functions use

# Example direct calls (S, T, K, r, q, sigma, option type)
put_value = bs_value(100.0, 1.0, 100.0, 0.04, 0.0, 0.20, PUT)
put_delta = bs_delta(100.0, 1.0, 100.0, 0.04, 0.0, 0.20, PUT)
put_gamma = bs_gamma(100.0, 1.0, 100.0, 0.04, 0.0, 0.20, PUT)
# [imports_end]

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
COLORS = {12: "#d62728", 52: "#1f77b4", 252: "#2ca02c"}


def save_table(df, name, float_format="%.4f", index=False):
    """Write a pandas DataFrame as a booktabs LaTeX tabular (no float)."""
    tex = df.to_latex(index=index, float_format=float_format, escape=False)
    with open(os.path.join("tables", name + ".tex"), "w") as f:
        f.write(tex)
    print(df.to_string(index=index), "\n")


print(f"Q1(a) check: put value {put_value:.4f}, delta {put_delta:.4f}, "
      f"gamma {put_gamma:.5f}")

# %% ==========================================================================
# QUESTION 1 -- Black-Scholes-Merton hedging
# =============================================================================

# [deltahedge_start]
def DeltaHedge(K, S, r, mu, sigma, T, N, sigma_real=None, q=0.0, rng=None):
    """Delta hedge of a European PUT with a self-financing portfolio.
    mu = real-world drift, sigma = vol used to price/hedge, sigma_real = vol
    of the simulated path (default sigma), N = hedges per year.
    Returns (S_T, payoff, realised_variance, replication_error, avg_gamma),
    replication_error = hedge portfolio value at T - option payoff."""
    rng = np.random.default_rng() if rng is None else rng
    sigma_real = sigma if sigma_real is None else sigma_real

    n_steps = int(round(N * T))            # number of hedging intervals
    dt = T / n_steps
    z = rng.standard_normal(n_steps)       # one shock per interval

    # t = 0: receive the BS premium and buy delta shares (delta < 0 for a put)
    stock = S
    delta = bs_delta(stock, T, K, r, q, sigma, PUT)
    gammas = [bs_gamma(stock, T, K, r, q, sigma, PUT)]
    cash = bs_value(stock, T, K, r, q, sigma, PUT) - delta * stock
    log_returns = np.empty(n_steps)

    for i in range(n_steps):
        # 1) the stock moves under the REAL-world lognormal dynamics
        new_stock = stock * np.exp((mu - 0.5 * sigma_real**2) * dt
                                   + sigma_real * np.sqrt(dt) * z[i])
        log_returns[i] = np.log(new_stock / stock)
        stock = new_stock

        # 2) cash accrues interest; dividends on the shares we hold are received
        cash = cash * np.exp(r * dt) + delta * stock * (np.exp(q * dt) - 1.0)

        # 3) rebalance to the new delta (not on the expiry date itself)
        tau = T - (i + 1) * dt              # remaining time to expiry
        if i < n_steps - 1:
            new_delta = bs_delta(stock, tau, K, r, q, sigma, PUT)
            cash -= (new_delta - delta) * stock   # self-financing trade
            delta = new_delta
            gammas.append(bs_gamma(stock, tau, K, r, q, sigma, PUT))

    portfolio = delta * stock + cash
    payoff = max(K - stock, 0.0)
    realised_variance = np.sum(log_returns**2) / T   # annualised
    return stock, payoff, realised_variance, portfolio - payoff, np.mean(gammas)
# [deltahedge_end]


# [runpaths_start]
def run_hedge_paths(hedge_fn=None, n_paths=10_000, seed=42, **params):
    """Run a hedging function (default DeltaHedge) on n_paths independent
    paths; returns a DataFrame with one row per path."""
    hedge_fn = DeltaHedge if hedge_fn is None else hedge_fn
    rng = np.random.default_rng(seed)       # same seed -> same shocks
    rows = [hedge_fn(rng=rng, **params) for _ in range(n_paths)]
    df = pd.DataFrame(rows, columns=["S_T", "payoff", "realised_var",
                                     "error", "avg_gamma"])
    df["realised_vol"] = np.sqrt(df["realised_var"])
    return df
# [runpaths_end]


# ---- 1(d)-(g): S0=100, K=100, r=4%, T=1, sigma=20%, mu=8% -----------------
BASE = dict(K=100.0, S=100.0, r=0.04, mu=0.08, sigma=0.20, T=1.0)
N_LIST = [12, 52, 252]

t0 = time.time()
q1 = {N: run_hedge_paths(N=N, **BASE) for N in N_LIST}
print(f"Q1 simulations done in {time.time() - t0:.0f}s")

# 1(d) hedging error versus terminal stock price
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for N in N_LIST:
    ax.scatter(q1[N]["S_T"], q1[N]["error"], s=2, alpha=0.35,
               color=COLORS[N], label=f"N = {N}")
ax.axvline(100, color="k", lw=0.6, ls="--")
ax.set_xlabel("Terminal stock price $S(T)$")
ax.set_ylabel("Hedging error (portfolio $-$ payoff)")
ax.set_xlim(40, 200)
ax.legend(markerscale=5)
fig.tight_layout()
fig.savefig("figures/q1_error_vs_ST.pdf", bbox_inches="tight")
plt.close(fig)

# 1(e) mean and variance of the hedging error, plus the Derman-Kamal
# approximation for its standard deviation: sqrt(pi/4) * vega * sigma / sqrt(N)
d1 = (np.log(1.0) + (0.04 + 0.5 * 0.2**2)) / 0.2
vega = 100 * norm.pdf(d1) * 1.0
q1_moments = pd.DataFrame({
    "$N$": N_LIST,
    "Mean": [q1[N]["error"].mean() for N in N_LIST],
    "Variance": [q1[N]["error"].var() for N in N_LIST],
    "Std.\\ dev.": [q1[N]["error"].std() for N in N_LIST],
    "Theory std.": [np.sqrt(np.pi / 4) * vega * 0.2 / np.sqrt(N)
                    for N in N_LIST],
})
save_table(q1_moments, "q1_moments")

# 1(f) replication error versus realised volatility
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for N in N_LIST:
    ax.scatter(q1[N]["realised_vol"], q1[N]["error"], s=2, alpha=0.35,
               color=COLORS[N], label=f"N = {N}")
ax.axvline(0.20, color="k", lw=0.6, ls="--")
ax.set_xlabel("Realised volatility (annualised)")
ax.set_ylabel("Replication error")
ax.legend(markerscale=5)
fig.tight_layout()
fig.savefig("figures/q1_error_vs_realvol.pdf", bbox_inches="tight")
plt.close(fig)

# 1(g) replication error versus average gamma (N = 52), coloured by the
# realised volatility of the path
d = q1[52]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
sc = ax.scatter(d["avg_gamma"], d["error"], s=3, c=d["realised_vol"],
                cmap="coolwarm", vmin=0.15, vmax=0.25)
fig.colorbar(sc, ax=ax, label="Realised volatility")
ax.set_xlabel(r"Average gamma along the path, $\bar\Gamma$")
ax.set_ylabel("Replication error")
fig.tight_layout()
fig.savefig("figures/q1_error_vs_gamma.pdf", bbox_inches="tight")
plt.close(fig)

# Quantify the patterns with simple regressions (used in the text)
for N in N_LIST:
    c_vol = np.corrcoef(q1[N]["realised_var"], q1[N]["error"])[0, 1]
    print(f"N={N:3d}: corr(error, realised variance) = {c_vol:.3f}")
# Second-order (Taylor) approximation: error ~ sum 0.5*Gamma*S^2*(sigma^2 -
# realised^2)*dt.  Its 'average' version uses the path-average gamma:
d = q1[52]
approx = 0.5 * d["avg_gamma"] * 100**2 * (0.2**2 - d["realised_var"])
print("N=52: corr(error, 0.5*avgGamma*S0^2*(sigma^2-realised var)) = "
      f"{np.corrcoef(approx, d['error'])[0, 1]:.3f}")
hi_g = d["avg_gamma"] > d["avg_gamma"].quantile(0.75)
lo_g = d["avg_gamma"] < d["avg_gamma"].quantile(0.25)
print(f"N=52: error std, top-quartile gamma {d.loc[hi_g, 'error'].std():.3f} "
      f"vs bottom-quartile gamma {d.loc[lo_g, 'error'].std():.3f}")

# ---- 1(h) sensitivity to the drift (N = 52, 10,000 paths) -----------------
# The same seed is used for every mu (common random numbers), so the only
# thing that changes between the rows is the drift.
MU_LIST = [0.025, 0.050, 0.075, 0.100]
q1_drift = {mu: run_hedge_paths(N=52, **{**BASE, "mu": mu}) for mu in MU_LIST}
q1_drift_tab = pd.DataFrame({
    "$\\mu$": [f"{100 * mu:.1f}\\%" for mu in MU_LIST],
    "Mean error": [q1_drift[mu]["error"].mean() for mu in MU_LIST],
    "Mean abs.\\ error": [q1_drift[mu]["error"].abs().mean() for mu in MU_LIST],
    "Variance": [q1_drift[mu]["error"].var() for mu in MU_LIST],
    "Mean $S(T)$": [q1_drift[mu]["S_T"].mean() for mu in MU_LIST],
})
save_table(q1_drift_tab, "q1_drift")

# PLAIN ENGLISH (Q1):
# * The hedge works on average (mean error ~ 0) but is never perfect because
#   we only rebalance N times a year.  The error's standard deviation roughly
#   halves every time N is multiplied by 4 (it scales like 1/sqrt(N)).
# * Paths whose realised volatility was ABOVE the 20% used for pricing lose
#   money (negative error): the option is long gamma, the hedge is not, and we
#   were paid a premium for 20% vol only.  Below 20% we make money.
# * The size of the error is driven by gamma: paths that spend a lot of time
#   near the strike (high average gamma) have the largest errors.
# * The drift hardly matters: delta hedging removes exposure to the direction
#   of the stock, so changing mu from 2.5% to 10% changes the error statistics
#   only marginally.


# %% ==========================================================================
# QUESTION 2 -- Time-varying volatility
# =============================================================================

# [tvhelpers_start]
def integrated_variance(t1, t2, vols, T):
    """
    Integral of sigma(u)^2 du from t1 to t2 for a piecewise-constant
    volatility: vols[j] applies on the j-th of len(vols) equal sub-periods
    of [0, T].
    """
    edges = np.linspace(0.0, T, len(vols) + 1)
    total = 0.0
    for j, v in enumerate(vols):
        overlap = max(0.0, min(t2, edges[j + 1]) - max(t1, edges[j]))
        total += v**2 * overlap
    return total


def remaining_vol(t, vols, T):
    """Volatility to price with at time t: the root-mean-square (average)
    volatility over the REMAINING life [t, T] of the option."""
    tau = T - t
    return np.sqrt(integrated_variance(t, T, vols, T) / tau)
# [tvhelpers_end]


# 2(a) put value: piecewise vol (4% then 28%) versus flat 20%
VOLS_LOW_FIRST = (0.04, 0.28)
VOLS_HIGH_FIRST = (0.28, 0.04)
sigma_eff = remaining_vol(0.0, VOLS_LOW_FIRST, 1.0)
p_flat = bs_value(100.0, 1.0, 100.0, 0.04, 0.0, 0.20, PUT)
p_tv = bs_value(100.0, 1.0, 100.0, 0.04, 0.0, sigma_eff, PUT)
# independent Monte-Carlo check under the risk-neutral measure (1,000,000 paths)
rng = np.random.default_rng(7)
z1, z2 = rng.standard_normal((2, 1_000_000))
logS = (np.log(100) + (0.04 - 0.5 * 0.04**2) * 0.5 + 0.04 * np.sqrt(0.5) * z1
        + (0.04 - 0.5 * 0.28**2) * 0.5 + 0.28 * np.sqrt(0.5) * z2)
disc_pay = np.exp(-0.04) * np.maximum(100 - np.exp(logS), 0)
q2_values = pd.DataFrame({
    "Volatility": ["Flat 20\\%", "4\\% then 28\\% (BS with $\\bar\\sigma$)",
                   "4\\% then 28\\% (MC, $10^6$ paths)"],
    "$\\bar\\sigma$": [0.20, sigma_eff, sigma_eff],
    "Put value": [p_flat, p_tv, disc_pay.mean()],
    "Std.\\ error": [0.0, 0.0, disc_pay.std() / np.sqrt(len(disc_pay))],
})
save_table(q2_values, "q2_values")


# [deltahedgetv_start]
def DeltaHedgeTV(K, S, r, mu, vols, T, N, q=0.0, rng=None, naive=False):
    """DeltaHedge with piecewise-constant vol (vols on equal sub-periods).
    Path: INSTANTANEOUS variance per step.  Pricing/hedging: AVERAGE vol over
    the REMAINING life.  naive=True (wrongly) hedges with today's spot vol.
    Returns the same 5-tuple as DeltaHedge."""
    rng = np.random.default_rng() if rng is None else rng
    n_steps = int(round(N * T))
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    edges = np.linspace(0.0, T, len(vols) + 1)

    def hedge_vol(t):
        if naive:   # instantaneous vol at time t
            return vols[min(np.searchsorted(edges, t, side="right") - 1,
                            len(vols) - 1)]
        return remaining_vol(t, vols, T)

    stock, t = S, 0.0
    sig = hedge_vol(t)
    delta = bs_delta(stock, T, K, r, q, sig, PUT)
    gammas = [bs_gamma(stock, T, K, r, q, sig, PUT)]
    cash = bs_value(stock, T, K, r, q, remaining_vol(0.0, vols, T), PUT) \
        - delta * stock                     # premium is always the fair value
    log_returns = np.empty(n_steps)

    for i in range(n_steps):
        var_step = integrated_variance(t, t + dt, vols, T)  # true variance
        new_stock = stock * np.exp(mu * dt - 0.5 * var_step
                                   + np.sqrt(var_step) * z[i])
        log_returns[i] = np.log(new_stock / stock)
        stock, t = new_stock, t + dt
        cash = cash * np.exp(r * dt) + delta * stock * (np.exp(q * dt) - 1.0)
        if i < n_steps - 1:
            tau = T - t
            sig = hedge_vol(t)
            new_delta = bs_delta(stock, tau, K, r, q, sig, PUT)
            cash -= (new_delta - delta) * stock
            delta = new_delta
            gammas.append(bs_gamma(stock, tau, K, r, q, sig, PUT))

    portfolio = delta * stock + cash
    payoff = max(K - stock, 0.0)
    return (stock, payoff, np.sum(log_returns**2) / T, portfolio - payoff,
            np.mean(gammas))
# [deltahedgetv_end]


# 2(c)-(d): repeat the hedging experiment of 1(d) for the two orderings.
# Same seed as in Q1 -> the three cases use exactly the same random shocks.
TV = dict(K=100.0, S=100.0, r=0.04, mu=0.08, T=1.0)
t0 = time.time()
q2_low = {N: run_hedge_paths(DeltaHedgeTV, N=N, vols=VOLS_LOW_FIRST, **TV)
          for N in N_LIST}
q2_high = {N: run_hedge_paths(DeltaHedgeTV, N=N, vols=VOLS_HIGH_FIRST, **TV)
           for N in N_LIST}
q2_naive = run_hedge_paths(DeltaHedgeTV, N=52, vols=VOLS_LOW_FIRST,
                           naive=True, **TV)
print(f"Q2 simulations done in {time.time() - t0:.0f}s")

cases = {"Constant 20\\%": q1, "Low first (4\\%, 28\\%)": q2_low,
         "High first (28\\%, 4\\%)": q2_high}
rows = []
for name, res in cases.items():
    for N in N_LIST:
        e = res[N]["error"]
        rows.append([name, N, e.mean(), e.var(), e.abs().mean(),
                     res[N]["avg_gamma"].mean()])
q2_compare = pd.DataFrame(rows, columns=["Case", "$N$", "Mean", "Variance",
                                         "Mean abs.", "Avg.\\ $\\bar\\Gamma$"])
save_table(q2_compare[q2_compare["Case"] != "High first (28\\%, 4\\%)"],
           "q2_compare")
save_table(q2_compare[q2_compare["Case"] == "High first (28\\%, 4\\%)"],
           "q2_reversed")
print(f"Naive hedge (instantaneous vol), low first, N=52: mean "
      f"{q2_naive['error'].mean():.3f}, variance {q2_naive['error'].var():.3f}")
naive_row = pd.DataFrame([["Low first, hedged with \\emph{instantaneous} vol",
                           52, q2_naive["error"].mean(), q2_naive["error"].var(),
                           q2_naive["error"].abs().mean(),
                           q2_naive["avg_gamma"].mean()]],
                         columns=q2_compare.columns)
save_table(naive_row, "q2_naive")

# Figure: hedging error vs S(T), N = 52, the three volatility cases
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (name, res), col in zip(axes, cases.items(),
                                ["#1f77b4", "#ff7f0e", "#9467bd"]):
    ax.scatter(res[52]["S_T"], res[52]["error"], s=1.5, alpha=0.35, color=col)
    ax.set_title(name.replace("\\%", "%"), fontsize=9)
    ax.set_xlabel("$S(T)$")
    ax.set_xlim(40, 200)
    ax.axvline(100, color="k", lw=0.6, ls="--")
axes[0].set_ylabel("Hedging error")
fig.tight_layout()
fig.savefig("figures/q2_error_vs_ST.pdf", bbox_inches="tight")
plt.close(fig)

# Where does the error come from?  Error variance by the half of the life in
# which the stock moves (diagnostic used in the discussion).
for name, res in cases.items():
    print(name, "N=52 error std:", round(res[52]["error"].std(), 3))

# PLAIN ENGLISH (Q2):
# * Only the total variance over the option's life matters for a European
#   option: 0.5*4%^2 + 0.5*28%^2 = 20%^2, so both prices are identical.
# * When hedging we must use the average vol over the REMAINING life, which
#   drifts away from 20% as time passes.  With that, the hedge is still
#   unbiased.  Hedging with today's spot vol instead gives a large error.
# * How large the error is depends on WHEN the volatility arrives: what
#   matters is gamma x (variance per period) at each rebalancing date.


# %% ==========================================================================
# QUESTION 3 -- Asian options and the control variate technique
# =============================================================================

# [asianpaths_start]
def simulate_averaging_prices(S0, r, q, sigma, T, T0, n_obs, P, rng):
    """Risk-neutral GBM prices on the averaging dates T0+dt, ..., T: one jump
    from 0 to T0, then n_obs daily steps.  Returns a (P x n_obs) array."""
    dt = (T - T0) / n_obs
    drift = r - q - 0.5 * sigma**2
    z0 = rng.standard_normal(P)                    # shock over [0, T0]
    z = rng.standard_normal((P, n_obs))            # daily shocks
    log_s_T0 = np.log(S0) + drift * T0 + sigma * np.sqrt(T0) * z0
    log_s = log_s_T0[:, None] + np.cumsum(drift * dt + sigma * np.sqrt(dt) * z,
                                          axis=1)
    return np.exp(log_s)


def asian_arithmetic_mc(S0, K, r, sigma, T=0.5, T0=0.25, n_obs=90, P=10_000,
                        q=0.0, seed=1):
    """Q3(a): Monte-Carlo price of an arithmetic-average Asian CALL.
    Returns (price, standard error)."""
    rng = np.random.default_rng(seed)
    prices = simulate_averaging_prices(S0, r, q, sigma, T, T0, n_obs, P, rng)
    arith_avg = prices.mean(axis=1)
    disc_payoff = np.exp(-r * T) * np.maximum(arith_avg - K, 0.0)
    return disc_payoff.mean(), disc_payoff.std(ddof=1) / np.sqrt(P)
# [asianpaths_end]


# [asianfull_start]
def asian_mc(S0, K, r, sigma, T=0.5, T0=0.25, n_obs=90, P=10_000, q=0.0,
             seed=1, return_payoffs=False):
    """Arithmetic AND geometric Asian calls on the SAME paths.  Returns
    [geo price, arith price, var geo payoff, var arith payoff, covariance]
    (per-path variances of discounted payoffs; estimate variance = var/P)."""
    rng = np.random.default_rng(seed)
    prices = simulate_averaging_prices(S0, r, q, sigma, T, T0, n_obs, P, rng)
    arith_avg = prices.mean(axis=1)
    geo_avg = np.exp(np.log(prices).mean(axis=1))
    disc = np.exp(-r * T)
    pay_a = disc * np.maximum(arith_avg - K, 0.0)
    pay_g = disc * np.maximum(geo_avg - K, 0.0)
    cov = np.cov(pay_g, pay_a, ddof=1)             # 2x2 covariance matrix
    out = np.array([pay_g.mean(), pay_a.mean(), cov[0, 0], cov[1, 1],
                    cov[0, 1]])
    if return_payoffs:
        return out, pay_g, pay_a, prices
    return out
# [asianfull_end]


# [geoexact_start]
def geometric_asian_exact(S0, K, r, sigma, T, T0, n_obs, q=0.0):
    """Exact price and delta of the DISCRETE geometric Asian call: log G is
    normal, mean m = log S0 + (r-q-sigma^2/2) mean(t_i),
    variance v = sigma^2/n^2 sum_ij min(t_i, t_j) -> Black formula."""
    t = T0 + (T - T0) / n_obs * np.arange(1, n_obs + 1)
    m = np.log(S0) + (r - q - 0.5 * sigma**2) * t.mean()
    v = sigma**2 * np.minimum.outer(t, t).sum() / n_obs**2
    d1 = (m - np.log(K) + v) / np.sqrt(v)
    d2 = d1 - np.sqrt(v)
    fwd_g = np.exp(m + 0.5 * v)                    # E[G]
    price = np.exp(-r * T) * (fwd_g * norm.cdf(d1) - K * norm.cdf(d2))
    delta = np.exp(-r * T) * fwd_g * norm.cdf(d1) / S0
    return price, delta


def kemna_vorst(S0, K, r, sigma, T, T0, q=0.0):
    """Continuous-averaging geometric Asian call (Kemna & Vorst / lecture 4)."""
    M = np.log(S0) + (r - q - 0.5 * sigma**2) * (T0 + 0.5 * (T - T0))
    V = sigma**2 * (T0 + (T - T0) / 3.0)
    d1 = (M - np.log(K) + V) / np.sqrt(V)
    return np.exp(-r * T) * (np.exp(M + 0.5 * V) * norm.cdf(d1)
                             - K * norm.cdf(d1 - np.sqrt(V)))


def control_variate(stats, geo_exact):
    """Optimal control-variate estimate of the arithmetic price from the
    5 outputs of asian_mc and the exact geometric price.
    A_cv = A_bar - beta (G_bar - G_exact),  beta = Cov(A,G) / Var(G).
    Returns (estimate, per-path variance of the CV estimator, beta, rho)."""
    g, a, var_g, var_a, cov = stats
    beta = cov / var_g
    est = a - beta * (g - geo_exact)
    var_cv = var_a - cov**2 / var_g                 # = var_a * (1 - rho^2)
    rho = cov / np.sqrt(var_g * var_a)
    return est, var_cv, beta, rho
# [geoexact_end]


# 3(b) S0=100, K=100, r=4%, sigma=25%, P=10,000
p3b, se3b = asian_arithmetic_mc(100, 100, 0.04, 0.25, P=10_000)
print(f"Q3(b): arithmetic Asian = {p3b:.4f} (s.e. {se3b:.4f})")

# 3(e) S0=100, r=5%, q=1%, sigma=30%, K=100, P=5,000 (averaging from 3m)
P3E = dict(S0=100, K=100, r=0.05, q=0.01, sigma=0.30, T=0.5)
stats_e = asian_mc(**P3E, T0=0.25, n_obs=90, P=5_000)
q3e = pd.DataFrame({
    "Quantity": ["Geometric price", "Arithmetic price",
                 "Var.\\ disc.\\ geometric payoff",
                 "Var.\\ disc.\\ arithmetic payoff", "Covariance",
                 "Std.\\ error geometric price",
                 "Std.\\ error arithmetic price"],
    "Value": list(stats_e) + [np.sqrt(stats_e[2] / 5_000),
                              np.sqrt(stats_e[3] / 5_000)],
})
save_table(q3e, "q3_results")

# 3(f) control variate.  Main case: T0 = 0, i.e. averaging starts today and
# runs daily (180 days) to expiry.  We also show the original contract
# (averaging from 3 months, 90 days) -- the exact discrete formula covers both.
rows = []
for label, T0, n_obs in [("$T_0=0$ (180 days)", 0.0, 180),
                         ("$T_0=0.25$ (90 days)", 0.25, 90)]:
    st = asian_mc(**P3E, T0=T0, n_obs=n_obs, P=5_000)
    g_exact, _ = geometric_asian_exact(**P3E, T0=T0, n_obs=n_obs)
    est, var_cv, beta, rho = control_variate(st, g_exact)
    rows.append([label, g_exact, kemna_vorst(**P3E, T0=T0), st[0], st[1],
                 np.sqrt(st[3] / 5_000), est, np.sqrt(var_cv / 5_000), beta,
                 rho])
q3f = pd.DataFrame(rows, columns=["Contract", "$G$ exact", "$G$ K--V",
                                  "$G$ MC", "$A$ MC", "s.e.", "$A$ CV",
                                  "s.e.\\ CV", "$\\beta$", "$\\rho$"])
save_table(q3f, "q3_cv")

# 3(g) how many paths does the CV method need to match the plain simulation
# with P = 10,000?  Var(estimate) = per-path variance / P, so
#   P_cv = 10,000 * Var_cv / Var_plain = 10,000 * (1 - rho^2).
rows = []
for label, kw in [("3(b): $r=4\\%$, $\\sigma=25\\%$, $T_0=0.25$",
                   dict(S0=100, K=100, r=0.04, q=0.0, sigma=0.25, T=0.5,
                        T0=0.25, n_obs=90)),
                  ("3(f): $r=5\\%$, $q=1\\%$, $\\sigma=30\\%$, $T_0=0$",
                   dict(**P3E, T0=0.0, n_obs=180))]:
    st = asian_mc(**kw, P=10_000, seed=11)
    g_exact, _ = geometric_asian_exact(**kw)
    _, var_cv, _, rho = control_variate(st, g_exact)
    rows.append([label, st[3], var_cv, rho, st[3] / var_cv,
                 int(np.ceil(10_000 * var_cv / st[3]))])
q3g = pd.DataFrame(rows, columns=["Setting", "Var plain", "Var CV", "$\\rho$",
                                  "Reduction", "Paths needed"])
save_table(q3g, "q3_paths", float_format="%.4f")

# PLAIN ENGLISH (Q3):
# * The geometric average is always below the arithmetic one, so the
#   geometric call is cheaper -- but the two payoffs move almost one for one
#   (correlation ~0.99+).
# * Because we know the geometric price exactly, we can measure the MC error
#   on the geometric option on each run and remove it from the arithmetic
#   price.  This cuts the variance by a factor 1/(1-rho^2) ~ 100+, i.e. a few
#   dozen paths with the control variate are as accurate as 10,000 without.


# %% ==========================================================================
# QUESTION 4 -- The delta of the Asian option
# =============================================================================
# Original contract with the 3(e) market data: S0=100, K=100, r=5%, q=1%,
# sigma=30%, T=0.5, averaging daily over the last 90 days (T0 = 0.25).
P4 = dict(K=100, r=0.05, q=0.01, sigma=0.30, T=0.5, T0=0.25, n_obs=90)
P4_PATHS = 10_000

# [fddelta_start]
def asian_fd_delta(h, S0=100.0, P=P4_PATHS, seed=21, common=True):
    """Forward FD delta (f(S0+h) - f(S0)) / h.  common=True re-uses the SAME
    random numbers (same seed) for both prices; False uses independent ones."""
    base = asian_mc(S0, P=P, seed=seed, **P4)[1]
    bumped = asian_mc(S0 + h, P=P, seed=seed if common else seed + 1000,
                      **P4)[1]
    return (bumped - base) / h
# [fddelta_end]


H_LIST = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
fd_common = [asian_fd_delta(h) for h in H_LIST]
fd_indep = [asian_fd_delta(h, common=False) for h in H_LIST]


# [pathwise_start]
def asian_pathwise_delta(S0=100.0, P=P4_PATHS, seed=21):
    """Pathwise delta samples exp(-rT) 1(avg > K) avg / S0 for the arithmetic
    and geometric averages (prices are proportional to S0)."""
    _, _, _, prices = asian_mc(S0, P=P, seed=seed, return_payoffs=True, **P4)
    disc = np.exp(-P4["r"] * P4["T"])
    a_bar = prices.mean(axis=1)
    g_bar = np.exp(np.log(prices).mean(axis=1))
    d_arith = disc * (a_bar > P4["K"]) * a_bar / S0
    d_geo = disc * (g_bar > P4["K"]) * g_bar / S0
    return d_arith, d_geo


d_arith, d_geo = asian_pathwise_delta()
delta_pw = d_arith.mean()
delta_pw_se = d_arith.std(ddof=1) / np.sqrt(P4_PATHS)
# [pathwise_end]
print(f"Q4(b): pathwise delta = {delta_pw:.4f} (s.e. {delta_pw_se:.4f})")

q4_fd = pd.DataFrame({"$h$": H_LIST, "Common RN": fd_common,
                      "Independent RN": fd_indep})
save_table(q4_fd, "q4_fd", float_format="%.4f")

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.semilogx(H_LIST, fd_common, "o-", label="Common random numbers")
ax.semilogx(H_LIST, fd_indep, "s--", label="Independent random numbers")
ax.axhline(delta_pw, color="k", lw=0.8, label="Pathwise estimate (4b)")
ax.set_ylim(-1.0, 2.0)
ax.set_xlabel("Bump size $h$ (log scale)")
ax.set_ylabel("Delta estimate")
ax.legend()
fig.tight_layout()
fig.savefig("figures/q4_fd_delta.pdf", bbox_inches="tight")
plt.close(fig)

# 4(c) control variate for the delta: the pathwise delta of the GEOMETRIC
# option has a known expectation (analytic geometric delta).
geo_price_exact, geo_delta_exact = geometric_asian_exact(100, **P4)


def cv_summary(y_a, y_g, mu_g):
    """Plain and control-variate estimates from per-path samples."""
    c = np.cov(y_g, y_a, ddof=1)
    beta, rho = c[0, 1] / c[0, 0], c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    est_cv = y_a.mean() - beta * (y_g.mean() - mu_g)
    var_cv = c[1, 1] * (1 - rho**2)
    n = len(y_a)
    return [y_a.mean(), np.sqrt(c[1, 1] / n), est_cv, np.sqrt(var_cv / n),
            rho, c[1, 1] / var_cv]


_, pay_g, pay_a, _ = asian_mc(100, P=P4_PATHS, seed=21, return_payoffs=True,
                              **P4)
q4_cv = pd.DataFrame(
    [["Price"] + cv_summary(pay_a, pay_g, geo_price_exact),
     ["Delta"] + cv_summary(d_arith, d_geo, geo_delta_exact)],
    columns=["", "Plain", "s.e.", "CV", "s.e.\\ CV", "$\\rho$",
             "Var.\\ reduction"])
save_table(q4_cv, "q4_cv")
mismatch = np.mean((pay_a > 0) != (pay_g > 0))
print(f"Q4(c): share of paths where only one of A>K, G>K holds: "
      f"{mismatch:.4f}")

# PLAIN ENGLISH (Q4):
# * Bump-and-revalue only works with the SAME random numbers for both prices;
#   with independent numbers the MC noise is divided by h and explodes for
#   small bumps.  With common numbers the estimate is stable and for small h
#   converges to the pathwise estimate.
# * The pathwise estimator gives the delta from ONE simulation, no bumping.
# * The geometric control variate still helps for the delta, but much less
#   than for the price: the delta payoff contains the indicator 1(avg > K),
#   which jumps, and the arithmetic and geometric indicators do not always
#   agree, so the correlation is lower.

print("\nAll figures saved to ./figures and tables to ./tables")
