# Assignment 3: Black–Litterman (QPM 2026-2027, Lecture 5)

Group: LOLLI Alessandro, MASTROGIACOMO Paolo, PERALDI Léandre. Due 25 November 2026.

Six stocks (AAPL, MSFT, AMZN, NVDA, TSLA, META), monthly simple excess returns from January 2015 to December 2022 (risk-free rate = 0).

| Part | Content |
|---|---|
| Q1.1 | Market weights (S&P 500 index weights, rescaled) and implied risk aversion γ_mkt = E[R_mkt]/V[R_mkt] |
| Q1.2 | Markowitz weights (1/γ) Σ⁻¹μ and why they are unreasonable |
| Q1.3 | CAPM-implied expected returns μ_CAPM = γ_mkt Σ w_mkt (reverse optimisation) |
| Q1.4 | Pick matrix P, views q (monthly), Ω = diag(P τΣ P'), τ = 1/T |
| Q1.5 | Posterior μ_BL and Σ_BL |
| Q1.6 | Black–Litterman weights (1/γ) Σ_BL⁻¹ μ_BL |
| Q1.7 | Comparison: Markowitz vs market vs Black–Litterman |

Run: open the notebook in Colab or Jupyter and choose Restart and Run All. Prices are downloaded with `yfinance`.
