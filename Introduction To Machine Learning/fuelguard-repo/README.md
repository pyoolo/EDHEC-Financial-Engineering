# FuelGuard — machine-learning signals to hedge diesel costs for small Italian hauliers

Group project, *Introduction to Machine Learning*, MSc in Financial Engineering, EDHEC Business School (2026–2027).

**Product.** A monthly signal, offered by a bank to its SME transport clients, that estimates the
probability that next month's diesel cost will exceed the price that can be locked today by more than 5%
(a *spike*), and recommends a hedge ratio (0%, 50% or 100%) for next month's consumption.

**Report:** [`paper/main.pdf`](paper/main.pdf) (9 pages + references + appendix).

## Main results (out of sample, walk-forward 2012–2026)

| | Result |
|---|---|
| Next month's premium over the futures price (regression) | **Not predictable**: best OOS R² vs futures 0.5%, no Diebold–Mariano test significant |
| Spikes (classification) | **Modestly predictable**: AUC 0.59–0.69; Brier skill +4.1% (random forest) and +5.1% (implied-volatility logistic), 90% bootstrap intervals above zero |
| What drives it | Mostly **volatility**: implied volatility alone (AUC 0.71) matches or beats the 57-feature models |
| Hedging backtest (178 months, 1% hedge margin) | Never hedging: CVaR95 of cost overrun 16.7%. ML rules: 9.4–11.3% while hedging 21–26% of months. Simple volatility rule: 6.8% |

What we can't claim: we don't beat the market on price, and ML doesn't beat a simple volatility rule.
See Section 6 of the report.

## Repository structure

```
data/
  raw/bloomberg_selected.csv   20 Bloomberg series, daily 2007-2026 (see data/README.md)
  raw/ecb_eurusd.xml           ECB EUR/USD reference rate
  processed/                   built by step 1 (panel, model dataset)
  bloomberg_xlsx/              (not committed) original Bloomberg Excel exports
src/fuelguard/
  config.py        all parameters: horizon, spike threshold, hedge cost, purge/embargo, seeds
  data.py          loading Bloomberg + ECB, daily panel, diesel in EUR/litre
  targets.py       market cost, lockable futures strip, premium and spike labels
  features.py      57 backward-looking features in 8 groups
  dataset.py       panel -> targets -> features
  validation.py    expanding walk-forward with purging + embargo; purged inner CV
  models.py        logistic / ridge, random forest, gradient boosting + grids
  experiment.py    yearly re-tuning and out-of-sample predictions, permutation importance
  evaluate.py      AUC, Brier skill, calibration, Diebold-Mariano (Newey-West), block bootstrap
  backtest.py      monthly hedging rules, cost overrun, CVaR, EUR figures
  report.py        figure style and LaTeX table export
scripts/
  00_extract_bloomberg.py   optional: rebuild the CSV from the Bloomberg Excel files
  01_build_dataset.py       step 1
  02_walk_forward.py        step 2 (~10-15 min)
  03_evaluate.py            step 3: tables, figures, backtest
  run_all.py                steps 1-3 in order
tests/test_leakage.py       automated checks that no future information enters features or training sets
results/                    predictions, tables (.csv + .tex), figures (.pdf), key_numbers.json
paper/                      LaTeX source of the report; tables and figures are read from results/
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python scripts/run_all.py          # steps 1 -> 3, about 15 minutes on 2 cores
python -m pytest -q                # leakage tests
cd paper && latexmk -pdf main.tex  # rebuild the report (needs a LaTeX distribution)
```

Order of execution: `01_build_dataset.py` → `02_walk_forward.py` → `03_evaluate.py` (or `run_all.py`).
Results are deterministic (fixed seeds); a full rerun reproduces every number in the report.

## Evaluation design in one paragraph

Decision at the close of day *t*, executed at *t+1*; the hedged month is *t+2 … t+22*. For each test
year 2012–2026, models are trained on every day whose label is fully known before 1 January of that year,
minus a 21-day embargo. Hyper-parameters are chosen by a purged three-fold time-series CV inside the
training window. The test year is never used for tuning. The parsimonious one-feature model was added after
the first run, and the report says so.

## Data licence

Bloomberg data were downloaded through the EDHEC terminal for academic use; they are included only to make
the project reproducible for the course and should not be redistributed. ECB data are public.
