# FuelGuard: forecasting volatility with machine learning to hedge diesel costs for small Italian hauliers

Group project, *Introduction to Machine Learning*, MSc in Financial Engineering, EDHEC Business School (2026–2027).

**Product.** A monthly signal, offered by a bank to its SME transport clients. It forecasts next month's
diesel volatility and turns it into (1) a hedge ratio (lock 0%, 50% or 100% of next month's diesel) and
(2) a 95% worst case ("5% chance that next month costs more than X% above the price you can fix today").

**Report:** [`paper/main.pdf`](paper/main.pdf) (8 pages + references + appendix).

## The story in three steps

1. **FuelGuard 1.0 (spike classification, kept in Appendix D).** Predicting whether diesel will jump by >5%
   next month failed. The size of the price move is not predictable beyond the futures price, and the small
   classification skill came from implied volatility alone.
2. **Pivot, following the volatility-forecasting literature** (Corsi 2009; Christensen, Siggaard & Veliyev 2023).
   Direction can't be predicted, but risk can. FuelGuard 2.0 forecasts next month's realised volatility.
3. **FuelGuard 2.0.** The setup is a panel of 22 futures markets (energy, metals, grains, softs, livestock,
   bonds), 109,000 market-days and 30 features including the implied-volatility smile. A purged walk-forward
   covers 15 test years (2012–2026). The ML design was chosen on a 2010–2011 validation period.

## Main results (out of sample, 2012–2026)

| | Result |
|---|---|
| Benchmarks | Random walk, HAR, HAR-IV (HAR + implied vol, per market). HAR-IV is very strong: R² 74% for diesel |
| ML alone (Lasso, RF, GB; pooled, scale-free) | Matches HAR-IV (−0.2%, n.s.); the neural network overfits (−31%) |
| **FuelGuard = HAR-IV + ML ensemble** | **−2.3% MSE vs HAR-IV, DM p < 0.001**; −26% vs HAR, −37% vs random walk; better in 20/22 markets (6 significant, 0 significantly worse) and 13/15 years |
| Diesel only | +0.7% vs HAR-IV, not significant; as good as the best benchmark |
| Hedging (178 months, 1% margin) | Worst-5%-months cost: 16.7% above lockable price if never hedging → **6.6%** with FuelGuard; average cost also lower (0.37% vs 0.88%); ~900 EUR/year saved for 25,000 l/month |
| 95% worst case | 7.9% breaches instead of 5% (Kupiec p = 0.10): slightly optimistic |

What we can't claim: price direction; a large ML gain on diesel alone; that FuelGuard beats HAR-IV in the
hedging backtest (all volatility forecasts give similar economic value). See Section 6.

## Repository structure

```
data/
  raw/bloomberg_futures_panel.csv  22 futures: OHLC, volume, 2nd month, IV smile (2007-2026)
  raw/bloomberg_common.csv         VIX, VIX3M, VSTOXX
  raw/bloomberg_selected.csv       20 series for FuelGuard 1.0 and the diesel cost
  raw/ecb_eurusd.xml               ECB EUR/USD reference rate
  processed/                       built by the scripts
  bloomberg_xlsx/                  (not committed) original Bloomberg Excel exports
src/fuelguard/
  config.py          all parameters (horizon, hedge cost, purge/embargo, seeds)
  panel.py           2.0: loading the 22-market panel
  vol_features.py    2.0: Garman-Klass volatility target + 30 features (HAR, IV smile, VRP, volume...)
  vol_experiment.py  2.0: benchmarks (RW, HAR, HAR-IV per market), pooled ML, walk-forward, importance
  data.py, targets.py, features.py, dataset.py   1.0 and the diesel cost (EUR/litre, futures strip)
  validation.py, models.py, experiment.py, evaluate.py, backtest.py   1.0 models + shared tools
  report.py          figure style and LaTeX table export
scripts/
  00_extract_bloomberg.py   optional: rebuild raw CSVs from the Bloomberg Excel files
  01-03_*.py                FuelGuard 1.0 (spike classification) + diesel cost series
  11_build_vol_dataset.py   2.0 step 1: volatility panel
  10_validation_design.py   2.0 step 2: design selection on 2010-2011 (before any test year)
  12_vol_walk_forward.py    2.0 step 3: walk-forward 2012-2026 (~11 min)
  13_vol_evaluate.py        2.0 step 4: accuracy, DM tests, hedging backtest, 95% bound, figures, tables
  run_all.py                everything in order (~30 min on 2 cores)
tests/                      leakage tests for both versions (features never use the future; purged folds)
results/                    predictions, tables (.csv + .tex), figures (.pdf), key numbers (.json)
paper/                      LaTeX source of the report (reads tables and figures from results/)
```

## How to run

```bash
pip install -r requirements.txt
python scripts/run_all.py          # all steps, about 30 minutes on 2 cores
python -m pytest -q                # leakage tests (10)
cd paper && latexmk -pdf main.tex  # rebuild the report
```

Order: `01 → 02 → 03` (1.0 and diesel cost) then `11 → 10 → 12 → 13` (2.0), or simply `run_all.py`.
Seeds are fixed. A clean clone reproduces every number in the report.

## Evaluation design in one paragraph

Each test year 2012–2026 is forecast by models trained only on market-days whose 21-day label was known
at least 30 days before 1 January of that year. HAR/HAR-IV are fitted per market. ML models are pooled
across markets, use scale-free inputs, train on every 5th day, and are tuned by purged time-series CV
inside the training window. The design (scale-free inputs, ensemble without the neural network,
combination with HAR-IV) was selected on 2010–2011, which precedes all test years. The report states that
a first run on the test years had shown raw ML losing to HAR-IV.

## Data licence

Bloomberg data were downloaded through the EDHEC terminal for academic use; they are included only to make
the project reproducible for the course and should not be redistributed. ECB data are public.
