# Volatility timing in commodity markets

Code for the EDHEC research thesis *Harvesting the Commodity Volatility
Premium: Does Volatility Timing Survive Supply Shocks?* (supervisor:
Prof. Enrique Schroth). It tests whether simple, real-time timing rules
improve a short variance-swap strategy on commodity futures. The same code
runs on any supported commodity; the first draft covers WTI crude oil.

**Data are not included.** The analysis uses licensed Bloomberg data; see
[`data/README.md`](data/README.md) for the file layout and how to export it.
A synthetic-data generator lets anyone run the code without Bloomberg.

## Supported commodities

| Code | Commodity | Exchange | Contract cycle |
|---|---|---|---|
| CL | WTI crude oil | NYMEX | every month |
| NG | Henry Hub natural gas | NYMEX | every month |
| GC | Gold | COMEX | Feb, Apr, Jun, Aug, Oct, Dec |
| SI | Silver | COMEX | Mar, May, Jul, Sep, Dec |
| HG | Copper | COMEX | Mar, May, Jul, Sep, Dec |
| C | Corn | CBOT | Mar, May, Jul, Sep, Dec |
| W | Wheat | CBOT | Mar, May, Jul, Sep, Dec |
| S | Soybeans | CBOT | Jan, Mar, May, Jul, Aug, Sep, Nov |

To add one, add an entry to `SPECS` in `src/commodities.py`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put your Bloomberg exports in data/raw/ as CL.xlsx, GC.xlsx, HG.xlsx, C.xlsx ...
python src/run_all.py --commodity CL               # one commodity
python src/run_all.py --commodity CL GC HG C       # several in a row
python src/run_all.py --data path/to/file.xlsx     # any location; commodity read from cell A1
python src/run_all.py --list                       # supported commodities

# without Bloomberg (simulated data, for testing only)
python scripts/make_synthetic_data.py --commodity GC
python src/run_all.py --commodity GC --data data/raw/synthetic_GC.xlsx
```

Results for commodity `CODE` go to `results/CODE/` (`tables/`, `figures/`,
`numbers.tex`). Then build the paper, which reads `results/CL/` (change
`\cmdty` at the top of `paper/main.tex` for another commodity):

```bash
cd paper
latexmk -pdf main.tex
```

Run the tests (no data needed): `pytest`

## Contract calendar and roll

Generic futures (F1, F2) jump when they switch contract. The code rebuilds
each contract's last trading day and first notice day from exchange rules,
holds a single contract, and rolls 5 trading days before the earlier of the
two dates. Each run writes `results/CODE/tables/tabA1_calendar.csv` and warns
if the F1–F2 spread does not jump where the calendar says it should. If the
rules do not match your data, export the calendar from Bloomberg
(`data/raw/CODE_calendar.csv`, see `data/README.md`) and it is used instead.

## What the code does

| Step | File | |
|---|---|---|
| 1 | `src/load_data.py` | Reads a Bloomberg export (F1, F2, ATM implied vol) |
| 2 | `src/commodities.py` | Commodity specs and contract-calendar rules |
| 3 | `src/rolls.py` | Daily returns of a position that rolls before expiry / first notice |
| 4 | `src/panel.py` | Monthly panel: signals at each month-end, realised variance over the next month |
| 5 | `src/strategies.py` | Short variance-swap proxy and seven timing rules |
| 6 | `src/evaluation.py` | Performance metrics, spanning regressions, block-bootstrap Sharpe tests |
| 7 | `src/plots.py`, `src/run_all.py` | Tables (`.csv`, `.tex`), figures, `numbers.tex` |

All research parameters (roll timing, thresholds, crisis dates, costs) are
in `src/config.py`.

## Repository layout

```
├── data/            # not tracked, except data/README.md
├── paper/           # LaTeX draft and bibliography
├── results/         # generated per commodity (not tracked)
├── scripts/         # synthetic-data generator
├── src/             # analysis code
└── tests/           # pytest tests on synthetic data
```

## Note on the text

Tables, figures and `numbers.tex` update automatically. The sentences that
interpret them in `paper/main.tex` do not: re-read them after changing data
or parameters.
