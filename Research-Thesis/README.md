# Volatility timing across commodities, rates and credit

Code for the EDHEC research thesis *Harvesting the Commodity Volatility
Premium: Does Volatility Timing Survive Supply Shocks?* (supervisor:
Prof. Enrique Schroth). It tests whether simple, real-time timing rules
improve a short variance-swap strategy. The same code runs on commodity
futures, government bond futures and credit (ETFs and the CDX IG index).

**Data are not included.** The analysis uses licensed Bloomberg data; see
[`data/README.md`](data/README.md) for the file layout and how to export it.
A synthetic-data generator lets anyone run the code without Bloomberg.

## Supported instruments

| Code | Instrument | Exchange | Contract cycle | Strike |
|---|---|---|---|---|
| CL | WTI crude oil | NYMEX | every month | smile |
| NG | Henry Hub natural gas | NYMEX | every month | smile |
| GC | Gold | COMEX | Feb, Apr, Jun, Aug, Oct, Dec | smile |
| SI | Silver | COMEX | Mar, May, Jul, Sep, Dec | smile |
| HG | Copper | COMEX | Mar, May, Jul, Sep, Dec | smile |
| C | Corn | CBOT | Mar, May, Jul, Sep, Dec | smile |
| W | Wheat | CBOT | Mar, May, Jul, Sep, Dec | smile |
| S | Soybeans | CBOT | Jan, Mar, May, Jul, Aug, Sep, Nov | smile |
| TU, FV, TY, US | US 2y, 5y, 10y notes and bond | CBOT | Mar, Jun, Sep, Dec | ATM |
| RX | Euro-Bund | Eurex | Mar, Jun, Sep, Dec | ATM |
| HYG, LQD | High-yield and IG bond ETFs | NYSE Arca | continuous | ATM |
| CDXIG | CDX NA IG 5Y, strike = Cboe VIXIG (bp) | OTC | continuous, series roll excluded | VIXIG |
| SPX | S&P 500 (equity benchmark) | Cboe | continuous | VIX |

Bond futures and ETFs use the ATM strike because 90%/110% moneyness is
many standard deviations from the money for them. For CDX IG, returns are
daily spread changes in basis points and the strike is the VIXIG index,
a model-free 30-day implied volatility of the spread in basis points.
To add a futures contract, add an entry to `SPECS` in `src/commodities.py`
(`SPOT_SPECS` for a continuous series).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put your Bloomberg exports in data/raw/ as CL.xlsx, GC.xlsx, HG.xlsx, C.xlsx ...
python src/run_all.py --commodity CL               # one commodity
python src/run_all.py --commodity CL GC HG C       # several in a row
python src/run_all.py --data path/to/file.xlsx     # one commodity per sheet, read from cell A1
python src/run_all.py --data FILE --strike atm     # ATM strike instead of the smile
python src/summarise.py                            # cross-asset tables and figures
python src/paper_exhibits.py                       # tables, figures, numbers for the paper
                                                   # (run both strikes first: --strike smile and atm)
python src/run_all.py --list                       # supported commodities

# without Bloomberg (simulated data, for testing only)
python scripts/make_synthetic_data.py --commodity GC
python src/run_all.py --commodity GC --data data/raw/synthetic_GC.xlsx
```

Results for commodity `CODE` go to `results/STRIKE/CODE/` (`tables/`,
`figures/`, `numbers.tex`), where `STRIKE` is `smile` (default: variance-swap
strike integrated over the 30-day smile, see `src/implied_variance.py`) or
`atm`; the cross-commodity comparison goes to `results/summary/STRIKE/`.
One workbook can hold several commodities, one per sheet. If a block's dates
jump backwards (rows left over from an older formula), the rows below the
jump are ignored and a warning is printed.

The paper is in `paper/`:

* `main.tex`: the full paper (framework, data, results, timing, mechanism);
* `summary.tex`: a two-page summary.

Every number, table and figure they show is written to `paper/generated/`
by `src/paper_exhibits.py`, so the text always matches the data. Those
files come from licensed data and are not tracked by git. To rebuild:

```bash
python src/run_all.py --data FILE.xlsx                 # smile strike
python src/run_all.py --data FILE.xlsx --strike atm    # ATM strike (robustness)
python src/summarise.py
python src/paper_exhibits.py
cd paper && latexmk -pdf main.tex summary.tex
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

## Note on the paper

Earlier versions (crude-oil draft, commodity research design) are in the
git history.
