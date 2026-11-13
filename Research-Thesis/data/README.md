# Data (not included)

The analysis uses Bloomberg data, which are licensed and cannot be
redistributed. Nothing in this folder except this file is tracked by git.

## One file per commodity

Save each export as `data/raw/CODE.xlsx` (`CL.xlsx`, `GC.xlsx`, `HG.xlsx`,
`C.xlsx`, ...). The code finds a file in this order:

1. `python src/run_all.py --commodity CODE --data path/to/file.xlsx`
2. the environment variable `VRP_DATA_FILE` (when running one commodity)
3. `data/raw/CODE.xlsx`

## Layout (same for every commodity)

One Excel sheet exported with `BDH`. Row 1 holds tickers, row 2 field names,
data start in row 3. Each block has its own date column. F1 and F2 are the
first and second generic futures.

| Columns | Ticker | Fields |
|---|---|---|
| A–F | F1 | date, `PX_SETTLE`, `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_VOLUME` |
| I–J | F2 | date, `PX_SETTLE` |
| L–O | F1 | date, `30DAY_IMPVOL_100.0%MNY_DF`, `1ST_MTH_IMPVOL_100.0%MNY_DF`, `2ND_MTH_IMPVOL_100.0%MNY_DF` |
| P– (optional) | F1 | more 30-day smile points in the same block, e.g. `30DAY_IMPVOL_90.0%MNY_DF`, `30DAY_IMPVOL_110.0%MNY_DF` (any order; read by field name; empty columns ignored) |

| Commodity | F1 (cells A1, L1) | F2 (cell I1) |
|---|---|---|
| WTI crude oil | `CL1 COMB Comdty` | `CL2 Comdty` |
| Natural gas | `NG1 Comdty` | `NG2 Comdty` |
| Gold | `GC1 Comdty` | `GC2 Comdty` |
| Silver | `SI1 Comdty` | `SI2 Comdty` |
| Copper | `HG1 Comdty` | `HG2 Comdty` |
| Corn | `C 1 Comdty` (with the space) | `C 2 Comdty` |
| Wheat | `W 1 Comdty` | `W 2 Comdty` |
| Soybeans | `S 1 Comdty` | `S 2 Comdty` |

The commodity is detected from the ticker in cell A1.

Bond futures use the same layout: `TU1`/`TU2`, `FV1`/`FV2`, `TY1`/`TY2`,
`US1`/`US2`, `RX1`/`RX2 Comdty`.

## Credit sheets

| Sheet | Cell A1 | Layout |
|---|---|---|
| ETF | `HYG US Equity` or `LQD US Equity` | A–F: date, `PX_LAST`, `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_VOLUME`; G–: date, `30DAY_IMPVOL_100.0%MNY_DF`, `1ST_MTH_IMPVOL_100.0%MNY_DF`, `2ND_MTH_IMPVOL_100.0%MNY_DF` (smile points optional) |
| CDX IG | `VIXIG Index` | two columns per ticker (date, `PX_LAST`): `VIXIG Index`, then any others, and `CDX IG CDSI GEN 5Y Corp` |

VIXIG is in basis points of spread volatility (the percentage version is
`VIXIGP`); its first print (100 on 29 February 2012) is dropped. The CDX
generic switches to the new series on the first trading day on or after
20 March and 20 September; those days are excluded from realised variance.

## Equity benchmark

A sheet with `VIX Index` in cell A1, two columns (date, `PX_LAST`) per
ticker: `VIX Index` and `SPX Index` (other tickers on the sheet are
ignored). It runs the same strategy on the S&P 500 with the VIX as strike.

Formulas (English Excel; in French Excel replace `,` with `;`):

```
A3: =BDH($A$1,$B$2:$F$2,DATE(2007,1,1),"","Days=T")
I3: =BDH($I$1,$J$2,DATE(2007,1,1),"","Days=T")
L3: =BDH($L$1,$M$2:$S$2,DATE(2007,1,1),"","Days=T")    (M2:S2 = ATM, 1st, 2nd, smile points)
```

The smile points give the variance-swap strike (`--strike smile`, the
default): implied variance integrated over the smile, flat beyond the last
point. Bloomberg provides 90% and 110% moneyness for these commodities
(80% and 120% came back empty). Without smile points the strike is ATM.

Common data problems the loader catches: a sheet whose formulas did not
calculate (`#NAME?`, skipped), F2 identical to F1 (cell I1 holding the
first generic instead of the second; skipped), and rows left below a block
by an older formula (dates jump back; ignored with a warning). Dates stored
as Excel numbers are converted.

## Optional: contract calendar from Bloomberg

By default the contract calendar is rebuilt from exchange rules. To use
Bloomberg's own dates instead, save `data/raw/CODE_calendar.csv` (or `.xlsx`)
with one row per contract and these columns:

| Column | Bloomberg |
|---|---|
| `LAST_TRADEABLE_DT` | required |
| `FUT_NOTICE_FIRST` | optional (physically delivered contracts) |

```
A2: =BDS("GC1 Comdty","FUT_CHAIN","INCLUDE_EXPIRED_CONTRACTS=Y")
B2: =BDP(A2,"LAST_TRADEABLE_DT")      C2: =BDP(A2,"FUT_NOTICE_FIRST")
```

Keep only the contract months the generic steps through (see the table in
the main README). Semicolon-separated files and dd/mm/yyyy dates from
French Excel are read correctly.

## No Bloomberg access?

`python scripts/make_synthetic_data.py --commodity CODE` writes simulated
data in the same layout to `data/raw/synthetic_CODE.xlsx`. The numbers are
not market data; they only show that the code runs.
