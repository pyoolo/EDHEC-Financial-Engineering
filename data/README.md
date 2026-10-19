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

Formulas (English Excel; in French Excel replace `,` with `;`):

```
A3: =BDH($A$1,$B$2:$F$2,DATE(2007,1,1),"","Days=T")
I3: =BDH($I$1,$J$2,DATE(2007,1,1),"","Days=T")
L3: =BDH($L$1,$M$2:$O$2,DATE(2007,1,1),"","Days=T")
```

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
