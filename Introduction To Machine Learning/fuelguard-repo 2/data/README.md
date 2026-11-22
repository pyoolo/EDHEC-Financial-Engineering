# Data

| File | Source | Content |
|---|---|---|
| `raw/bloomberg_selected.csv` | Bloomberg terminal (EDHEC), `BDH` formulas, downloaded 17–19 Nov 2026 | 20 daily series, 2007–2026 (see table below) |
| `raw/ecb_eurusd.xml` | European Central Bank, euro foreign exchange reference rates (SDMX-ML), extracted 19 Nov 2026 | EUR/USD daily reference rate, 1999–2026 |
| `processed/panel_daily.csv` | built by `scripts/01_build_dataset.py` | aligned daily panel + targets |
| `processed/model_dataset.csv` | built by `scripts/01_build_dataset.py` | 57 features + targets, one row per decision day |
| `processed/latest_unlabelled.csv` | built by `scripts/01_build_dataset.py` | last month: features known, outcome not yet known |

`bloomberg_xlsx/` is where the original Excel exports go if you want to rebuild the CSV
with `scripts/00_extract_bloomberg.py`. They are not committed (licensed data, and they
contain series from other projects that this product does not use).

## Raw series (21, before feature engineering)

| Column | Bloomberg ticker | Field | Description |
|---|---|---|---|
| HO1 | HO1 Comdty | PX_SETTLE | NYMEX NY Harbor ULSD (diesel) futures, 1st month, US cents/gallon |
| HO2 | HO2 Comdty | PX_SETTLE | same, 2nd month |
| HO_iv | HO1 Comdty | 30DAY_IMPVOL_100.0%MNY_DF | diesel 30-day at-the-money implied volatility, % |
| CO1, CO2 | CO1/CO2 Comdty | PX_SETTLE | ICE Brent crude futures, 1st/2nd month, USD/bbl |
| CO_iv | CO1 Comdty | 30DAY_IMPVOL_100.0%MNY_DF | Brent 30-day ATM implied volatility, % |
| CL1, CL2 | CL1/CL2 COMB Comdty | PX_SETTLE | NYMEX WTI crude futures, 1st/2nd month |
| NG1, NG2 | NG1/NG2 COMB Comdty | PX_SETTLE | Henry Hub natural gas futures, 1st/2nd month |
| HG1 | HG1 COMB Comdty | PX_SETTLE | COMEX copper, 1st month |
| GC1 | GC1 COMB Comdty | PX_SETTLE | COMEX gold, 1st month |
| TY1 | TY1 Comdty | PX_SETTLE | US 10-year Treasury note futures |
| VIX, VIX3M | VIX / VIX3M Index | PX_LAST | CBOE equity implied volatility, 1 and 3 months |
| SPX | SPX Index | PX_LAST | S&P 500 |
| US_BE5 | USGGBE05 Index | PX_LAST | US 5-year breakeven inflation |
| EU_INF5 | EUSWI5 Index | PX_LAST | Euro 5-year inflation swap |
| SP_ENERGY | S5ENRS Index | PX_LAST | S&P 500 Energy |
| STOXX_OILGAS | SXEP Index | PX_LAST | STOXX Europe 600 Oil & Gas |
| EURUSD | ECB EXR.D.USD.EUR.SP00.A | — | ECB reference rate, USD per EUR |

## Known data issues and how they are handled

* Calendars differ across exchanges: the panel uses NYMEX diesel trading days; other
  series are carried **forward only**, for at most 5 days.
* One implied-volatility value of 1.7% (2007-09-12) is an error: implied vols below 5% are
  set to missing and carried forward.
* WTI front month was negative on 2020-04-20: WTI enters only through the 2nd-month
  return and a curve slope scaled by the 2nd month.
* Generic futures (HO1, HO2) roll on the first trading day of each month. The lockable
  price is therefore built as a strip of the 1st and 2nd month (see `src/fuelguard/targets.py`).
