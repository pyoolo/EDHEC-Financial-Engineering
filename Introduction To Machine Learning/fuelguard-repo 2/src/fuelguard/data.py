"""Loading the raw data and building the daily panel.

Two sources:
* Bloomberg (exported to Excel with BDH formulas, then reduced to the series we use
  in ``data/raw/bloomberg_selected.csv`` by ``scripts/00_extract_bloomberg.py``);
* the ECB euro reference rate EUR/USD (SDMX-ML file from the ECB website).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

# (file, sheet, column layout, {output name: (date col, value col)}, first data row)
# Futures sheets: col 0 = date, 1 = 1st-month settle, 8/9 = date/2nd-month settle,
# 11/12 = date/30-day ATM implied volatility.
_FUT = {"1": (0, 1), "2": (8, 9), "_iv": (11, 12)}
BLOOMBERG_LAYOUT = [
    ("BBG_Download_Blocco2.xlsx", "Heating oil", "HO", ["1", "2", "_iv"], 2),
    ("BBG_Download_Blocco2.xlsx", "Brent", "CO", ["1", "2", "_iv"], 2),
    ("BBG_Download_Blocco1.xlsx", "Oil", "CL", ["1", "2"], 2),
    ("BBG_Download_Blocco1.xlsx", "NatGas", "NG", ["1", "2"], 2),
    ("BBG_Download_Blocco1.xlsx", "Copper", "HG", ["1"], 2),
    ("BBG_Download_Blocco1.xlsx", "Gold", "GC", ["1"], 2),
    ("BBG_Download_Blocco1.xlsx", "Treasury 10 anni", "TY", ["1"], 2),
]
BLOOMBERG_PAIRS = [
    ("BBG_Download_Blocco1.xlsx", "Market", {"VIX": 0, "VIX3M": 2, "SPX": 4}, 2),
    ("Bloomberg_download_oil_inflation.xlsx", "GIORNALIERO",
     {"US_BE5": 6, "EU_INF5": 12, "SP_ENERGY": 18, "STOXX_OILGAS": 21}, 3),
]

# Human-readable description of every raw series (used in the README and the report)
SERIES = {
    "HO1": ("HO1 Comdty", "NYMEX NY Harbor ULSD (diesel) futures, 1st month, USc/gal"),
    "HO2": ("HO2 Comdty", "NYMEX NY Harbor ULSD (diesel) futures, 2nd month, USc/gal"),
    "HO_iv": ("HO1 Comdty", "Diesel 30-day ATM implied volatility, %"),
    "CO1": ("CO1 Comdty", "ICE Brent crude futures, 1st month, USD/bbl"),
    "CO2": ("CO2 Comdty", "ICE Brent crude futures, 2nd month, USD/bbl"),
    "CO_iv": ("CO1 Comdty", "Brent 30-day ATM implied volatility, %"),
    "CL1": ("CL1 COMB Comdty", "NYMEX WTI crude futures, 1st month, USD/bbl"),
    "CL2": ("CL2 COMB Comdty", "NYMEX WTI crude futures, 2nd month, USD/bbl"),
    "NG1": ("NG1 COMB Comdty", "Henry Hub natural gas futures, 1st month, USD/MMBtu"),
    "NG2": ("NG2 COMB Comdty", "Henry Hub natural gas futures, 2nd month, USD/MMBtu"),
    "HG1": ("HG1 COMB Comdty", "COMEX copper futures, 1st month, USc/lb"),
    "GC1": ("GC1 COMB Comdty", "COMEX gold futures, 1st month, USD/oz"),
    "TY1": ("TY1 Comdty", "US 10-year Treasury note futures, price"),
    "VIX": ("VIX Index", "CBOE VIX (S&P 500 30-day implied volatility)"),
    "VIX3M": ("VIX3M Index", "CBOE 3-month VIX"),
    "SPX": ("SPX Index", "S&P 500 index"),
    "US_BE5": ("USGGBE05 Index", "US 5-year breakeven inflation, %"),
    "EU_INF5": ("EUSWI5 Index", "Euro 5-year inflation swap rate, %"),
    "SP_ENERGY": ("S5ENRS Index", "S&P 500 Energy sector index"),
    "STOXX_OILGAS": ("SXEP Index", "STOXX Europe 600 Oil & Gas index"),
    "EURUSD": ("ECB EXR D.USD.EUR.SP00.A", "ECB euro reference rate, USD per EUR"),
}


def _series(rows, dcol, vcol, name, start):
    out = {}
    for r in rows[start:]:
        d = r[dcol] if dcol < len(r) else None
        v = r[vcol] if vcol < len(r) else None
        if hasattr(d, "year") and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[pd.Timestamp(d)] = float(v)
    return pd.Series(out, name=name, dtype=float)


def extract_bloomberg(xlsx_dir=C.DATA_XLSX) -> pd.DataFrame:
    """Read the Bloomberg Excel exports and keep only the series used by the product."""
    import openpyxl

    books, series = {}, []

    def rows(fname, sheet):
        if fname not in books:
            books[fname] = openpyxl.load_workbook(xlsx_dir / fname, read_only=True, data_only=True)
        return list(books[fname][sheet].iter_rows(values_only=True))

    for fname, sheet, prefix, parts, start in BLOOMBERG_LAYOUT:
        rr = rows(fname, sheet)
        for p in parts:
            series.append(_series(rr, *_FUT[p], prefix + p, start))
    for fname, sheet, cols, start in BLOOMBERG_PAIRS:
        rr = rows(fname, sheet)
        for name, c in cols.items():
            series.append(_series(rr, c, c + 1, name, start))
    df = pd.concat(series, axis=1, sort=True).sort_index()
    df.index.name = "date"
    return df[df.index >= C.SAMPLE_START]


def load_ecb_fx(path=C.ECB_XML) -> pd.Series:
    """Parse the ECB SDMX-ML file into a daily USD-per-EUR series."""
    txt = path.read_text(encoding="utf-8")
    obs = re.findall(r'TIME_PERIOD="([\d-]+)" OBS_VALUE="([\d.]+)"', txt)
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in obs}, name="EURUSD", dtype=float)
    s.index.name = "date"
    return s.sort_index()


def build_panel(ffill_limit: int = 5) -> pd.DataFrame:
    """Daily panel on NYMEX diesel trading days.

    Missing values (holidays on other exchanges, late publications) are filled
    *forward only* and for at most ``ffill_limit`` days, so a value is never
    taken from the future.
    """
    bbg = pd.read_csv(C.BLOOMBERG_CSV, index_col=0, parse_dates=True)
    fx = load_ecb_fx()
    df = bbg.join(fx, how="outer").sort_index()
    df = df.ffill(limit=ffill_limit)
    df = df[bbg["HO1"].reindex(df.index).notna() & bbg["HO2"].reindex(df.index).notna()]
    df = df[df.index >= C.SAMPLE_START]
    # Diesel in EUR per litre: market price (1st month) and lockable price (2nd month)
    to_eur_litre = 1 / 100 / C.LITRES_PER_GALLON / df["EURUSD"]
    df["diesel_eur"] = df["HO1"] * to_eur_litre
    df["lock_eur"] = df["HO2"] * to_eur_litre
    return df


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SERIES if c in df.columns] + ["diesel_eur", "lock_eur"]
    d = df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    d.insert(0, "description", [SERIES[c][1] if c in SERIES else
                                "Diesel, EUR/litre (" + ("market" if c == "diesel_eur" else "lockable") + ")"
                                for c in d.index])
    return d
