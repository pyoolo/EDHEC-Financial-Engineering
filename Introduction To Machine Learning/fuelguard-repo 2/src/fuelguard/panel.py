"""FuelGuard 2.0 data layer: a panel of 22 futures markets with OHLC and the implied-volatility smile.

Each Bloomberg futures sheet has the layout
  col 0 date | 1 PX_SETTLE | 2 PX_OPEN | 3 PX_HIGH | 4 PX_LOW | 5 PX_VOLUME      (1st generic contract)
  col 8 date | 9 PX_SETTLE                                                      (2nd generic contract)
  col 11 date| 12 30d IV 100% | 13 1st-month IV | 14 2nd-month IV | 15..20 30d IV at 90, 95, 97.5,
             102.5, 105, 110% moneyness
"""
from __future__ import annotations

import pandas as pd

from . import config as C

# market code: (file, sheet, ticker, asset class)
MARKETS = {
    "HO": ("BBG_Download_Blocco2.xlsx", "Heating oil", "HO1 Comdty", "Energy"),
    "CO": ("BBG_Download_Blocco2.xlsx", "Brent", "CO1 Comdty", "Energy"),
    "CL": ("BBG_Download_Blocco1.xlsx", "Oil", "CL1 COMB Comdty", "Energy"),
    "NG": ("BBG_Download_Blocco1.xlsx", "NatGas", "NG1 COMB Comdty", "Energy"),
    "GC": ("BBG_Download_Blocco1.xlsx", "Gold", "GC1 COMB Comdty", "Metals"),
    "SI": ("BBG_Download_Blocco1.xlsx", "Silver", "SI1 COMB Comdty", "Metals"),
    "HG": ("BBG_Download_Blocco1.xlsx", "Copper", "HG1 COMB Comdty", "Metals"),
    "C": ("BBG_Download_Blocco1.xlsx", "Corn", "C 1 COMB Comdty", "Grains"),
    "W": ("BBG_Download_Blocco1.xlsx", "Wheat", "W 1 COMB Comdty", "Grains"),
    "S": ("BBG_Download_Blocco1.xlsx", "Soybean", "S 1 COMB Comdty", "Grains"),
    "SB": ("BBG_Download_Blocco2.xlsx", "Sugar", "SB1 Comdty", "Softs"),
    "KC": ("BBG_Download_Blocco2.xlsx", "Coffee", "KC1 Comdty", "Softs"),
    "CT": ("BBG_Download_Blocco2.xlsx", "Cotton", "CT1 Comdty", "Softs"),
    "LC": ("BBG_Download_Blocco2.xlsx", "Live cattle", "LC1 Comdty", "Livestock"),
    "LH": ("BBG_Download_Blocco2.xlsx", "Lean hogs", "LH1 Comdty", "Livestock"),
    "TU": ("BBG_Download_Blocco1.xlsx", "Treasury 2 anni", "TU1 Comdty", "Rates"),
    "FV": ("BBG_Download_Blocco1.xlsx", "Treasury 5 anni", "FV1 Comdty", "Rates"),
    "TY": ("BBG_Download_Blocco1.xlsx", "Treasury 10 anni", "TY1 Comdty", "Rates"),
    "US": ("BBG_Download_Blocco1.xlsx", "Treasury 30 anni", "US1 Comdty", "Rates"),
    "RX": ("BBG_Download_Blocco1.xlsx", "Bund 10Y", "RX1 Comdty", "Rates"),
    "OE": ("BBG_Download_Blocco2.xlsx", "Bobl", "OE1 Comdty", "Rates"),
    "DU": ("BBG_Download_Blocco2.xlsx", "Schatz", "DU1 Comdty", "Rates"),
}
NAMES = {"HO": "Diesel (NY Harbor ULSD)", "CO": "Brent crude", "CL": "WTI crude", "NG": "Natural gas",
         "GC": "Gold", "SI": "Silver", "HG": "Copper", "C": "Corn", "W": "Wheat", "S": "Soybeans",
         "SB": "Sugar", "KC": "Coffee", "CT": "Cotton", "LC": "Live cattle", "LH": "Lean hogs",
         "TU": "US Treasury 2y", "FV": "US Treasury 5y", "TY": "US Treasury 10y", "US": "US Treasury 30y",
         "RX": "Euro Bund 10y", "OE": "Euro Bobl 5y", "DU": "Euro Schatz 2y"}

PRICE_COLS = {"settle": 1, "open": 2, "high": 3, "low": 4, "volume": 5}
IV_COLS = {"iv": 12, "iv_m1": 13, "iv_m2": 14, "iv90": 15, "iv95": 16, "iv975": 17,
           "iv1025": 18, "iv105": 19, "iv110": 20}
COMMON = [("BBG_Download_Blocco1.xlsx", "Market", {"VIX": 0, "VIX3M": 2}, 2),
          ("BBG_Download_Blocco1.xlsx", "Euro equity", {"V2X": 0}, 2)]

PANEL_CSV = C.DATA_RAW / "bloomberg_futures_panel.csv"
COMMON_CSV = C.DATA_RAW / "bloomberg_common.csv"


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def extract_panel(xlsx_dir=C.DATA_XLSX):
    import openpyxl
    books = {}
    frames = []
    for code, (fname, sheet, _, _) in MARKETS.items():
        if fname not in books:
            books[fname] = openpyxl.load_workbook(xlsx_dir / fname, read_only=True, data_only=True)
        rows = list(books[fname][sheet].iter_rows(values_only=True))[2:]
        px, f2, iv = {}, {}, {}
        for r in rows:
            r = list(r) + [None] * 21
            if hasattr(r[0], "year"):
                px[pd.Timestamp(r[0])] = {k: _num(r[c]) for k, c in PRICE_COLS.items()}
            if hasattr(r[8], "year") and _num(r[9]) is not None:
                f2[pd.Timestamp(r[8])] = _num(r[9])
            if hasattr(r[11], "year"):
                iv[pd.Timestamp(r[11])] = {k: _num(r[c]) for k, c in IV_COLS.items()}
        df = pd.DataFrame.from_dict(px, orient="index")
        df["settle2"] = pd.Series(f2)
        df = df.join(pd.DataFrame.from_dict(iv, orient="index"), how="left")
        df.index.name = "date"
        df.insert(0, "market", code)
        frames.append(df.reset_index())
    panel = pd.concat(frames, ignore_index=True).sort_values(["market", "date"])
    panel = panel[panel["date"] >= C.SAMPLE_START]

    common = []
    for fname, sheet, cols, start in COMMON:
        rows = list(books.setdefault(fname, openpyxl.load_workbook(xlsx_dir / fname, read_only=True,
                                                                    data_only=True))[sheet]
                    .iter_rows(values_only=True))[start:]
        for name, c in cols.items():
            s = {pd.Timestamp(r[c]): _num(r[c + 1]) for r in rows
                 if c + 1 < len(r) and hasattr(r[c], "year") and _num(r[c + 1]) is not None}
            common.append(pd.Series(s, name=name))
    common = pd.concat(common, axis=1).sort_index()
    common.index.name = "date"
    return panel, common[common.index >= C.SAMPLE_START]


def load_panel():
    p = pd.read_csv(PANEL_CSV, parse_dates=["date"])
    c = pd.read_csv(COMMON_CSV, index_col=0, parse_dates=True)
    return p, c
