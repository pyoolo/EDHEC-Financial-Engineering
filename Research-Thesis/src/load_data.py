"""Read a Bloomberg Excel export into clean daily series.

Same layout for every commodity (one sheet, as exported with BDH), where
F1 and F2 are the first and second generic futures (e.g. CL1/CL2, GC1/GC2):
  columns A-F : F1 date, PX_SETTLE, PX_OPEN, PX_HIGH, PX_LOW, PX_VOLUME
  columns I-J : F2 date, PX_SETTLE
  columns L-O : F1 date, 30DAY_IMPVOL_100.0%MNY_DF,
                1ST_MTH_IMPVOL_100.0%MNY_DF, 2ND_MTH_IMPVOL_100.0%MNY_DF
Row 1 holds the tickers, row 2 the fields. Each block has its own date
column, so blocks are merged on dates.
"""
import pandas as pd

from commodities import detect_code


def _block(raw, cols, names):
    df = raw.iloc[:, cols].copy()
    df.columns = names
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    for c in names[1:]:
        # Bloomberg writes "#N/A N/A" for missing values
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def load_daily(path, sheet=0):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=2)
    f1 = _block(raw, [0, 1, 2, 3, 4, 5],
                ["date", "f1", "f1_open", "f1_high", "f1_low", "f1_volume"])
    f2 = _block(raw, [8, 9], ["date", "f2"])
    iv = _block(raw, [11, 12, 13, 14], ["date", "iv30", "iv1m", "iv2m"])

    df = f1.join(f2, how="outer").join(iv, how="outer")
    # keep days on which both futures settled
    df = df.dropna(subset=["f1", "f2"])
    return df


def read_ticker(path, sheet=0):
    """Ticker in cell A1, e.g. 'CL1 COMB Comdty'."""
    return pd.read_excel(path, sheet_name=sheet, header=None, nrows=1).iloc[0, 0]


def detect_commodity(path):
    """Commodity code from the ticker in cell A1 ('GC1 Comdty' -> 'GC')."""
    return detect_code(read_ticker(path))


if __name__ == "__main__":
    import sys
    d = load_daily(sys.argv[1])
    print(d.head())
    print(d.tail())
    print(d.isna().sum())
