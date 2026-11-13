"""Read a Bloomberg Excel export into clean daily series.

Same layout for every commodity (one sheet, as exported with BDH), where
F1 and F2 are the first and second generic futures (e.g. CL1/CL2, GC1/GC2):
  columns A-F : F1 date, PX_SETTLE, PX_OPEN, PX_HIGH, PX_LOW, PX_VOLUME
  columns I-J : F2 date, PX_SETTLE
  columns L-O : F1 date, 30DAY_IMPVOL_100.0%MNY_DF,
                1ST_MTH_IMPVOL_100.0%MNY_DF, 2ND_MTH_IMPVOL_100.0%MNY_DF
  columns P-  : optional, more points of the 30-day smile in the same block,
                e.g. 30DAY_IMPVOL_90.0%MNY_DF, 30DAY_IMPVOL_110.0%MNY_DF
                (read by field name, in any order; empty columns are ignored)
Row 1 holds the tickers, row 2 the fields. Each block has its own date
column, so blocks are merged on dates.
"""
import re

import pandas as pd

from commodities import SPOT_SPECS, detect_code, is_spot


def _to_dates(col):
    """Dates as Excel writes them: datetimes, or day numbers (e.g. 46314)
    when a cell is formatted as a number."""
    num = pd.to_numeric(col, errors="coerce")
    serial = num.between(20000, 80000)
    out = pd.to_datetime(col.where(~serial), errors="coerce")
    out[serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(num[serial], unit="D")
    return out


def _block(raw, cols, names, label=""):
    df = raw.iloc[:, cols].copy()
    df.columns = names
    df["date"] = _to_dates(df["date"])
    df = df.dropna(subset=["date"])
    for c in names[1:]:
        # Bloomberg writes "#N/A N/A" for missing values
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # A BDH block is in date order. If the dates jump back, the rows below are
    # left over from an earlier, longer formula (e.g. a sheet copied from
    # another commodity): keep only the rows before the jump.
    back = (df["date"].diff().dt.days <= 0).to_numpy().nonzero()[0]
    if len(back):
        cut = back[0]
        print(f"  warning: {label} block: dates go back from "
              f"{df['date'].iloc[cut - 1].date()} to {df['date'].iloc[cut].date()}; "
              f"ignoring the {len(df) - cut} rows below (stale output?). "
              "Delete them in Excel.")
        df = df.iloc[:cut]
    return df.set_index("date")


IV_FIELDS = {"30DAY_IMPVOL_100.0%MNY_DF": "iv30",
             "1ST_MTH_IMPVOL_100.0%MNY_DF": "iv1m",
             "2ND_MTH_IMPVOL_100.0%MNY_DF": "iv2m"}


def _iv_names(header):
    """Column names for the implied-vol block from the field names in row 2.
    30-day smile points become iv_k<moneyness>, e.g. iv_k90, iv_k110."""
    names = []
    for h in header:
        h = str(h).strip().upper()
        m = re.fullmatch(r"30DAY_IMPVOL_(\d+(?:\.\d+)?)%MNY_DF", h)
        if h in IV_FIELDS:
            names.append(IV_FIELDS[h])
        elif m:
            names.append(f"iv_k{float(m.group(1)):g}")
        else:
            names.append(None)
    return names


def load_daily(path, sheet=0):
    full = pd.read_excel(path, sheet_name=sheet, header=None)
    header, raw = full.iloc[1], full.iloc[2:].reset_index(drop=True)
    f1 = _block(raw, [0, 1, 2, 3, 4, 5],
                ["date", "f1", "f1_open", "f1_high", "f1_low", "f1_volume"], "F1 (columns A-F)")
    f2 = _block(raw, [8, 9], ["date", "f2"], "F2 (columns I-J)")

    # implied-vol block: column L (dates) and every field to its right
    cols = [11]
    for j in range(12, full.shape[1]):
        if pd.isna(header.iloc[j]):
            break
        cols.append(j)
    names = _iv_names(header.iloc[cols[1:]])
    if not any(names):                      # no field names: assume the basic layout
        names = ["iv30", "iv1m", "iv2m"][:len(cols) - 1]
    keep = [c for c, n in zip(cols[1:], names) if n]
    iv = _block(raw, [11] + keep, ["date"] + [n for n in names if n], "implied vol (columns L-)")
    iv = iv.dropna(axis=1, how="all")       # e.g. smile points Bloomberg does not provide

    df = f1.join(f2, how="outer").join(iv, how="outer")
    # keep days on which both futures settled
    df = df.dropna(subset=["f1", "f2"])
    return df


def load_spot(path, sheet, code):
    """Continuous series (no futures roll). Columns returned: f1 (price, or
    spread in bp), f2 (empty), iv30, iv1m, iv2m and any smile points.

    ETF layout (HYG, LQD), as exported with BDH:
      columns A-F : date, PX_LAST, PX_OPEN, PX_HIGH, PX_LOW, PX_VOLUME
      columns G-  : date, 30DAY_IMPVOL_100.0%MNY_DF, 1ST_MTH_..., 2ND_MTH_..., smile points
    Paired layout, two columns (date, value) per ticker, the implied-vol
    index in A1: credit (A-B VIXIG Index, bp vol; ... CDX IG CDSI GEN 5Y Corp,
    spread in bp) or equity (A-B VIX Index; C-D SPX Index).
    """
    full = pd.read_excel(path, sheet_name=sheet, header=None)
    header, raw = full.iloc[1], full.iloc[2:].reset_index(drop=True)
    if "pair" in SPOT_SPECS[code]:
        # two columns per ticker: (date, value) for the implied vol index and
        # for the underlying (price or spread)
        iv_t, px_t = SPOT_SPECS[code]["pair"]
        tickers = [" ".join(str(t).upper().split()) for t in full.iloc[0]]
        j_iv = tickers.index(iv_t)
        j_s = next(j for j, t in enumerate(tickers) if t.startswith(px_t))
        iv = _block(raw, [j_iv, j_iv + 1], ["date", "iv30"], iv_t)
        # the first VIXIG print (100 on 29 Feb 2012) is a placeholder, not a quote
        iv = iv[~((iv.index == iv.index[0]) & (iv["iv30"] == 100))]
        spread = _block(raw, [j_s, j_s + 1], ["date", "f1"], px_t)
        df = spread.join(iv, how="outer")
        df["iv1m"] = df["iv2m"] = float("nan")
    else:
        px = _block(raw, [0, 1, 2, 3, 4, 5],
                    ["date", "f1", "f1_open", "f1_high", "f1_low", "f1_volume"], "price (A-F)")
        cols = [6] + [j for j in range(7, full.shape[1]) if pd.notna(header.iloc[j])]
        names = _iv_names(header.iloc[cols[1:]])
        keep = [c for c, n in zip(cols[1:], names) if n]
        iv = _block(raw, [6] + keep, ["date"] + [n for n in names if n], "implied vol (G-)")
        df = px.join(iv.dropna(axis=1, how="all"), how="outer")
    df["f2"] = float("nan")
    return df.dropna(subset=["f1"])


def load_instrument(path, sheet, code):
    return load_spot(path, sheet, code) if is_spot(code) else load_daily(path, sheet)


def smile_columns(df):
    """{moneyness: column} for the 30-day smile points present, incl. ATM."""
    out = {float(c[4:]): c for c in df.columns if c.startswith("iv_k")}
    if "iv30" in df.columns:
        out[100.0] = "iv30"
    return dict(sorted(out.items()))


def read_ticker(path, sheet=0):
    """Ticker in cell A1, e.g. 'CL1 COMB Comdty'."""
    return pd.read_excel(path, sheet_name=sheet, header=None, nrows=1).iloc[0, 0]


def detect_commodity(path, sheet=0):
    """Commodity code from the ticker in cell A1 ('GC1 Comdty' -> 'GC')."""
    return detect_code(read_ticker(path, sheet))


def list_sheets(path):
    """[(sheet name, commodity code)] for every sheet of a workbook, so one
    file can hold several commodities (one per sheet)."""
    return [(name, detect_commodity(path, name)) for name in pd.ExcelFile(path).sheet_names]


def check_daily(df, label):
    """Problems that make a sheet unusable; returns a list of messages."""
    issues = []
    if len(df) < 250:
        issues.append(f"{label}: only {len(df)} usable days (formulas not calculated? #NAME?)")
        return issues
    same = (df["f1"] == df["f2"]).mean() if df["f2"].notna().any() else 0.0
    if same > 0.5:
        issues.append(f"{label}: F2 is identical to F1 on {100 * same:.0f}% of days; "
                      "cell I1 should hold the second generic (e.g. W 2 COMB Comdty)")
    if "iv30" not in df.columns or df["iv30"].notna().mean() < 0.5:
        issues.append(f"{label}: implied volatility (column M) missing")
    return issues


if __name__ == "__main__":
    import sys
    d = load_daily(sys.argv[1])
    print(d.head())
    print(d.tail())
    print(d.isna().sum())
