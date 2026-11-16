"""Instrument specifications and futures contract calendars.

Futures (commodities, government bond futures) are in SPECS; instruments
traded as a single continuous series (credit ETFs, the CDX IG index) are in
SPOT_SPECS. INSTRUMENTS holds both.

Each commodity needs two dates per contract:
  * last trading day (LTD): when the Bloomberg generics F1/F2 switch to
    the next contract (the generic rolls on the day after the LTD);
  * first notice day (FND), for physically delivered contracts whose
    liquidity moves to the next contract before delivery starts.
The position leaves the front contract ROLL_DAYS_BEFORE_EXPIRY trading
days before the earlier of the two dates.

Dates are computed from exchange rules, using the trading days in the data
as the business-day calendar. Rules follow CME Group contract specifications;
check them (and the listed contract cycle the Bloomberg generic follows)
with the calendar validation table that run_all.py writes for every
commodity, or override them with a calendar exported from Bloomberg
(see data/README.md).

To add a commodity: add an entry to SPECS with its contract cycle and the
names of its LTD and FND rules below.
"""
import re

import pandas as pd

ALL_MONTHS = "FGHJKMNQUVXZ"           # futures month codes, January..December

SPECS = {
    # code: name, exchange, months the generic steps through, LTD rule, FND rule
    "CL": dict(name="WTI crude oil", exchange="NYMEX", cycle=ALL_MONTHS,
               ltd="cl_25th_prior_month", fnd=None),
    "NG": dict(name="Henry Hub natural gas", exchange="NYMEX", cycle=ALL_MONTHS,
               ltd="three_bd_before_month", fnd=None),
    "GC": dict(name="Gold", exchange="COMEX", cycle="GJMQVZ",
               ltd="third_last_bd", fnd="last_bd_prior_month"),
    "SI": dict(name="Silver", exchange="COMEX", cycle="HKNUZ",
               ltd="third_last_bd", fnd="last_bd_prior_month"),
    "HG": dict(name="Copper", exchange="COMEX", cycle="HKNUZ",
               ltd="third_last_bd", fnd="last_bd_prior_month"),
    "C": dict(name="Corn", exchange="CBOT", cycle="HKNUZ",
              ltd="bd_before_15th", fnd="last_bd_prior_month"),
    "W": dict(name="Wheat", exchange="CBOT", cycle="HKNUZ",
              ltd="bd_before_15th", fnd="last_bd_prior_month"),
    "S": dict(name="Soybeans", exchange="CBOT", cycle="FHKNQUX",
              ltd="bd_before_15th", fnd="last_bd_prior_month"),
    # Government bond futures. Options are quoted far closer to the money than
    # 90%/110% moneyness, so the variance strike is ATM (smile=False).
    "TU": dict(name="US 2-year note", exchange="CBOT", cycle="HMUZ", asset="Rates",
               ltd="last_bd_of_month", fnd="last_bd_prior_month", smile=False),
    "FV": dict(name="US 5-year note", exchange="CBOT", cycle="HMUZ", asset="Rates",
               ltd="last_bd_of_month", fnd="last_bd_prior_month", smile=False),
    "TY": dict(name="US 10-year note", exchange="CBOT", cycle="HMUZ", asset="Rates",
               ltd="seventh_bd_before_last_bd", fnd="last_bd_prior_month", smile=False),
    "US": dict(name="US Treasury bond", exchange="CBOT", cycle="HMUZ", asset="Rates",
               ltd="seventh_bd_before_last_bd", fnd="last_bd_prior_month", smile=False),
    "RX": dict(name="Euro-Bund", exchange="Eurex", cycle="HMUZ", asset="Rates",
               ltd="eurex_two_days_before_10th", fnd=None, smile=False),
}
for _c in ["CL", "NG"]:
    SPECS[_c]["asset"] = "Energy"
for _c in ["GC", "SI", "HG"]:
    SPECS[_c]["asset"] = "Metals"
for _c in ["C", "W", "S"]:
    SPECS[_c]["asset"] = "Grains"

# Continuous series: no contract calendar. `units`:
#   "log" -> daily log returns of the price, implied vol in % of price;
#   "bp"  -> daily spread changes in bp/100, implied vol in bp (normal vol),
#            so one "vol point" is one basis point of annualised spread vol.
SPOT_SPECS = {
    "HYG": dict(name="High-yield bond ETF (HYG)", exchange="NYSE Arca", asset="Credit",
                units="log", smile=False, ticker="HYG US EQUITY"),
    "LQD": dict(name="Investment-grade bond ETF (LQD)", exchange="NYSE Arca", asset="Credit",
                units="log", smile=False, ticker="LQD US EQUITY"),
    "CDXIG": dict(name="CDX North America IG 5Y", exchange="OTC", asset="Credit",
                  units="bp", smile=False, ticker="VIXIG INDEX",
                  pair=("VIXIG INDEX", "CDX IG"), roll_dates=((3, 20), (9, 20))),
    # Equity benchmark: S&P 500 index with the VIX (model-free 30-day implied
    # vol) as the variance strike, from a sheet with 'VIX Index' in cell A1
    "SPX": dict(name="S&P 500 (VIX strike)", exchange="Cboe", asset="Equity",
                units="log", smile=False, ticker="VIX INDEX", pair=("VIX INDEX", "SPX INDEX")),
}
INSTRUMENTS = {**SPECS, **SPOT_SPECS}


def is_spot(code):
    return code in SPOT_SPECS


def uses_smile(code):
    return INSTRUMENTS[code].get("smile", True)


def bloomberg_ticker(code, n):
    """Generic ticker as Bloomberg writes it: 'CL1 Comdty', 'C 1 Comdty'."""
    return f"{code:<2}{n} Comdty"


def detect_code(ticker):
    """'CL1 COMB Comdty' -> 'CL', 'C 1 Comdty' -> 'C', 'HYG US Equity' -> 'HYG',
    'VIXIG Index' (credit sheet) -> 'CDXIG'."""
    t = " ".join(str(ticker).upper().split())
    for code, s in SPOT_SPECS.items():
        if t == s["ticker"]:
            return code
    m = re.match(r"\s*([A-Z]{1,3})\s*\d", str(ticker).upper())
    return m.group(1) if m else None


# ----------------------------------------------------------------- date rules
# Each rule takes the delivery month (pd.Period) and the trading days, and
# returns (date, observed_until): the date is valid only if the data run at
# least until `observed_until`.

def _month_days(days, m):
    return days[(days >= m.start_time) & (days <= m.end_time)]


def cl_25th_prior_month(m, days):
    """NYMEX CL: 3 business days before the 25th of the month before delivery
    (4 if the 25th is not a business day)."""
    p = m - 1
    d25 = pd.Timestamp(p.year, p.month, 25)
    before = days[days < d25]
    if len(before) < 4:
        return None, d25
    return (before[-3] if d25 in days else before[-4]), d25


def three_bd_before_month(m, days):
    """NYMEX NG: 3 business days before the first calendar day of delivery."""
    first = m.start_time
    before = days[days < first]
    return (before[-3] if len(before) >= 3 else None), first


def third_last_bd(m, days):
    """COMEX metals: third-last business day of the delivery month."""
    d = _month_days(days, m)
    return (d[-3] if len(d) >= 3 else None), m.end_time.normalize()


def bd_before_15th(m, days):
    """CBOT grains: business day before the 15th of the delivery month."""
    d15 = pd.Timestamp(m.year, m.month, 15)
    before = days[(days < d15) & (days >= m.start_time)]
    return (before[-1] if len(before) else None), d15


def last_bd_prior_month(m, days):
    """First notice day of physically delivered contracts: last business day
    of the month before delivery."""
    d = _month_days(days, m - 1)
    return (d[-1] if len(d) else None), (m - 1).end_time.normalize()


def last_bd_of_month(m, days):
    """CBOT 2- and 5-year notes: last business day of the delivery month."""
    d = _month_days(days, m)
    return (d[-1] if len(d) else None), m.end_time.normalize()


def seventh_bd_before_last_bd(m, days):
    """CBOT 10-year note and bond: seventh business day before the last
    business day of the delivery month."""
    d = _month_days(days, m)
    return (d[-8] if len(d) >= 8 else None), m.end_time.normalize()


def eurex_two_days_before_10th(m, days):
    """Eurex Bund: delivery on the 10th calendar day of the delivery month
    (next exchange day if not one); last trading day two exchange days before."""
    d10 = pd.Timestamp(m.year, m.month, 10)
    after = days[days >= d10]
    if not len(after) or after[0].to_period("M") != m:
        return None, d10
    delivery = after[0]
    before = days[days < delivery]
    return (before[-2] if len(before) >= 2 else None), delivery


RULES = {f.__name__: f for f in [cl_25th_prior_month, three_bd_before_month, third_last_bd,
                                  bd_before_15th, last_bd_prior_month, last_bd_of_month,
                                  seventh_bd_before_last_bd, eurex_two_days_before_10th]}


def spot_excluded_days(code, trading_days):
    """Days whose return is not a market move: the CDX generic switching to
    the new series (first trading day on or after 20 March / 20 September)."""
    days = pd.DatetimeIndex(trading_days)
    out = []
    for mon, day in SPOT_SPECS[code].get("roll_dates", ()):
        for y in range(days[0].year, days[-1].year + 1):
            after = days[days >= pd.Timestamp(y, mon, day)]
            if len(after) and (after[0] - pd.Timestamp(y, mon, day)).days < 7:
                out.append(after[0])
    return pd.DatetimeIndex(sorted(out))


def contract_calendar(code, trading_days):
    """Last trading day and roll reference date of every contract in the data.

    Returns a DataFrame indexed by delivery month with columns
    `last_trade` (when the generic switches) and `roll_ref`
    (min of last trade and first notice).
    """
    spec = SPECS[code]
    days = pd.DatetimeIndex(trading_days).sort_values()
    ltd_rule, fnd_rule = RULES[spec["ltd"]], RULES.get(spec["fnd"]) if spec["fnd"] else None
    rows = []
    for m in pd.period_range(days[0].to_period("M"), days[-1].to_period("M") + 2, freq="M"):
        if ALL_MONTHS[m.month - 1] not in spec["cycle"]:
            continue
        ltd, need = ltd_rule(m, days)
        if ltd is None or days[-1] < need or ltd < days[0]:
            continue
        ref = ltd
        if fnd_rule is not None:
            fnd, need_f = fnd_rule(m, days)
            if fnd is not None and days[-1] >= need_f:
                ref = min(ltd, fnd)
        rows.append({"delivery": m, "last_trade": ltd, "roll_ref": ref})
    return pd.DataFrame(rows).set_index("delivery")


def _read_table(path):
    """CSV (comma, semicolon or tab separated, as Excel writes it in any
    language) or Excel file."""
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = max([";", "\t", ","], key=header.count) if any(c in header for c in ";\t,") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def _to_dates(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    text = s.astype(str)
    dayfirst = text.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").any()     # French Excel: 21/04/2020
    return pd.to_datetime(text, dayfirst=bool(dayfirst))


def calendar_from_file(path, trading_days):
    """Contract calendar exported from Bloomberg: one row per contract, a
    LAST_TRADEABLE_DT column and optionally FUT_NOTICE_FIRST (CSV or Excel).
    Keep only the contracts the generic steps through."""
    raw = _read_table(path)
    cols = {str(c).lower().strip(): c for c in raw.columns}
    ltd_col = cols.get("last_tradeable_dt") or cols.get("last_trade")
    if ltd_col is None:
        raise SystemExit(f"{path}: needs a LAST_TRADEABLE_DT column (found {list(raw.columns)})")
    ltd = _to_dates(raw[ltd_col])
    fnd_col = cols.get("fut_notice_first") or cols.get("first_notice")
    fnd = _to_dates(raw[fnd_col]) if fnd_col else ltd
    cal = pd.DataFrame({"last_trade": ltd, "roll_ref": pd.concat([ltd, fnd], axis=1).min(axis=1)})
    cal = cal.dropna(subset=["last_trade"])
    days = pd.DatetimeIndex(trading_days)
    cal = cal[(cal["last_trade"] >= days[0]) & (cal["last_trade"] <= days[-1])]
    cal = cal.sort_values("last_trade").drop_duplicates("last_trade")
    cal.index = cal["last_trade"].dt.to_period("M").rename("last_trade_month")
    return cal
