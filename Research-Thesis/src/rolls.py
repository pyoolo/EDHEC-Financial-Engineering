"""Roll-adjusted daily returns of a position that always holds one contract.

Inputs are the Bloomberg generics F1 (front) and F2 (second) and the
contract calendar from commodities.py. The generics switch contract on
the day after each last trading day; the position switches earlier, a
fixed number of trading days before the roll reference date (the earlier
of last trading day and first notice day).
"""
import numpy as np
import pandas as pd

from config import ROLL_DAYS_BEFORE_EXPIRY


def check_calendar(df, last_trade, offsets=range(-3, 4)):
    """Validate the calendar against the data.

    When F1 and F2 switch contract, the F1-F2 spread jumps. The median
    absolute daily change in log(F1/F2) should therefore peak on the day
    after each last trading day (offset +1). Returns a table by offset.
    """
    dspread = np.log(df["f1"].where(df["f1"] > 0) / df["f2"]).diff().abs()
    pos = np.searchsorted(df.index.values, pd.DatetimeIndex(last_trade).values)
    rows = []
    for off in offsets:
        p = pos + off
        p = p[(p > 0) & (p < len(df))]
        rows.append({"offset_days": off,
                     "median_abs_spread_change": dspread.iloc[p].median(),
                     "ratio_to_all_days": dspread.iloc[p].median() / dspread.median()})
    return pd.DataFrame(rows)


def front_returns(df, calendar, roll_days=ROLL_DAYS_BEFORE_EXPIRY):
    """Daily log returns of a position that holds F1, switching to F2
    `roll_days` trading days before F1's roll reference date.

    Each return is computed on the contract held at the previous close:
      * F1 held             -> log F1_t / F1_{t-1}
      * F2 held             -> log F2_t / F2_{t-1}
      * generics just rolled -> log F1_t / F2_{t-1} (yesterday's F2 is today's F1)
    """
    dates = df.index
    pos = np.arange(len(dates))
    ltd_pos = np.searchsorted(dates.values, pd.DatetimeIndex(calendar["last_trade"]).values)
    ref_pos = np.searchsorted(dates.values, pd.DatetimeIndex(calendar["roll_ref"]).values)
    keep = ltd_pos < len(dates)
    ltd_pos, ref_pos = ltd_pos[keep], ref_pos[keep]

    # the contract currently in F1 is the first whose last trading day is today or later
    nxt = np.searchsorted(ltd_pos, pos, side="left")
    has_next = nxt < len(ltd_pos)
    ref_next = np.where(has_next, ref_pos[np.minimum(nxt, len(ltd_pos) - 1)], len(dates) + 999)
    in_window = pd.Series(pos >= ref_next - roll_days, index=dates)   # holding F2 at close
    is_ltd = pd.Series(np.isin(pos, ltd_pos), index=dates)

    f1, f2 = np.log(df["f1"].where(df["f1"] > 0)), np.log(df["f2"])
    r_f1, r_f2 = f1 - f1.shift(1), f2 - f2.shift(1)
    r_cross = f1 - f2.shift(1)

    held_f2 = in_window.shift(1, fill_value=False)
    generic_rolled = is_ltd.shift(1, fill_value=False)
    r = np.where(generic_rolled, r_cross, np.where(held_f2, r_f2, r_f1))
    out = pd.DataFrame({"ret": r, "held_f2": held_f2, "generic_roll": generic_rolled},
                       index=dates)
    return out.iloc[1:]
