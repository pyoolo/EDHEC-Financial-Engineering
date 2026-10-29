"""
Python Labs -- Trump tariff threats: event study on the US Treasury curve
=========================================================================
Answers Part II of the group assignment (Q1.1 - Q1.4 and Q2.1).

Run from anywhere:      python 06_event_study.py
Outputs (figures + CSV tables) are written to  ./outputs/  with prefix "pl_".

Data
----
Daily Treasury constant-maturity yields (FRED DGS3MO ... DGS30), downloaded
with the course helper `functions_library/module_1/fred.py` and the API key in
`data/fred/api_key.py` (git-ignored).  The first successful download is cached
in `data/fred/treasury_yields_cache.csv`, so the script also runs without the
key / without internet (the cache is public FRED data, not a secret).

Structure of the script
-----------------------
  0. Configuration and data loading
  1. Event table  (the notebook's two dictionaries, copied verbatim)
  2. Event-study engine of the assignment (copied verbatim from the notebook)
  3. Q1.1 - Q1.4 : the assignment's methodology, original sample (to 2025-08-01)
  4. Q2.1        : methodological improvements and how they change Q1

Conventions and assumptions (highlighted for the grader)
--------------------------------------------------------
A1  Yields are in percent; all results are reported in basis points (bp).
A2  "Business days" = trading days present in the FRED data.  An event that
    falls on a weekend/holiday is moved to the next trading day (t = 0).
A3  The notebook measures y(t+k) - y(t).  For k < 0 this is MINUS the move
    into the event.  To avoid sign confusion we report three windows, all
    as "yield change over the window" (positive = yields rose):
        pre  [-30, 0] : y(t)    - y(t-30)   (run-up into the event)
        ann  [-1, +1] : y(t+1)  - y(t-1)    (announcement effect)
        post [0, +30] : y(t+30) - y(t)      (post-event drift)
    each minus the same-horizon non-event baseline (= the assignment's
    "abnormal change").
A4  "Reaction of targeted countries" (Q1.3) is measured on the US curve on
    the dates of the foreign responses (we only have US data).
A5  Level = 10Y, slope = 10Y - 2Y, curvature = 2*5Y - 2Y - 10Y, as in the
    notebook.  A PCA-based level is used as a robustness check in Q2.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # no window needed, figures go to files
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION AND DATA
# =============================================================================
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
CACHE = HERE / "data" / "fred" / "treasury_yields_cache.csv"
sys.path.insert(0, str(HERE))            # so that functions_library / data import

WINDOW = 30                               # [-30, +30] trading days (assignment)
START = "2015-01-01"
END_ORIGINAL = "2025-08-01"               # sample used in the notebook (Q1)
END_EXTENDED = "2025-10-31"               # extended sample (Q2 improvement)
RNG = np.random.default_rng(2764)         # fixed seed: placebo tests reproducible
N_PLACEBO = 5000
PLACEBO_EXCL = 1                          # placebo/drift days: no event within +-1 day

# Colours: EDHEC red / grey as in Part I, plus a blue and a green for maturities
RED, GREY, BLUE, GREEN, ORANGE = "#9e1b32", "#4d4d4d", "#2a78d6", "#1baf7a", "#eb6834"
MAT_COLORS = {"2Y": BLUE, "10Y": RED, "30Y": GREEN}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def load_yields() -> pd.DataFrame:
    """FRED download (course helper) with a CSV cache as fallback."""
    try:
        from data.fred.api_key import FRED_API_KEY
        from functions_library.module_1.fred import fetch_yield_curve_data
        pl_df = fetch_yield_curve_data(start_date=START, end_date=END_EXTENDED,
                                       FRED_API_KEY=FRED_API_KEY)
        df = pl_df.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(CACHE, index=False)
        print(f"FRED download OK, cache refreshed: {CACHE.name}")
    except Exception as exc:                       # no key / no internet
        print(f"FRED not available ({type(exc).__name__}); using cache {CACHE.name}")
        df = pd.read_csv(CACHE, parse_dates=["date"])
    # Missing values (single-maturity gaps) are forward-filled within 3 days
    return df.sort_values("date").ffill(limit=3).reset_index(drop=True)


# =============================================================================
# 1. EVENTS -- the notebook's dictionaries, copied verbatim (incl. its issues:
#    duplicated keys are silently overwritten by Python; see Q2.1)
# =============================================================================
trump_actions = {
    '2017-01-23': 'TPP Withdrawal announced',
    '2017-04-20': 'Commerce Dept. Steel dumping investigation launched',
    '2018-01-22': 'Solar panel and washing machine tariffs (30% and 20%)',
    '2018-03-01': 'Steel and Aluminum Tariffs Announced (25% steel, 10% aluminum)',
    '2018-03-22': 'China Tariffs Announced ($60B worth of Chinese goods)',
    '2018-04-03': 'China tariff list published ($50B in goods, 25%)',
    '2018-06-15': 'China Tariffs List Published (25% on $50B of goods)',
    '2018-07-06': 'First China tariffs take effect ($34B)',
    '2018-08-23': 'Second tranche China tariffs ($16B)',
    '2018-09-17': 'China Tariffs Phase 2 Announced (10% on additional $200B)',
    '2018-09-24': 'China tariffs Phase 2 implemented ($200B at 10%)',
    '2019-05-05': 'China Tariff Escalation Tweet (25% on $200B)',
    '2019-05-10': 'China tariffs raised to 25% ($200B worth)',
    '2019-08-01': 'China Tariffs Phase 3 Announced (10% on remaining $300B)',
    '2019-08-23': 'China tariff escalation tweet (existing and new tariffs)',
    '2019-09-01': 'China tariffs on $300B begin (15% on $112B)',
    '2019-12-15': 'Planned China tariff increase postponed (trade deal progress)',
    '2020-01-15': 'Phase One China trade deal signed',
    '2025-01-20': 'Inauguration - pledged immediate trade system overhaul',
    '2025-01-22': 'Executive order directing tariff review within 60 days',
    '2025-02-01': 'National emergency declared - 25% tariffs on Canada/Mexico, 10% on China (IEEPA)',
    '2025-02-10': 'Steel/aluminum tariff expansion - ended exemptions, raised aluminum to 25%',
    '2025-02-13': 'Announced reciprocal tariffs on all countries with trade barriers (April implementation)',
    '2025-02-21': 'Presidential memorandum investigating digital service taxes for retaliation',
    '2025-03-04': 'Chinese "fentanyl tariff" raised to 20% (USMCA tariffs suspended)',
    '2025-04-02': 'National emergency on trade deficit - 10% baseline tariff on all countries',
    '2025-04-05': 'Universal 10% tariff takes effect on nearly all imports',
    '2025-05-02': 'China/Hong Kong exemption removed again',
    '2025-07-02': 'Trade deal announced with Vietnam',
    '2025-07-04': 'One Big Beautiful Bill Act - eliminates de minimis exemption (2027)',
    '2025-07-07': 'Reciprocal tariffs delayed to August 1',
    '2025-07-08': 'Letters sent to 14 countries (Japan, S.Korea) threatening tariffs without deals',
    '2025-07-30': 'Cathode copper exempted from tariffs',
    '2025-08-06': 'Secondary tariff on India (25% additional) for buying Russian oil',
    '2025-08-06': 'Tariff rates finalized: Canada 35%, Mexico 30%, China 34%, EU 20%, Japan 24%',  # noqa: F601 duplicate key (kept as in notebook)
    '2025-08-15': 'Brazil hit with 50% tariffs, India total 50%, various Asian countries 25-36%',
    '2025-08-29': 'De minimis exemption ended early (originally scheduled 2027)',
}

country_responses = {
    '2018-04-02': 'China retaliatory tariffs announced ($3B US goods)',
    '2018-04-04': 'China escalates retaliation ($50B US goods, 25%)',
    '2018-06-16': 'China announces retaliatory tariffs ($50B)',
    '2018-07-06': 'China implements retaliatory tariffs ($34B)',
    '2018-08-03': 'China tariffs on $60B US goods (5-25%)',
    '2018-08-23': 'China retaliates on $16B US goods',
    '2018-09-18': 'China announces tariffs on $60B US goods',
    '2018-09-24': 'China implements tariffs on $60B US goods',
    '2019-05-13': 'China raises tariffs on $60B US goods to 25%',
    '2019-08-05': 'China allows yuan to weaken past 7/USD',
    '2019-08-23': 'China announces tariffs on $75B US goods',
    '2019-09-01': 'China implements tariffs on $75B US goods',
    '2019-12-19': 'China announces tariff reductions (Phase One)',
    '2018-03-02': 'EU threatens retaliation on steel/aluminum tariffs',
    '2018-06-22': 'EU implements retaliatory tariffs ($3.2B US goods)',
    '2018-11-02': 'EU complains to WTO about US auto tariff threats',
    '2018-05-31': 'Canada announces retaliatory tariffs ($16.6B)',
    '2018-07-01': 'Canada implements steel/aluminum retaliation',
    '2018-06-05': 'Mexico announces retaliatory tariffs (steel, pork, bourbon)',
    '2018-03-26': 'India raises tariffs on US almonds, apples',
    '2018-06-21': 'Turkey raises tariffs on US goods (rice, cars, alcohol)',
    '2018-08-10': 'Russia considers tariffs on US goods',
    '2025-02-02': 'Canada, Mexico, China condemn tariffs as "unjustified"',
    '2025-03-04': 'Canada implements immediate 25% counter-tariffs on $20B US imports',
    '2025-03-04': 'China retaliates with 10-15% tariffs on US agricultural imports',  # noqa: F601 duplicate key
    '2025-03-05': 'Mexico announces retaliatory measures (5-20% on pork, cheese, steel)',
    '2025-03-15': 'China adds 15 US companies to export control list',
    '2025-04-04': 'EU announces $100B counter-tariffs if 30% rate implemented',
    '2025-04-06': 'China escalates to 145% tariffs on US goods (US at 145% on China)',
    '2025-04-10': 'Brazil passes reciprocity bill allowing tariff retaliation',
    '2025-05-28': 'Court of International Trade declares Trump tariff orders invalid',
    '2025-06-15': 'Canada provinces boycott US spirits amid trade war',
    '2025-07-12': 'Japan threatens "all options on table" for tariff response',
    '2025-07-20': 'EU confirms 30% counter-tariffs on $100B US goods',
    '2025-08-01': 'Malaysian government rules out retaliatory tariffs',
    '2025-08-05': 'New Zealand announces no retaliation plans',
    '2025-08-08': 'India maintains higher tariffs than US on various products (80% rice, 50% apples)',
    '2025-08-10': 'Brazil president vows reciprocation but holds back on steel/aluminum',
    '2025-08-12': 'China continues agricultural tariff escalation (chicken, wheat, corn 15%)',
    '2025-08-15': 'EU-Canada-China coordination on $330B retaliatory measures',
}


def build_events_table(trump_actions: dict, country_responses: dict) -> pd.DataFrame:
    """Notebook version (verbatim)."""
    rows = []
    for d, desc in trump_actions.items():
        rows.append({'date': pd.to_datetime(d), 'description': desc, 'event_type': 'trump_action'})
    for d, desc in country_responses.items():
        rows.append({'date': pd.to_datetime(d), 'description': desc, 'event_type': 'country_response'})
    events = (pd.DataFrame(rows).drop_duplicates(subset=['date', 'event_type'])
              .sort_values('date').reset_index(drop=True))
    events['term'] = np.where(events['date'] < pd.Timestamp('2021-01-01'), 'first_term', 'second_term')
    return events


# =============================================================================
# 2. EVENT-STUDY ENGINE OF THE ASSIGNMENT (verbatim from the notebook)
# =============================================================================
def event_study(dates, series, event_dates, window: int = WINDOW) -> pd.DataFrame:
    """Mean change y(t+k)-y(t) over events, minus the same-horizon mean change
    over non-event dates (dates more than `window` days away from any event)."""
    dates = pd.DatetimeIndex(dates)
    s = pd.Series(np.asarray(series), index=dates).sort_index()
    s = s[~s.index.duplicated(keep='first')]
    n = len(s)

    event_positions = []
    for ed in pd.DatetimeIndex(event_dates):
        pos = s.index.searchsorted(ed)           # next trading day on/after event
        if pos < n:
            event_positions.append(pos)
    event_positions = sorted(set(event_positions))  # same-day events counted once

    is_near_event = np.zeros(n, dtype=bool)
    for p in event_positions:
        is_near_event[max(0, p - window):min(n, p + window + 1)] = True
    non_event_positions = np.where(~is_near_event)[0]
    non_event_set = set(non_event_positions.tolist())

    rows, values = [], s.values
    for offset in range(-window, window + 1):
        changes = [values[p + offset] - values[p] for p in event_positions if 0 <= p + offset < n]
        base = [values[p + offset] - values[p] for p in non_event_positions
                if 0 <= p + offset < n and (p + offset) in non_event_set]
        ev, bs = (np.mean(changes) if changes else np.nan), (np.mean(base) if base else np.nan)
        rows.append({'offset': offset, 'event_mean_change': ev, 'n_events_used': len(changes),
                     'baseline_mean_change': bs, 'abnormal_change': ev - bs})
    return pd.DataFrame(rows).set_index('offset')


def level_slope_curvature(df: pd.DataFrame) -> pd.DataFrame:
    out = df[['date']].copy()
    out['level'] = df['10Y']
    out['slope'] = df['10Y'] - df['2Y']
    out['curvature'] = 2 * df['5Y'] - (df['2Y'] + df['10Y'])
    return out


# ---------- small helpers on top of the engine --------------------------------
SERIES = ["3M", "2Y", "5Y", "10Y", "30Y", "level", "slope", "curvature"]


def add_lsc(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, level_slope_curvature(df).drop(columns="date")], axis=1)


def run_paths(df, dates, cols=SERIES):
    """Event-study path (abnormal change, bp) for several series."""
    return pd.DataFrame({c: event_study(df["date"], df[c], dates)["abnormal_change"] * 100
                         for c in cols})


def windows_from_path(path: pd.DataFrame) -> pd.DataFrame:
    """Convert the notebook's 'relative to t' path into the three windows of A3."""
    return pd.DataFrame({
        "pre [-30,0]": -path.loc[-WINDOW],
        "ann [-1,+1]": path.loc[1] - path.loc[-1],
        "post [0,+30]": path.loc[WINDOW],
    }).T


def n_used(df, dates):
    es = event_study(df["date"], df["10Y"], dates)
    return int(es.loc[-WINDOW, "n_events_used"]), int(es.loc[WINDOW, "n_events_used"])


def plot_paths(ax, path, cols, title, styles=None):
    for c in cols:
        st = (styles or {}).get(c, {})
        ax.plot(path.index, path[c], lw=1.8, color=st.get("color", MAT_COLORS.get(c, GREY)),
                ls=st.get("ls", "-"), label=st.get("label", c))
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8, ls=":")
    ax.set_title(title)
    ax.set_xlabel("Event time (trading days)")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


# =============================================================================
# 3. Q1 -- ASSIGNMENT METHODOLOGY, ORIGINAL SAMPLE
# =============================================================================
def part_q1(ycd_full: pd.DataFrame):
    ycd = add_lsc(ycd_full[ycd_full["date"] <= END_ORIGINAL].reset_index(drop=True))
    events = build_events_table(trump_actions, country_responses)
    ta = events[events.event_type == "trump_action"]
    cr = events[events.event_type == "country_response"]

    print(f"\nQ1 sample {ycd.date.min():%Y-%m-%d} -> {ycd.date.max():%Y-%m-%d}, {len(ycd)} days")
    print(f"Events in table: {len(events)} (trump_action {len(ta)}, country_response {len(cr)})")
    print("Distinct event days inside the sample (n at -30 / +30):",
          "all", n_used(ycd, events.date), "| TA", n_used(ycd, ta.date), "| CR", n_used(ycd, cr.date))

    paths = {"all": run_paths(ycd, events.date), "TA": run_paths(ycd, ta.date),
             "CR": run_paths(ycd, cr.date)}
    for typ, sub in [("TA", ta), ("CR", cr)]:
        for term in ["first_term", "second_term"]:
            paths[f"{typ}_{term}"] = run_paths(ycd, sub[sub.term == term].date)

    # ---- tables ------------------------------------------------------------
    tables = {k: windows_from_path(v) for k, v in paths.items()}
    t11 = tables["TA"][["3M", "2Y", "5Y", "10Y", "30Y"]].round(1)
    t12 = pd.concat({"Trump actions": tables["TA"][["level", "slope", "curvature"]],
                     "All events": tables["all"][["level", "slope", "curvature"]]}, axis=1).round(1)
    t13 = tables["CR"][["3M", "2Y", "5Y", "10Y", "30Y", "slope", "curvature"]].round(1)
    rows = []
    for typ, lab in [("TA", "Trump actions"), ("CR", "Country responses")]:
        for term, tl in [("first_term", "1st term"), ("second_term", "2nd term")]:
            sub = (ta if typ == "TA" else cr)
            n = n_used(ycd, sub[sub.term == term].date)
            w = tables[f"{typ}_{term}"]
            rows.append({"group": f"{lab}, {tl}", "n (-30/+30)": f"{n[0]}/{n[1]}",
                         **{f"{c} pre": w.loc["pre [-30,0]", c] for c in ["2Y", "10Y", "30Y", "slope"]},
                         **{f"{c} post": w.loc["post [0,+30]", c] for c in ["2Y", "10Y", "30Y", "slope"]}})
    t14 = pd.DataFrame(rows).set_index("group").round(1)
    for name, t in [("q11_trump_actions", t11), ("q12_lsc", t12),
                    ("q13_country_responses", t13), ("q14_terms", t14)]:
        t.to_csv(OUT / f"pl_{name}.csv")
        print(f"\n{name} (bp):\n{t.to_string()}")
    base = event_study(ycd["date"], ycd["10Y"], events.date)["baseline_mean_change"] * 100
    print(f"\nNon-event baseline 10Y change at +30: {base.loc[30]:.1f} bp")

    # ---- figures -----------------------------------------------------------
    mats = ["2Y", "10Y", "30Y"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    plot_paths(axes[0], paths["TA"], mats, "Q1.1 Trump actions")
    plot_paths(axes[1], paths["CR"], mats, "Q1.3 Country responses")
    axes[0].set_ylabel("Abnormal change vs. day 0 (bp)")
    axes[0].legend(loc="lower left")
    save(fig, "pl_q11_q13_paths.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, c, t in zip(axes, ["level", "slope", "curvature"],
                        ["Level (10Y)", "Slope (10Y-2Y)", "Curvature (2x5Y-2Y-10Y)"]):
        plot_paths(ax, paths["TA"].rename(columns={c: "TA"}), ["TA"], t,
                   {"TA": {"color": RED, "label": "Trump actions"}})
        ax.plot(paths["all"].index, paths["all"][c], color=GREY, ls="--", lw=1.4, label="All events")
    axes[0].set_ylabel("Abnormal change vs. day 0 (bp)")
    axes[0].legend(loc="lower left")
    save(fig, "pl_q12_lsc.png")

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
    for i, (typ, lab) in enumerate([("TA", "Trump actions"), ("CR", "Country responses")]):
        for j, c in enumerate(["2Y", "10Y", "30Y", "slope"]):
            ax = axes[i, j]
            ax.plot(paths[f"{typ}_first_term"].index, paths[f"{typ}_first_term"][c], color=BLUE, lw=1.8, label="1st term")
            ax.plot(paths[f"{typ}_second_term"].index, paths[f"{typ}_second_term"][c], color=ORANGE, lw=1.8, label="2nd term")
            ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8, ls=":")
            ax.set_title(f"{lab}: {c if c != 'slope' else 'slope 10Y-2Y'}")
            if i == 1:
                ax.set_xlabel("Event time (trading days)")
        axes[i, 0].set_ylabel("Abnormal change (bp)")
    axes[0, 0].legend(loc="lower left")
    save(fig, "pl_q14_terms.png")
    return ycd, events


# =============================================================================
# 4. Q2.1 -- IMPROVED METHODOLOGY
# =============================================================================
# (a) Cleaned event list.  sign = +1 escalation, -1 de-escalation.
#     Dropped (not a tariff shock / not a foreign response): see DROPPED.
#     Added: three de-escalation events missing from the notebook list.
SIGN_OVERRIDES = {  # (date, type) -> sign ; default is +1 (escalation)
    ("2019-12-15", "trump_action"): -1, ("2020-01-15", "trump_action"): -1,
    ("2025-07-02", "trump_action"): -1, ("2025-07-07", "trump_action"): -1,
    ("2025-07-30", "trump_action"): -1, ("2019-12-19", "country_response"): -1,
}
DROPPED = {  # reason for removal
    ("2025-05-28", "country_response"): "US court ruling, not a foreign response",
    ("2025-08-01", "country_response"): "announces NO retaliation",
    ("2025-08-05", "country_response"): "announces NO retaliation",
    ("2025-08-08", "country_response"): "status quo, no new measure",
    ("2018-08-10", "country_response"): "only 'considers' tariffs",
    ("2025-02-02", "country_response"): "verbal condemnation only",
}
ADDED = [  # missing de-escalation milestones (public, widely reported dates)
    ("2025-04-09", "trump_action", -1, "90-day pause of reciprocal tariffs (ex-China)"),
    ("2025-05-12", "trump_action", -1, "US-China Geneva agreement: tariffs cut for 90 days"),
    ("2025-07-27", "trump_action", -1, "US-EU framework deal (15% tariff ceiling)"),
]
# FOMC decision dates (scheduled + emergency) -- confounding macro news
FOMC = pd.to_datetime([
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-03-15",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29"])
TERMS = {"first_term": ("2017-01-20", "2021-01-19"), "second_term": ("2025-01-20", END_EXTENDED)}


def clean_events(dates_index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for typ, dct in [("trump_action", trump_actions), ("country_response", country_responses)]:
        for d, desc in dct.items():
            if (d, typ) in DROPPED:
                continue
            rows.append({"date": pd.Timestamp(d), "event_type": typ, "description": desc,
                         "sign": SIGN_OVERRIDES.get((d, typ), +1)})
    for d, typ, sgn, desc in ADDED:
        rows.append({"date": pd.Timestamp(d), "event_type": typ, "description": desc, "sign": sgn})
    ev = pd.DataFrame(rows)
    ev["term"] = np.where(ev.date < pd.Timestamp("2021-01-01"), "first_term", "second_term")
    ev["pos"] = dates_index.searchsorted(ev.date)              # trading day t = 0
    ev = ev[ev.pos < len(dates_index)].copy()
    # same trading day for a Trump action and a foreign response -> contaminated
    both = ev.groupby("pos").event_type.nunique()
    ev["overlap_other_type"] = ev.pos.map(both) > 1
    # within +-1 trading day of an FOMC decision
    fomc_pos = dates_index.searchsorted(FOMC)
    ev["near_fomc"] = ev.pos.apply(lambda p: np.any(np.abs(fomc_pos - p) <= 1))
    # one observation per (trading day, type, sign)
    ev = (ev.sort_values("date").groupby(["pos", "event_type", "sign"], as_index=False)
          .agg(date=("date", "first"), term=("term", "first"),
               description=("description", " | ".join),
               overlap_other_type=("overlap_other_type", "max"), near_fomc=("near_fomc", "max")))
    # first event of a cluster: no other event in the previous 5 trading days
    allpos = np.sort(ev.pos.unique())
    ev["cluster_head"] = ev.pos.apply(lambda p: not np.any((allpos < p) & (allpos >= p - 5)))
    return ev.sort_values("pos").reset_index(drop=True)


IMP_WINDOWS = {"[0]": (0, 0), "[-1,+1]": (-1, 1), "[0,+5]": (0, 5), "[0,+30]": (0, 30), "[-30,-1]": (-30, -1)}


def car(values: np.ndarray, pos: np.ndarray, a: int, b: int) -> np.ndarray:
    """Cumulative change over trading days a..b around t: y(t+b) - y(t+a-1)."""
    ok = (pos + a - 1 >= 0) & (pos + b < len(values))
    out = np.full(len(pos), np.nan)
    out[ok] = values[pos[ok] + b] - values[pos[ok] + a - 1]
    return out


def improved_test(df, ev, col, win, pools, drift=None):
    """Abnormal change over window `win` and its placebo p-value.

    Expected change (default, drift=None): the average change over windows of
    the same length that start on placebo days of the same mandate.  Estimate
    and test then measure the same thing: are event windows different from
    ordinary windows of that mandate?  If `drift` is given (old design), the
    expected change is drift (bp/day) x window length instead.
    Placebo: N_PLACEBO draws of the same number of pseudo-events per mandate
    from that mandate's pool; two-sided p-value."""
    a, b = IMP_WINDOWS[win]
    L = b - a + 1
    vals = df[col].values * 100
    raw = car(vals, ev.pos.values, a, b)
    keep = ~np.isnan(raw)
    raw, terms = raw[keep], ev.term.values[keep]
    if len(raw) == 0:
        return dict(n=0, mean=np.nan, t=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    ab = np.empty_like(raw)
    sims = np.zeros(N_PLACEBO)
    for term in np.unique(terms):
        k = int((terms == term).sum())
        pc = car(vals, pools[term], a, b)
        pc = pc[~np.isnan(pc)]
        expected = pc.mean() if drift is None else drift[(term, col)] * L
        ab[terms == term] = raw[terms == term] - expected
        draws = RNG.choice(pc - expected, size=(N_PLACEBO, k), replace=True)
        sims += draws.sum(axis=1)
    sims /= len(ab)
    m = ab.mean()
    p = (np.sum(np.abs(sims - sims.mean()) >= abs(m - sims.mean())) + 1) / (N_PLACEBO + 1)
    t = m / (ab.std(ddof=1) / np.sqrt(len(ab))) if len(ab) > 1 else np.nan
    return dict(n=len(ab), mean=m, t=t, p=p, lo=np.percentile(sims, 2.5), hi=np.percentile(sims, 97.5))


def part_q2(ycd_full: pd.DataFrame, events_q1: pd.DataFrame):
    df = add_lsc(ycd_full.copy())
    # PCA level (first principal component of daily changes) as alternative level
    mats = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    dchg = df[mats].diff().dropna()
    w = np.linalg.eigh(np.cov(dchg.T.values))[1][:, -1]       # 1st eigenvector
    df["pca_level"] = df[mats].values @ (w / w.sum())          # weights rescaled to sum to one
    idx = pd.DatetimeIndex(df.date)
    ev = clean_events(idx)
    ev.to_csv(OUT / "pl_q2_clean_events.csv", index=False)

    # Placebo pools by mandate.
    #  * Main design: pool = days of the same mandate with no event within
    #    +-PLACEBO_EXCL (=1) days; expected change = mean change over windows of
    #    the same length starting on pool days (see improved_test).
    #  * Old design (sensitivity): pool = days with no event within +-5 days and
    #    expected change = mean daily change on that pool x window length.  In
    #    2025 that pool has only 56 unusually calm days, so the placebo
    #    distribution is too narrow and the p-values too optimistic.
    def make_pools(excl):
        near = np.zeros(len(df), bool)
        for p in ev.pos:
            near[max(0, p - excl):p + excl + 1] = True
        pl, dr = {}, {}
        for term, (s, e) in TERMS.items():
            in_term = ((df.date >= s) & (df.date <= e)).values
            pl[term] = np.where(in_term & ~near)[0]
            for c in SERIES + ["pca_level"]:
                dd = df[c].diff().values * 100
                dr[(term, c)] = np.nanmean(dd[pl[term]])
        return pl, dr

    pools, _ = make_pools(PLACEBO_EXCL)
    pools5, drift5 = make_pools(5)
    drift = None                                     # main design: window-mean expectation
    print("Placebo pool size (days) per term, +-%d / +-5 exclusion:" % PLACEBO_EXCL,
          {t: (len(pools[t]), len(pools5[t])) for t in TERMS})

    print("\nQ2 extended sample to", df.date.max().date(), "| cleaned events:", len(ev),
          "| escalations:", int((ev.sign > 0).sum()), "| de-escalations:", int((ev.sign < 0).sum()))

    # ---- why the notebook baseline is biased: mean 30-day change by period --
    ch30 = (df.set_index("date")["10Y"].shift(-30) - df.set_index("date")["10Y"]) * 100
    periods = {"2015-16 (pre-Trump)": ("2015-01-01", "2016-12-31"),
               "2017-20 (1st term)": ("2017-01-20", "2021-01-19"),
               "2021-24 (Biden)": ("2021-01-20", "2025-01-19"),
               "2025 (2nd term)": ("2025-01-20", END_ORIGINAL)}
    base_tab = pd.Series({k: ch30.loc[s:e].mean() for k, (s, e) in periods.items()}).round(1)
    base_tab.to_csv(OUT / "pl_q2_drift_by_period.csv")
    print("\nMean 30-day 10Y change by period (bp):\n", base_tab.to_string())

    # ---- main improved results ------------------------------------------------
    esc = ev[ev.sign > 0]
    groups = {
        "Trump actions (esc.)": esc[esc.event_type == "trump_action"],
        "  1st term": esc[(esc.event_type == "trump_action") & (esc.term == "first_term")],
        "  2nd term": esc[(esc.event_type == "trump_action") & (esc.term == "second_term")],
        "Country responses (esc.)": esc[esc.event_type == "country_response"],
        "  1st term ": esc[(esc.event_type == "country_response") & (esc.term == "first_term")],
        "  2nd term ": esc[(esc.event_type == "country_response") & (esc.term == "second_term")],
        "De-escalations (all)": ev[ev.sign < 0],
    }
    res = []
    for g, sub in groups.items():
        for col in ["2Y", "10Y", "30Y", "slope"]:
            for win in ["[-1,+1]", "[0,+5]", "[0,+30]"]:
                sub_w = sub[sub.cluster_head] if win == "[0,+30]" else sub   # de-cluster long window
                r = improved_test(df, sub_w, col, win, pools, drift)
                res.append({"group": g, "series": col, "window": win, **r})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "pl_q2_improved_results.csv", index=False)

    piv = res.pivot_table(index=["group"], columns=["window", "series"], values="mean", sort=False).round(1)
    pv = res.pivot_table(index=["group"], columns=["window", "series"], values="p", sort=False)
    stars = pv.map(lambda p: "**" if p < 0.05 else ("*" if p < 0.10 else ""))
    nn = res[res.series == "10Y"].pivot_table(index="group", columns="window", values="n", sort=False)
    print("\nImproved: abnormal change (bp), * p<0.10, ** p<0.05 (placebo)")
    print((piv.astype(str) + stars).to_string())
    print("n events:\n", nn.to_string())

    # ---- sensitivity: placebo pool +-1 (main) vs. +-5 (calm days only) ---------
    sens = []
    for g, sub in groups.items():
        for col in ["2Y", "10Y", "30Y", "slope"]:
            for win in ["[-1,+1]", "[0,+5]", "[0,+30]"]:
                sub_w = sub[sub.cluster_head] if win == "[0,+30]" else sub
                r5 = improved_test(df, sub_w, col, win, pools5, drift5)
                r1 = res[(res.group == g) & (res.series == col) & (res.window == win)].iloc[0]
                sens.append({"group": g.strip(), "series": col, "window": win, "n": int(r1["n"]),
                             "mean_new": r1["mean"], "p_new": r1["p"],
                             "mean_old": r5["mean"], "p_old": r5["p"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "pl_q2_placebo_sensitivity.csv", index=False)
    lost = sens[(sens.p_old < 0.05) & (sens.p_new >= 0.05)]
    gained = sens[(sens.p_old >= 0.05) & (sens.p_new < 0.05)]
    print(f"\nSensitivity to the placebo design: {int((sens.p_old < 0.05).sum())} cells significant at 5% "
          f"with the old design (+-5 calm-day pool) vs {int((sens.p_new < 0.05).sum())} with the new one.")
    print("Lose significance:\n", lost.round(3).to_string(index=False))
    print("Gain significance:\n", gained.round(3).to_string(index=False))

    # ---- robustness of the announcement effect --------------------------------
    rob = []
    ta_esc = groups["Trump actions (esc.)"]
    variants = {
        "Baseline improved (esc. Trump actions)": ta_esc,
        "excl. same-day foreign response": ta_esc[~ta_esc.overlap_other_type],
        "excl. +-1 day of FOMC": ta_esc[~ta_esc.near_fomc],
        "cluster heads only": ta_esc[ta_esc.cluster_head],
        "original sample only (to 2025-08-01)": ta_esc[ta_esc.date <= END_ORIGINAL],
    }
    for v, sub in variants.items():
        for col in ["2Y", "10Y", "30Y", "pca_level"]:
            r = improved_test(df, sub, col, "[-1,+1]", pools, drift)
            rob.append({"variant": v, "series": col, "n": r["n"], "mean": r["mean"], "p": r["p"]})
    # second-term results: do they survive without April 2025 (Liberation Day,
    # basis-trade deleveraging) or without overlapping/FOMC days?
    apr = (ev.date >= "2025-04-01") & (ev.date <= "2025-04-30")
    for typ, lab in [("trump_action", "2nd term Trump esc."), ("country_response", "2nd term foreign esc.")]:
        base = esc[(esc.event_type == typ) & (esc.term == "second_term")]
        for v, sub in {f"{lab}": base,
                       f"{lab}, excl. April 2025": base[~apr.loc[base.index]],
                       f"{lab}, excl. same-day overlap": base[~base.overlap_other_type],
                       f"{lab}, excl. +-1 day of FOMC": base[~base.near_fomc]}.items():
            for col in ["2Y", "10Y", "30Y", "pca_level"]:
                r = improved_test(df, sub, col, "[-1,+1]", pools, drift)
                rob.append({"variant": v, "series": col, "n": r["n"], "mean": r["mean"], "p": r["p"]})
    rob = pd.DataFrame(rob)
    rob.to_csv(OUT / "pl_q2_robustness.csv", index=False)
    print("\nRobustness, Trump escalations, [-1,+1] (bp, placebo p):")
    print(rob.pivot_table(index="variant", columns="series", values=["mean", "p"], sort=False).round(3).to_string())

    # ---- figures ----------------------------------------------------------------
    # (1) dot plot with placebo 95% bands, [-1,+1] and [0,+30], 10Y and 2Y
    show = ["Trump actions (esc.)", "  1st term", "  2nd term",
            "Country responses (esc.)", "  1st term ", "  2nd term ", "De-escalations (all)"]
    n_short = res[(res.series == "10Y") & (res.window == "[-1,+1]")].set_index("group").loc[show, "n"]
    n_long = res[(res.series == "2Y") & (res.window == "[0,+30]")].set_index("group").loc[show, "n"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.0), sharey=True)
    y = np.arange(len(show))[::-1]
    for ax, (col, win) in zip(axes, [("2Y", "[-1,+1]"), ("10Y", "[-1,+1]"), ("30Y", "[-1,+1]"), ("2Y", "[0,+30]")]):
        sub = res[(res.series == col) & (res.window == win)].set_index("group").loc[show]
        ax.hlines(y, sub.lo, sub.hi, color="#c8c8c8", lw=5, label="placebo 95% band")
        sig = (sub.p < 0.05).values
        ax.scatter(sub["mean"].values[~sig], y[~sig], color=GREY, s=28, zorder=3, label="not significant")
        ax.scatter(sub["mean"].values[sig], y[sig], color=RED, s=34, zorder=3, label="p < 0.05 (placebo)")
        ax.axvline(0, color="#999", lw=0.8)
        ttl = f"{col}, window {win}"
        if win == "[0,+30]":
            ttl += "\n(cluster heads; n = " + "/".join(str(int(v)) for v in n_long.values) + ")"
        ax.set_title(ttl)
        ax.set_xlabel("Abnormal change (bp)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([(g.strip() if not g.startswith("  ") else "   " + g.strip()) + f" (n={int(n)})"
                             for g, n in zip(show, n_short.values)])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "pl_q2_improved_dotplot.png")

    # (2) Liberation Day case study: timing and long-end sell-off
    s, e = "2025-03-24", "2025-05-16"
    sub = df[(df.date >= s) & (df.date <= e)].set_index("date")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for c in ["2Y", "10Y", "30Y"]:
        ax.plot(sub.index, sub[c], color=MAT_COLORS[c], lw=1.8, label=c)
    top = sub[["2Y", "10Y", "30Y"]].values.max()
    for d, lab, dy, ha in [("2025-04-02", "Apr 2: 'Liberation Day'\n(announced after the close)", 0.20, "right"),
                           ("2025-04-09", "Apr 9: 90-day pause", 0.20, "left"),
                           ("2025-05-12", "May 12: Geneva deal", 0.20, "right")]:
        ax.axvline(pd.Timestamp(d), color=GREY, ls=":", lw=1)
        ax.text(pd.Timestamp(d) + pd.Timedelta(days=-0.6 if ha == "right" else 0.6), top + dy, lab,
                fontsize=7, ha=ha, va="top", color=GREY)
    ax.set_ylim(None, top + 0.24)
    ax.set_ylabel("Yield (%)")
    ax.set_title("April-May 2025: Treasury yields around the reciprocal-tariff shock")
    ax.legend(loc="lower left", ncol=3)
    save(fig, "pl_q2_liberation_day.png")

    daily = (sub.loc["2025-04-01":"2025-04-11", ["2Y", "10Y", "30Y", "slope"]].diff() * 100).round(1)
    daily.to_csv(OUT / "pl_q2_liberation_day_daily.csv")
    print("\nDaily changes around Liberation Day (bp):\n", daily.to_string())
    return ev, res, rob


if __name__ == "__main__":
    data = load_yields()
    ycd_q1, events_q1 = part_q1(data)
    part_q2(data, events_q1)
    print("\nDone. Outputs in", OUT)
