"""Central configuration: paths and research parameters.

Every number that drives a result lives here, so a robustness check
is a one-line change.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Bloomberg data are licensed and never committed to the repository.
# For commodity CODE (e.g. CL, GC, HG, C) the code looks for the data in
# this order:
#   1. python src/run_all.py --commodity CODE --data path/to/file.xlsx
#   2. environment variable VRP_DATA_FILE (only when running one commodity)
#   3. data/raw/CODE.xlsx (git-ignored)
# An optional Bloomberg contract calendar can sit next to it:
#   data/raw/CODE_calendar.csv   (see data/README.md)
DATA_ENV_VAR = "VRP_DATA_FILE"
RAW_DIR = ROOT / "data" / "raw"


def default_data_file(code):
    return RAW_DIR / f"{code}.xlsx"


def default_calendar_file(code):
    return RAW_DIR / f"{code}_calendar.csv"


def processed_dir(code):
    return ROOT / "data" / "processed" / code


def results_dir(code):
    return ROOT / "results" / code


# --- Futures roll -----------------------------------------------------------
# Leave the front contract this many trading days before its roll reference
# date (last trading day, or first notice day for physical delivery) and
# hold the second contract instead. Avoids expiry-week illiquidity, e.g. the
# negative WTI settlement of 20 April 2020.
ROLL_DAYS_BEFORE_EXPIRY = 5

# --- Variance-swap proxy -----------------------------------------------------
ANNUALISATION = 252           # trading days per year
IV_FIELD = "iv30"             # 30-day constant-maturity ATM implied vol

# --- Timing rules ------------------------------------------------------------
MIN_HISTORY_MONTHS = 12       # expanding-window rules switch on after this
                              # (12 keeps the 2008 crisis in the test; 36 is
                              # reported as a robustness check)
PERCENTILE = 0.80             # "high" = above the expanding 80th percentile
VOL_SCALING_CAP = 2.0         # maximum weight in the Moreira-Muir rule
IV_JUMP_THRESHOLD = 0.20      # IV up more than 20% over 5 trading days
IV_JUMP_WINDOW = 5
IV_TREND_WINDOW = 63          # ~3 months of trading days
REDUCED_WEIGHT = 0.5          # "reduce exposure" = halve it

# --- Evaluation --------------------------------------------------------------
TCOST_VOL_POINTS = [0.0, 0.5, 1.0, 2.0]   # bid-ask cost per trade, vol points
BOOTSTRAP_REPS = 5000
BOOTSTRAP_BLOCK = 6           # months, circular block bootstrap
NEWEY_WEST_LAGS = 3
SPLIT_DATE = "2016-12-31"     # in-sample / out-of-sample split
RANDOM_SEED = 42

# Stress episodes reported for every commodity (months of realised returns)
CRISES = {
    "GFC 2008": ("2008-09", "2008-12"),
    "Commodity crash 2014-16": ("2014-07", "2016-02"),
    "COVID 2020": ("2020-02", "2020-04"),
    "Ukraine 2022": ("2022-02", "2022-06"),
    "Tariffs 2025": ("2025-03", "2025-04"),
}


def resolve_data_path(code, cli_path=None, use_env=True):
    """Return the data file for `code`, or stop with instructions if missing."""
    env = os.environ.get(DATA_ENV_VAR) if use_env else None
    candidates = [cli_path, env, default_data_file(code)]
    path = Path(next(c for c in candidates if c)).expanduser()
    if not path.exists():
        raise SystemExit(
            f"[{code}] data file not found: {path}\n"
            "The Bloomberg data are not in the repository (see data/README.md).\n"
            f"Copy your export to data/raw/{code}.xlsx, set {DATA_ENV_VAR},\n"
            "or pass --data PATH.\n"
            "To test the code without Bloomberg data:\n"
            f"  python scripts/make_synthetic_data.py --commodity {code}\n"
            f"  python src/run_all.py --commodity {code} --data data/raw/synthetic_{code}.xlsx"
        )
    return path
