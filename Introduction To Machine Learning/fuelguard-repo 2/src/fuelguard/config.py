"""Central configuration: paths, product parameters and evaluation design.

Every number that shapes the product or the evaluation lives here, so that the
report can point to a single place and a reviewer can change one value and rerun.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_XLSX = ROOT / "data" / "bloomberg_xlsx"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

BLOOMBERG_CSV = DATA_RAW / "bloomberg_selected.csv"
ECB_XML = DATA_RAW / "ecb_eurusd.xml"

# ---------------------------------------------------------------- product
LITRES_PER_GALLON = 3.785411784
HORIZON = 21            # trading days in the hedged month
EXECUTION_LAG = 1       # decision at close of t, price locked at settlement of t+1
SPIKE_THRESHOLD = 0.05  # "spike" = next month costs >5% more than the price lockable today
HEDGE_COST = 0.01       # supplier margin on a fixed-price contract, % of the lockable price
HEDGE_COST_GRID = [0.0, 0.01, 0.02]  # sensitivity analysis
MONTHLY_LITRES = 25_000 # diesel used by a ~10-truck Italian haulier, used to express results in EUR

# ---------------------------------------------------------------- evaluation
SAMPLE_START = "2007-01-01"
FIRST_TEST_YEAR = 2012      # walk-forward: test years 2012, 2013, ..., last full year of data
PURGE = HORIZON + EXECUTION_LAG + 1   # a training label must be fully observed before the test year starts
EMBARGO = 21                # extra gap against serial correlation in features
INNER_FOLDS = 3             # purged time-series CV folds used to tune hyper-parameters
SEED = 42
N_JOBS = 2
