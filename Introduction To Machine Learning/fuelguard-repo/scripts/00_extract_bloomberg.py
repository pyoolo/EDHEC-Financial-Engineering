"""OPTIONAL. Rebuild data/raw/bloomberg_selected.csv from the Bloomberg Excel exports.

The CSV is already in the repository, so this step is only needed if you
re-download the data on a Bloomberg terminal. Put the three Excel files
(values saved, not formulas) in data/bloomberg_xlsx/ and run:

    python scripts/00_extract_bloomberg.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.data import extract_bloomberg  # noqa: E402

if __name__ == "__main__":
    df = extract_bloomberg()
    C.BLOOMBERG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.BLOOMBERG_CSV, float_format="%.6g")
    print(f"Saved {C.BLOOMBERG_CSV} {df.shape}, {df.index.min().date()} -> {df.index.max().date()}")
