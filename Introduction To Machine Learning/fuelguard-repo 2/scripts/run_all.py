"""Run the whole pipeline in order:  python scripts/run_all.py   (about 30 minutes on 2 cores)"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    # FuelGuard 1.0 (spike classification) - kept as the first attempt (report, Appendix D)
    "01_build_dataset.py", "02_walk_forward.py", "03_evaluate.py",
    # FuelGuard 2.0 (volatility forecasting) - the product
    "11_build_vol_dataset.py", "10_validation_design.py", "12_vol_walk_forward.py", "13_vol_evaluate.py",
]

if __name__ == "__main__":
    for s in STEPS:
        print(f"\n=== {s} ===", flush=True)
        sys.argv = [str(HERE / s)]
        runpy.run_path(str(HERE / s), run_name="__main__")
