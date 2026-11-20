"""Run the whole pipeline in order:  python scripts/run_all.py"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_build_dataset.py", "02_walk_forward.py", "03_evaluate.py"]

if __name__ == "__main__":
    for s in STEPS:
        print(f"\n=== {s} ===", flush=True)
        sys.argv = [str(HERE / s)]
        runpy.run_path(str(HERE / s), run_name="__main__")
