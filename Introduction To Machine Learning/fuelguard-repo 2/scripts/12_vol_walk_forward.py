"""FuelGuard 2.0 - step 2. Walk-forward 2012-2026: HAR benchmarks vs pooled ML (about 25-40 min)."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.vol_experiment import run  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    d = pd.read_csv(C.DATA_PROCESSED / "vol_dataset.csv.gz", parse_dates=["date", "label_end_date"])
    preds, chosen, imp = run(d)
    preds.to_csv(C.RESULTS / "vol_oos_predictions.csv.gz", index=False, float_format="%.6g")
    chosen.to_csv(C.TABLES / "vol_chosen_hyperparameters.csv", index=False)
    imp.to_csv(C.RESULTS / "vol_group_importance_by_year.csv")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. {len(preds)} out-of-sample market-days.")
