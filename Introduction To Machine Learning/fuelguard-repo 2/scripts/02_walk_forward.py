"""Step 2. Expanding-window walk-forward (2012 -> last year), with purged inner CV.

Saves out-of-sample predictions, chosen hyper-parameters and permutation importances.
Runtime: about 15-25 minutes on a laptop (2 cores).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.experiment import run_walk_forward  # noqa: E402
from fuelguard.features import GROUPS  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    data = pd.read_csv(C.DATA_PROCESSED / "model_dataset.csv", index_col=0, parse_dates=True)
    feats = [c for g in GROUPS.values() for c in g]
    preds, chosen, imp = run_walk_forward(data, feats)
    C.RESULTS.mkdir(exist_ok=True)
    preds.to_csv(C.RESULTS / "oos_predictions.csv")
    chosen.to_csv(C.TABLES / "chosen_hyperparameters.csv", index=False)
    imp.columns = [f"{y}|{m}" for y, m in imp.columns]
    imp.to_csv(C.RESULTS / "permutation_importance_by_year.csv")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. {len(preds)} out-of-sample days.")
