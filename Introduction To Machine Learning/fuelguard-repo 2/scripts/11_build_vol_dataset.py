"""FuelGuard 2.0 - step 1. Build the volatility panel dataset (22 futures markets)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.panel import NAMES, load_panel  # noqa: E402
from fuelguard.vol_features import FEATURES, GROUPS, build_vol_dataset  # noqa: E402

if __name__ == "__main__":
    panel, common = load_panel()
    d = build_vol_dataset(panel, common)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    d.to_csv(C.DATA_PROCESSED / "vol_dataset.csv.gz", index=False, float_format="%.6g")
    ok = d.dropna(subset=["y", "lrv_22"])
    desc = ok.groupby("market").agg(start=("date", "min"), end=("date", "max"), days=("y", "size"),
                                    vol_median=("y", lambda s: np.exp(s.median())),
                                    vol_p95=("y", lambda s: np.exp(s.quantile(0.95))),
                                    iv_median=("liv", lambda s: np.exp(s.median())))
    desc.insert(0, "name", [NAMES[m] for m in desc.index])
    C.TABLES.mkdir(parents=True, exist_ok=True)
    desc.to_csv(C.TABLES / "vol_markets.csv")
    print(f"Volatility panel: {len(ok)} market-days, {ok['market'].nunique()} markets, "
          f"{len(FEATURES)} features in {len(GROUPS)} groups")
