"""FuelGuard 2.0 - design selection on a VALIDATION period (2010-2011), before any test year.

A first walk-forward run with raw pooled ML models showed they lose to the per-market HAR-IV.
To choose a better design without looking at the 2012-2026 test years, we train on labels known before
December 2009 and compare designs on 2010-2011 only:
  raw       ML on raw features, target = log vol
  hybrid    ML on the residual of HAR-IV
  relative  scale-free features, target = change of log vol vs its current 22-day level
and equal-weight combinations with HAR-IV. The chosen design (relative ML ensemble of Lasso, random
forest and gradient boosting, combined with HAR-IV) is then used unchanged in the walk-forward.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.vol_experiment import (HAR, HAR_IV, TRAIN_STRIDE, _per_market_ols, ml_models,  # noqa: E402
                                      relative)
from fuelguard.vol_features import FEATURES  # noqa: E402

SETTINGS = {"Lasso": {"lasso__alpha": 1e-3}, "Random forest": {"max_depth": 12, "min_samples_leaf": 20},
            "Gradient boosting": {"max_iter": 200, "max_depth": 3}, "Neural network": {"mlpregressor__alpha": 1e-2}}

if __name__ == "__main__":
    d = pd.read_csv(C.DATA_PROCESSED / "vol_dataset.csv.gz", parse_dates=["date", "label_end_date"])
    d = d.dropna(subset=["y", "lrv_1", "lrv_5", "lrv_22", "label_end_date"]).reset_index(drop=True)
    tr = d[d["label_end_date"] < pd.Timestamp("2010-01-01") - pd.Timedelta(days=30)]
    va = d[d["date"].dt.year.isin([2010, 2011])]
    sub = tr[tr.groupby("market").cumcount() % TRAIN_STRIDE == 0]
    base_va = _per_market_ols(tr, va, HAR_IV).fillna(_per_market_ols(tr, va, HAR))
    base_tr = _per_market_ols(tr, sub, HAR_IV).fillna(_per_market_ols(tr, sub, HAR))
    mse = lambda p: float(np.mean((va["y"] - p) ** 2))  # noqa: E731
    ref = mse(base_va)
    rows, P = [{"design": "HAR-IV (benchmark)", "model": "", "MSE": ref}], {}
    for name, (est, _) in ml_models().items():
        e = clone(est).set_params(**SETTINGS[name])
        raw = clone(e).fit(sub[FEATURES], sub["y"]).predict(va[FEATURES])
        hyb = base_va + clone(e).fit(sub[FEATURES], sub["y"] - base_tr).predict(va[FEATURES])
        rel = va["lrv_22"] + clone(e).fit(relative(sub), sub["y"] - sub["lrv_22"]).predict(relative(va))
        P[("raw", name)], P[("relative", name)] = pd.Series(raw, index=va.index), rel
        for design, p in [("raw", raw), ("hybrid (residual of HAR-IV)", hyb), ("relative", rel)]:
            rows.append({"design": design, "model": name, "MSE": mse(p)})
    for design in ["raw", "relative"]:
        ens = pd.concat([P[(design, m)] for m in ["Lasso", "Random forest", "Gradient boosting"]], axis=1).mean(axis=1)
        rows.append({"design": design, "model": "ML ensemble (Lasso, RF, GB)", "MSE": mse(ens)})
        rows.append({"design": design + " + HAR-IV (equal weights)", "model": "ML ensemble (Lasso, RF, GB)",
                     "MSE": mse((ens + base_va) / 2)})
    out = pd.DataFrame(rows)
    out["vs HAR-IV (%)"] = (1 - out["MSE"] / ref) * 100
    C.TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.TABLES / "vol_validation_design.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"Validation 2010-2011: {len(va)} market-days, training labels known before Dec 2009 ({len(sub)} rows)")
    print(out.round(4).to_string(index=False))
