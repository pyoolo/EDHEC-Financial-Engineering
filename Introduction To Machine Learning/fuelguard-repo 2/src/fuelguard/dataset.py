"""Assemble the modelling dataset: panel -> cleaning -> targets -> features."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .data import build_panel
from .features import build_features, clean_implied_vol
from .targets import add_targets

TARGET_COLS = ["premium", "spike", "market_cost", "lock_price", "label_end_date"]


def build_dataset(save: bool = True):
    panel = clean_implied_vol(build_panel())
    panel = add_targets(panel)
    le = panel["label_end"].to_numpy()
    panel["label_end_date"] = pd.NaT
    ok = le >= 0
    panel.loc[ok, "label_end_date"] = panel.index[le[ok]]
    X = build_features(panel)
    data = pd.concat([X, panel[TARGET_COLS + ["diesel_eur", "HO_iv"]].add_prefix("_")], axis=1)
    # keep rows with all features; rows without label (the last month) are kept apart
    data = data.dropna(subset=list(X.columns))
    labelled = data.dropna(subset=["_premium"])
    unlabelled = data[data["_premium"].isna()]
    if save:
        C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        panel.to_csv(C.DATA_PROCESSED / "panel_daily.csv")
        labelled.to_csv(C.DATA_PROCESSED / "model_dataset.csv")
        unlabelled.to_csv(C.DATA_PROCESSED / "latest_unlabelled.csv")
    return panel, labelled, unlabelled, list(X.columns)
