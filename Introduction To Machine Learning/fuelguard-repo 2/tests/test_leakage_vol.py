"""FuelGuard 2.0 leakage tests (volatility panel)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard.panel import load_panel  # noqa: E402
from fuelguard.vol_experiment import PurgedDateSplit, folds  # noqa: E402
from fuelguard.vol_features import FEATURES, build_vol_dataset  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return load_panel()


@pytest.fixture(scope="module")
def dataset(raw):
    d = build_vol_dataset(*raw)
    d = d.dropna(subset=["y", "lrv_1", "lrv_5", "lrv_22", "label_end_date"]).reset_index(drop=True)
    d["label_end_date"] = pd.to_datetime(d["label_end_date"])
    return d


@pytest.mark.parametrize("cut", ["2014-06-30", "2020-03-02"])
def test_vol_features_do_not_use_future_data(raw, cut):
    panel, common = raw
    base = build_vol_dataset(panel, common).set_index(["date", "market"])
    p2, c2 = panel.copy(), common.copy()
    rng = np.random.default_rng(1)
    num = [c for c in p2.columns if c not in ("date", "market")]
    after = p2["date"] > cut
    p2.loc[after, num] = p2.loc[after, num].to_numpy() * rng.uniform(0.5, 1.5, (after.sum(), len(num)))
    ca = c2.index > cut
    c2.loc[ca] = c2.loc[ca].to_numpy() * rng.uniform(0.5, 1.5, (ca.sum(), c2.shape[1]))
    new = build_vol_dataset(p2, c2).set_index(["date", "market"])
    m = base.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(base.loc[m, FEATURES], new.loc[m, FEATURES], rtol=1e-10)
    # and the target before the cut DOES change near the cut (it uses the future)
    near = (base.index.get_level_values("date") > pd.Timestamp(cut) - pd.Timedelta(days=10)) & m
    assert not np.allclose(base.loc[near, "y"].dropna(), new.loc[near, "y"].reindex(base.loc[near, "y"].dropna().index))


def test_outer_folds_are_purged(dataset):
    for y, tr, te in folds(dataset):
        start = pd.Timestamp(y, 1, 1)
        assert dataset["label_end_date"].iloc[tr].max() < start - pd.Timedelta(days=29)
        assert (dataset["date"].iloc[te].dt.year == y).all()


def test_inner_cv_is_purged(dataset):
    sub = dataset[dataset["date"] < "2016-01-01"]
    cv = PurgedDateSplit(sub["date"].to_numpy(), sub["label_end_date"].to_numpy())
    for tr, va in cv.split(sub):
        assert sub["label_end_date"].to_numpy()[tr].max() < sub["date"].to_numpy()[va].min()
