"""Leakage tests: run with  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.data import build_panel  # noqa: E402
from fuelguard.dataset import build_dataset  # noqa: E402
from fuelguard.features import build_features, clean_implied_vol  # noqa: E402
from fuelguard.targets import add_targets  # noqa: E402
from fuelguard.validation import PurgedTimeSeriesSplit, walk_forward  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return add_targets(clean_implied_vol(build_panel()))


@pytest.fixture(scope="module")
def dataset():
    _, data, _, feats = build_dataset(save=False)
    return data, feats


@pytest.mark.parametrize("cut", ["2012-06-29", "2016-03-31", "2022-02-24"])
def test_features_do_not_use_future_prices(panel, cut):
    """Scramble every market series after `cut`: features up to `cut` must not change."""
    base = build_features(panel)
    raw_cols = [c for c in panel.columns if c not in ("label_end",)]
    shocked = build_panel()
    rng = np.random.default_rng(0)
    after = shocked.index > cut
    for c in shocked.columns:
        shocked.loc[after, c] = shocked.loc[after, c] * rng.uniform(0.5, 1.5, after.sum())
    shocked = add_targets(clean_implied_vol(shocked))
    new = build_features(shocked)
    upto = base.index <= cut
    pd.testing.assert_frame_equal(base.loc[upto], new.loc[upto], check_exact=False, rtol=1e-12)
    assert raw_cols  # sanity


def test_labels_need_future_prices(panel):
    """Conversely, labels DO depend on the future (otherwise the target would be trivial)."""
    s = panel.index.get_loc(pd.Timestamp("2015-06-01"))
    assert panel["label_end"].iloc[s] == s + C.EXECUTION_LAG + C.HORIZON


def test_walk_forward_has_no_overlap(dataset):
    data, _ = dataset
    led = pd.to_datetime(data["_label_end_date"])
    folds = walk_forward(data.index, led)
    assert len(folds) >= 10
    for f in folds:
        test_start = data.index[f.test_idx[0]]
        assert led.iloc[f.train_idx].max() < test_start          # every training label was known
        assert data.index[f.train_idx].max() < test_start
        assert (data.index[f.test_idx].year == f.year).all()


def test_inner_cv_is_purged(dataset):
    data, _ = dataset
    led = pd.to_datetime(data["_label_end_date"]).to_numpy()
    cv = PurgedTimeSeriesSplit(led, data.index.to_numpy())
    for tr, va in cv.split(data):
        assert led[tr].max() < data.index[va[0]]
        assert tr.max() < va.min()
