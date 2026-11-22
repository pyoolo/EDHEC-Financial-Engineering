"""Time-ordered evaluation: expanding-window walk-forward with purging and embargo.

For each test year Y:
* test set  = all decision dates in Y whose label is observed (label_end < n);
* train set = all dates s whose label is fully observed before Y starts
  (label_end(s) < first test position), minus an extra EMBARGO of days.

Inside each training set, hyper-parameters are chosen with a *purged* time-series
cross-validation (``PurgedTimeSeriesSplit``): validation blocks always come after
their training block, with the same purge and embargo gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Fold:
    year: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward(index: pd.DatetimeIndex, label_end_date: pd.Series,
                 first_year: int = C.FIRST_TEST_YEAR, embargo: int = C.EMBARGO,
                 min_train: int = 750) -> list[Fold]:
    """Walk-forward folds using the *date* at which each label becomes known.

    A training row s is admissible for test year Y only if label_end_date[s] is
    strictly before the first test date of Y; then the last ``embargo`` admissible
    rows are dropped as well.
    """
    pos = np.arange(len(index))
    years = np.asarray(index.year)
    led = label_end_date.to_numpy()
    folds = []
    for y in range(first_year, int(years.max()) + 1):
        test = pos[years == y]
        if len(test) < 20:
            continue
        start = index[test[0]]
        train = pos[(led < np.datetime64(start)) & (np.asarray(index) < np.datetime64(start))]
        if embargo:
            train = train[: max(0, len(train) - embargo)]
        if len(train) >= min_train:
            folds.append(Fold(y, train, test))
    return folds


class PurgedTimeSeriesSplit:
    """K contiguous validation blocks at the end of the training window; each block's
    training part contains only rows whose label is known before the block starts,
    minus an embargo. Compatible with scikit-learn's ``cv=`` argument."""

    def __init__(self, label_end_date: np.ndarray, dates: np.ndarray,
                 n_splits: int = C.INNER_FOLDS, embargo: int = C.EMBARGO, val_frac: float = 0.15):
        self.led = np.asarray(label_end_date)
        self.dates = np.asarray(dates)
        self.n_splits, self.embargo, self.val_frac = n_splits, embargo, val_frac

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(self.dates)
        size = int(n * self.val_frac)
        for k in range(self.n_splits, 0, -1):
            v0 = n - k * size
            val = np.arange(v0, v0 + size)
            start = self.dates[v0]
            tr = np.where((self.led < start) & (self.dates < start))[0]
            tr = tr[: max(0, len(tr) - self.embargo)]
            yield tr, val
