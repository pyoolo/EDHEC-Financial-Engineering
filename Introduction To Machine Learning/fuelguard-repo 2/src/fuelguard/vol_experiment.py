"""FuelGuard 2.0 walk-forward: benchmarks (random walk, HAR, HAR-IV per market) and pooled ML models."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .vol_features import FEATURES, GROUPS

EMBARGO_DAYS = 30          # calendar days between the last known training label and the test start
TRAIN_STRIDE = 5           # ML models train on every 5th day of each market (labels overlap 21 days)
HAR = ["lrv_1", "lrv_5", "lrv_22"]
HAR_IV = HAR + ["liv"]
ENSEMBLE = ["Lasso", "Random forest", "Gradient boosting"]   # the neural network overfits (validation 2010-11)


def relative(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features for pooling markets whose volatility differs by a factor of 50 (Schatz vs gas):
    every log-volatility level is expressed relative to the market's current 22-day volatility, and the
    current level relative to the market's own long-run average. The ML target is the change
    y - lrv_22; the forecast is lrv_22 + prediction. Design chosen on the 2010-2011 validation period
    (scripts/10_validation_design.py), before any test year."""
    X = df[FEATURES].copy()
    for c in ["lrv_1", "lrv_5", "lrv_63", "liv", "lrv_longrun"]:
        X[c] = df[c] - df["lrv_22"]
    X["lrv_22"] = df["lrv_22"] - df["lrv_longrun"]
    return X


def ml_models():
    imp_scale = [SimpleImputer(strategy="median"), StandardScaler()]
    return {
        "Lasso": (make_pipeline(*imp_scale, Lasso(max_iter=20000)),
                  {"lasso__alpha": [1e-4, 1e-3, 1e-2]}),
        "Random forest": (RandomForestRegressor(n_estimators=300, max_features=0.33, max_samples=0.5,
                                                random_state=C.SEED, n_jobs=C.N_JOBS),
                          {"max_depth": [6, 12], "min_samples_leaf": [20, 100]}),
        "Gradient boosting": (HistGradientBoostingRegressor(learning_rate=0.05, l2_regularization=1.0,
                                                            early_stopping=False, random_state=C.SEED),
                              {"max_iter": [200, 500], "max_depth": [3, 5]}),
        "Neural network": (make_pipeline(*imp_scale, MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3,
                                                                  early_stopping=True, max_iter=300,
                                                                  random_state=C.SEED)),
                           {"mlpregressor__alpha": [1e-4, 1e-2]}),
    }


class PurgedDateSplit:
    """Inner CV on a panel: validation = contiguous blocks of dates at the end of the training window;
    training rows must have their label known EMBARGO_DAYS before the block starts."""

    def __init__(self, dates, label_end, n_splits=C.INNER_FOLDS, val_frac=0.15):
        self.dates, self.led = np.asarray(dates), np.asarray(label_end)
        self.n_splits, self.val_frac = n_splits, val_frac

    def get_n_splits(self, *a, **k):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        ud = np.unique(self.dates)
        size = int(len(ud) * self.val_frac)
        for k in range(self.n_splits, 0, -1):
            v0, v1 = ud[len(ud) - k * size], ud[min(len(ud) - 1, len(ud) - (k - 1) * size - 1)]
            val = np.where((self.dates >= v0) & (self.dates <= v1))[0]
            cut = v0 - np.timedelta64(EMBARGO_DAYS, "D")
            tr = np.where(self.led < cut)[0]
            yield tr, val


def folds(d: pd.DataFrame, first_year=C.FIRST_TEST_YEAR):
    out = []
    for y in range(first_year, int(d["date"].dt.year.max()) + 1):
        start = pd.Timestamp(y, 1, 1)
        test = np.where(d["date"].dt.year == y)[0]
        train = np.where(d["label_end_date"] < start - pd.Timedelta(days=EMBARGO_DAYS))[0]
        if len(test) > 100 and len(train) > 5000:
            out.append((y, train, test))
    return out


def _per_market_ols(tr: pd.DataFrame, te: pd.DataFrame, cols) -> pd.Series:
    pred = pd.Series(np.nan, index=te.index)
    for mk, g in te.groupby("market"):
        t = tr[tr["market"] == mk].dropna(subset=cols + ["y"])
        ok = g[cols].notna().all(axis=1)
        if len(t) > 250 and ok.any():
            lr = LinearRegression().fit(t[cols], t["y"])
            pred.loc[g.index[ok]] = lr.predict(g.loc[ok, cols])
    return pred


def group_importance(model, X: pd.DataFrame, y: pd.Series, rng, n_repeats=3) -> pd.Series:
    base = np.mean((y - model.predict(X)) ** 2)
    out = {}
    for g, cols in GROUPS.items():
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            inc.append(np.mean((y - model.predict(Xp)) ** 2) - base)
        out[g] = np.mean(inc) / base * 100          # % increase of MSE
    return pd.Series(out)


def run(d: pd.DataFrame, verbose=True):
    d = d.dropna(subset=["y", "lrv_1", "lrv_5", "lrv_22", "label_end_date"]).reset_index(drop=True)
    d["label_end_date"] = pd.to_datetime(d["label_end_date"])
    preds = d[["date", "market", "y"]].copy()
    preds["fold_year"] = np.nan
    chosen, imps = [], []
    rng = np.random.default_rng(C.SEED)
    within = d.groupby("market").cumcount()
    for y, tr, te in folds(d):
        t0 = time.time()
        Dtr, Dte = d.iloc[tr], d.iloc[te]
        preds.loc[Dte.index, "fold_year"] = y
        # --- benchmarks
        preds.loc[Dte.index, "Random walk"] = Dte["lrv_22"]
        preds.loc[Dte.index, "HAR"] = _per_market_ols(Dtr, Dte, HAR)
        har_iv = _per_market_ols(Dtr, Dte, HAR_IV)
        preds.loc[Dte.index, "HAR-IV"] = har_iv.fillna(preds.loc[Dte.index, "HAR"])
        # --- pooled ML, trained on every TRAIN_STRIDE-th day of each market
        sub = Dtr[(within.iloc[tr] % TRAIN_STRIDE) == 0]
        cv = PurgedDateSplit(sub["date"].to_numpy(), sub["label_end_date"].to_numpy())
        ml_cols = []
        for name, (est, grid) in ml_models().items():
            gs = GridSearchCV(clone(est), grid, scoring="neg_mean_squared_error", cv=cv, n_jobs=1)
            gs.fit(relative(sub), sub["y"] - sub["lrv_22"])
            preds.loc[Dte.index, name] = Dte["lrv_22"] + gs.best_estimator_.predict(relative(Dte))
            ml_cols.append(name)
            chosen.append({"year": y, "model": name, "params": str(gs.best_params_),
                           "cv_mse": -gs.best_score_, "n_train": len(sub)})
            if name == "Gradient boosting":
                samp = Dte.sample(min(4000, len(Dte)), random_state=C.SEED)
                gi = group_importance(gs.best_estimator_, relative(samp), samp["y"] - samp["lrv_22"], rng)
                gi.name = y
                imps.append(gi)
        preds.loc[Dte.index, "ML ensemble"] = preds.loc[Dte.index, ENSEMBLE].mean(axis=1)
        preds.loc[Dte.index, "FuelGuard (HAR-IV + ML)"] = preds.loc[Dte.index, ["HAR-IV", "ML ensemble"]].mean(axis=1)
        if verbose:
            print(f"  test {y}: train {len(sub):6d} rows (ML, every {TRAIN_STRIDE}th day) / {len(Dtr):6d} (HAR), "
                  f"test {len(Dte):5d} rows, {time.time() - t0:5.1f}s", flush=True)
    return preds.dropna(subset=["fold_year"]), pd.DataFrame(chosen), pd.concat(imps, axis=1)
