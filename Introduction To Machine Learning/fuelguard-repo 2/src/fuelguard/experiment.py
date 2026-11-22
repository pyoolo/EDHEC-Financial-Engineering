"""Walk-forward experiment: tune in each training window, predict the next year."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

from . import config as C
from .models import classifiers, regressors
from .validation import PurgedTimeSeriesSplit, walk_forward


def run_walk_forward(data: pd.DataFrame, features: list[str], verbose: bool = True):
    X = data[features]
    y_clf = data["_spike"].astype(int)
    y_reg = data["_premium"]
    led = pd.to_datetime(data["_label_end_date"])
    folds = walk_forward(data.index, led)

    preds = pd.DataFrame(index=data.index)
    preds["fold_year"] = np.nan
    chosen, importances = [], []
    tasks = [("clf", classifiers(), y_clf, "neg_log_loss"),
             ("reg", regressors(), y_reg, "neg_mean_squared_error")]

    for f in folds:
        t0 = time.time()
        tr, te = f.train_idx, f.test_idx
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        cv = PurgedTimeSeriesSplit(led.iloc[tr].to_numpy(), data.index[tr].to_numpy())
        preds.iloc[te, preds.columns.get_loc("fold_year")] = f.year
        # benchmarks that also only use the training window
        preds.loc[Xte.index, "clf|Climatology"] = y_clf.iloc[tr].mean()
        preds.loc[Xte.index, "reg|Historical mean"] = y_reg.iloc[tr].mean()
        preds.loc[Xte.index, "reg|Futures (zero premium)"] = 0.0
        for task, specs, y, scoring in tasks:
            for name, (est, grid) in specs.items():
                gs = GridSearchCV(clone(est), grid, scoring=scoring, cv=cv, n_jobs=1, refit=True)
                gs.fit(Xtr, y.iloc[tr])
                model = gs.best_estimator_
                col = f"{task}|{name}"
                preds.loc[Xte.index, col] = (model.predict_proba(Xte)[:, 1] if task == "clf"
                                             else model.predict(Xte))
                chosen.append({"year": f.year, "task": task, "model": name,
                               "params": str(gs.best_params_), "cv_score": gs.best_score_,
                               "n_train": len(tr), "n_test": len(te),
                               "train_start": data.index[tr[0]].date(),
                               "train_end": data.index[tr[-1]].date()})
                if task == "clf":
                    pi = permutation_importance(model, Xte, y.iloc[te], scoring="roc_auc"
                                                if y.iloc[te].nunique() > 1 else "neg_log_loss",
                                                n_repeats=5, random_state=C.SEED, n_jobs=1)
                    importances.append(pd.Series(pi.importances_mean, index=features,
                                                 name=(f.year, name)))
        if verbose:
            print(f"  test year {f.year}: train {len(tr):5d} rows "
                  f"({data.index[tr[0]].date()} -> {data.index[tr[-1]].date()}), "
                  f"test {len(te):3d} rows, {time.time() - t0:5.1f}s", flush=True)

    imp = pd.concat(importances, axis=1) if importances else pd.DataFrame()
    return preds.dropna(subset=["fold_year"]), pd.DataFrame(chosen), imp


def fit_final(data: pd.DataFrame, features: list[str], model_name: str = "Gradient boosting"):
    """Fit the chosen classifier on all labelled data (used for today's signal)."""
    est, grid = classifiers()[model_name]
    led = pd.to_datetime(data["_label_end_date"])
    cv = PurgedTimeSeriesSplit(led.to_numpy(), data.index.to_numpy())
    gs = GridSearchCV(clone(est), grid, scoring="neg_log_loss", cv=cv, n_jobs=1)
    gs.fit(data[features], data["_spike"].astype(int))
    return gs.best_estimator_, gs.best_params_
