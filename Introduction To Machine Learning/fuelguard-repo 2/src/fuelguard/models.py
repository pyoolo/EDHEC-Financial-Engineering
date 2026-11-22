"""Model specifications and hyper-parameter grids.

Three families, from most to least restrictive assumptions:
* regularised linear models (logistic / ridge): linear, additive effects;
* random forests: bagged deep-ish trees, non-linear, interactions, variance reduction;
* gradient boosting (histogram-based): sequential shallow trees, bias reduction.
Grids are small on purpose: with ~200 effective independent months per window,
large grids overfit the validation folds.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


# Parsimonious challenger, added after the first walk-forward run showed that implied
# volatility alone ranks spikes better than the 57-feature models (disclosed in the report).
PARSIMONIOUS = ["HO_iv"]


def _select(cols):
    return ColumnTransformer([("keep", "passthrough", cols)], remainder="drop")


def classifiers():
    return {
        "Logistic (implied vol only)": (
            make_pipeline(_select(PARSIMONIOUS), SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.01, 1.0, 100.0]},
        ),
        "Logistic (L2)": (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]},
        ),
        "Random forest": (
            RandomForestClassifier(n_estimators=300, max_features="sqrt", max_samples=0.5,
                                   random_state=C.SEED, n_jobs=C.N_JOBS),
            {"max_depth": [3, 6], "min_samples_leaf": [25, 100]},
        ),
        "Gradient boosting": (
            HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1.0,
                                           early_stopping=False, random_state=C.SEED),
            {"max_iter": [100, 300], "max_depth": [2, 3]},
        ),
    }


def regressors():
    return {
        "Ridge": (
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
            {"ridge__alpha": [10.0, 100.0, 1000.0, 10000.0]},
        ),
        "Random forest": (
            RandomForestRegressor(n_estimators=300, max_features="sqrt", max_samples=0.5,
                                  random_state=C.SEED, n_jobs=C.N_JOBS),
            {"max_depth": [3, 6], "min_samples_leaf": [25, 100]},
        ),
        "Gradient boosting": (
            HistGradientBoostingRegressor(learning_rate=0.05, l2_regularization=1.0,
                                          early_stopping=False, random_state=C.SEED),
            {"max_iter": [100, 300], "max_depth": [2, 3]},
        ),
    }
