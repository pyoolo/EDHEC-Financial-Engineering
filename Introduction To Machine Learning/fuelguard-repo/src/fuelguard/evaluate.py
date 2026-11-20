"""Out-of-sample metrics and statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (brier_score_loss, log_loss, precision_score, recall_score,
                             roc_auc_score)


def classification_table(preds: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y = y.loc[preds.index].astype(int)
    clim = preds["clf|Climatology"]
    bs_ref = brier_score_loss(y, clim)
    rows = []
    for col in [c for c in preds.columns if c.startswith("clf|")]:
        p = preds[col].clip(1e-4, 1 - 1e-4)
        flag = p > clim   # "elevated risk" = probability above the historical base rate
        rows.append({
            "model": col.split("|")[1],
            "AUC": roc_auc_score(y, p) if col != "clf|Climatology" else 0.5,
            "Brier": brier_score_loss(y, p),
            "Brier skill": 1 - brier_score_loss(y, p) / bs_ref,
            "Log loss": log_loss(y, p),
            "Precision": precision_score(y, flag, zero_division=0) if col != "clf|Climatology" else np.nan,
            "Recall": recall_score(y, flag) if col != "clf|Climatology" else np.nan,
            "Share flagged": flag.mean() if col != "clf|Climatology" else np.nan,
        })
    return pd.DataFrame(rows).set_index("model")


def auc_by_year(preds: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y = y.loc[preds.index].astype(int)
    out = {}
    for yr, idx in preds.groupby("fold_year").groups.items():
        yy = y.loc[idx]
        row = {}
        for col in [c for c in preds.columns if c.startswith("clf|") and c != "clf|Climatology"]:
            row[col.split("|")[1]] = roc_auc_score(yy, preds.loc[idx, col]) if yy.nunique() > 1 else np.nan
        row["Spike rate"] = yy.mean()
        out[int(yr)] = row
    return pd.DataFrame(out).T


def newey_west_var(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return v


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, lag: int) -> tuple[float, float]:
    """DM test on squared-error loss differential with Newey-West variance
    (overlapping horizons make the differential autocorrelated up to the horizon)."""
    d = e_model ** 2 - e_bench ** 2
    stat = d.mean() / np.sqrt(newey_west_var(d, lag) / len(d))
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def regression_table(preds: pd.DataFrame, y: pd.Series, lag: int = 21) -> pd.DataFrame:
    y = y.loc[preds.index]
    bench = preds["reg|Futures (zero premium)"]
    e_b = (y - bench).to_numpy()
    rows = []
    for col in [c for c in preds.columns if c.startswith("reg|")]:
        e = (y - preds[col]).to_numpy()
        dm, pv = diebold_mariano(e, e_b, lag) if col != "reg|Futures (zero premium)" else (np.nan, np.nan)
        rows.append({
            "model": col.split("|")[1],
            "RMSE (%)": np.sqrt(np.mean(e ** 2)) * 100,
            "MAE (%)": np.mean(np.abs(e)) * 100,
            "OOS R2 vs futures (%)": (1 - np.sum(e ** 2) / np.sum(e_b ** 2)) * 100,
            "Hit rate (sign)": np.mean(np.sign(preds[col]) == np.sign(y)) if col != "reg|Futures (zero premium)" else np.nan,
            "DM stat": dm, "DM p-value": pv,
        })
    return pd.DataFrame(rows).set_index("model")


def calibration_bins(p: pd.Series, y: pd.Series, n_bins: int = 8) -> pd.DataFrame:
    q = pd.qcut(p, n_bins, duplicates="drop")
    return pd.DataFrame({"predicted": p.groupby(q, observed=True).mean(),
                         "observed": y.loc[p.index].groupby(q, observed=True).mean(),
                         "n": p.groupby(q, observed=True).size()})


def block_bootstrap_ci(preds: pd.DataFrame, y: pd.Series, n_boot: int = 2000, seed: int = 42,
                       level: float = 0.90) -> pd.DataFrame:
    """Confidence intervals for AUC and Brier skill by resampling whole calendar months
    (labels overlap within a month, so days are not independent)."""
    y = y.loc[preds.index].astype(int)
    months = preds.index.to_period("M")
    groups = {m: np.where(months == m)[0] for m in months.unique()}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    models = [c for c in preds.columns if c.startswith("clf|") and c != "clf|Climatology"]
    yv, clim = y.to_numpy(), preds["clf|Climatology"].to_numpy()
    P = {c: preds[c].to_numpy() for c in models}
    res = {c: {"auc": [], "bss": []} for c in models}
    for _ in range(n_boot):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        yy = yv[idx]
        if yy.min() == yy.max():
            continue
        bs_ref = np.mean((clim[idx] - yy) ** 2)
        for c in models:
            res[c]["auc"].append(roc_auc_score(yy, P[c][idx]))
            res[c]["bss"].append(1 - np.mean((P[c][idx] - yy) ** 2) / bs_ref)
    a = (1 - level) / 2
    rows = []
    for c in models:
        au, bs = np.array(res[c]["auc"]), np.array(res[c]["bss"])
        rows.append({"model": c.split("|")[1],
                     "AUC low": np.quantile(au, a), "AUC high": np.quantile(au, 1 - a),
                     "P(AUC<=0.5)": np.mean(au <= 0.5),
                     "BSS low": np.quantile(bs, a), "BSS high": np.quantile(bs, 1 - a),
                     "P(BSS<=0)": np.mean(bs <= 0)})
    return pd.DataFrame(rows).set_index("model")
