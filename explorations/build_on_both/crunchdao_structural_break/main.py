"""
CrunchDAO — ADIA Lab Structural Break (Open Benchmark) starter submission.

Task: for each univariate series, decide whether a *permanent structural break*
occurred at a known boundary. Each series is a DataFrame with two columns:
    value   float   the observation
    period  int     0 = before the boundary, 1 = after
You return ONE score per series in [0, 1] (higher = more likely a break).
Metric: ROC AUC across series, so only the *ranking* of scores matters — we do
not need calibrated probabilities, just a monotone "break-ness" signal.

Why this beats the shipped baseline
-----------------------------------
The official baseline is a single Welch t-test (mean shift only). A structural
break can also be a shift in variance, distribution shape, dependence, or trend.
This starter extracts a *battery* of two-sample statistics comparing the before
and after segments, then learns how to weight them with a gradient-boosted
classifier trained on the provided labels (y_train). That is the "reward marginal
signal, ensemble weak detectors" idea from RETHINK.md, applied to a supervised,
ROC-AUC-ranked task.

Contract (verified against the Open Benchmark baseline notebook):
    train(X_train, y_train, model_directory_path)      -> persists model.joblib
    infer(X_test, model_directory_path)                -> GENERATOR:
        first `yield` is a readiness handshake, then yield one score per dataset,
        consuming the iterate-once X_test stream sequentially.
"""

from __future__ import annotations

import os
import typing
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.spatial.distance import jensenshannon

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression


def _base_estimators():
    """Diverse, regularized base learners. LightGBM is added when available."""
    estimators = [
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0)),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=0)),
    ]
    try:
        from lightgbm import LGBMClassifier

        estimators.insert(0, ("lgbm", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            random_state=0, n_jobs=-1, deterministic=True,
            force_row_wise=True, verbose=-1)))
    except Exception:
        pass
    return estimators


def _make_model():
    """Stacked ensemble -> logistic meta-learner.

    Mirrors the published winning solutions (XGB/RF/LGBM stack) while staying
    runnable without lightgbm installed. Stacking with an out-of-fold meta-learner
    guards against any single base model overfitting one distribution family — the
    failure mode the independent benchmark flagged (top models fell 5+ ranks on a
    second dataset).
    """
    return StackingClassifier(
        estimators=_base_estimators(),
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
#  Feature extraction: compare the "before" (period 0) and "after" (period 1)  #
#  segments of a single series with a range of two-sample statistics.          #
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    "n0", "n1", "len_ratio",
    "mean_shift", "median_shift", "welch_t", "mannwhitney_z",
    "log_std_ratio", "levene_stat", "iqr_ratio",
    "ks_stat", "wasserstein", "energy_dist", "ad_stat", "js_dist",
    "skew_diff", "kurt_diff", "tail_range_ratio",
    "acf1_diff", "slope_diff", "vol_shift", "cusum_max",
    "spec_centroid_diff", "lowband_frac_diff",
]


def _safe(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else default


def _acf1(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return 0.0
    return _safe(np.dot(x[:-1], x[1:]) / denom)


def _spectrum_stats(x: np.ndarray) -> tuple[float, float]:
    """Spectral centroid and low-frequency energy fraction of a segment."""
    if x.size < 8:
        return 0.0, 0.0
    x = x - x.mean()
    ps = np.abs(np.fft.rfft(x)) ** 2
    total = ps.sum()
    if total <= 0:
        return 0.0, 0.0
    freqs = np.fft.rfftfreq(x.size)
    centroid = float((freqs * ps).sum() / total)
    lowband_frac = float(ps[freqs <= 0.1].sum() / total)
    return centroid, lowband_frac


def _slope(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=float)
    t -= t.mean()
    denom = np.dot(t, t)
    if denom <= 0:
        return 0.0
    return _safe(np.dot(t, x - x.mean()) / denom)


def extract_features(dataset: pd.DataFrame) -> np.ndarray:
    """Return a fixed-length feature vector for one series (before vs after)."""
    v = dataset["value"].to_numpy(dtype=float)
    p = dataset["period"].to_numpy()
    a = v[p == 0]
    b = v[p == 1]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n0, n1 = a.size, b.size
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats["n0"], feats["n1"] = float(n0), float(n1)
    feats["len_ratio"] = _safe(n1 / n0, 1.0) if n0 else 0.0

    if n0 < 2 or n1 < 2:
        return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)

    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(0.5 * (sa ** 2 + sb ** 2)) or 1.0

    # Location
    feats["mean_shift"] = _safe((b.mean() - a.mean()) / pooled)
    feats["median_shift"] = _safe((np.median(b) - np.median(a)) / pooled)
    try:
        feats["welch_t"] = _safe(stats.ttest_ind(a, b, equal_var=False).statistic)
    except Exception:
        pass
    try:
        u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        mu = n0 * n1 / 2.0
        sig = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12.0) or 1.0
        feats["mannwhitney_z"] = _safe((u - mu) / sig)
    except Exception:
        pass

    # Scale / dispersion
    feats["log_std_ratio"] = _safe(np.log((sb + 1e-9) / (sa + 1e-9)))
    try:
        feats["levene_stat"] = _safe(stats.levene(a, b, center="median").statistic)
    except Exception:
        pass
    iqr_a = stats.iqr(a) or 1e-9
    feats["iqr_ratio"] = _safe(np.log((stats.iqr(b) + 1e-9) / iqr_a))

    # Distribution shape / distance
    try:
        feats["ks_stat"] = _safe(stats.ks_2samp(a, b).statistic)
    except Exception:
        pass
    try:
        feats["wasserstein"] = _safe(stats.wasserstein_distance(a, b) / pooled)
    except Exception:
        pass
    try:
        feats["energy_dist"] = _safe(stats.energy_distance(a, b) / pooled)
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats["ad_stat"] = _safe(stats.anderson_ksamp([a, b]).statistic)
    except Exception:
        pass
    try:  # Jensen-Shannon distance between histograms on a shared grid
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if hi > lo:
            edges = np.linspace(lo, hi, 21)
            ha = np.histogram(a, bins=edges)[0] + 1e-9
            hb = np.histogram(b, bins=edges)[0] + 1e-9
            feats["js_dist"] = _safe(jensenshannon(ha, hb))
    except Exception:
        pass
    feats["skew_diff"] = _safe(stats.skew(b) - stats.skew(a))
    feats["kurt_diff"] = _safe(stats.kurtosis(b) - stats.kurtosis(a))
    ra = (np.quantile(a, 0.95) - np.quantile(a, 0.05)) or 1e-9
    rb = np.quantile(b, 0.95) - np.quantile(b, 0.05)
    feats["tail_range_ratio"] = _safe(np.log((rb + 1e-9) / ra))

    # Dependence / dynamics
    feats["acf1_diff"] = _safe(_acf1(b) - _acf1(a))
    feats["slope_diff"] = _safe((_slope(b) - _slope(a)) / pooled)
    feats["vol_shift"] = _safe(np.log((sb + 1e-9) / (sa + 1e-9)))

    # CUSUM: max cumulative deviation of the full series from its mean,
    # standardized. Sensitive to a persistent level shift at the boundary.
    full = np.concatenate([a, b])
    fstd = full.std(ddof=1) or 1.0
    cusum = np.cumsum(full - full.mean())
    feats["cusum_max"] = _safe(np.max(np.abs(cusum)) / (fstd * np.sqrt(full.size)))

    # Spectral shift (change in frequency content before vs after)
    ca, la = _spectrum_stats(a)
    cb, lb = _spectrum_stats(b)
    feats["spec_centroid_diff"] = _safe(cb - ca)
    feats["lowband_frac_diff"] = _safe(lb - la)

    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)


def _features_for_all(X: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Build a feature matrix over every series id in a MultiIndex frame."""
    ids, rows = [], []
    for series_id, dataset in X.groupby(level="id", sort=False):
        ids.append(series_id)
        rows.append(extract_features(dataset))
    return np.asarray(rows, dtype=float), ids


# --------------------------------------------------------------------------- #
#  CrunchDAO entry points                                                      #
# --------------------------------------------------------------------------- #

def train(X_train: pd.DataFrame, y_train: pd.Series, model_directory_path: str) -> None:
    X, ids = _features_for_all(X_train)
    y = y_train.loc[ids].astype(int).to_numpy()

    model = _make_model()
    model.fit(X, y)
    joblib.dump(model, os.path.join(model_directory_path, "model.joblib"))


def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
) -> typing.Iterator[float]:
    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))

    yield  # readiness handshake required by the runner

    for dataset in X_test:
        feats = extract_features(dataset).reshape(1, -1)
        try:
            score = float(model.predict_proba(feats)[0, 1])
        except Exception:
            # Unsupervised fallback: a KS-based break signal keeps us > random
            # even if the model failed to load/predict.
            score = float(np.clip(extract_features(dataset)[FEATURE_NAMES.index("ks_stat")], 0, 1))
        yield score


# --------------------------------------------------------------------------- #
#  Local smoke test (not run on the platform). Synthesises break / no-break    #
#  series so you can verify the pipeline end-to-end before `crunch push`.      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def make_series(sid, break_it):
        n0, n1 = rng.integers(200, 400), rng.integers(200, 400)
        before = rng.normal(0, 1, n0)
        shift = rng.normal(1.5, 0.4) if break_it else 0.0
        scale = rng.uniform(1.6, 2.2) if break_it else 1.0
        after = rng.normal(shift, scale, n1)
        idx = pd.MultiIndex.from_product([[sid], range(n0 + n1)], names=["id", "time"])
        return pd.DataFrame(
            {"value": np.concatenate([before, after]),
             "period": [0] * n0 + [1] * n1},
            index=idx,
        )

    labels = {i: bool(i % 2) for i in range(200)}
    X = pd.concat([make_series(i, labels[i]) for i in range(200)])
    y = pd.Series(labels, name="structural_breakpoint")
    y.index.name = "id"

    # Honest estimate: 5-fold CV ROC AUC on the feature matrix.
    from sklearn.model_selection import cross_val_score
    Xmat, ids = _features_for_all(X)
    yv = y.loc[ids].astype(int).to_numpy()
    cv_auc = cross_val_score(_make_model(), Xmat, yv, cv=5, scoring="roc_auc").mean()
    print(f"5-fold CV ROC AUC on synthetic data: {cv_auc:.3f}")

    # Also exercise the exact train()/infer() generator contract end to end.
    os.makedirs("resources", exist_ok=True)
    train(X, y, "resources")
    test_ids = list(range(200))
    gen = infer((X.loc[[i]] for i in test_ids), "resources")
    next(gen)  # consume readiness handshake
    scores = np.array([next(gen) for _ in test_ids])
    from sklearn.metrics import roc_auc_score
    truth = np.array([labels[i] for i in test_ids])
    print(f"in-sample ROC AUC (contract check): {roc_auc_score(truth, scores):.3f}")
