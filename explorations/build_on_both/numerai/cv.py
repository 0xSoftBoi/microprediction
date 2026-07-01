"""
Era-wise cross-validation and Numerai-Corr scoring helpers.

Random K-fold LEAKS on Numerai: rows within an era are correlated and targets are
forward-looking (20/60 days), so adjacent eras share information. The unit of
cross-validation must be the *era*, split in time order, with a purge gap between
train and test. This module provides:

  * TimeSeriesSplitGroups  - walk-forward, era-atomic splitter (after mdo's
    classic forum post) with a one-era purge.
  * numerai_corr           - the official CORR transform (rank -> gaussianize ->
    power 1.5 -> Pearson).
  * era_scores / summarize  - per-era correlation, mean, std, and Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class TimeSeriesSplitGroups:
    """Walk-forward CV where each fold's test set is a future block of eras.

    A one-era purge is dropped from the end of each training block so the
    (forward-looking) training target cannot overlap the test eras.
    """

    def __init__(self, n_splits: int = 4, purge: int = 1):
        self.n_splits = n_splits
        self.purge = purge

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups (era per row) is required")
        groups = np.asarray(groups)
        idx = np.arange(len(groups))
        unique_eras = np.unique(groups)  # np.unique returns sorted
        blocks = np.array_split(unique_eras, self.n_splits + 1)

        for i in range(self.n_splits):
            train_eras = np.concatenate(blocks[: i + 1])
            if self.purge and len(train_eras) > self.purge:
                train_eras = train_eras[: -self.purge]
            test_eras = blocks[i + 1]
            train_idx = idx[np.isin(groups, train_eras)]
            test_idx = idx[np.isin(groups, test_eras)]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def numerai_corr(preds: pd.Series, target: pd.Series) -> float:
    """Official Numerai CORR: rank -> gaussianize -> power 1.5 -> Pearson."""
    ranked = preds.rank(pct=True).clip(1e-6, 1 - 1e-6)
    gauss = norm.ppf(ranked)
    gauss = np.sign(gauss) * np.abs(gauss) ** 1.5
    t = target.to_numpy(dtype=float)
    t = t - t.mean()
    t = np.sign(t) * np.abs(t) ** 1.5
    if gauss.std() == 0 or t.std() == 0:
        return 0.0
    return float(np.corrcoef(gauss, t)[0, 1])


def era_scores(preds: pd.Series, target: pd.Series, eras: pd.Series) -> pd.Series:
    """Per-era Numerai CORR."""
    df = pd.DataFrame({"p": preds.to_numpy(), "t": target.to_numpy(), "era": eras.to_numpy()})
    return df.groupby("era").apply(
        lambda g: numerai_corr(g["p"].reset_index(drop=True), g["t"].reset_index(drop=True))
    )


def summarize(scores: pd.Series) -> dict:
    """Mean per-era corr, volatility, and Sharpe (mean/std)."""
    mean, std = float(scores.mean()), float(scores.std())
    return {
        "mean_corr": mean,
        "std_corr": std,
        "sharpe": mean / std if std else 0.0,
        "n_eras": int(scores.shape[0]),
    }
