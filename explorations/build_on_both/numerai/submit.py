"""
Numerai Classic — starter (download -> era-wise CV -> multi-target blend -> upload).

What this does
--------------
1. Downloads the current v5 ("Atlas") data with numerapi (small feature set).
2. Trains a LightGBM model per target and BLENDS them (rank-averaged, weighted by
   each target's walk-forward validation Sharpe) — this is Numerai's own flagship
   pattern (`V5_LGBM_CT_BLEND` = Cyrus + Teager), with Sharpe-weighting so a
   consistently-better target dominates instead of being diluted by a noisier one.
   Blending across targets diversifies which residual return you predict and
   improves out-of-sample consistency.
3. Optional feature neutralization (orthogonalize predictions vs features so you
   are rewarded for signal, not raw feature exposure).
4. `--cv` runs proper ERA-WISE cross-validation (never random K-fold) and reports
   mean per-era CORR + Sharpe, so you can trust a change before staking.
5. Packages a `predict(live_features, live_benchmark_models)` for Model Upload,
   which Numerai runs for you every round.

Verified vs docs.numer.ai (2026): confirm the current version string via
`napi.list_datasets()`; only per-era ranks matter; Model Upload and manual upload
are mutually exclusive per model slot; payouts 0.75*CORR + 2.25*MMC, +/-5%/round.

Usage:
    pip install -r requirements.txt
    python submit.py --cv          # download + era-wise CV report (no upload)
    python submit.py               # build predict.pkl
    NUMERAI_PUBLIC_ID=... NUMERAI_SECRET_KEY=... NUMERAI_MODEL_ID=... \
        python submit.py --upload  # build + model-upload
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from numerapi import NumerAPI

import cv as cvmod

DATA_VERSION = "v5.2"            # confirm current via napi.list_datasets()
FEATURE_SET = "small"           # "small" (~42) to start; "medium"/"all" later
# Primary payout target + a diversifier. Any missing target is skipped gracefully.
TARGETS = ["target", "target_teager2b_20", "target_cyrusd_20"]
NEUTRALIZE_PROPORTION = 0.5     # 0.0 = none, 1.0 = full feature neutralization


def _lgbm():
    from lightgbm import LGBMRegressor

    # Fast-to-iterate config; scale up to the official large config
    # (n_estimators=20000, learning_rate=0.001, max_depth=6, num_leaves=64,
    #  colsample_bytree=0.1) once you're ready to spend the compute.
    return LGBMRegressor(
        n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=31,
        colsample_bytree=0.1, random_state=0, n_jobs=-1, verbose=-1,
    )


def neutralize(predictions: pd.Series, features: pd.DataFrame, proportion: float) -> pd.Series:
    """Subtract the linear feature-exposure component from predictions."""
    if proportion <= 0:
        return predictions
    f = features.to_numpy(dtype=np.float32)
    p = predictions.to_numpy(dtype=np.float32).reshape(-1, 1)
    exposure = f @ np.linalg.pinv(f) @ p
    out = (p - proportion * exposure).ravel()
    out = (out - out.mean()) / (out.std() + 1e-12)
    return pd.Series(out, index=predictions.index)


def _blend(models: dict, features_df: pd.DataFrame, features: list[str],
           weights: dict | None = None) -> pd.Series:
    """Weighted rank-average of each per-target model (ranks are all that count).

    `weights` maps target->weight; when None, all models weigh equally. Weighting
    by validation Sharpe (see `_target_weights`) lets a consistently-better target
    dominate the blend instead of being diluted by a noisier one.
    """
    keys = list(models.keys())
    if weights is None:
        weights = {k: 1.0 for k in keys}
    wsum = sum(weights.get(k, 0.0) for k in keys) or float(len(keys))
    blended = None
    for k in keys:
        r = pd.Series(models[k].predict(features_df[features]),
                      index=features_df.index).rank(pct=True)
        contrib = r * (weights.get(k, 0.0) / wsum)
        blended = contrib if blended is None else blended + contrib
    return blended


def build_predict_fn(models: dict, features: list[str], weights: dict | None = None):
    """Return the predict() function Numerai runs each round."""

    def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        blended = _blend(models, live_features, features, weights)
        if NEUTRALIZE_PROPORTION > 0:
            blended = neutralize(blended, live_features[features], NEUTRALIZE_PROPORTION)
        return blended.rank(pct=True).to_frame("prediction")

    return predict


def _target_weights(train: pd.DataFrame, features: list[str], targets: list[str]) -> dict:
    """Weight each target by its walk-forward validation Sharpe (clipped >=0).

    Uses the LAST walk-forward split per target (cheap: one extra fit each) as an
    out-of-sample proxy. Falls back to equal weights if no target scores positive.
    """
    if len(targets) == 1:
        return {targets[0]: 1.0}
    splitter = cvmod.TimeSeriesSplitGroups(n_splits=4, purge=1)
    sharpes = {}
    for tgt in targets:
        df = train.dropna(subset=[tgt]).reset_index(drop=True)
        tr, te = list(splitter.split(df, groups=df["era"]))[-1]
        m = _lgbm()
        m.fit(df.iloc[tr][features], df.iloc[tr][tgt])
        preds = pd.Series(m.predict(df.iloc[te][features]), index=df.index[te])
        s = cvmod.summarize(cvmod.era_scores(preds, df.iloc[te][tgt], df.iloc[te]["era"]))
        sharpes[tgt] = max(s["sharpe"], 0.0)
    total = sum(sharpes.values())
    if total <= 0:
        return {t: 1.0 / len(targets) for t in targets}
    return {t: v / total for t, v in sharpes.items()}


def _available_targets(path: str) -> list[str]:
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(path).schema.names)
    present = [t for t in TARGETS if t in cols]
    return present or ["target"]


def main(upload: bool, run_cv: bool) -> None:
    napi = NumerAPI()

    print(f"Downloading {DATA_VERSION} data ({FEATURE_SET} feature set)...")
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    napi.download_dataset(f"{DATA_VERSION}/train.parquet")

    with open(f"{DATA_VERSION}/features.json") as fh:
        features = json.load(fh)["feature_sets"][FEATURE_SET]

    train_path = f"{DATA_VERSION}/train.parquet"
    targets = _available_targets(train_path)
    print(f"Targets to blend: {targets}")

    train = pd.read_parquet(train_path, columns=["era"] + targets + features)

    if run_cv:
        _era_wise_cv(train, features, targets[0])

    print("Computing Sharpe-based blend weights (walk-forward validation)...")
    weights = _target_weights(train, features, targets)
    print("Blend weights: " + ", ".join(f"{t}={w:.3f}" for t, w in weights.items()))

    print(f"Training {len(targets)} model(s) on {len(features)} features...")
    models = {}
    for tgt in targets:
        sub = train.dropna(subset=[tgt])
        m = _lgbm()
        m.fit(sub[features], sub[tgt])
        models[tgt] = m

    predict = build_predict_fn(models, features, weights)

    sample = train[features].head(1000)
    demo = predict(sample, pd.DataFrame(index=sample.index))
    assert list(demo.columns) == ["prediction"]
    assert demo["prediction"].between(0, 1).all()
    print("predict() smoke test OK.")

    import cloudpickle

    with open("predict.pkl", "wb") as fh:
        fh.write(cloudpickle.dumps(predict))
    print("Wrote predict.pkl")

    if upload:
        napi_auth = NumerAPI(
            public_id=os.environ["NUMERAI_PUBLIC_ID"],
            secret_key=os.environ["NUMERAI_SECRET_KEY"],
        )
        upload_id = napi_auth.model_upload(
            "predict.pkl", model_id=os.environ["NUMERAI_MODEL_ID"], data_version=DATA_VERSION
        )
        print(f"Uploaded. model_upload_id={upload_id}")
    else:
        print("Skipped upload. Re-run with --upload and NUMERAI_* env vars set.")


def _era_wise_cv(train: pd.DataFrame, features: list[str], target: str) -> None:
    """Walk-forward, era-atomic CV reporting mean per-era CORR + Sharpe."""
    print("Running era-wise CV (walk-forward, purged)...")
    df = train.dropna(subset=[target]).reset_index(drop=True)
    splitter = cvmod.TimeSeriesSplitGroups(n_splits=4, purge=1)
    fold_summaries = []
    for k, (tr, te) in enumerate(splitter.split(df, groups=df["era"]), 1):
        m = _lgbm()
        m.fit(df.iloc[tr][features], df.iloc[tr][target])
        preds = pd.Series(m.predict(df.iloc[te][features]), index=df.index[te])
        scores = cvmod.era_scores(preds, df.iloc[te][target], df.iloc[te]["era"])
        s = cvmod.summarize(scores)
        fold_summaries.append(s)
        print(f"  fold {k}: mean_corr={s['mean_corr']:.4f}  sharpe={s['sharpe']:.2f}  "
              f"({s['n_eras']} eras)")
    mean_corr = np.mean([s["mean_corr"] for s in fold_summaries])
    mean_sharpe = np.mean([s["sharpe"] for s in fold_summaries])
    print(f"CV summary: mean_corr={mean_corr:.4f}  mean_sharpe={mean_sharpe:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="model-upload predict.pkl")
    ap.add_argument("--cv", action="store_true", help="run era-wise CV and report")
    args = ap.parse_args()
    main(args.upload, args.cv)
