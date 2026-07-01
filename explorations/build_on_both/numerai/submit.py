"""
Numerai Classic — first-model starter (download → train → model-upload).

What this does
--------------
1. Downloads the current v5 ("Atlas") data with numerapi (small feature set).
2. Trains a LightGBM regressor on the primary `target` (Cyrus 20-day).
3. Optionally reduces feature exposure via feature neutralization (RETHINK.md's
   "reward orthogonal signal" idea — here, orthogonalize predictions against the
   features so you're not just re-selling raw feature exposure).
4. Wraps it in the `predict(live_features, live_benchmark_models)` function that
   Numerai's "Model Upload" flow runs for you every round, and cloudpickles it.

Notes verified against docs.numer.ai (2026):
- Data version string advances over time; confirm the current one via
  `napi.list_datasets()` or the Data page. `v5.2` is current at time of writing.
- Only *ranks per era* matter for scoring (Numerai Corr), so absolute prediction
  scale is irrelevant.
- Model Upload and manual CSV upload are mutually exclusive per model slot.
- Payouts (2026): 0.75*CORR + 2.25*MMC, capped at +/-5% of stake per round.
  Staking is optional — run unstaked first.

Usage:
    pip install -r requirements.txt
    # set NUMERAI_PUBLIC_ID / NUMERAI_SECRET_KEY and NUMERAI_MODEL_ID to upload
    python submit.py --upload      # build, cloudpickle, and model-upload
    python submit.py               # build + save predict.pkl only (no upload)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from numerapi import NumerAPI

DATA_VERSION = "v5.2"            # confirm current via napi.list_datasets()
FEATURE_SET = "small"           # "small" (~42) to start; "medium"/"all" later
TARGET = "target"               # alias of the Cyrus 20-day payout target
NEUTRALIZE_PROPORTION = 0.5     # 0.0 = none, 1.0 = full feature neutralization


def _lgbm():
    from lightgbm import LGBMRegressor

    # Community "fast to iterate" config; swap to the official large config
    # (n_estimators=20000, learning_rate=0.001, max_depth=6, num_leaves=64,
    #  colsample_bytree=0.1) once you are ready to spend the compute.
    return LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.1,
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    )


def neutralize(predictions: pd.Series, features: pd.DataFrame, proportion: float) -> pd.Series:
    """Subtract the linear feature-exposure component from predictions."""
    if proportion <= 0:
        return predictions
    f = features.to_numpy(dtype=np.float32)
    p = predictions.to_numpy(dtype=np.float32).reshape(-1, 1)
    exposure = f @ np.linalg.pinv(f) @ p
    residual = p - proportion * exposure
    out = residual.ravel()
    out = (out - out.mean()) / (out.std() + 1e-12)
    return pd.Series(out, index=predictions.index)


def build_predict_fn(model, features: list[str]):
    """Return the predict() function Numerai will run each round."""

    def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        raw = pd.Series(model.predict(live_features[features]), index=live_features.index)
        if NEUTRALIZE_PROPORTION > 0:
            raw = neutralize(raw, live_features[features], NEUTRALIZE_PROPORTION)
        # Rank to (0,1) — only order matters, and this guarantees a valid range.
        ranked = raw.rank(pct=True)
        return ranked.to_frame("prediction")

    return predict


def main(upload: bool) -> None:
    napi = NumerAPI()

    print(f"Downloading {DATA_VERSION} data ({FEATURE_SET} feature set)...")
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    napi.download_dataset(f"{DATA_VERSION}/train.parquet")

    with open(f"{DATA_VERSION}/features.json") as fh:
        features = json.load(fh)["feature_sets"][FEATURE_SET]

    train = pd.read_parquet(
        f"{DATA_VERSION}/train.parquet", columns=["era", TARGET] + features
    ).dropna(subset=[TARGET])

    print(f"Training LightGBM on {len(features)} features, {len(train):,} rows...")
    model = _lgbm()
    model.fit(train[features], train[TARGET])

    predict = build_predict_fn(model, features)

    # Sanity-check the predict fn on a slice of training features.
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
        model_id = os.environ["NUMERAI_MODEL_ID"]
        napi_auth = NumerAPI(
            public_id=os.environ["NUMERAI_PUBLIC_ID"],
            secret_key=os.environ["NUMERAI_SECRET_KEY"],
        )
        upload_id = napi_auth.model_upload(
            "predict.pkl", model_id=model_id, data_version=DATA_VERSION
        )
        print(f"Uploaded. model_upload_id={upload_id}")
    else:
        print("Skipped upload. Re-run with --upload and NUMERAI_* env vars set.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="model-upload predict.pkl")
    main(ap.parse_args().upload)
