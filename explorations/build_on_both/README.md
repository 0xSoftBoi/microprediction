# Build on Both: Numerai + CrunchDAO — Exploration

*Companion to `RETHINK.md` and `GO_NO_GO.md`. Those concluded: don't revive the
defunct microprediction network — instead **build on the living platforms that
already solved its incentive problem** (real stakes / real payouts). This folder
turns that into concrete, runnable starting points for the two most relevant
platforms, using the modern methods from the research. All mechanics verified
against 2026 primary sources (docs, PyPI, competition pages).*

## Why these two

Both are the intellectual descendants of microprediction's "crowd of algorithms"
idea — and Peter Cotton (microprediction's founder) is now **CrunchDAO's Chief
Scientific Officer**. The difference from microprediction is the part that
matters: **real money and low onboarding friction.**

| | **CrunchDAO — Structural Break** | **Numerai — Classic** | **Numerai — Signals** |
|---|---|---|---|
| Task | Did a break occur at a known boundary in a series? (score 0–1) | Rank ~global stock universe from obfuscated features | Rank stocks from **your own** data/signal |
| You need your own data? | **No** (data provided) | **No** (data provided) | **Yes** (bring a unique signal) |
| Submit | `train()`/`infer()` code, `crunch push` | Upload a pickled `predict()` (Model Upload) | Upload ticker→signal CSV / predict fn |
| Scoring | **ROC AUC** across series | 0.75·CORR + 2.25·MMC (per-era rank corr) | Alpha + MPC (neutralized residual) |
| Incentive | **USDC**, leaderboard/performance | NMR (crypto) | NMR (crypto) |
| Staking required? | **No** | Optional (run unstaked first) | Optional |
| Onboarding friction | **Lowest** — CLI + notebook, minutes | Low — one notebook, model-upload automates it | **Higher** — you must source & map ticker data |

**Recommendation: start with the two "no proprietary data" on-ramps in parallel —
CrunchDAO Structural Break (lowest friction, real USDC, no staking) and Numerai
Classic via Model Upload (Numerai runs your model every round automatically).**
Treat Numerai Signals as a later step once you have a genuinely original data
source — otherwise you're doing data engineering, not modeling.

---

## 1. CrunchDAO — Structural Break  (`crunchdao_structural_break/`)

**What's here:** `main.py` — a complete `train()`/`infer()` submission that beats
the shipped t-test baseline. It extracts a **battery of two-sample before/after
statistics** (KS, Mann–Whitney, Welch t, Levene, Wasserstein, energy distance,
Anderson–Darling, Jensen–Shannon, autocorrelation/slope/variance shifts, CUSUM,
moment differences) and learns to weight them with a gradient-boosted classifier
on the provided labels.

**Why this design:** the published winning solutions and an independent 25-method
benchmark all converge on the same recipe — **engineered two-sample features →
regularized GBDT (LightGBM/XGBoost), validated for cross-distribution
robustness.** Crucially, the benchmark found that *pure hypothesis testing is
unstable* (use tests as **features**, not as the classifier) and that **neural
nets / time-series foundation models are near-random on this short-univariate
task.** So the modern-forecasting stack from `RETHINK.md` says, correctly, *not*
to reach for a foundation model here — GBDT-on-features is the right tool.

**Run it locally (works today, no account needed):**
```bash
cd crunchdao_structural_break
pip install numpy scipy scikit-learn pandas joblib   # lightgbm optional
python main.py        # synthesises break/no-break series, prints a sanity AUC
```

**Submit for real:**
```bash
pip install crunch-cli --upgrade
# copy the `crunch setup ...` line (with your token) from the competition's
# Submit tab at hub.crunchdao.com, then:
crunch setup structural-break-open-benchmark my-model --token <TOKEN>
# drop main.py into the project, then:
crunch test            # local dry run against provided data
crunch push -m "two-sample features + GBDT"
```

**Improve from here (in rough ROI order):** add Jensen–Shannon/Hellinger on more
bins, spectral (FFT/PSD) and wavelet (PyWavelets) features; stack XGBoost + RF +
LightGBM; tune regularization for the *worst* fold, not the mean (the benchmark's
top models overfit and fell 5+ ranks on a second dataset). For the separate
**2026 streaming "Real-Time Edition"**, this batch approach doesn't apply — that's
where online **conformal test martingales** and Cotton's **`midone`** `Attacker`
(`pip install midone`) become first-class.

---

## 2. Numerai — Classic  (`numerai/`)

**What's here:** `submit.py` — downloads the current v5 ("Atlas") data, trains a
LightGBM regressor on the small feature set predicting the Cyrus target, applies
optional **feature neutralization** (orthogonalize predictions against features so
you're rewarded for signal, not raw feature exposure — the same "reward the
orthogonal component" theme as CrunchDAO/Numerai's contribution metrics), and
packages a `predict(live_features, live_benchmark_models)` function that Numerai's
**Model Upload** runs for you every round.

**Run it:**
```bash
cd numerai
pip install -r requirements.txt
python submit.py                 # download → train → write predict.pkl
# then, with a model created at numer.ai and API keys in env:
NUMERAI_PUBLIC_ID=... NUMERAI_SECRET_KEY=... NUMERAI_MODEL_ID=... \
  python submit.py --upload
```

**Improve from here:** ensemble across targets (e.g. Cyrus + Teager, rank-averaged
— this is literally Numerai's own flagship `V5_LGBM_CT_BLEND`); tune neutralization
proportion (~0.5–1.0) against **FNCv3**; validate **era-wise** (each era is one
atomic unit; never random-split — targets are forward-looking and leak) with
purge between train/test. Scale the LightGBM up (official large config:
`n_estimators=20000, lr=0.001, max_depth=6, num_leaves=64, colsample_bytree=0.1`).

---

## Honest expectations

- **Effort to first submission:** an afternoon for either. Both starters here run
  today.
- **Effort to *payout* / leaderboard money:** weeks-to-months of iteration. Numerai
  payouts are noisy (scored over ~20 days, ±5%/round cap) and denominated in NMR,
  which carries crypto price risk. CrunchDAO Structural Break Open Benchmark is a
  small pool ($6k/quarter, top-3) but no staking risk — good for *proving the
  method* before the $100k streaming edition.
- **Where the edge is:** not raw model horsepower — it's disciplined validation
  (era-wise / cross-distribution), feature/idea originality (both platforms
  explicitly pay for the *orthogonal* contribution, not correlated accuracy), and
  showing up every round.

## Key sources
- CrunchDAO participate/CLI: https://docs.crunchdao.com/competitions/participate ·
  Structural Break: https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge ·
  Open Benchmark: https://hub.crunchdao.com/competitions/structural-break-open-benchmark
- Winning-solution recipes: https://humbertobrandao.medium.com/how-far-can-we-push-the-winning-model-of-the-adia-lab-structural-break-challenge-87ebf3d0ff67 ·
  https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution ·
  benchmark: https://waddadaa.github.io/structural_break_detection/
- `midone` (streaming edition): https://github.com/microprediction/midone
- Numerai data/scoring/model-upload: https://docs.numer.ai/numerai-tournament/data ·
  https://docs.numer.ai/numerai-tournament/scoring/correlation-corr ·
  https://docs.numer.ai/numerai-tournament/submissions/model-uploads ·
  example scripts: https://github.com/numerai/example-scripts
- Numerai Signals: https://docs.numer.ai/numerai-signals/signals-overview ·
  https://github.com/numerai/signals-example-scripts
