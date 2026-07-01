# Microprediction — First-Principles Rethink (Research Memo)

*Status: research / design memo. No code changed. Branch: `claude/first-principles-rethink`. Date: 2026-06-29.*

This memo rethinks the `microprediction` package from first principles. It (1) states what the
project actually is and which design choices are load-bearing, (2) surveys the 2024–2026 state of
the art across the three layers that matter — the **forecasting/statistics** layer, the
**mechanism/incentive** layer, and the **client/SDK engineering** layer — and (3) recommends a
concrete direction with a phased roadmap.

It is deliberately opinionated. The TL;DR is at the top; evidence and citations follow.

---

## 0. TL;DR

- **The core idea is still good.** A thin scoring server that decouples *publishing a live stream*
  from *predicting it*, turning prediction into an open, continuously-scored market, remains a
  genuinely good architecture. Don't throw the idea away.
- **Three things have aged badly**, in descending order of how attackable / outdated they are:
  1. **The incentive layer** (proof-of-work "MUID" key mining, *relative-to-community* scoring,
     "bankruptcy"). This is the most attackable part of the design and the part the field has moved
     furthest on. **Numerai is the living blueprint** for what to do instead: stake-based identity +
     reward for *marginal contribution to the ensemble*, not raw accuracy.
  2. **The prediction primitive** (a fixed list of **225 raw samples** per forecast). The field has
     converged on **quantile output + CRPS/pinball scoring**, wrapped in **online conformal
     calibration**, with **time-series foundation models** (Chronos-Bolt, TimesFM, IBM TTM) as the
     new zero-shot baseline that replaces hand-built bootstrap/Gaussian/copula samplers.
  3. **The client engineering** (Python 3.7/3.8, synchronous `requests`, a ~10-deep crawler
     inheritance tree, all-heavy-deps-required, a 67 MB repo with 34 committed notebooks). This is
     a straightforward, low-risk modernization: `pyproject.toml`/`uv`/`ruff`, async `httpx`, a
     `Forecaster` **Protocol** injected into one engine, and splitting examples into a cookbook repo.
- **Recommended path: a clean rewrite of the *client* (`microprediction` v2) around a small typed
  core + a strategy protocol, done as a new package alongside the old one** — *not* an in-place
  refactor, and *not* a big-bang rewrite of the whole network. The forecasting and incentive
  redesigns are larger and partly **server-side** (out of this repo's control); this memo specifies
  them so the client v2 is built to support them, and flags what needs server cooperation.
- **One honest caveat up front:** much of the project's recent energy has visibly moved to adjacent
  efforts (MontePrediction, CrunchDAO, Numerai-style contribution scoring — see the current
  `README.md`). A "whole package rewrite" is worth doing only if the live network is still the
  thing you want to invest in. If the goal is really "capture the good ideas in a modern form,"
  the client-v2 + the Numerai-style incentive redesign below are where the leverage is.

---

## 1. What microprediction is today (the as-is)

**The system.** Contributors publish live numeric **streams** (a value every few minutes). A
community of algorithms (**crawlers**) navigate streams and submit **distributional predictions** at
four fixed horizons (~70 s, 310 s, 910 s, 3555 s ahead). A Redis-backed server at
`api.microprediction.org` pools predictions, builds CDFs *relative to the community*, and scores
them once the horizon elapses. Z-score and copula streams are derived automatically; multivariate
prediction is folded to univariate via a Morton/Z-curve space-filling embedding.

**This repo is the client**, not the server. The scoring, aggregation, and incentive logic live
server-side. That is a hard scope boundary for any rewrite (see §6).

**The load-bearing design choices, made explicit:**

| Choice | Where | Implication |
|---|---|---|
| Prediction = exactly **225 float samples**, sent as a comma-separated string via `PUT /submit/` | `writer.py:352`, `num_predictions` | Empirical-distribution wire format; no quantiles, no parametric form, no calibration metadata |
| Scoring is **relative to the community** at fixed horizons | server | Elegant, self-calibrating *in theory*; the single most attackable choice in practice (§5) |
| Identity/anti-spam = **proof-of-work MUID** ("mine" a write_key for hours) | `conventions.new_key` | Memorable IDs, tunable cost — but PoW-for-identity is a minority choice in 2026 |
| Incentives = **balance + bankruptcy**; sponsor **budgets/prizes** | server | Skin-in-the-game via losable key; no explicit stake, no contribution-based reward |
| Samplers = **bootstrap / differenced bootstrap / Gaussian / t-digest / copula** | `samplers.py` | 2020-era; no learned models, no foundation models, no conformal calibration |
| Client = `MicroReader → MicroWriter → {MicroPoll, MicroCrawler}` + ~10 crawler subclasses | `microprediction/*.py` | Deep inheritance; users `subclass` and override `sample()`; sync `requests`; long-polling |
| Packaging = `setup.py`, Python 3.7/3.8, **all heavy deps required** (sklearn, statsmodels, hyperopt, copulas…) | `setup.py` | Slow, fragile installs; 67 MB repo; 34 committed notebooks (several >1 MB); 40 `*_examples` dirs |

**What's genuinely good and must be preserved:**

- The **decoupling** of publishing from predicting via a thin scoring layer (the "market" framing).
- **Delayed-but-real ground truth**: numeric outcomes *do* arrive. This is a major asset most
  elicitation mechanisms lack — lean on it (§5).
- A **uniform, language-agnostic REST surface** (contributors use Python, Julia, R).
- The **"derived streams" idea** (z-scores, copulas auto-generated from a base stream) is clever and
  worth keeping in some form.

---

## 2. First-principles restatement

Strip the implementation away and microprediction is solving three separable problems. The 2020
codebase fuses them; a rethink should keep them clean and let each evolve independently.

1. **Forecasting:** given a live stream's history, produce a *calibrated predictive distribution* at
   several horizons. (A statistics/ML problem.)
2. **Mechanism:** elicit those distributions *honestly* from self-interested strangers, resist
   Sybils/spam, and aggregate them into one community distribution that beats any individual. (A
   mechanism-design / incentives problem.)
3. **Plumbing:** make it ergonomic to read streams, run a strategy "forever," and submit — in a
   maintainable, modern SDK. (A software-engineering problem.)

The rest of the memo takes each in turn: *what's wrong now → what's SOTA in 2026 → what to do.*

---

## 3. Layer 1 — Forecasting / statistics

### 3.1 What's wrong now
- The wire primitive is **225 raw samples**. It is bulky, awkward to score with proper rules, and
  out of step with every modern probabilistic forecaster (which emit **quantiles**).
- The built-in samplers are **hand-rolled heuristics** (bootstrap, differenced bootstrap, Gaussian,
  t-digest). They have no learned component and no calibration guarantees.
- Multivariate dependence is captured by **Morton/Z-curve copula embeddings** — ingenious for 2020,
  but superseded by learned joint models and in-context multivariate foundation models.

### 3.2 What's SOTA in 2026
- **Represent distributions as quantiles, not samples.** A dense quantile grid (or a continuous
  quantile function) trained with **pinball/quantile loss** is the current default output head;
  nearly all foundation models emit it. It is compact, directly scorable, and trivially converted
  back to samples if the API must keep them. ([Any-Quantile Forecasting](https://arxiv.org/abs/2404.17451),
  [MQ-RNN](https://arxiv.org/pdf/1711.11053))
- **Score with strictly proper rules.** **CRPS** (= ∫ pinball loss over quantile levels) is the de
  facto standard for distributional forecasts — scale-aware, robust, decomposes into
  calibration + sharpness. Use **weighted pinball / CRPS** as the atomic score; report **PIT
  histograms / calibration error** separately; use the **energy score** only for genuine
  multivariate/joint streams. Avoid unbounded **log score** as the *ranking* primitive in an open
  adversarial setting (one tail miss dominates — which is, not coincidentally, exactly the current
  "bankruptcy" dynamic). ([Gneiting & Raftery 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf),
  [Forecaster's Dilemma](https://arxiv.org/pdf/1512.09244))
- **Time-series foundation models are the new baseline.** Zero-shot, quantile-output models now beat
  hand-built heuristics out of the box. For *many short, high-frequency streams*, the relevant axis
  is cost: prefer **small, fast, quantile-output** models.
  - **Amazon Chronos-Bolt** — quantiles, ~250× faster than original Chronos, CPU-viable; best
    price/perf for many small streams. ([AWS](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/))
  - **IBM Tiny Time Mixers (TTM)** — ≤ a few M params, 40× smaller than TimesFM; TTM-r3 adds
    quantiles. Excellent fit for cheap per-stream inference. ([arXiv](https://arxiv.org/pdf/2401.03955))
  - **Google TimesFM**, **Salesforce Moirai-2**, **Chronos-2** — heavier; Chronos-2/Moirai-2 add
    **in-context multivariate + covariates**, the modern replacement for hand-built copulas.
    ([Chronos-2](https://arxiv.org/pdf/2510.15821), [Moirai-2](https://arxiv.org/html/2511.11698v1))
  - Track the **GIFT-Eval** leaderboard (high-frequency emphasis, scores probabilistic WQL).
  - Caveat: foundation models are **not always calibrated** zero-shot
    ([arXiv](https://arxiv.org/html/2510.16060v1)) — which motivates the next point.
- **Online conformal calibration is the principled successor to "calibration emerges from the
  community."** For a drift-prone, community-scored network this is arguably the single most
  important modern ingredient: wrap *any* predictor to get distribution-free coverage that adapts
  online.
  - **Adaptive Conformal Inference (ACI)** — adjusts miscoverage online under arbitrary drift.
    ([Gibbs & Candès](https://arxiv.org/abs/2106.00170))
  - **Conformal PID control** — most stable online variant; integral term = quantile tracking via
    online pinball-loss gradient descent. Strong default. ([Angelopoulos et al.](https://arxiv.org/pdf/2307.16895),
    [code](https://github.com/aangelopoulos/conformal-time-series))
  - Libraries: **MAPIE** (sklearn-compatible, has time-series/ACI), **TorchCP** (PyTorch, GPU,
    online).
- **Streaming/online learners** per stream match the "value every few minutes" cadence: **River**
  (`learn_one`/`predict_one`, drift detectors) is the standard pure-Python option.
- **For the multivariate frontier**, learned **diffusion** (TimeGrad) / **normalizing-flow** joint
  models, or in-context multivariate foundation models, replace Morton-curve copulas.
  ([TimeGrad](https://arxiv.org/abs/2101.12072))

### 3.3 Recommendation (Layer 1)
1. **Change the prediction primitive to a quantile grid** (keep a samples-compat shim). This is the
   highest-leverage change and touches the wire format, so it needs server cooperation — but the
   *client v2* should model predictions as quantiles natively and down-convert to 225 samples only
   at the legacy boundary.
2. **Ship a foundation-model baseline strategy** (Chronos-Bolt or TTM, quantile output) so a new
   contributor's "hello world" is a genuinely strong forecaster, not a bootstrap toy.
3. **Provide an online-conformal wrapper** (`ConformalForecaster`) any strategy can opt into for
   per-stream calibrated coverage under drift.
4. **Score with CRPS/pinball; report PIT calibration separately.** (Server-side, but the client
   should expose these so contributors can self-evaluate locally before submitting.)

---

## 4. Layer 2 — Mechanism / incentives (the highest-stakes rethink)

### 4.1 What's wrong now
- **Relative-to-community scoring** is the most attackable choice in the whole design. Peer-prediction
  theory shows verification-free / relative scoring invites **herding** (copy the consensus),
  **collusion** (coordinate to capture the pool), and **self-prediction Sybils** (one actor runs many
  crawlers as their own "community"). Even market-scoring rules lose incentive compatibility for
  non-myopic, repeat players. ([Manipulation & Peer Mechanisms survey](https://arxiv.org/pdf/2210.01984))
- **Proof-of-work MUID** as the Sybil mechanism is dominated in 2026: burning compute for identity is
  costly, wasteful, and *weak* (an attacker with compute mints many keys).
- **Reward = raw relative accuracy** incentivizes accuracy *correlated with everyone else*, not
  independent signal — the opposite of what makes an ensemble good.

### 4.2 What's SOTA in 2026
- **The scoring core is essentially solved — use it.** CRPS / log-score **against realized stream
  values** is the proper, honest atomic primitive. The 2020 instinct here was right; the danger is
  entirely in the *relative-to-community wrapper*. Microprediction's **delayed-but-real ground truth**
  means it does *not* have to rely on verification-free elicitation — it should anchor scoring to
  realized outcomes and treat peer-relative scoring as a secondary, manipulation-audited layer.
- **Numerai is the single most relevant living blueprint** and solves the herding, Sybil, and
  aggregation problems in one coherent economic loop:
  - **Stake** (bonded capital, slashable) replaces PoW as skin-in-the-game and Sybil cost.
  - **Reward marginal contribution to the ensemble** (MMC / Meta-Portfolio Contribution), *not* raw
    accuracy — pays for *orthogonal* signal that improves the meta-model. This directly neutralizes
    the herding incentive.
  - Stake lockups + burn-on-failure provide the "bankruptcy" dynamic in a principled form.
  ([Numerai MMC/MPC](https://blog.numer.ai/signals-alpha-and-mpc/), [staking](https://docs.numer.ai/numerai-signals/staking))
- **Sybil resistance has moved to stake-based bonding and/or proof-of-personhood** (World ID, Idena,
  Human Passport's ZK credential aggregation). Keep the MUID only as *optional human-readable
  identity UX*; move the actual anti-Sybil weight to stake (which doubles as the balance) and/or PoP.
  ([PoP overview](https://digitap.app/news/guide/proof-of-personhood-solving-sybil-attacks))
- **Aggregation is the highest-confidence win in the entire literature.** Performance/stake-weighted,
  **extremized**, **proper-score-optimized** ensembling *reliably* beats individuals (FluSight,
  Good Judgment). Make the **community distribution**, not the leaderboard, the primary product, and
  pay contributors for their contribution to *it*. ([Bayesian stacking via proper scores](https://arxiv.org/html/2509.04203),
  [Forecast Aggregation via Peer Prediction](https://arxiv.org/pdf/1910.03779))
- **A sobering empirical caution on the thesis.** In the M6 financial-forecasting competition only
  ~7% of teams beat *both* the forecasting and portfolio benchmarks — most underperformed. The
  "open network of algorithms reliably beats the market" framing should be tempered accordingly.
  ([M6 results](https://arxiv.org/abs/2310.13357))

### 4.3 Recommendation (Layer 2)
1. **Anchor scoring to realized outcomes with a proper rule (CRPS).** Demote relative-to-community
   scoring to a secondary, audited signal.
2. **Replace PoW identity with stake-based bonding** (merging "balance/bankruptcy" into a single
   stake mechanic, à la Numerai); keep MUID as optional vanity identity; consider PoP for a
   one-human baseline.
3. **Pay for marginal contribution to the community ensemble (MMC-style), not raw accuracy.** This is
   the structural fix for herding and the strongest lever for forecast quality.
4. **Make the calibrated community distribution the headline product.**

> ⚠️ **Scope:** items 1–4 are predominantly **server-side**. This repo (the client) cannot change
> them unilaterally. The memo specifies them so client v2 is *built to support* stake, quantile
> submission, and contribution feedback — and so there's a clear spec if/when the server is rebuilt.

---

## 5. Layer 3 — Client / SDK engineering (the in-scope, low-risk modernization)

### 5.1 What's wrong now
Python 3.7/3.8; `setup.py`; synchronous `requests` long-polling; a ~10-deep crawler inheritance tree
where users must *subclass* and override `sample()`; every heavy dependency (sklearn, statsmodels,
hyperopt, copulas, tdigest, pycoingecko…) is a *required* install; a 67 MB repo with **34 committed
notebooks** (several > 1 MB) and **40 `*_examples` directories**; minimal typing; no async.

### 5.2 What's SOTA in 2026 (with named tools)
- **Packaging & tooling:** `pyproject.toml` (PEP 621), **hatchling** or **uv** build backend, **uv**
  as installer/resolver, **ruff** (lint + format), **mypy/pyright**, a **`src/` layout** so examples
  never ship in the wheel, **optional-dependency extras** to slim the core install (foundation
  models, sklearn, copulas all become `pip install microprediction[chronos]` etc.). Minimum Python
  **3.10+**. ([uv](https://docs.astral.sh/uv/), setuptools `src` layout)
- **HTTP / streaming:** migrate `requests` → **httpx** (sync+async, HTTP/2, pooling — the base layer
  of the OpenAI/Anthropic SDKs). Design **async-first**, derive a sync facade (via **unasync** or
  dual generated clients). Replace sync long-polling with **Server-Sent Events** (`httpx-sse`) where
  the server supports push; reserve **websockets** for true duplex. One shared `AsyncClient`, never
  per-request. ([httpx](https://www.python-httpx.org/async/), [httpx-sse](https://pypi.org/project/httpx-sse/))
- **Resilience:** **stamina** (backoff + jitter over tenacity) for retries; **Idempotency-Key** on
  every submit; **pyrate-limiter** for per-stream + global rate limits; **pybreaker** per endpoint so
  one bad stream can't drown healthy ones. ([stamina](https://stamina.hynek.me/))
- **Data modeling:** **pydantic v2** for request/response models and **pydantic-settings** for config
  at the trust boundary; `TypedDict` for request params; `model_construct()` to skip re-validation on
  trusted responses (the openai-python pattern). ([pydantic v2](https://docs.pydantic.dev/latest/concepts/performance/))
- **Strategy architecture — replace the inheritance tree with a Protocol + engine.** Define a
  `Forecaster` **`typing.Protocol`** (`def predict(history, horizon) -> Quantiles`). Make the crawler
  a **runtime/engine** that takes an **injected** `forecaster` (composition / DI), owning the
  poll→predict→submit loop — instead of being subclassed. This is exactly how Nixtla
  (`StatsForecast(models=[...])`), sktime (stable public `fit/predict`, private `_fit/_predict`),
  darts, GluonTS, and River structure their public APIs. Add a **callback/hook list** (Lightning-style)
  for cross-cutting concerns (logging, rate-limit handling) instead of god-subclasses. Expose
  third-party strategies via **`importlib.metadata` entry points**; escalate to **pluggy** only if
  strategies must hook multiple lifecycle stages. The ~10-deep tree collapses into *one Protocol +
  one engine + injected strategies + optional callbacks*. ([PEP 544](https://peps.python.org/pep-0544/),
  [sktime extension](https://www.sktime.net/en/latest/developer_guide/add_estimators.html))
- **Concurrency:** one **asyncio task per stream** supervised under **`asyncio.TaskGroup`** (3.11+)
  for structured concurrency; bound with a semaphore/bulkhead.
- **Ops:** multi-stage Dockerfile (uv + slim/distroless, non-root); run "forever" via Docker
  `--restart=unless-stopped` or systemd `Restart=always` (single host) / Fargate or Cloud Run
  `min-instances≥1` (managed) / K8s Deployment + liveness probe (fleet); **avoid Lambda** for a
  long-poller. **structlog** JSON → stdout + **OpenTelemetry** auto-instrumentation; a heartbeat /
  dead-man's-switch for a silently-hung crawler.
- **Testing:** **respx** (httpx mocking) + **VCR.py** (recorded cassettes) + **pytest-asyncio** +
  **Hypothesis** (property-based for the samplers/serializers) + **Pact** (consumer-driven contract
  test against the live API).
- **Repo hygiene:** move the 40 example dirs + 34 notebooks to a **separate `microprediction-cookbook`
  repo** (the OpenAI/HuggingFace/LangChain pattern); keep ≤5 tiny curated examples in-repo. Store
  notebooks via **jupytext** (`.py`/`.md`) + **nbstripout** pre-commit + **nbval** in CI. Build docs
  with **Material for MkDocs + mkdocstrings** (API from docstrings). This alone takes the repo from
  67 MB to a few MB. ([openai-cookbook](https://github.com/openai/openai-cookbook),
  [jupytext](https://jupytext.readthedocs.io/))

### 5.3 Recommendation (Layer 3)
Build **`microprediction` v2 as a new, clean package** (async httpx core, pydantic models,
`Forecaster` Protocol + engine, extras-based optional deps, src layout, modern tooling) rather than
refactoring the existing tree in place. Ship it alongside v1 so existing crawlers keep running.
Split examples into a cookbook repo. This layer is fully in-scope for this repository and carries the
least risk.

---

## 6. Synthesis — three options, and the recommendation

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **A. Incremental refresh** | Modernize in place: pyproject/uv/ruff, add type hints, swap `requests`→httpx, strip notebooks | Lowest effort; preserves everything | Inheritance tree + 225-sample primitive survive; doesn't address the real rot |
| **B. Modular rebuild of the client (recommended)** | New `microprediction` v2 package: typed async core + `Forecaster` Protocol + engine + extras; quantile-native with a samples-compat shim; cookbook split. Forecasting & incentive specs (§3, §4) written down for the server | Fixes Layer 3 fully; *prepares* for Layers 1–2; ships next to v1; reversible | Layers 1–2 only land when the server cooperates |
| **C. Full from-scratch rewrite of the network** | New client **and** new server: quantile wire format, CRPS scoring, stake-based identity, MMC-style contribution rewards, ensemble-as-product | Realizes the entire modern design | Large; the server isn't in this repo; only worth it if the live network is the strategic priority (see §0 caveat) |

**Recommendation: Option B now, with the §3–§4 specs as the contract for an eventual Option C.**
It is the honest match to (a) what this repo actually controls, (b) the risk you should take before
re-validating that the live network is still the goal, and (c) where the engineering leverage is.
Do **not** do a big-bang Option C rewrite before deciding the network is the thing worth rebuilding.

---

## 7. Proposed target architecture (client v2 sketch)

```
microprediction/                      # src/ layout, py>=3.10, hatchling+uv, ruff+mypy
├── core/
│   ├── client.py        # async httpx; AsyncClient pool; SSE for live feeds; Idempotency-Key
│   ├── models.py        # pydantic v2: Stream, Prediction(Quantiles), Submission, Score
│   └── config.py        # pydantic-settings (env-driven)
├── forecast/
│   ├── protocol.py      # Forecaster Protocol: predict(history, horizon) -> Quantiles
│   ├── baselines.py     # bootstrap (ported), Gaussian — zero-dep defaults
│   ├── foundation.py    # [extra] Chronos-Bolt / TTM quantile baseline
│   └── conformal.py     # online-conformal wrapper (ACI / Conformal-PID)
├── engine/
│   ├── runtime.py       # the loop: discover→poll→predict→submit; TaskGroup per stream
│   └── callbacks.py     # Lightning-style hooks (logging, rate-limit, lifecycle)
└── compat.py            # v1 shim: Quantiles <-> 225 samples; legacy submit path

# Optional extras:  microprediction[chronos], [ttm], [sklearn], [copula], [all]
# Examples & notebooks → separate microprediction-cookbook repo (jupytext + nbval)
```

Contributor "hello world" becomes injection, not subclassing:

```python
from microprediction import Engine
from microprediction.forecast import ConformalForecaster, ChronosBaseline

engine = Engine(
    forecaster=ConformalForecaster(ChronosBaseline()),  # strong, calibrated, zero-shot
    write_key=KEY,
)
await engine.run()   # async-first; sync facade also provided
```

---

## 8. Phased roadmap

1. **Phase 0 — Repo hygiene (days).** `src/` layout, `pyproject.toml`, uv/ruff/mypy, CI; move
   notebooks + 40 example dirs to `microprediction-cookbook`; nbstripout/jupytext. *Pure win, no
   behavior change.*
2. **Phase 1 — Typed async core (1–2 wks).** httpx client, pydantic models, stamina retries,
   idempotent submit, rate limiting. Keep v1 wire format (225 samples).
3. **Phase 2 — Forecaster Protocol + engine (1–2 wks).** Collapse the crawler tree; DI; callbacks;
   entry-point strategy discovery; port the existing samplers as baselines.
4. **Phase 3 — Modern forecasting (2–4 wks).** Quantile-native `Prediction` type + compat shim;
   foundation-model baseline (extra); online-conformal wrapper; local CRPS/PIT self-evaluation.
5. **Phase 4 — Mechanism (server-dependent, scoped separately).** Quantile submission endpoint;
   CRPS scoring against realized values; stake-based identity; MMC-style contribution rewards;
   ensemble-as-product. *Requires server work; specs in §3–§4.*

---

## 9. Open questions for you

1. **Is the live network still the priority**, or is the goal to capture the good ideas in a modern
   form (in which case client v2 + the incentive spec are the deliverables, and the server is moot)?
2. **Do you control the server** (`api.microprediction.org` + scoring/Redis), or is it effectively
   frozen? This decides whether Layers 1–2 are buildable or merely specifiable.
3. **Greenfield `v2` package vs in-place evolution** — confirm Option B (new package alongside v1).
4. **How much does backward compatibility matter** — are there live crawlers in the wild that must
   keep running unchanged?
5. **Appetite for the Numerai-style stake/contribution economics** — is that a direction you want, or
   is the open/no-stake/PoW ethos a deliberate value to preserve?

Answers to (1)–(2) most change what gets built next; I can turn the chosen option into a concrete
implementation plan (or start Phase 0) on this branch once you point.

---

## 10. References

**Forecasting / statistics**
- Gneiting & Raftery, *Strictly Proper Scoring Rules* (2007) — https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Any-Quantile Probabilistic Forecasting — https://arxiv.org/abs/2404.17451
- Chronos-Bolt (AWS) — https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/
- Chronos-2 — https://arxiv.org/pdf/2510.15821 · Moirai-2 — https://arxiv.org/html/2511.11698v1
- IBM Tiny Time Mixers — https://arxiv.org/pdf/2401.03955
- Are TSFMs well-calibrated? — https://arxiv.org/html/2510.16060v1
- Adaptive Conformal Inference — https://arxiv.org/abs/2106.00170 · Conformal PID — https://arxiv.org/pdf/2307.16895 (code: https://github.com/aangelopoulos/conformal-time-series)
- River (online ML) — https://riverml.xyz/ · TimeGrad (diffusion) — https://arxiv.org/abs/2101.12072

**Mechanism / incentives**
- Manipulation & Peer Mechanisms: A Survey — https://arxiv.org/pdf/2210.01984
- Prelec et al., "Surprisingly Popular" (Nature 2017) — https://www.nature.com/articles/nature21054
- Forecast Aggregation via Peer Prediction — https://arxiv.org/pdf/1910.03779
- Numerai MMC/MPC — https://blog.numer.ai/signals-alpha-and-mpc/ · staking — https://docs.numer.ai/numerai-signals/staking
- M6 competition results — https://arxiv.org/abs/2310.13357
- Proof-of-Personhood overview — https://digitap.app/news/guide/proof-of-personhood-solving-sybil-attacks
- Bayesian stacking via proper scores — https://arxiv.org/html/2509.04203

**Client / SDK engineering**
- httpx — https://www.python-httpx.org/async/ · httpx-sse — https://pypi.org/project/httpx-sse/ · stamina — https://stamina.hynek.me/
- pydantic v2 performance — https://docs.pydantic.dev/latest/concepts/performance/
- PEP 544 Protocols — https://peps.python.org/pep-0544/ · sktime extension — https://www.sktime.net/en/latest/developer_guide/add_estimators.html
- uv — https://docs.astral.sh/uv/ · openai-cookbook (examples split) — https://github.com/openai/openai-cookbook · jupytext — https://jupytext.readthedocs.io/
