"""
CrunchDAO — Structural Break "Real-Time Edition" ($100k, streaming) starter.

Unlike the Open Benchmark (batch: whole before/after series at once), the 2026
Real-Time edition reveals observations ONE AT A TIME after a historical reference
segment, and you emit a cumulative "a break has occurred" score after each point,
with no look-ahead. That is an online change-point problem, which is exactly what
Peter Cotton's `midone` `Attacker` interface (and online conformal test
martingales) are built for.

This file provides a `BreakAttacker` that:
  * builds a reference distribution from the pre-break history,
  * for each new streamed point, updates an online CUSUM + a conformal-style
    exceedance martingale against that reference,
  * returns a monotone, ever-growing break score (good for ROC-AUC ranking).

It subclasses midone's `Attacker` when installed (so it drops straight into the
midone/CrunchDAO streaming runner), and falls back to a self-contained base class
so the file is runnable and testable on its own.

Install for real use:  pip install midone
"""

from __future__ import annotations

import numpy as np

try:
    from midone import Attacker as _MidoneAttacker  # streaming runner base
    HAVE_MIDONE = True
except Exception:  # self-contained fallback so this file runs anywhere
    HAVE_MIDONE = False

    class _MidoneAttacker:  # minimal stand-in mirroring the midone surface
        def __init__(self, *args, **kwargs):
            self.history: list[float] = []

        def tick(self, x: float) -> None:  # override point for assimilation
            pass

        def predict(self, horizon: int = 1) -> float:
            return 0.0

        def tick_and_predict(self, x: float, horizon: int = 1) -> float:
            self.tick(x)
            return self.predict(horizon)


class BreakAttacker(_MidoneAttacker):
    """Online structural-break detector with a monotone break score.

    Two complementary online signals, combined:
      * Standardized CUSUM of deviations from the reference mean (level shifts).
      * A conformal-style martingale on |z| exceedances vs the reference scale
        (distribution/variance shifts). Under "no break" (exchangeable with the
        reference) it stays ~bounded; it grows when the stream departs.
    """

    def __init__(self, warmup: int = 50, cusum_k: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.warmup = warmup
        self.cusum_k = cusum_k          # slack: ignore drifts smaller than k*sigma
        self.epsilon = epsilon
        self._ref: list[float] = []
        self._mu = 0.0
        self._sigma = 1.0
        self._ready = False
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._log_wealth = 0.0          # log of the betting martingale
        self._score = 0.0

    def _finalize_reference(self) -> None:
        arr = np.asarray(self._ref, dtype=float)
        self._mu = float(arr.mean())
        self._sigma = float(arr.std(ddof=1)) or 1.0
        self._ready = True

    def tick(self, x: float) -> None:
        """Assimilate one streamed observation and update the break score."""
        if not np.isfinite(x):
            return

        if not self._ready:
            self._ref.append(float(x))
            if len(self._ref) >= self.warmup:
                self._finalize_reference()
            return

        z = (x - self._mu) / self._sigma

        # Two-sided CUSUM with slack k (Page's scheme), standardized.
        self._cusum_pos = max(0.0, self._cusum_pos + z - self.cusum_k)
        self._cusum_neg = max(0.0, self._cusum_neg - z - self.cusum_k)
        cusum = max(self._cusum_pos, self._cusum_neg)

        # Conformal-style bet: reference exceedance prob p ~ 2*(1 - Phi(|z|)).
        # Bet against exchangeability; wealth compounds when |z| is surprising.
        from math import erf, sqrt, log
        p = max(self.epsilon, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))))
        self._log_wealth += log((1.0 - 0.5) + 0.5 * (1.0 / p) * 0.5)  # calibrated small bet
        self._log_wealth = max(0.0, self._log_wealth)  # reset floor (no credit for calm)

        # Monotone score: never decreases, so it ranks "broke earlier/harder" high.
        self._score = max(self._score, cusum + self._log_wealth)

    def predict(self, horizon: int = 1) -> float:
        """Return the current cumulative break score (>= 0, non-decreasing)."""
        return float(self._score)

    # Convenience for the batch/local case: score a full stream.
    def score_stream(self, values) -> float:
        for x in values:
            self.tick_and_predict(x) if HAVE_MIDONE else (self.tick(x), self.predict())
        return self.predict()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    from sklearn.metrics import roc_auc_score

    def stream(break_it):
        ref = rng.normal(0, 1, 200)
        tail = rng.normal(1.5, 2.0, 200) if break_it else rng.normal(0, 1, 200)
        return np.concatenate([ref, tail])

    labels = [i % 2 == 0 for i in range(120)]
    scores = []
    for lab in labels:
        att = BreakAttacker(warmup=150)
        scores.append(att.score_stream(stream(lab)))

    print(f"midone available: {HAVE_MIDONE}")
    print(f"streaming detector ROC AUC on synthetic data: "
          f"{roc_auc_score(labels, scores):.3f}")
