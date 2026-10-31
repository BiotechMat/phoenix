"""Online residual layer, abstention and outlier guard for the meta-labeller (REINFORCEMENT_PLAN task P5.3).

- ``OnlineResidualRidge``: ridge regression of the label residual ``y - p`` on standardised continuous
  features, accumulated as forgetting-weighted normal equations (recursive least squares with exponential
  forgetting), so it can be warm-started with new labelled rows without refitting from scratch.
  The adjusted probability is ``clip(p + ridge(x), eps, 1 - eps)``.
- Abstention: no prediction when the live vector's Mahalanobis distance exceeds the 99th percentile of the
  training distances (a dissimilarity-index cut, stricter than nothing and data-driven unlike a constant).
- Outlier guard: skip a retrain when more than 30% of the new training rows are beyond the current gate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-4


class OnlineResidualRidge:
    def __init__(self, n_features: int, l2: float = 10.0, forgetting: float = 0.999) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_features = n_features
        self.l2 = float(l2)
        self.forgetting = float(forgetting)
        self._xtx = np.zeros((n_features + 1, n_features + 1), dtype=np.float64)
        self._xty = np.zeros(n_features + 1, dtype=np.float64)
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._std = np.ones(n_features, dtype=np.float64)
        self._coef = np.zeros(n_features + 1, dtype=np.float64)
        self._n_seen = 0

    @property
    def coef(self) -> NDArray[np.float64]:
        return self._coef.copy()

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def _design(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = (np.atleast_2d(np.asarray(x, dtype=np.float64)) - self._mean) / self._std
        return np.column_stack([np.ones(z.shape[0]), z])

    def fit(self, x: NDArray[np.float64], residual: NDArray[np.float64]) -> OnlineResidualRidge:
        """Reset and fit with forgetting weights (the newest row has weight 1)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self._mean = np.mean(x, axis=0)
        sd = np.std(x, axis=0)
        self._std = np.where(sd > 1e-12, sd, 1.0)
        self._xtx[:] = 0.0
        self._xty[:] = 0.0
        self._n_seen = 0
        return self.update(x, residual)

    def update(self, x: NDArray[np.float64], residual: NDArray[np.float64]) -> OnlineResidualRidge:
        """Warm-start with new rows, oldest first, decaying earlier information by ``forgetting`` per row."""
        d = self._design(x)
        r = np.asarray(residual, dtype=np.float64).reshape(-1)
        n = d.shape[0]
        decay_all = self.forgetting**n
        w = self.forgetting ** np.arange(n - 1, -1, -1, dtype=np.float64)
        self._xtx = decay_all * self._xtx + (d * w[:, None]).T @ d
        self._xty = decay_all * self._xty + (d * w[:, None]).T @ r
        reg = self.l2 * np.eye(self.n_features + 1)
        reg[0, 0] = 0.0  # the intercept is not penalised
        self._coef = np.linalg.solve(self._xtx + reg + 1e-9 * np.eye(self.n_features + 1), self._xty)
        self._n_seen += n
        return self

    def adjust(self, p: NDArray[np.float64] | float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        adj = self._design(x) @ self._coef
        return np.clip(np.asarray(p, dtype=np.float64) + adj, EPS, 1.0 - EPS)


def abstention_threshold(train_di: NDArray[np.float64], percentile: float = 99.0) -> float:
    vals = np.asarray(train_di, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, percentile)) if vals.size else float("inf")


def outlier_guard_skip(new_di: NDArray[np.float64], gate_threshold: float, max_fraction: float = 0.30) -> tuple[bool, float]:
    """(skip, flagged fraction): skip a retrain when too many new rows are beyond the current gate."""
    vals = np.asarray(new_di, dtype=np.float64)
    if vals.size == 0:
        return False, 0.0
    frac = float(np.mean(vals > gate_threshold))
    return frac > max_fraction, frac


__all__ = ["OnlineResidualRidge", "abstention_threshold", "outlier_guard_skip"]
