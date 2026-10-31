"""Statistical jump model for market regimes (REINFORCEMENT_PLAN task P3.3).

Penalised k-means with a switching cost (Nystrup, Kolm, Lindström; Shu, Yu, Mulvey):

    minimise  sum_t ||x_t - mu_{s_t}||^2  +  lambda * sum_t 1[s_t != s_{t-1}]

fitted by coordinate descent: a Viterbi-style dynamic programme assigns states given centroids, then the
centroids are re-estimated as state means. Several seeded k-means++ starts are tried and the lowest
objective kept. ``filter_states`` runs the forward recursion only, which is causal: the state at ``t``
uses observations up to ``t``. Larger ``lambda`` gives more persistent states.
"""

from __future__ import annotations

from collections.abc import Sequence

from numba import njit
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class JumpModel(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    centroids: NDArray[np.float64]
    jump_penalty: float = Field(ge=0.0)
    feature_mean: NDArray[np.float64]
    feature_std: NDArray[np.float64]
    objective: float
    n_iter: int

    @property
    def n_states(self) -> int:
        return int(self.centroids.shape[0])

    def standardise(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (np.asarray(x, dtype=np.float64) - self.feature_mean) / self.feature_std

    def filter_states(self, x_raw: NDArray[np.float64]) -> NDArray[np.int64]:
        """Causal online states: argmin of the forward cost at every ``t``."""
        loss = _sq_dist(self.standardise(x_raw), self.centroids)
        return _forward_filter(loss, self.jump_penalty)

    def smooth_states(self, x_raw: NDArray[np.float64]) -> NDArray[np.int64]:
        """Full-sample (non-causal) Viterbi states, for description only."""
        loss = _sq_dist(self.standardise(x_raw), self.centroids)
        return _viterbi(loss, self.jump_penalty)


def _sq_dist(x: NDArray[np.float64], centroids: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = x[:, None, :] - centroids[None, :, :]
    out: NDArray[np.float64] = np.einsum("tkd,tkd->tk", diff, diff)
    return out


@njit(cache=True)
def _viterbi_kernel(loss: NDArray[np.float64], penalty: float) -> NDArray[np.int64]:
    t_len, k = loss.shape
    cost = loss[0].copy()
    back = np.zeros((t_len, k), dtype=np.int64)
    for t in range(1, t_len):
        best_prev = 0
        for j in range(1, k):
            if cost[j] < cost[best_prev]:
                best_prev = j
        switch = cost[best_prev] + penalty
        new_cost = np.empty(k)
        for j in range(k):
            if cost[j] <= switch:
                back[t, j] = j
                new_cost[j] = loss[t, j] + cost[j]
            else:
                back[t, j] = best_prev
                new_cost[j] = loss[t, j] + switch
        cost = new_cost
    states = np.empty(t_len, dtype=np.int64)
    best = 0
    for j in range(1, k):
        if cost[j] < cost[best]:
            best = j
    states[t_len - 1] = best
    for t in range(t_len - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


@njit(cache=True)
def _forward_kernel(loss: NDArray[np.float64], penalty: float) -> NDArray[np.int64]:
    t_len, k = loss.shape
    cost = loss[0].copy()
    out = np.empty(t_len, dtype=np.int64)
    out[0] = np.argmin(cost)
    for t in range(1, t_len):
        m = cost.min()
        switch = m + penalty
        for j in range(k):
            c = cost[j] if cost[j] <= switch else switch
            cost[j] = loss[t, j] + c
        m2 = cost.min()
        for j in range(k):
            cost[j] -= m2  # keep numbers bounded; argmin unchanged
        out[t] = np.argmin(cost)
    return out


def _viterbi(loss: NDArray[np.float64], penalty: float) -> NDArray[np.int64]:
    out: NDArray[np.int64] = _viterbi_kernel(np.ascontiguousarray(loss, dtype=np.float64), float(penalty))
    return out


def _forward_filter(loss: NDArray[np.float64], penalty: float) -> NDArray[np.int64]:
    out: NDArray[np.int64] = _forward_kernel(np.ascontiguousarray(loss, dtype=np.float64), float(penalty))
    return out


def _kmeans_pp(x: NDArray[np.float64], k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    centers = [x[int(rng.integers(0, x.shape[0]))]]
    for _ in range(1, k):
        d = np.min(np.stack([np.sum((x - c) ** 2, axis=1) for c in centers]), axis=0)
        total = float(np.sum(d))
        idx = int(rng.choice(x.shape[0], p=d / total)) if total > 0 else int(rng.integers(0, x.shape[0]))
        centers.append(x[idx])
    return np.stack(centers)


def fit_jump_model(
    x_raw: NDArray[np.float64],
    n_states: int,
    jump_penalty: float,
    *,
    n_init: int = 8,
    max_iter: int = 30,
    seed: int = 20261031,
) -> JumpModel:
    """Fit on raw features (standardised internally with the fitting sample's mean and std)."""
    x_raw = np.asarray(x_raw, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[0] < n_states * 10:
        raise ValueError(f"need a (T, d) matrix with T >= {n_states * 10}, got {x_raw.shape}")
    mean = np.mean(x_raw, axis=0)
    std = np.std(x_raw, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x = (x_raw - mean) / std
    rng = np.random.default_rng(seed)  # research fitting only; never in a live path
    best: tuple[float, NDArray[np.float64], int] | None = None
    for _ in range(n_init):
        centroids = _kmeans_pp(x, n_states, rng)
        states = np.full(x.shape[0], -1, dtype=np.int64)
        it = 0
        for it in range(1, max_iter + 1):
            new_states = _viterbi(_sq_dist(x, centroids), jump_penalty)
            if np.array_equal(new_states, states):
                break
            states = new_states
            for k in range(n_states):
                members = x[states == k]
                if members.shape[0] > 0:
                    centroids[k] = members.mean(axis=0)
        loss = _sq_dist(x, centroids)
        obj = float(np.sum(loss[np.arange(x.shape[0]), states]) + jump_penalty * np.sum(states[1:] != states[:-1]))
        if best is None or obj < best[0]:
            best = (obj, centroids.copy(), it)
    assert best is not None
    order = np.argsort(best[1][:, 0])  # sort states by the first feature (e.g. volatility) for stable labels
    return JumpModel(centroids=best[1][order], jump_penalty=jump_penalty, feature_mean=mean, feature_std=std, objective=best[0], n_iter=best[2])


def mean_dwell(states: Sequence[int] | NDArray[np.int64]) -> float:
    """Average run length of consecutive identical states."""
    s = np.asarray(states)
    if s.size == 0:
        return 0.0
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    return float(s.size / runs)


__all__ = ["JumpModel", "fit_jump_model", "mean_dwell"]
