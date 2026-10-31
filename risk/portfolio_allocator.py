"""Unsigned HRP-CVaR Risk Budgeting and Signed HRP-CVaR + Fractional Kelly Portfolio Allocator.

Implements:
1. Ledoit-Wolf shrinkage covariance (`sklearn.covariance.LedoitWolf`) and correlation distance matrix
   D_ij = sqrt(0.5 * (1 - rho_ij)).
2. SciPy hierarchical tree clustering (`single` or `ward` linkage) and quasi-diagonalization (`get_quasi_diag`).
3. Recursive bisection inversely proportional to 95% Historical/Parametric Cluster Conditional Value at Risk (CVaR_0.95).
4. Mapping unsigned HRP-CVaR weights w_{HRP,i} > 0 (sum = 1) to signed target weights via `compute_signed_target_weights`:
       w_{final,i} = y_{primary,i} * w_{HRP,i} * f_{Kelly}(p_{meta,i}) * M_{regime}
   subject to expected gross return > 2 * Maker_Fee and sum_i |w_{final,i}| <= 2.5x gross leverage cap.
"""

from __future__ import annotations

import logging
import math
from typing import Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from core.contracts import (
    AssetAllocationInput,
    ChampionChallengerState,
    ExecutionRegimeMode,
    FeeSchedule,
    PortfolioAllocationOutput,
    SystemicRiskOutput,
    compute_fractional_kelly,
    compute_regime_multiplier_from_asri,
    compute_signed_target_weights,
)

DEFAULT_CVAR_ALPHA: Final[float] = 0.95
DEFAULT_GAMMA_KELLY: Final[float] = 0.5
DEFAULT_MAX_GROSS_LEVERAGE: Final[float] = 2.5
MIN_CVAR_FLOOR: Final[float] = 1e-8


def compute_ledoit_wolf_cov_corr(
    returns: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Ledoit-Wolf shrinkage covariance and correlation matrices from (T, M) returns."""
    ret = np.asarray(returns, dtype=np.float64)
    if ret.ndim != 2:
        raise ValueError(f"returns must be a 2D array of shape (n_samples, n_assets), got shape {ret.shape}")
    n_samples, n_assets = ret.shape
    if n_assets < 1:
        raise ValueError("returns must contain at least 1 asset column.")
    if n_samples < 2:
        raise ValueError(f"returns must contain at least 2 observations, got {n_samples}")

    clean_ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    if n_assets == 1:
        var_0 = max(float(np.var(clean_ret[:, 0], ddof=1)), 1e-10)
        return np.array([[var_0]], dtype=np.float64), np.array([[1.0]], dtype=np.float64)

    lw = LedoitWolf(store_precision=False)
    lw.fit(clean_ret)
    cov = np.asarray(lw.covariance_, dtype=np.float64)

    # Ensure strictly positive diagonal variances
    diag_var = np.maximum(np.diag(cov), 1e-12)
    std = np.sqrt(diag_var)
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.eye(n_assets, dtype=np.float64), where=denom > 1e-15)
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return cov, corr


def compute_correlation_distance(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute correlation distance matrix D_ij = sqrt(0.5 * (1 - rho_ij))."""
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"corr must be a square 2D matrix, got shape {c.shape}")

    clipped = np.clip(0.5 * (c + c.T), -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - clipped)))
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def get_quasi_diag(link: NDArray[np.float64]) -> list[int]:
    """Quasi-diagonalize a SciPy hierarchical linkage matrix into an ordered list of leaf asset indices.

    Follows Lopez de Prado (2016) tree seriation algorithm: iteratively replaces non-leaf
    cluster identifiers (>= n_assets) with their left and right children until only leaf
    indices in [0, n_assets - 1] remain.
    """
    link_arr = np.asarray(link)
    if link_arr.ndim != 2 or link_arr.shape[1] < 2:
        raise ValueError(f"Invalid linkage matrix shape {link_arr.shape}; expected (M - 1, 4).")

    num_items = link_arr.shape[0] + 1
    sort_ix: list[int] = [int(link_arr[-1, 0]), int(link_arr[-1, 1])]

    while max(sort_ix) >= num_items:
        expanded: list[int] = []
        for idx in sort_ix:
            if idx >= num_items:
                row = idx - num_items
                expanded.append(int(link_arr[row, 0]))
                expanded.append(int(link_arr[row, 1]))
            else:
                expanded.append(idx)
        sort_ix = expanded

    return sort_ix


def _compute_single_series_cvar(
    series_returns: NDArray[np.float64],
    variance: float,
    alpha: float = DEFAULT_CVAR_ALPHA,
    method: Literal["historical", "parametric", "hybrid"] = "historical",
) -> float:
    """Compute 95% Conditional Value at Risk (Expected Shortfall) of a 1D return series."""
    if not (0.50 <= alpha < 1.0):
        raise ValueError(f"CVaR alpha must be in [0.50, 1.0), got {alpha}")

    losses = -np.asarray(series_returns, dtype=np.float64).ravel()
    var_threshold = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_threshold]
    hist_cvar = float(np.mean(tail_losses)) if tail_losses.size > 0 else var_threshold

    sigma = math.sqrt(max(variance, 1e-12))
    mu = float(np.mean(series_returns))
    z_alpha = float(norm.ppf(alpha))
    phi_z = float(norm.pdf(z_alpha))
    param_cvar = -mu + sigma * (phi_z / (1.0 - alpha))

    if method == "historical":
        raw_cvar = hist_cvar if hist_cvar > MIN_CVAR_FLOOR else param_cvar
    elif method == "parametric":
        raw_cvar = param_cvar
    elif method == "hybrid":
        raw_cvar = 0.5 * max(hist_cvar, MIN_CVAR_FLOOR) + 0.5 * max(param_cvar, MIN_CVAR_FLOOR)
    else:
        raise ValueError(f"Unsupported cvar_method '{method}'")

    return float(max(raw_cvar, MIN_CVAR_FLOOR))


def compute_cluster_cvar(
    returns: NDArray[np.float64],
    cov: NDArray[np.float64],
    cluster_indices: Sequence[int],
    alpha: float = DEFAULT_CVAR_ALPHA,
    method: Literal["historical", "parametric", "hybrid"] = "historical",
) -> float:
    """Compute 95% Historical/Parametric Conditional Value at Risk (CVaR_0.95) for a sub-cluster."""
    idx_list = [int(i) for i in cluster_indices]
    if not idx_list:
        raise ValueError("cluster_indices must not be empty.")

    sub_ret = returns[:, idx_list]
    sub_cov = cov[np.ix_(idx_list, idx_list)]

    if len(idx_list) == 1:
        return _compute_single_series_cvar(
            series_returns=sub_ret[:, 0],
            variance=float(sub_cov[0, 0]),
            alpha=alpha,
            method=method,
        )

    # Intra-cluster inverse-tail-risk sub-allocation
    asset_cvars = np.array(
        [
            _compute_single_series_cvar(
                series_returns=sub_ret[:, k],
                variance=float(sub_cov[k, k]),
                alpha=alpha,
                method=method,
            )
            for k in range(len(idx_list))
        ],
        dtype=np.float64,
    )
    inv_cvar = 1.0 / np.maximum(asset_cvars, MIN_CVAR_FLOOR)
    sub_weights = inv_cvar / float(np.sum(inv_cvar))

    cluster_ret = sub_ret @ sub_weights
    cluster_var = float(sub_weights.T @ sub_cov @ sub_weights)
    return _compute_single_series_cvar(
        series_returns=cluster_ret,
        variance=cluster_var,
        alpha=alpha,
        method=method,
    )


def compute_hrp_cvar_weights(
    returns: NDArray[np.float64],
    alpha: float = DEFAULT_CVAR_ALPHA,
    linkage_method: Literal["single", "ward"] = "single",
    cvar_method: Literal["historical", "parametric", "hybrid"] = "historical",
) -> NDArray[np.float64]:
    """Compute unsigned HRP-CVaR risk budget weights w_{HRP,i} > 0 with sum_{i=1}^M w_{HRP,i} = 1.

    Pipeline:
    1. Fit `sklearn.covariance.LedoitWolf` shrinkage covariance and correlation matrix rho.
    2. Compute correlation distance matrix D_ij = sqrt(0.5 * (1 - rho_ij)).
    3. Cluster assets via SciPy hierarchical tree linkage (`single` or `ward`).
    4. Quasi-diagonalize linkage matrix via `get_quasi_diag`.
    5. Recursively bisect sorted clusters inversely proportional to 95% Cluster CVaR.
    """
    if linkage_method not in ("single", "ward"):
        raise ValueError(f"linkage_method must be 'single' or 'ward', got '{linkage_method}'")

    ret = np.asarray(returns, dtype=np.float64)
    if ret.ndim != 2:
        raise ValueError(f"returns must be a 2D array of shape (n_samples, n_assets), got {ret.shape}")

    _, n_assets = ret.shape
    if n_assets == 0:
        raise ValueError("returns must contain at least 1 asset column.")
    if n_assets == 1:
        return np.ones(1, dtype=np.float64)

    cov, corr = compute_ledoit_wolf_cov_corr(ret)
    dist = compute_correlation_distance(corr)

    # Pairwise Euclidean distance of distance matrix columns for tree clustering
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method=linkage_method)
    sort_ix = get_quasi_diag(link)

    weights = np.ones(n_assets, dtype=np.float64)
    clusters: list[list[int]] = [sort_ix]

    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left_cluster = cluster[:mid]
            right_cluster = cluster[mid:]

            cvar_left = compute_cluster_cvar(
                returns=ret,
                cov=cov,
                cluster_indices=left_cluster,
                alpha=alpha,
                method=cvar_method,
            )
            cvar_right = compute_cluster_cvar(
                returns=ret,
                cov=cov,
                cluster_indices=right_cluster,
                alpha=alpha,
                method=cvar_method,
            )

            # Recursive bisection inversely proportional to 95% Cluster CVaR
            total_cvar = cvar_left + cvar_right
            alpha_left = 1.0 - (cvar_left / total_cvar) if total_cvar > 0.0 else 0.5
            alpha_left = min(max(alpha_left, 1e-6), 1.0 - 1e-6)
            alpha_right = 1.0 - alpha_left

            weights[left_cluster] *= alpha_left
            weights[right_cluster] *= alpha_right

            if len(left_cluster) > 1:
                next_clusters.append(left_cluster)
            if len(right_cluster) > 1:
                next_clusters.append(right_cluster)
        clusters = next_clusters

    weights = np.maximum(weights, 1e-12)
    weights /= float(np.sum(weights))
    return weights


def compute_hrp_cvar_weight_dict(
    returns: NDArray[np.float64],
    asset_ids: Sequence[str],
    alpha: float = DEFAULT_CVAR_ALPHA,
    linkage_method: Literal["single", "ward"] = "single",
    cvar_method: Literal["historical", "parametric", "hybrid"] = "historical",
) -> dict[str, float]:
    """Compute unsigned HRP-CVaR risk budgets mapped by asset_id."""
    ret = np.asarray(returns, dtype=np.float64)
    if ret.ndim != 2 or ret.shape[1] != len(asset_ids):
        raise ValueError(
            f"Number of asset_ids ({len(asset_ids)}) must match returns columns ({ret.shape[1] if ret.ndim == 2 else 'N/A'})"
        )
    w_arr = compute_hrp_cvar_weights(
        returns=ret,
        alpha=alpha,
        linkage_method=linkage_method,
        cvar_method=cvar_method,
    )
    return {str(asset_id): float(w_arr[i]) for i, asset_id in enumerate(asset_ids)}


# ==============================================================================
# Rolling hourly return matrix and hourly HRP-CVaR weight schedule (REINFORCEMENT_PLAN P0.3)
# ==============================================================================

HRP_MIN_RETURN_ROWS: Final[int] = 96
NANOSECONDS_PER_HOUR: Final[int] = 3_600_000_000_000

_hrp_logger = logging.getLogger("phoenix.hrp")


class RollingHourlyReturnMatrix:
    """Rolling matrix of 1-hour log returns for a fixed asset universe.

    Rows are appended either from a seed of hourly closes (``seed_from_hourly_closes``) or
    online from live prices (``update``), which appends one row per completed hour bucket.
    The matrix is strictly causal: a row for hour ``h`` is formed from the close observed at
    the end of hour ``h`` against the close at the end of hour ``h - 1``.
    """

    def __init__(self, asset_ids: Sequence[str], max_rows: int = 720) -> None:
        if not asset_ids:
            raise ValueError("asset_ids must not be empty")
        if max_rows < 2:
            raise ValueError(f"max_rows must be >= 2, got {max_rows}")
        self._asset_ids: tuple[str, ...] = tuple(str(a) for a in asset_ids)
        self._max_rows = max_rows
        self._rows: list[list[float]] = []
        self._close_rows: list[list[float]] = []
        self._last_hour_bucket: int | None = None
        self._last_hourly_close: dict[str, float] = {}
        self._completed_bucket: int | None = None
        self._completed_close: dict[str, float] | None = None

    @property
    def asset_ids(self) -> tuple[str, ...]:
        """Asset universe in column order."""
        return self._asset_ids

    @property
    def n_rows(self) -> int:
        """Number of hourly return rows currently held."""
        return len(self._rows)

    @property
    def last_hour_bucket(self) -> int | None:
        """Hour bucket (unix hours) of the most recent row, or None."""
        return self._last_hour_bucket

    def closes_by_asset(self) -> dict[str, list[float]]:
        """Completed hourly closes per asset, oldest first (up to ``max_rows + 1`` per asset)."""
        return {a: [row[i] for row in self._close_rows] for i, a in enumerate(self._asset_ids)}

    def matrix(self) -> NDArray[np.float64]:
        """Return the (n_rows, n_assets) matrix of hourly log returns."""
        if not self._rows:
            return np.empty((0, len(self._asset_ids)), dtype=np.float64)
        return np.asarray(self._rows, dtype=np.float64)

    def seed_from_hourly_closes(self, hourly_closes: Mapping[str, Sequence[tuple[int, float]]]) -> int:
        """Seed from ``{asset: [(ts_ms, close), ...]}`` hourly closes aligned on common timestamps."""
        per_asset: dict[str, dict[int, float]] = {}
        for asset in self._asset_ids:
            series = hourly_closes.get(asset)
            if not series:
                return self.n_rows
            per_asset[asset] = {
                int(ts_ms) // 3_600_000: float(px) for ts_ms, px in series if float(px) > 0.0
            }
        common = sorted(set.intersection(*(set(d.keys()) for d in per_asset.values())))
        if len(common) < 2:
            return self.n_rows
        self._rows = []
        self._close_rows = [[per_asset[a][b] for a in self._asset_ids] for b in common][-(self._max_rows + 1) :]
        prev = common[0]
        for bucket in common[1:]:
            self._rows.append(
                [math.log(per_asset[a][bucket] / per_asset[a][prev]) for a in self._asset_ids]
            )
            prev = bucket
        self._rows = self._rows[-self._max_rows :]
        self._last_hour_bucket = common[-1]
        self._last_hourly_close = {a: per_asset[a][common[-1]] for a in self._asset_ids}
        self._completed_bucket = common[-1]
        self._completed_close = dict(self._last_hourly_close)
        return self.n_rows

    def update(self, ts_ns: int, prices: Mapping[str, float]) -> bool:
        """Observe live prices at ``ts_ns``; append one return row each time an hour completes.

        The last observation inside an hour is that hour's close. When the first observation of a later
        hour arrives, the previous hour is complete and its return against the previous completed hour is
        appended. Callers should pass a bar's close time minus 1 ns so a bar closing exactly on the hour
        belongs to the hour it closes. Returns True when a return row was appended.
        """
        if any(a not in prices or float(prices[a]) <= 0.0 for a in self._asset_ids):
            return False
        bucket = int(ts_ns) // NANOSECONDS_PER_HOUR
        if self._last_hour_bucket is None:
            self._last_hour_bucket = bucket
            self._last_hourly_close = {a: float(prices[a]) for a in self._asset_ids}
            return False
        if bucket <= self._last_hour_bucket:
            self._last_hourly_close = {a: float(prices[a]) for a in self._asset_ids}
            return False
        appended = self._complete_hour(self._last_hour_bucket, dict(self._last_hourly_close))
        self._last_hour_bucket = bucket
        self._last_hourly_close = {a: float(prices[a]) for a in self._asset_ids}
        return appended

    def _complete_hour(self, bucket: int, closes: dict[str, float]) -> bool:
        if self._completed_bucket is not None and bucket <= self._completed_bucket:
            return False  # already recorded (e.g. the last seeded hour)
        appended = False
        if self._completed_close is not None:
            prev = self._completed_close
            self._rows.append([math.log(closes[a] / prev[a]) for a in self._asset_ids])
            self._rows = self._rows[-self._max_rows :]
            appended = True
        self._close_rows.append([closes[a] for a in self._asset_ids])
        self._close_rows = self._close_rows[-(self._max_rows + 1) :]
        self._completed_bucket = bucket
        self._completed_close = closes
        return appended


class HRPWeightScheduler:
    """Compute unsigned HRP-CVaR weights at most once per hour from a rolling return matrix.

    When fewer than ``min_rows`` hourly rows exist, equal weights are returned and a warning
    is logged; ``last_result_is_fallback`` reports which branch produced the live weights.
    """

    def __init__(
        self,
        asset_ids: Sequence[str],
        min_rows: int = HRP_MIN_RETURN_ROWS,
        recompute_interval_ns: int = NANOSECONDS_PER_HOUR,
        alpha: float = DEFAULT_CVAR_ALPHA,
        linkage_method: Literal["single", "ward"] = "single",
        cvar_method: Literal["historical", "parametric", "hybrid"] = "historical",
    ) -> None:
        if not asset_ids:
            raise ValueError("asset_ids must not be empty")
        if min_rows < 2:
            raise ValueError(f"min_rows must be >= 2, got {min_rows}")
        self._asset_ids: tuple[str, ...] = tuple(str(a) for a in asset_ids)
        self._min_rows = min_rows
        self._recompute_interval_ns = max(1, int(recompute_interval_ns))
        self._alpha = alpha
        self._linkage_method: Literal["single", "ward"] = linkage_method
        self._cvar_method: Literal["historical", "parametric", "hybrid"] = cvar_method
        self._last_compute_ts_ns: int | None = None
        self._last_weights: dict[str, float] = self.equal_weights()
        self._last_is_fallback: bool = True
        self._last_rows_used: int = 0

    @property
    def asset_ids(self) -> tuple[str, ...]:
        """Asset universe."""
        return self._asset_ids

    @property
    def min_rows(self) -> int:
        """Minimum hourly rows required before HRP-CVaR is trusted."""
        return self._min_rows

    @property
    def last_weights(self) -> dict[str, float]:
        """Most recently produced weights."""
        return dict(self._last_weights)

    @property
    def last_result_is_fallback(self) -> bool:
        """True when the most recent weights are the equal-weight fallback."""
        return self._last_is_fallback

    @property
    def last_rows_used(self) -> int:
        """Rows in the matrix when the weights were last computed."""
        return self._last_rows_used

    def equal_weights(self) -> dict[str, float]:
        """Equal-weight fallback summing to 1."""
        n = float(len(self._asset_ids))
        return {a: 1.0 / n for a in self._asset_ids}

    def compute(self, ts_ns: int, returns: NDArray[np.float64], force: bool = False) -> dict[str, float]:
        """Return HRP-CVaR weights for ``returns`` (n_rows, n_assets), recomputing at most hourly."""
        ret = np.asarray(returns, dtype=np.float64)
        if (
            not force
            and self._last_compute_ts_ns is not None
            and int(ts_ns) - self._last_compute_ts_ns < self._recompute_interval_ns
        ):
            return dict(self._last_weights)

        self._last_compute_ts_ns = int(ts_ns)
        n_rows = int(ret.shape[0]) if ret.ndim == 2 else 0
        self._last_rows_used = n_rows
        if ret.ndim != 2 or ret.shape[1] != len(self._asset_ids) or n_rows < self._min_rows:
            _hrp_logger.warning(
                "HRP-CVaR fallback to equal weights: %d hourly return rows available, %d required.",
                n_rows,
                self._min_rows,
            )
            self._last_weights = self.equal_weights()
            self._last_is_fallback = True
            return dict(self._last_weights)

        try:
            weights = compute_hrp_cvar_weight_dict(
                returns=ret,
                asset_ids=self._asset_ids,
                alpha=self._alpha,
                linkage_method=self._linkage_method,
                cvar_method=self._cvar_method,
            )
        except Exception as exc:
            _hrp_logger.warning("HRP-CVaR computation failed (%s); using equal weights.", exc)
            self._last_weights = self.equal_weights()
            self._last_is_fallback = True
            return dict(self._last_weights)

        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0.0:
            self._last_weights = self.equal_weights()
            self._last_is_fallback = True
            return dict(self._last_weights)
        self._last_weights = {a: float(w) / total for a, w in weights.items()}
        self._last_is_fallback = False
        return dict(self._last_weights)


class SignedHRPCVaRAllocator:
    """Signed HRP-CVaR & Fractional Kelly Target Portfolio Allocator.

    Governing allocation equation:
        w_{final,i} = y_{primary,i} * w_{HRP,i} * f_{Kelly}(p_{meta,i}) * M_{regime}

    Enforces:
    1. Unsigned HRP-CVaR cluster risk budgets w_{HRP,i} > 0 (sum_{i=1}^M w_{HRP,i} = 1).
    2. Fractional Kelly meta-probability conviction scalar f_{Kelly}(p_{meta,i}) = clip(gamma * (2 * p_{meta,i} - 1), 0, 1).
    3. Expected gross return hurdle > 2 * Maker_Fee (rejects sub-hurdle trades).
    4. Champion/Challenger sleeve regime fallback (zeros directional exposure when in CARRY_ONLY_FALLBACK).
    5. Global gross leverage ceiling sum_{i=1}^M |w_{final,i}| <= 2.5x.
    """

    def __init__(
        self,
        gamma_frac: float = DEFAULT_GAMMA_KELLY,
        max_gross_leverage: float = DEFAULT_MAX_GROSS_LEVERAGE,
        cvar_alpha: float = DEFAULT_CVAR_ALPHA,
        linkage_method: Literal["single", "ward"] = "single",
        cvar_method: Literal["historical", "parametric", "hybrid"] = "historical",
        default_fee_schedule: FeeSchedule | None = None,
    ) -> None:
        if not (0.0 < gamma_frac <= 1.0):
            raise ValueError(f"gamma_frac must be in (0.0, 1.0], got {gamma_frac}")
        if not (0.0 < max_gross_leverage <= 2.5):
            raise ValueError(f"max_gross_leverage must be in (0.0, 2.5], got {max_gross_leverage}")
        if not (0.50 <= cvar_alpha < 1.0):
            raise ValueError(f"cvar_alpha must be in [0.50, 1.0), got {cvar_alpha}")

        self.gamma_frac: float = gamma_frac
        self.max_gross_leverage: float = max_gross_leverage
        self.cvar_alpha: float = cvar_alpha
        self.linkage_method: Literal["single", "ward"] = linkage_method
        self.cvar_method: Literal["historical", "parametric", "hybrid"] = cvar_method
        self.default_fee_schedule: FeeSchedule = default_fee_schedule or FeeSchedule.usdt_perp()

    def _resolve_regime_multiplier(
        self,
        regime_multiplier: float = 1.0,
        asri_output: SystemicRiskOutput | None = None,
    ) -> float:
        """Combine explicit regime_multiplier with top-down ASRI macro risk scalar."""
        if not (0.0 <= regime_multiplier <= 1.0):
            raise ValueError(f"regime_multiplier must be in [0.0, 1.0], got {regime_multiplier}")
        if asri_output is None:
            return regime_multiplier
        asri_mult = compute_regime_multiplier_from_asri(
            asri_score=asri_output.score,
            tail_risk_flag=asri_output.tail_risk_flag,
        )
        return float(min(regime_multiplier, asri_mult))

    def allocate_from_inputs(
        self,
        inputs: Sequence[AssetAllocationInput],
        regime_multiplier: float = 1.0,
        asri_output: SystemicRiskOutput | None = None,
        sleeve_states: Mapping[str, ChampionChallengerState] | None = None,
    ) -> PortfolioAllocationOutput:
        """Compute signed target weights from pre-constructed AssetAllocationInput objects."""
        effective_m_regime = self._resolve_regime_multiplier(
            regime_multiplier=regime_multiplier,
            asri_output=asri_output,
        )

        adjusted_inputs: list[AssetAllocationInput] = []
        for item in inputs:
            if sleeve_states is not None and item.asset_id in sleeve_states:
                sleeve_state = sleeve_states[item.asset_id]
                if sleeve_state.regime_mode != ExecutionRegimeMode.DIRECTIONAL_PLUS_CARRY:
                    adjusted_inputs.append(
                        AssetAllocationInput(
                            asset_id=item.asset_id,
                            y_primary=0,
                            w_hrp=item.w_hrp,
                            p_meta=item.p_meta,
                            expected_gross_return_bps=item.expected_gross_return_bps,
                            fee_schedule=item.fee_schedule,
                            epistemic_gate_passed=item.epistemic_gate_passed,
                        )
                    )
                    continue
            adjusted_inputs.append(item)

        return compute_signed_target_weights(
            inputs=adjusted_inputs,
            regime_multiplier=effective_m_regime,
            gamma_frac=self.gamma_frac,
            max_gross_leverage=self.max_gross_leverage,
        )

    def allocate_from_returns_and_signals(
        self,
        returns: NDArray[np.float64],
        asset_ids: Sequence[str],
        primary_directions: Mapping[str, Literal[-1, 0, 1]] | Sequence[Literal[-1, 0, 1]],
        meta_probabilities: Mapping[str, float] | Sequence[float],
        expected_gross_returns_bps: Mapping[str, float] | Sequence[float],
        epistemic_gates: Mapping[str, bool] | Sequence[bool] | None = None,
        fee_schedules: Mapping[str, FeeSchedule] | None = None,
        regime_multiplier: float = 1.0,
        asri_output: SystemicRiskOutput | None = None,
        sleeve_states: Mapping[str, ChampionChallengerState] | None = None,
    ) -> PortfolioAllocationOutput:
        """Compute unsigned HRP-CVaR weights from historical returns and map to signed target weights."""
        hrp_map = compute_hrp_cvar_weight_dict(
            returns=returns,
            asset_ids=asset_ids,
            alpha=self.cvar_alpha,
            linkage_method=self.linkage_method,
            cvar_method=self.cvar_method,
        )

        allocation_inputs: list[AssetAllocationInput] = []
        for idx, asset_id in enumerate(asset_ids):
            y_dir: Literal[-1, 0, 1] = (
                primary_directions[asset_id]
                if isinstance(primary_directions, Mapping)
                else primary_directions[idx]
            )
            p_meta_val = (
                float(meta_probabilities[asset_id])
                if isinstance(meta_probabilities, Mapping)
                else float(meta_probabilities[idx])
            )
            gross_bps_val = (
                float(expected_gross_returns_bps[asset_id])
                if isinstance(expected_gross_returns_bps, Mapping)
                else float(expected_gross_returns_bps[idx])
            )
            if epistemic_gates is None:
                gate_ok = True
            elif isinstance(epistemic_gates, Mapping):
                gate_ok = bool(epistemic_gates.get(asset_id, True))
            else:
                gate_ok = bool(epistemic_gates[idx])

            schedule = (
                fee_schedules[asset_id]
                if (fee_schedules is not None and asset_id in fee_schedules)
                else self.default_fee_schedule
            )

            allocation_inputs.append(
                AssetAllocationInput(
                    asset_id=asset_id,
                    y_primary=y_dir,
                    w_hrp=hrp_map[asset_id],
                    p_meta=p_meta_val,
                    expected_gross_return_bps=gross_bps_val,
                    fee_schedule=schedule,
                    epistemic_gate_passed=gate_ok,
                )
            )

        return self.allocate_from_inputs(
            inputs=allocation_inputs,
            regime_multiplier=regime_multiplier,
            asri_output=asri_output,
            sleeve_states=sleeve_states,
        )

    def allocate(
        self,
        inputs: Sequence[AssetAllocationInput],
        regime_multiplier: float = 1.0,
        asri_output: SystemicRiskOutput | None = None,
        sleeve_states: Mapping[str, ChampionChallengerState] | None = None,
    ) -> PortfolioAllocationOutput:
        """Primary entrypoint delegating to `allocate_from_inputs`."""
        return self.allocate_from_inputs(
            inputs=inputs,
            regime_multiplier=regime_multiplier,
            asri_output=asri_output,
            sleeve_states=sleeve_states,
        )


__all__ = [
    "DEFAULT_CVAR_ALPHA",
    "DEFAULT_GAMMA_KELLY",
    "DEFAULT_MAX_GROSS_LEVERAGE",
    "SignedHRPCVaRAllocator",
    "compute_cluster_cvar",
    "compute_correlation_distance",
    "compute_fractional_kelly",
    "compute_hrp_cvar_weight_dict",
    "compute_hrp_cvar_weights",
    "compute_ledoit_wolf_cov_corr",
    "compute_regime_multiplier_from_asri",
    "compute_signed_target_weights",
    "get_quasi_diag",
]
