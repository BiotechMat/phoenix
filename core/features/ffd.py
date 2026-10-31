"""Fixed-Width Window Fractional Differentiation (FFD) with Frozen Per-Split d*.

Implements Marcos Lopez de Prado's Fixed-Width Window Fractional Differentiation (AFML Ch. 5)
with strict causality, Numba JIT convolution, and per-CPCV-split frozen calibration to prevent
weight-length and scale drift between training and out-of-sample evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import warnings

import numba
import numpy as np
import numpy.typing as npt
from scipy import stats

from core.contracts import FFDWeightsConfig

try:
    from statsmodels.tsa.stattools import adfuller as _sm_adfuller

    _HAS_STATSMODELS = True
except ImportError:
    _sm_adfuller = None
    _HAS_STATSMODELS = False


@numba.njit(cache=True)
def _compute_ffd_weights_numba(
    d: float,
    tau: float = 1e-5,
    max_len: int = 10000,
) -> npt.NDArray[np.float64]:
    """Compute truncated binomial FFD weights: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k.

    Truncates as soon as |w_k| < tau (for k >= 1) or k reaches max_len.
    """
    buf = np.empty(max_len, dtype=np.float64)
    buf[0] = 1.0
    length = 1
    for k in range(1, max_len):
        w_k = -buf[k - 1] * (d - float(k) + 1.0) / float(k)
        if abs(w_k) < tau:
            break
        buf[k] = w_k
        length += 1
    return buf[:length].copy()


def compute_ffd_weights(
    d: float,
    tau: float = 1e-5,
    max_len: int = 10000,
) -> npt.NDArray[np.float64]:
    """Compute Fixed-Width Window Fractional Differentiation weights for order d in (0, 1).

    Parameters
    ----------
    d : float
        Fractional differentiation order in (0.0, 1.0).
    tau : float, default 1e-5
        Weight truncation threshold tau = 10^-5 (L* = max{k : |w_k| >= tau}).
    max_len : int, default 10000
        Hard upper bound on memory window length.

    Returns
    -------
    npt.NDArray[np.float64]
        1D array of weights [w_0, w_1, ..., w_{L*}] where w_0 = 1.0 and w_k < 0 for k >= 1.
    """
    if not (0.0 < d < 1.0):
        raise ValueError(f"Fractional differentiation order d must be in (0, 1), got {d}")
    if tau <= 0.0:
        raise ValueError(f"Truncation threshold tau must be > 0, got {tau}")
    if max_len < 2:
        raise ValueError(f"max_len must be >= 2, got {max_len}")
    return _compute_ffd_weights_numba(float(d), float(tau), int(max_len))


@numba.njit(cache=True)
def apply_ffd_weights_numba(
    series: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply FFD weights via strictly causal convolution.

    At index t, the output depends ONLY on series[0 : t + 1] and NEVER on future indices > t.
    For warm-up bars t < L* (where L* = len(weights) - 1), unobserved pre-sample lags (t - k < 0)
    are causally anchored to series[0] so the effective weight sum is constant across all t >= 0
    while preserving strict causal invariance to 10^-12 precision.
    """
    n = series.shape[0]
    w_len = weights.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    if w_len == 0:
        for i in range(n):
            out[i] = 0.0
        return out

    x0 = series[0]
    for t in range(n):
        acc = 0.0
        max_k = t if t < (w_len - 1) else (w_len - 1)
        for k in range(max_k + 1):
            acc += weights[k] * series[t - k]
        if max_k < w_len - 1:
            rem_w = 0.0
            for k in range(max_k + 1, w_len):
                rem_w += weights[k]
            acc += rem_w * x0
        out[t] = acc
    return out


def _mackinnon_adf_pvalue_fallback(
    series: npt.NDArray[np.float64],
    maxlag: int = 1,
) -> float:
    """Deterministic OLS Augmented Dickey-Fuller (ADF) test with MacKinnon p-value interpolation.

    Used when statsmodels is unavailable or when a fast fallback is required.
    Estimates: delta_y_t = alpha + beta * y_{t-1} + sum_{i=1}^p gamma_i * delta_y_{t-i} + eps_t.
    """
    y = np.asarray(series, dtype=np.float64)
    n = y.shape[0]
    if n < max(maxlag + 5, 10):
        return 1.0

    dy = np.diff(y)
    if float(np.std(y)) < 1e-12 or float(np.std(dy)) < 1e-12:
        return 1.0

    p = max(0, int(maxlag))
    target = dy[p:]
    t_eff = target.shape[0]
    if t_eff <= p + 3:
        return 1.0

    cols = [np.ones(t_eff, dtype=np.float64), y[p : n - 1]]
    for lag in range(1, p + 1):
        cols.append(dy[p - lag : n - 1 - lag])
    x_mat = np.column_stack(cols)

    try:
        beta_hat, residuals, _, _ = np.linalg.lstsq(x_mat, target, rcond=None)
        if residuals.size > 0:
            rss = float(residuals[0])
        else:
            resid_vec = target - x_mat @ beta_hat
            rss = float(np.dot(resid_vec, resid_vec))
        dof = max(1, t_eff - x_mat.shape[1])
        sigma2 = rss / float(dof)
        cov_beta = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
        se_beta = math.sqrt(max(float(cov_beta[1, 1]), 1e-24))
        t_stat = float(beta_hat[1]) / se_beta
    except np.linalg.LinAlgError:
        return 1.0

    # MacKinnon (1994, 2010) asymptotic critical values for constant-only ADF ("c"):
    # tau_0.01 = -3.43035 - 6.5393/T - 16.786/T^2
    # tau_0.05 = -2.86154 - 2.8903/T - 4.234/T^2
    # tau_0.10 = -2.56677 - 1.5384/T - 2.809/T^2
    # Median (p=0.50) ~ -1.57, p=0.90 ~ -0.44, p=0.99 ~ 0.60
    inv_t = 1.0 / float(t_eff)
    tau_01 = -3.43035 - 6.5393 * inv_t - 16.786 * (inv_t * inv_t)
    tau_05 = -2.86154 - 2.8903 * inv_t - 4.234 * (inv_t * inv_t)
    tau_10 = -2.56677 - 1.5384 * inv_t - 2.809 * (inv_t * inv_t)

    if t_stat <= -6.0:
        return 1e-6
    if t_stat >= 1.0:
        return 0.999

    tau_knots = np.array([-6.0, -4.5, tau_01, tau_05, tau_10, -1.94, -1.57, -0.44, 0.60, 1.0])
    p_knots = np.array([1e-6, 1e-4, 0.01, 0.05, 0.10, 0.30, 0.50, 0.90, 0.99, 0.999])
    pval = float(np.interp(t_stat, tau_knots, p_knots))
    return float(min(max(pval, 0.0), 1.0))


def compute_adf_pvalue(
    series: npt.NDArray[np.float64],
    maxlag: int = 1,
) -> float:
    """Compute the Augmented Dickey-Fuller (ADF) unit-root p-value on a 1D series.

    Uses `statsmodels.tsa.stattools.adfuller` when available, with a deterministic
    MacKinnon OLS fallback if statsmodels is not installed or encounters degenerate inputs.
    """
    clean = np.asarray(series, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.shape[0] < 12 or float(np.std(clean)) < 1e-12:
        return 1.0

    if _HAS_STATSMODELS and _sm_adfuller is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = _sm_adfuller(clean, maxlag=maxlag, regression="c", autolag=None)
            pval = float(getattr(res, "pvalue", res[1]))
            if math.isfinite(pval):
                return float(min(max(pval, 0.0), 1.0))
        except Exception:
            pass

    return _mackinnon_adf_pvalue_fallback(clean, maxlag=maxlag)


def compute_level_correlation(
    raw_series: npt.NDArray[np.float64],
    ffd_series: npt.NDArray[np.float64],
    burn_in: int = 0,
) -> float:
    """Compute Pearson correlation between raw log-levels and FFD stationary series."""
    if raw_series.shape[0] != ffd_series.shape[0]:
        raise ValueError("raw_series and ffd_series must have identical length")
    n = raw_series.shape[0]
    start = min(max(0, burn_in), max(0, n - 10))
    x = raw_series[start:]
    y = ffd_series[start:]
    if x.shape[0] < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    r_val, _ = stats.pearsonr(x, y)
    if not math.isfinite(float(r_val)):
        return 0.0
    return float(min(max(float(r_val), -1.0), 1.0))


def calibrate_optimal_d(
    log_series: npt.NDArray[np.float64],
    split_id: str = "split_0",
    d_grid: Sequence[float] | None = None,
    tau: float = 1e-5,
    max_adf_pvalue: float = 0.01,
) -> FFDWeightsConfig:
    """Sweep d in [0.05, 0.60] on an in-sample training split to find minimum stationary d*.

    Selects the smallest d in `d_grid` that achieves `adf_pvalue <= max_adf_pvalue` (0.01),
    preserving maximum memory (correlation with raw levels). If no candidate in the grid
    crosses the p-value threshold (e.g. on very short synthetic slices), selects the candidate
    with the lowest ADF p-value (typically d = 0.60).

    Parameters
    ----------
    log_series : npt.NDArray[np.float64]
        1D array of in-sample log-prices or log-open-interest.
    split_id : str, default "split_0"
        Identifier for the CPCV training split.
    d_grid : Sequence[float] | None, default None
        Candidate fractional orders in [0.05, 0.60]. Defaults to 0.05 .. 0.60 in steps of 0.05.
    tau : float, default 1e-5
        Weight truncation threshold (10^-5).
    max_adf_pvalue : float, default 0.01
        Maximum ADF p-value required for stationarity (99% confidence).

    Returns
    -------
    FFDWeightsConfig
        Frozen Pydantic v2 configuration containing d_star, weights, adf_pvalue, and correlation.
    """
    arr = np.asarray(log_series, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 16:
        raise ValueError(
            f"log_series must be a 1D array with at least 16 observations, got shape {arr.shape}"
        )

    if d_grid is None:
        candidates = [round(0.05 * i, 4) for i in range(1, 13)]  # 0.05, 0.10, ..., 0.60
    else:
        candidates = sorted(float(d) for d in d_grid)
        if not candidates:
            raise ValueError("d_grid must not be empty")
        for d_val in candidates:
            if not (0.05 <= d_val <= 0.60):
                raise ValueError(f"All d_grid candidates must lie in [0.05, 0.60], got {d_val}")

    best_fallback_cfg: FFDWeightsConfig | None = None
    best_fallback_pval: float = 2.0

    n = arr.shape[0]
    for d_cand in candidates:
        weights = compute_ffd_weights(d=d_cand, tau=tau, max_len=max(n, 2048))
        ffd_vals = apply_ffd_weights_numba(arr, weights)
        burn_in = min( max(0, len(weights) - 1), n // 3 )
        eval_slice = ffd_vals[burn_in:]
        pval = compute_adf_pvalue(eval_slice, maxlag=1)
        corr = compute_level_correlation(arr, ffd_vals, burn_in=burn_in)

        cfg = FFDWeightsConfig(
            split_id=split_id,
            d_star=d_cand,
            threshold_tau=tau,
            weights=tuple(float(w) for w in weights),
            adf_pvalue=pval,
            level_correlation=corr,
        )

        if pval <= max_adf_pvalue:
            return cfg

        if pval < best_fallback_pval:
            best_fallback_pval = pval
            best_fallback_cfg = cfg

    assert best_fallback_cfg is not None
    return best_fallback_cfg


class FrozenFFDTransformer:
    """Stateful FFD transformer that freezes d* and weights per CPCV training split.

    Guarantees that identical FFD weights (computed solely on in-sample training data)
    are applied to both in-sample and out-of-sample / live bars without re-estimation
    or lookahead leakage.
    """

    def __init__(self, config: FFDWeightsConfig | None = None) -> None:
        self._config: FFDWeightsConfig | None = config
        self._weights_arr: npt.NDArray[np.float64] | None = (
            np.asarray(config.weights, dtype=np.float64) if config is not None else None
        )

    @property
    def is_fitted(self) -> bool:
        """Return True if a frozen FFDWeightsConfig is bound to this transformer."""
        return self._config is not None

    @property
    def config(self) -> FFDWeightsConfig:
        """Return the frozen FFDWeightsConfig, raising RuntimeError if unfitted."""
        if self._config is None:
            raise RuntimeError("FrozenFFDTransformer has not been fitted on a training split.")
        return self._config

    @property
    def d_star(self) -> float:
        """Return the frozen fractional differentiation order d*."""
        return self.config.d_star

    @property
    def split_id(self) -> str:
        """Return the split_id on which this transformer was calibrated."""
        return self.config.split_id

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Return a read-only view of the frozen FFD weight vector."""
        if self._weights_arr is None:
            raise RuntimeError("FrozenFFDTransformer has not been fitted on a training split.")
        view = self._weights_arr.view()
        view.flags.writeable = False
        return view

    def fit(
        self,
        in_sample_log_series: npt.NDArray[np.float64],
        split_id: str = "split_0",
        d_grid: Sequence[float] | None = None,
        tau: float = 1e-5,
        max_adf_pvalue: float = 0.01,
    ) -> FrozenFFDTransformer:
        """Calibrate and freeze d* on an in-sample training log-series."""
        cfg = calibrate_optimal_d(
            log_series=in_sample_log_series,
            split_id=split_id,
            d_grid=d_grid,
            tau=tau,
            max_adf_pvalue=max_adf_pvalue,
        )
        self._config = cfg
        self._weights_arr = np.asarray(cfg.weights, dtype=np.float64)
        return self

    @classmethod
    def from_training_split(
        cls,
        in_sample_log_series: npt.NDArray[np.float64],
        split_id: str = "split_0",
        d_grid: Sequence[float] | None = None,
        tau: float = 1e-5,
        max_adf_pvalue: float = 0.01,
    ) -> FrozenFFDTransformer:
        """Construct and freeze a FrozenFFDTransformer directly from an in-sample training split."""
        transformer = cls()
        return transformer.fit(
            in_sample_log_series=in_sample_log_series,
            split_id=split_id,
            d_grid=d_grid,
            tau=tau,
            max_adf_pvalue=max_adf_pvalue,
        )

    def transform(self, log_series: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the frozen FFD weights causally across a 1D log-series."""
        if self._weights_arr is None:
            raise RuntimeError("FrozenFFDTransformer must be fitted before calling transform().")
        arr = np.asarray(log_series, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D log_series array, got shape {arr.shape}")
        return apply_ffd_weights_numba(arr, self._weights_arr)

    def transform_last(self, log_history: npt.NDArray[np.float64]) -> float:
        """Compute the causal FFD value for the latest bar at the end of `log_history` in O(L*)."""
        if self._weights_arr is None:
            raise RuntimeError("FrozenFFDTransformer must be fitted before calling transform_last().")
        arr = np.asarray(log_history, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] == 0:
            raise ValueError("log_history must be a non-empty 1D array")
        n = arr.shape[0]
        w = self._weights_arr
        w_len = w.shape[0]
        t = n - 1
        max_k = t if t < (w_len - 1) else (w_len - 1)
        acc = float(np.dot(w[: max_k + 1], arr[t - max_k : t + 1][::-1]))
        if max_k < w_len - 1:
            acc += float(np.sum(w[max_k + 1 :])) * float(arr[0])
        return acc
