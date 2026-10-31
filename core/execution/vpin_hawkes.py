"""Real-Time Bulk Volume Classification (BVC) VPIN and O(1) Recursive Markovian Hawkes Monitor.

Implements:
1. RollingBVCVPINMonitor: 50-bucket volume clock Bulk Volume Classification VPIN
   with exact multi-bucket residual volume spillover and strict causality.
2. @numba.njit O(1) recursive Markovian Hawkes intensity updates:
   R_i = exp(-beta * (t_i - t_{i-1})) * (1 + R_{i-1})
   lambda(t_i) = mu + alpha * R_i
   eta_Hawkes = alpha / beta
3. Asynchronous 30-second MLE parameter refit via L-BFGS-B using the exact O(N)
   analytical Ozaki (1979) log-likelihood over a trailing 5-minute event ring-buffer.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import math
from typing import Final

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from core.contracts import HawkesStateSnapshot

_SQRT_2: Final[float] = math.sqrt(2.0)
_NS_TO_SEC: Final[float] = 1e-9
_EPS: Final[float] = 1e-12


@numba.njit(cache=True, fastmath=True)
def hawkes_recursive_step_numba(
    dt: float,
    r_prev: float,
    mu: float,
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    """Perform a single O(1) Markovian Hawkes recursive state update.

    Governing equations:
        R_i = exp(-beta * dt) * (1.0 + R_{i-1})
        lambda(t_i) = mu + alpha * R_i
        eta_Hawkes = alpha / beta

    Args:
        dt: Elapsed time (t_i - t_{i-1}) in seconds (clamped to >= 0.0).
        r_prev: Previous recursive accumulator R_{i-1}.
        mu: Baseline exogenous intensity (> 0).
        alpha: Self-excitation jump magnitude (>= 0).
        beta: Exponential decay rate (> 0).

    Returns:
        Tuple of (r_i, lambda_i, eta_hawkes).
    """
    dt_clamped = dt if dt > 0.0 else 0.0
    r_i = math.exp(-beta * dt_clamped) * (1.0 + r_prev)
    lambda_i = mu + alpha * r_i
    eta_hawkes = alpha / beta if beta > 0.0 else 0.0
    return r_i, lambda_i, eta_hawkes


@numba.njit(cache=True, fastmath=True)
def hawkes_recursive_batch_numba(
    timestamps_sec: NDArray[np.float64],
    mu: float,
    alpha: float,
    beta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute recursive Hawkes state R_i and intensity lambda(t_i) across a batch of timestamps.

    At index 0: R_0 = 0.0, lambda(t_0) = mu.
    For i >= 1: R_i = exp(-beta * (t_i - t_{i-1})) * (1.0 + R_{i-1}), lambda(t_i) = mu + alpha * R_i.

    Args:
        timestamps_sec: 1D float64 array of monotonically non-decreasing arrival timestamps in seconds.
        mu: Baseline intensity (> 0).
        alpha: Jump size (>= 0).
        beta: Decay rate (> 0).

    Returns:
        Tuple of (r_array, lambda_array), both 1D float64 arrays of shape (N,).
    """
    n = timestamps_sec.shape[0]
    r_arr = np.empty(n, dtype=np.float64)
    lambda_arr = np.empty(n, dtype=np.float64)
    if n == 0:
        return r_arr, lambda_arr

    r_prev = 0.0
    r_arr[0] = 0.0
    lambda_arr[0] = mu

    for i in range(1, n):
        dt = float(timestamps_sec[i] - timestamps_sec[i - 1])
        if dt < 0.0:
            dt = 0.0
        r_curr = math.exp(-beta * dt) * (1.0 + r_prev)
        r_arr[i] = r_curr
        lambda_arr[i] = mu + alpha * r_curr
        r_prev = r_curr

    return r_arr, lambda_arr


@numba.njit(cache=True, fastmath=True)
def hawkes_analytical_neg_loglik_numba(
    timestamps_sec: NDArray[np.float64],
    mu: float,
    alpha: float,
    beta: float,
) -> float:
    """Exact O(N) analytical negative log-likelihood of a univariate exponential Hawkes process.

    By Ozaki (1979):
        log L(mu, alpha, beta) = sum_{i=0}^{N-1} log(mu + alpha * R_i)
                                 - mu * (t_{N-1} - t_0)
                                 - (alpha / beta) * sum_{i=0}^{N-1} (1 - exp(-beta * (t_{N-1} - t_i)))

    Args:
        timestamps_sec: 1D float64 array of sorted event timestamps in seconds.
        mu: Baseline intensity (> 0).
        alpha: Excitation amplitude (>= 0).
        beta: Decay parameter (> 0).

    Returns:
        Negative log-likelihood value (to be minimized).
    """
    n = timestamps_sec.shape[0]
    if n < 2 or mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        return 1e18

    horizon = float(timestamps_sec[n - 1] - timestamps_sec[0])
    if horizon <= 0.0:
        return 1e18

    r_prev = 0.0
    log_intensity_sum = math.log(mu)

    for i in range(1, n):
        dt = float(timestamps_sec[i] - timestamps_sec[i - 1])
        if dt < 0.0:
            dt = 0.0
        r_curr = math.exp(-beta * dt) * (1.0 + r_prev)
        lam = mu + alpha * r_curr
        if lam <= 1e-15:
            return 1e18
        log_intensity_sum += math.log(lam)
        r_prev = r_curr

    # Exact identity: sum_{i=0}^{n-1} (1 - exp(-beta * (t_{n-1} - t_i))) = (n - 1) - R_{n-1}
    compensator_integral = mu * horizon + (alpha / beta) * (float(n - 1) - r_prev)
    log_likelihood = log_intensity_sum - compensator_integral
    return float(-log_likelihood)


def refit_parameters_mle_lbfgsb(
    event_timestamps_sec: NDArray[np.float64],
    default_mu: float = 1.0,
    default_alpha: float = 0.25,
    default_beta: float = 1.0,
    max_eta: float = 0.98,
) -> tuple[float, float, float]:
    """Fit (mu, alpha, beta) via L-BFGS-B minimizing the exact O(N) analytical negative log-likelihood.

    Designed to be called asynchronously once every 30 seconds over the trailing 5-minute
    event ring-buffer (never on every tick).

    Args:
        event_timestamps_sec: 1D float64 array of event arrival times in seconds.
        default_mu: Fallback mu when insufficient events are present.
        default_alpha: Fallback alpha when insufficient events are present.
        default_beta: Fallback beta when insufficient events are present.
        max_eta: Upper bound on branching ratio eta = alpha / beta for stability.

    Returns:
        Tuple of fitted (mu, alpha, beta).
    """
    ts = np.asarray(event_timestamps_sec, dtype=np.float64)
    if ts.ndim != 1 or ts.shape[0] < 5:
        return float(default_mu), float(default_alpha), float(default_beta)

    ts_sorted = np.sort(ts)
    horizon = float(ts_sorted[-1] - ts_sorted[0])
    if horizon <= 1e-9:
        return float(default_mu), float(default_alpha), float(default_beta)

    n_events = ts_sorted.shape[0]
    empirical_rate = float(n_events) / horizon
    dts = np.diff(ts_sorted)
    positive_dts = dts[dts > 1e-9]
    mean_dt = float(np.mean(positive_dts)) if positive_dts.size > 0 else (1.0 / max(empirical_rate, 1e-3))
    beta_init = float(np.clip(1.0 / max(mean_dt, 1e-4), 0.1, 200.0))

    # Parameterize optimization as x = (mu, eta, beta) where alpha = eta * beta
    # so box constraints directly control branching ratio eta in [0.0, max_eta].
    def _objective(params: NDArray[np.float64]) -> float:
        mu_val = float(params[0])
        eta_val = float(params[1])
        beta_val = float(params[2])
        alpha_val = eta_val * beta_val
        return float(hawkes_analytical_neg_loglik_numba(ts_sorted, mu_val, alpha_val, beta_val))

    mu_lower = max(1e-4, 0.02 * empirical_rate)
    mu_upper = max(10.0, 5.0 * empirical_rate)
    bounds = [
        (mu_lower, mu_upper),
        (0.0, float(max_eta)),
        (0.05, 500.0),
    ]

    # Evaluate two candidate starting regimes (calm vs. self-exciting burst)
    candidates: list[NDArray[np.float64]] = [
        np.array([max(mu_lower, 0.85 * empirical_rate), 0.15, beta_init], dtype=np.float64),
        np.array([max(mu_lower, 0.25 * empirical_rate), 0.78, beta_init * 1.5], dtype=np.float64),
    ]

    best_params: tuple[float, float, float] = (
        float(max(mu_lower, 0.85 * empirical_rate)),
        float(0.15 * beta_init),
        float(beta_init),
    )
    best_nll = float("inf")

    for x0 in candidates:
        res = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60, "ftol": 1e-7},
        )
        if np.all(np.isfinite(res.x)) and float(res.fun) < best_nll:
            best_nll = float(res.fun)
            mu_fit = float(max(res.x[0], 1e-6))
            eta_fit = float(np.clip(res.x[1], 0.0, max_eta))
            beta_fit = float(max(res.x[2], 1e-4))
            alpha_fit = float(eta_fit * beta_fit)
            best_params = (mu_fit, alpha_fit, beta_fit)

    return best_params


class RollingBVCVPINMonitor:
    """Real-time Bulk Volume Classification (BVC) VPIN across a rolling 50-bucket volume clock.

    Explicitly handles multi-bucket residual volume spillover when a large trade crosses
    multiple volume buckets and enforces strict causality (updating bucket state at trade i
    uses only trades <= i).
    """

    def __init__(
        self,
        bucket_volume: float = 100.0,
        num_buckets: int = 50,
        sigma_window: int = 50,
        toxic_threshold: float = 0.60,
    ) -> None:
        if bucket_volume <= 0.0:
            raise ValueError(f"bucket_volume must be > 0.0, got {bucket_volume}")
        if num_buckets < 1:
            raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
        if sigma_window < 2:
            raise ValueError(f"sigma_window must be >= 2, got {sigma_window}")
        if not (0.0 < toxic_threshold <= 1.0):
            raise ValueError(f"toxic_threshold must be in (0.0, 1.0], got {toxic_threshold}")

        self._bucket_volume: float = float(bucket_volume)
        self._num_buckets: int = int(num_buckets)
        self._sigma_window: int = int(sigma_window)
        self._toxic_threshold: float = float(toxic_threshold)

        self._delta_p_window: deque[float] = deque(maxlen=self._sigma_window)
        self._bucket_imbalances: deque[float] = deque(maxlen=self._num_buckets)
        self._last_price: float | None = None

        self._current_bucket_volume: float = 0.0
        self._current_bucket_buy_volume: float = 0.0
        self._current_bucket_sell_volume: float = 0.0
        self._total_buckets_completed: int = 0
        self._vpin: float = 0.0

    @property
    def bucket_volume(self) -> float:
        """Target volume size per volume bucket."""
        return self._bucket_volume

    @property
    def num_buckets(self) -> int:
        """Number of rolling volume buckets (default 50)."""
        return self._num_buckets

    @property
    def sigma_window(self) -> int:
        """Causal rolling window length for price change standard deviation."""
        return self._sigma_window

    @property
    def toxic_threshold(self) -> float:
        """VPIN threshold above which order flow is flagged as toxic (default 0.60)."""
        return self._toxic_threshold

    @property
    def vpin(self) -> float:
        """Current rolling BVC VPIN in [0.0, 1.0]."""
        return self._vpin

    @property
    def is_toxic(self) -> bool:
        """True if current VPIN strictly exceeds toxic_threshold."""
        return self._vpin > self._toxic_threshold

    @property
    def completed_buckets_count(self) -> int:
        """Number of completed buckets currently in the rolling window (0..num_buckets)."""
        return len(self._bucket_imbalances)

    @property
    def total_buckets_completed(self) -> int:
        """Cumulative number of volume buckets completed since initialization or reset."""
        return self._total_buckets_completed

    @property
    def current_bucket_volume(self) -> float:
        """Residual volume accumulated inside the active incomplete bucket."""
        return self._current_bucket_volume

    @property
    def current_bucket_buy_volume(self) -> float:
        """Buy-classified volume accumulated inside the active incomplete bucket."""
        return self._current_bucket_buy_volume

    @property
    def current_bucket_sell_volume(self) -> float:
        """Sell-classified volume accumulated inside the active incomplete bucket."""
        return self._current_bucket_sell_volume

    @property
    def bucket_imbalances(self) -> tuple[float, ...]:
        """Snapshot of completed bucket imbalances |V_buy - V_sell| / V_bucket."""
        return tuple(self._bucket_imbalances)

    def reset(self) -> None:
        """Reset all rolling state and active bucket accumulators."""
        self._delta_p_window.clear()
        self._bucket_imbalances.clear()
        self._last_price = None
        self._current_bucket_volume = 0.0
        self._current_bucket_buy_volume = 0.0
        self._current_bucket_sell_volume = 0.0
        self._total_buckets_completed = 0
        self._vpin = 0.0

    def _classify_causal_bvc(self, price: float) -> tuple[float, float]:
        """Causally classify trade volume into (w_buy, w_sell) using only trades <= current."""
        if self._last_price is None:
            self._last_price = price
            return 0.5, 0.5

        delta_p = price - self._last_price
        self._last_price = price
        self._delta_p_window.append(delta_p)

        if abs(delta_p) <= 1e-15:
            return 0.5, 0.5

        n = len(self._delta_p_window)
        if n >= 2:
            mean_dp = sum(self._delta_p_window) / float(n)
            var_dp = sum((dp - mean_dp) ** 2 for dp in self._delta_p_window) / float(n)
            sigma_dp = math.sqrt(max(var_dp, 0.0))
        else:
            sigma_dp = 0.0

        sigma_floor = 1e-6 * max(abs(delta_p), 1.0)
        sigma_eff = max(sigma_dp, sigma_floor)
        z = delta_p / sigma_eff
        w_buy = 0.5 * (1.0 + math.erf(z / _SQRT_2))
        w_buy = min(max(w_buy, 0.0), 1.0)
        return w_buy, 1.0 - w_buy

    def update(self, price: float, volume: float, ts_ns: int | None = None) -> float:
        """Process a single trade (price, volume) with exact multi-bucket residual spillover.

        Args:
            price: Trade execution price (> 0).
            volume: Trade volume (>= 0).
            ts_ns: Optional timestamp in nanoseconds.

        Returns:
            Updated rolling BVC VPIN in [0.0, 1.0].
        """
        _ = ts_ns
        if price <= 0.0:
            raise ValueError(f"Trade price must be > 0.0, got {price}")
        if volume < 0.0:
            raise ValueError(f"Trade volume must be >= 0.0, got {volume}")

        w_buy, w_sell = self._classify_causal_bvc(price)
        if volume == 0.0:
            return self._vpin

        remaining_vol = float(volume)

        # Step 1: If there is a partially filled bucket and remaining_vol can complete it
        if self._current_bucket_volume > 0.0:
            space_in_bucket = self._bucket_volume - self._current_bucket_volume
            if remaining_vol >= space_in_bucket - _EPS:
                fill_vol = max(0.0, space_in_bucket)
                buy_total = self._current_bucket_buy_volume + fill_vol * w_buy
                sell_total = self._current_bucket_sell_volume + fill_vol * w_sell
                imbalance = min(max(abs(buy_total - sell_total) / self._bucket_volume, 0.0), 1.0)
                self._bucket_imbalances.append(imbalance)
                self._total_buckets_completed += 1
                remaining_vol = max(0.0, remaining_vol - fill_vol)
                self._current_bucket_volume = 0.0
                self._current_bucket_buy_volume = 0.0
                self._current_bucket_sell_volume = 0.0
            else:
                self._current_bucket_volume += remaining_vol
                self._current_bucket_buy_volume += remaining_vol * w_buy
                self._current_bucket_sell_volume += remaining_vol * w_sell
                remaining_vol = 0.0

        # Step 2: Spillover across any number of full buckets in O(1)
        if remaining_vol >= self._bucket_volume - _EPS:
            n_full = int((remaining_vol + _EPS) // self._bucket_volume)
            if n_full > 0:
                full_imbalance = min(max(abs(w_buy - w_sell), 0.0), 1.0)
                self._total_buckets_completed += n_full
                append_count = min(n_full, self._num_buckets)
                for _ in range(append_count):
                    self._bucket_imbalances.append(full_imbalance)
                remaining_vol = max(0.0, remaining_vol - float(n_full) * self._bucket_volume)
                if remaining_vol < _EPS:
                    remaining_vol = 0.0

        # Step 3: Store any final fractional bucket remainder
        if remaining_vol > _EPS:
            self._current_bucket_volume = remaining_vol
            self._current_bucket_buy_volume = remaining_vol * w_buy
            self._current_bucket_sell_volume = remaining_vol * w_sell

        if len(self._bucket_imbalances) > 0:
            self._vpin = float(
                min(max(sum(self._bucket_imbalances) / float(len(self._bucket_imbalances)), 0.0), 1.0)
            )
        else:
            self._vpin = 0.0

        return self._vpin

    def on_trade(self, price: float, volume: float, ts_ns: int | None = None) -> float:
        """Alias for update(price, volume, ts_ns)."""
        return self.update(price=price, volume=volume, ts_ns=ts_ns)

    def update_batch(
        self,
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sequentially and causally update BVC VPIN across arrays of trade prices and volumes."""
        if prices.shape[0] != volumes.shape[0]:
            raise ValueError("prices and volumes must have identical length")
        n = prices.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.update(float(prices[i]), float(volumes[i]))
        return out


class RecursiveHawkesMonitor:
    """O(1) Recursive Markovian Hawkes Intensity Monitor with asynchronous 30s L-BFGS-B refit.

    Maintains a trailing 5-minute (300s) ring-buffer of event timestamps and triggers
    an L-BFGS-B maximum likelihood parameter refit at most once every 30 seconds.
    """

    def __init__(
        self,
        mu: float = 1.0,
        alpha: float = 0.25,
        beta: float = 1.0,
        window_sec: float = 300.0,
        refit_interval_sec: float = 30.0,
        min_events_for_refit: int = 25,
        eta_toxic_threshold: float = 0.70,
        vpin_toxic_threshold: float = 0.60,
        async_refit: bool = True,
    ) -> None:
        if mu <= 0.0:
            raise ValueError(f"mu must be > 0.0, got {mu}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0.0, got {alpha}")
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0.0, got {beta}")
        if window_sec <= 0.0:
            raise ValueError(f"window_sec must be > 0.0, got {window_sec}")
        if refit_interval_sec <= 0.0:
            raise ValueError(f"refit_interval_sec must be > 0.0, got {refit_interval_sec}")

        self._mu: float = float(mu)
        self._alpha: float = float(alpha)
        self._beta: float = float(beta)
        self._window_sec: float = float(window_sec)
        self._refit_interval_sec: float = float(refit_interval_sec)
        self._min_events_for_refit: int = int(min_events_for_refit)
        self._eta_toxic_threshold: float = float(eta_toxic_threshold)
        self._vpin_toxic_threshold: float = float(vpin_toxic_threshold)
        self._async_refit: bool = bool(async_refit)

        self._r_state: float = 0.0
        self._lambda_intensity: float = float(mu)
        self._last_ts_ns: int | None = None
        self._last_refit_ts_sec: float | None = None
        self._refit_count: int = 0

        self._event_ring_sec: deque[float] = deque()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hawkes_mle_refit")
            if self._async_refit
            else None
        )
        self._pending_refit_future: Future[tuple[float, float, float]] | None = None

    @property
    def mu(self) -> float:
        """Current baseline intensity mu (> 0)."""
        self._collect_completed_async_refit()
        return self._mu

    @property
    def alpha(self) -> float:
        """Current self-excitation parameter alpha (>= 0)."""
        self._collect_completed_async_refit()
        return self._alpha

    @property
    def beta(self) -> float:
        """Current exponential decay rate beta (> 0)."""
        self._collect_completed_async_refit()
        return self._beta

    @property
    def r_state(self) -> float:
        """Current recursive Markovian accumulator R_i (>= 0)."""
        return self._r_state

    @property
    def lambda_intensity(self) -> float:
        """Current conditional intensity lambda(t_i) = mu + alpha * R_i."""
        return self._lambda_intensity

    @property
    def branching_ratio_eta(self) -> float:
        """Current branching ratio eta_Hawkes = alpha / beta."""
        self._collect_completed_async_refit()
        return self._alpha / self._beta if self._beta > 0.0 else 0.0

    @property
    def refit_count(self) -> int:
        """Number of completed 30-second parameter refits."""
        self._collect_completed_async_refit()
        return self._refit_count

    @property
    def window_event_count(self) -> int:
        """Number of events currently retained in the trailing 5-minute ring-buffer."""
        return len(self._event_ring_sec)

    def _collect_completed_async_refit(self) -> None:
        """Non-blocking check to harvest completed asynchronous L-BFGS-B refit parameters."""
        if self._pending_refit_future is not None and self._pending_refit_future.done():
            try:
                mu_new, alpha_new, beta_new = self._pending_refit_future.result()
                self._mu = mu_new
                self._alpha = alpha_new
                self._beta = beta_new
                self._lambda_intensity = self._mu + self._alpha * self._r_state
                self._refit_count += 1
            finally:
                self._pending_refit_future = None

    def update(self, ts_ns: int, vpin: float = 0.0) -> HawkesStateSnapshot:
        """Perform an O(1) recursive state update on event arrival at ts_ns.

        Also maintains the trailing 5-minute ring-buffer and triggers an asynchronous
        L-BFGS-B parameter refit at most once every 30 seconds.

        Args:
            ts_ns: Event timestamp in nanoseconds.
            vpin: Current BVC VPIN value in [0.0, 1.0] to include in the snapshot.

        Returns:
            Validated HawkesStateSnapshot.
        """
        self._collect_completed_async_refit()

        ts_sec = float(ts_ns) * _NS_TO_SEC
        if self._last_ts_ns is None:
            self._r_state = 0.0
            self._lambda_intensity = self._mu
            eta = self._alpha / self._beta
        else:
            dt_sec = max(0.0, float(ts_ns - self._last_ts_ns) * _NS_TO_SEC)
            r_new, lam_new, eta = hawkes_recursive_step_numba(
                dt_sec, self._r_state, self._mu, self._alpha, self._beta
            )
            self._r_state = float(r_new)
            self._lambda_intensity = float(lam_new)

        self._last_ts_ns = ts_ns

        # Maintain trailing 5-minute event ring-buffer
        self._event_ring_sec.append(ts_sec)
        cutoff_sec = ts_sec - self._window_sec
        while len(self._event_ring_sec) > 0 and self._event_ring_sec[0] < cutoff_sec:
            self._event_ring_sec.popleft()

        if self._last_refit_ts_sec is None:
            self._last_refit_ts_sec = ts_sec
        elif (
            ts_sec - self._last_refit_ts_sec >= self._refit_interval_sec
            and len(self._event_ring_sec) >= self._min_events_for_refit
        ):
            self._last_refit_ts_sec = ts_sec
            self._trigger_refit()
            eta = self._alpha / self._beta

        vpin_clamped = min(max(float(vpin), 0.0), 1.0)
        toxic_active = (vpin_clamped > self._vpin_toxic_threshold) or (eta > self._eta_toxic_threshold)

        return HawkesStateSnapshot(
            ts_ns=ts_ns,
            r_state=self._r_state,
            lambda_intensity=self._lambda_intensity,
            mu=self._mu,
            alpha=self._alpha,
            beta=self._beta,
            branching_ratio_eta=float(eta),
            vpin=vpin_clamped,
            toxic_sweep_active=toxic_active,
        )

    def on_event(self, ts_ns: int, vpin: float = 0.0) -> HawkesStateSnapshot:
        """Alias for update(ts_ns, vpin)."""
        return self.update(ts_ns=ts_ns, vpin=vpin)

    def _trigger_refit(self) -> None:
        """Dispatch L-BFGS-B parameter refit over trailing 5-minute event snapshot."""
        events_arr = np.fromiter(
            self._event_ring_sec, dtype=np.float64, count=len(self._event_ring_sec)
        )
        if self._async_refit and self._executor is not None:
            if self._pending_refit_future is None or self._pending_refit_future.done():
                self._pending_refit_future = self._executor.submit(
                    refit_parameters_mle_lbfgsb,
                    events_arr,
                    self._mu,
                    self._alpha,
                    self._beta,
                )
        else:
            mu_new, alpha_new, beta_new = refit_parameters_mle_lbfgsb(
                events_arr,
                default_mu=self._mu,
                default_alpha=self._alpha,
                default_beta=self._beta,
            )
            self._mu = mu_new
            self._alpha = alpha_new
            self._beta = beta_new
            self._lambda_intensity = self._mu + self._alpha * self._r_state
            self._refit_count += 1

    def refit_now(self) -> tuple[float, float, float]:
        """Synchronously execute an L-BFGS-B parameter refit on the current 5-minute buffer."""
        if self._pending_refit_future is not None:
            try:
                mu_new, alpha_new, beta_new = self._pending_refit_future.result(timeout=5.0)
                self._mu = mu_new
                self._alpha = alpha_new
                self._beta = beta_new
                self._lambda_intensity = self._mu + self._alpha * self._r_state
                self._refit_count += 1
            finally:
                self._pending_refit_future = None
            return self._mu, self._alpha, self._beta

        events_arr = np.fromiter(
            self._event_ring_sec, dtype=np.float64, count=len(self._event_ring_sec)
        )
        mu_new, alpha_new, beta_new = refit_parameters_mle_lbfgsb(
            events_arr,
            default_mu=self._mu,
            default_alpha=self._alpha,
            default_beta=self._beta,
        )
        self._mu = mu_new
        self._alpha = alpha_new
        self._beta = beta_new
        self._lambda_intensity = self._mu + self._alpha * self._r_state
        self._refit_count += 1
        return self._mu, self._alpha, self._beta

    def snapshot(self, ts_ns: int | None = None, vpin: float = 0.0) -> HawkesStateSnapshot:
        """Return current HawkesStateSnapshot without recording a new arrival event."""
        self._collect_completed_async_refit()
        eff_ts = ts_ns if ts_ns is not None else (self._last_ts_ns or 0)
        eta = self._alpha / self._beta
        vpin_clamped = min(max(float(vpin), 0.0), 1.0)
        toxic_active = (vpin_clamped > self._vpin_toxic_threshold) or (eta > self._eta_toxic_threshold)
        return HawkesStateSnapshot(
            ts_ns=eff_ts,
            r_state=self._r_state,
            lambda_intensity=self._lambda_intensity,
            mu=self._mu,
            alpha=self._alpha,
            beta=self._beta,
            branching_ratio_eta=float(eta),
            vpin=vpin_clamped,
            toxic_sweep_active=toxic_active,
        )

    def shutdown(self) -> None:
        """Cleanly shut down the background refit executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


# Pre-warm Numba JIT functions on module import so the first call has zero compilation overhead
_WARMUP_TS = np.array([0.0, 0.1, 0.25], dtype=np.float64)
hawkes_recursive_step_numba(0.1, 0.0, 1.0, 0.25, 1.0)
hawkes_recursive_batch_numba(_WARMUP_TS, 1.0, 0.25, 1.0)
hawkes_analytical_neg_loglik_numba(_WARMUP_TS, 1.0, 0.25, 1.0)
