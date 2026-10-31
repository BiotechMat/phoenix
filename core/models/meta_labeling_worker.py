"""Two-Stage SVD + Calibrated LightGBM Meta-Labelling Worker with Mahalanobis Outlier Gate.

Stage 1 (`ContinuousSVDOrthogonalizer`):
    Fits thin SVD strictly on continuous training features (FFD price/OI trend z-scores
    and continuous OI-Funding-Basis divergence z-scores) with deterministic column-sign
    alignment so the dominant loading of every right singular vector V_j is strictly
    positive. Explicitly excludes the sparse binary 99th-percentile
    `liquidation_exhaustion_flag` from the SVD rotation matrix and applies it as a
    direct regime override when emitting y_primary in {-1, 0, +1} and as a raw feature
    column into Stage 2.

Stage 2 (`LightGBMMetaLabelWorker`):
    Trains a calibrated LightGBM classifier (`n_jobs=4`, with optional XGBoost `n_jobs=4`
    ensemble and scikit-learn probability calibration) on the SVD-orthogonalized
    continuous factors, raw `liquidation_exhaustion_flag`, bar volatility, and VPIN
    toxicity to predict p_meta in [0, 1].

Epistemic Gate (`MahalanobisOutlierGate`):
    Computes the Ledoit-Wolf shrinkage Mahalanobis distance D_M(x_live) strictly from
    the training distribution and hard-blocks directional entries
    (`epistemic_gate_passed = False`) whenever D_M(x_live) > D_thresh.
"""

from __future__ import annotations

import os
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

from collections.abc import Sequence
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
import xgboost as xgb

from core.contracts import (
    FeeSchedule,
    MetaPredictionOutput,
    PositioningSignalSnapshot,
)

CONTINUOUS_SIGNAL_COLUMNS: tuple[str, ...] = (
    "ffd_close_zscore",
    "ffd_oi_zscore",
    "oi_funding_basis_zscore",
)

EXTENDED_CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "ffd_close_zscore",
    "ffd_oi_zscore",
    "oi_funding_basis_zscore",
    "basis_pct",
    "funding_rate_8h",
    "oi_pct_change",
    "liquidation_zscore",
)

EXCLUDED_BINARY_COLUMNS: tuple[str, ...] = ("liquidation_exhaustion_flag",)


def extract_continuous_and_binary_from_snapshots(
    snapshots: Sequence[PositioningSignalSnapshot],
    extended: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Extract continuous SVD input matrix, raw binary exhaustion flag, and primary directions.

    Strictly keeps `liquidation_exhaustion_flag` separate from the continuous matrix.
    """
    n_rows = len(snapshots)
    if n_rows == 0:
        n_cols = len(EXTENDED_CONTINUOUS_COLUMNS) if extended else len(CONTINUOUS_SIGNAL_COLUMNS)
        return (
            np.empty((0, n_cols), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    if extended:
        x_cont = np.array(
            [
                [
                    s.ffd_close_zscore,
                    s.ffd_oi_zscore,
                    s.oi_funding_basis_zscore,
                    s.basis_pct,
                    s.funding_rate_8h,
                    s.oi_pct_change,
                    s.liquidation_zscore,
                ]
                for s in snapshots
            ],
            dtype=np.float64,
        )
    else:
        x_cont = np.array(
            [
                [
                    s.ffd_close_zscore,
                    s.ffd_oi_zscore,
                    s.oi_funding_basis_zscore,
                ]
                for s in snapshots
            ],
            dtype=np.float64,
        )

    exhaustion_flags = np.array(
        [int(s.liquidation_exhaustion_flag) for s in snapshots],
        dtype=np.int64,
    )
    primary_dirs = np.array(
        [int(s.primary_direction) for s in snapshots],
        dtype=np.int64,
    )
    return x_cont, exhaustion_flags, primary_dirs


class ContinuousSVDOrthogonalizer:
    """Stage 1 Continuous-Only SVD orthogonalizer with deterministic column-sign alignment.

    Fits strictly on training continuous medium-frequency features (excluding the
    sparse 99th-percentile binary `liquidation_exhaustion_flag`) and aligns each right
    singular vector V_j so its dominant loading (largest absolute entry) is strictly
    positive.
    """

    def __init__(
        self,
        n_components: int | None = None,
        whiten: bool = False,
        signal_threshold: float = 0.25,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.signal_threshold = signal_threshold

        self._mean: npt.NDArray[np.float64] | None = None
        self._std: npt.NDArray[np.float64] | None = None
        self._right_singular_vectors: npt.NDArray[np.float64] | None = None
        self._singular_values: npt.NDArray[np.float64] | None = None
        self._explained_variance_ratio: npt.NDArray[np.float64] | None = None
        self._feature_names: tuple[str, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the SVD orthogonalizer has been fit on training data."""
        return self._right_singular_vectors is not None

    @property
    def mean_(self) -> npt.NDArray[np.float64]:
        """Training feature means of shape (n_features,)."""
        if self._mean is None:
            raise RuntimeError("ContinuousSVDOrthogonalizer is not fitted.")
        return self._mean

    @property
    def std_(self) -> npt.NDArray[np.float64]:
        """Training feature standard deviations of shape (n_features,)."""
        if self._std is None:
            raise RuntimeError("ContinuousSVDOrthogonalizer is not fitted.")
        return self._std

    @property
    def right_singular_vectors_(self) -> npt.NDArray[np.float64]:
        """Sign-aligned right singular vectors V of shape (n_features, n_components).

        Each column V_j is guaranteed to have a strictly positive dominant loading.
        """
        if self._right_singular_vectors is None:
            raise RuntimeError("ContinuousSVDOrthogonalizer is not fitted.")
        return self._right_singular_vectors

    @property
    def components_(self) -> npt.NDArray[np.float64]:
        """Right singular vectors transposed V^T of shape (n_components, n_features)."""
        return self.right_singular_vectors_.T

    @property
    def singular_values_(self) -> npt.NDArray[np.float64]:
        """Singular values of the standardized training matrix."""
        if self._singular_values is None:
            raise RuntimeError("ContinuousSVDOrthogonalizer is not fitted.")
        return self._singular_values

    @property
    def explained_variance_ratio_(self) -> npt.NDArray[np.float64]:
        """Fraction of total variance explained by each orthogonal component."""
        if self._explained_variance_ratio is None:
            raise RuntimeError("ContinuousSVDOrthogonalizer is not fitted.")
        return self._explained_variance_ratio

    @property
    def n_components_(self) -> int:
        """Number of retained orthogonal SVD components."""
        return int(self.right_singular_vectors_.shape[1])

    @staticmethod
    def _validate_continuous_matrix(
        x_arr: npt.ArrayLike,
        feature_names: Sequence[str] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Validate 2D continuous feature matrix and ensure binary exhaustion flag is excluded."""
        if feature_names is not None:
            for forbidden in EXCLUDED_BINARY_COLUMNS:
                if forbidden in feature_names:
                    raise ValueError(
                        f"Binary column '{forbidden}' must be excluded from ContinuousSVDOrthogonalizer."
                    )

        arr = np.asarray(x_arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D continuous feature matrix, got shape {arr.shape}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Continuous feature matrix must be non-empty, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Continuous feature matrix contains NaN or infinite values.")
        return arr

    def fit(
        self,
        X_continuous: npt.ArrayLike,
        feature_names: Sequence[str] | None = None,
    ) -> ContinuousSVDOrthogonalizer:
        """Fit SVD strictly on the training split continuous features with deterministic sign alignment."""
        x_train = self._validate_continuous_matrix(X_continuous, feature_names=feature_names)
        n_samples, n_features = x_train.shape
        if n_samples < 2:
            raise ValueError(f"At least 2 training samples required for SVD, got {n_samples}")

        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"len(feature_names)={len(feature_names)} != n_features={n_features}"
                )
            self._feature_names = tuple(feature_names)
        else:
            self._feature_names = None

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0, ddof=1)
        std_safe = np.where(std > 1e-8, std, 1.0)

        z_train = (x_train - mean) / std_safe
        scale = float(np.sqrt(max(n_samples - 1, 1)))
        _, s_vals, vt = np.linalg.svd(z_train / scale, full_matrices=False)

        max_k = vt.shape[0]
        k = max_k if self.n_components is None else min(max(1, self.n_components), max_k)

        v_mat = vt[:k, :].T.copy()  # shape (n_features, k), columns are right singular vectors V_j
        s_ret = s_vals[:k].copy()

        # Deterministic column-sign alignment: ensure dominant loading of each V_j is strictly > 0
        for j in range(k):
            col = v_mat[:, j]
            dominant_idx = int(np.argmax(np.abs(col)))
            if col[dominant_idx] < 0.0:
                v_mat[:, j] *= -1.0
            elif col[dominant_idx] == 0.0:
                v_mat[dominant_idx, j] = 1.0

        total_var = float(np.sum(s_vals * s_vals))
        if total_var > 0.0:
            ev_ratio = (s_ret * s_ret) / total_var
        else:
            ev_ratio = np.full(k, 1.0 / float(k), dtype=np.float64)

        self._mean = mean
        self._std = std_safe
        self._right_singular_vectors = v_mat
        self._singular_values = np.maximum(s_ret, 1e-6)
        self._explained_variance_ratio = ev_ratio
        return self

    def transform(self, X_continuous: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Project continuous features onto the frozen, sign-aligned right singular vectors."""
        if (
            self._mean is None
            or self._std is None
            or self._right_singular_vectors is None
            or self._singular_values is None
        ):
            raise RuntimeError("ContinuousSVDOrthogonalizer must be fitted before transform().")

        arr = np.asarray(X_continuous, dtype=np.float64)
        single_row = arr.ndim == 1
        x_mat = self._validate_continuous_matrix(arr)

        if x_mat.shape[1] != self._mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self._mean.shape[0]}, got {x_mat.shape[1]}"
            )

        z_mat = (x_mat - self._mean) / self._std
        factors: npt.NDArray[np.float64] = z_mat @ self._right_singular_vectors
        if self.whiten:
            factors = factors / self._singular_values

        if single_row:
            return factors.reshape(-1)
        return factors

    def fit_transform(
        self,
        X_continuous: npt.ArrayLike,
        feature_names: Sequence[str] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Fit on training continuous features and return the orthogonalized factors."""
        return self.fit(X_continuous, feature_names=feature_names).transform(X_continuous)

    def emit_primary_direction(
        self,
        x_continuous_row: npt.ArrayLike,
        liquidation_exhaustion_flag: int = 0,
        price_trend_sign: float | None = None,
        signal_threshold: float | None = None,
    ) -> Literal[-1, 0, 1]:
        """Emit primary directional signal y_primary in {-1, 0, +1} with exhaustion override.

        - If `liquidation_exhaustion_flag == 1`, directly overrides the continuous SVD
          factor to emit a cascade exhaustion reversal signal (+1 if prior trend <= 0
          else -1).
        - Otherwise, thresholds the primary SVD continuous composite score.
        """
        if liquidation_exhaustion_flag not in (0, 1):
            raise ValueError(
                f"liquidation_exhaustion_flag must be 0 or 1, got {liquidation_exhaustion_flag}"
            )

        row_arr = np.asarray(x_continuous_row, dtype=np.float64).reshape(-1)
        factors = self.transform(row_arr)

        if liquidation_exhaustion_flag == 1:
            ref_trend = float(row_arr[0]) if price_trend_sign is None else float(price_trend_sign)
            return 1 if ref_trend <= 0.0 else -1

        thresh = self.signal_threshold if signal_threshold is None else float(signal_threshold)
        primary_score = float(factors[0])
        if primary_score > thresh:
            return 1
        if primary_score < -thresh:
            return -1
        return 0

    def emit_primary_directions(
        self,
        X_continuous: npt.ArrayLike,
        liquidation_exhaustion_flags: Sequence[int] | npt.NDArray[np.int_] | None = None,
        signal_threshold: float | None = None,
    ) -> npt.NDArray[np.int64]:
        """Vectorized primary directional signal emission across a batch of bars."""
        x_mat = self._validate_continuous_matrix(X_continuous)
        n_rows = x_mat.shape[0]
        if liquidation_exhaustion_flags is None:
            flags = np.zeros(n_rows, dtype=np.int64)
        else:
            flags = np.asarray(liquidation_exhaustion_flags, dtype=np.int64).reshape(-1)
            if flags.shape[0] != n_rows:
                raise ValueError(
                    f"Length mismatch: X_continuous has {n_rows} rows, flags has {flags.shape[0]}"
                )

        out = np.zeros(n_rows, dtype=np.int64)
        for i in range(n_rows):
            out[i] = int(
                self.emit_primary_direction(
                    x_continuous_row=x_mat[i],
                    liquidation_exhaustion_flag=int(flags[i]),
                    price_trend_sign=float(x_mat[i, 0]),
                    signal_threshold=signal_threshold,
                )
            )
        return out


def compute_di_values(
    X: npt.ArrayLike,
    mean: npt.ArrayLike,
    precision: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute Mahalanobis distance D_M(x) = sqrt((x - mu)^T Sigma^{-1} (x - mu))."""
    x_arr = np.asarray(X, dtype=np.float64)
    single = x_arr.ndim == 1
    if single:
        x_arr = x_arr.reshape(1, -1)

    mu = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    prec = np.asarray(precision, dtype=np.float64)
    diff = x_arr - mu
    quad = np.einsum("ij,jk,ik->i", diff, prec, diff)
    di: npt.NDArray[np.float64] = np.asarray(np.sqrt(np.maximum(quad, 0.0)), dtype=np.float64)
    return di


# Alias matching architecture state terminology (`DI_values`)
DI_values = compute_di_values


class MahalanobisOutlierGate:
    """Epistemic outlier gate using Ledoit-Wolf shrinkage covariance fit strictly on training data.

    Hard-blocks directional entries (`epistemic_gate_passed = False`) whenever
    D_M(x_live) > di_threshold.
    """

    def __init__(
        self,
        di_threshold: float = 4.5,
        assume_centered: bool = False,
    ) -> None:
        if di_threshold <= 0.0:
            raise ValueError(f"di_threshold must be > 0.0, got {di_threshold}")
        self.di_threshold = float(di_threshold)
        self.assume_centered = assume_centered

        self._ledoit_wolf: LedoitWolf | None = None
        self._mean: npt.NDArray[np.float64] | None = None
        self._covariance: npt.NDArray[np.float64] | None = None
        self._precision: npt.NDArray[np.float64] | None = None
        self._shrinkage: float = 0.0
        self._train_di_values: npt.NDArray[np.float64] | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if the Ledoit-Wolf estimator has been fit on training data."""
        return self._precision is not None

    @property
    def mean_(self) -> npt.NDArray[np.float64]:
        """Training distribution centroid mu_train."""
        if self._mean is None:
            raise RuntimeError("MahalanobisOutlierGate is not fitted.")
        return self._mean

    @property
    def covariance_(self) -> npt.NDArray[np.float64]:
        """Ledoit-Wolf shrinkage covariance matrix Sigma_train."""
        if self._covariance is None:
            raise RuntimeError("MahalanobisOutlierGate is not fitted.")
        return self._covariance

    @property
    def precision_(self) -> npt.NDArray[np.float64]:
        """Inverse shrinkage covariance matrix Sigma_train^{-1}."""
        if self._precision is None:
            raise RuntimeError("MahalanobisOutlierGate is not fitted.")
        return self._precision

    @property
    def shrinkage_(self) -> float:
        """Fitted Ledoit-Wolf shrinkage coefficient in [0, 1]."""
        return self._shrinkage

    @property
    def train_di_values_(self) -> npt.NDArray[np.float64]:
        """In-sample Mahalanobis distances D_M(x_train) on the training split."""
        if self._train_di_values is None:
            raise RuntimeError("MahalanobisOutlierGate is not fitted.")
        return self._train_di_values

    def fit(self, X_train: npt.ArrayLike) -> MahalanobisOutlierGate:
        """Fit Ledoit-Wolf shrinkage covariance strictly on `X_train`."""
        x_arr = np.asarray(X_train, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.ndim != 2 or x_arr.shape[0] < 2:
            raise ValueError(
                f"Expected 2D training array with at least 2 samples, got shape {x_arr.shape}"
            )
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("X_train contains NaN or infinite values.")

        lw = LedoitWolf(store_precision=True, assume_centered=self.assume_centered)
        lw.fit(x_arr)

        mean = np.asarray(lw.location_, dtype=np.float64)
        cov = np.asarray(lw.covariance_, dtype=np.float64)
        prec = np.asarray(lw.precision_, dtype=np.float64)

        # Ensure strictly positive-definite precision fallback if all features were constant
        if not np.all(np.isfinite(prec)) or np.allclose(prec, 0.0):
            prec = np.eye(x_arr.shape[1], dtype=np.float64)

        self._ledoit_wolf = lw
        self._mean = mean
        self._covariance = cov
        self._precision = prec
        self._shrinkage = float(getattr(lw, "shrinkage_", 0.0))
        self._train_di_values = compute_di_values(x_arr, mean, prec)
        return self

    def compute_di(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute Mahalanobis distance D_M(x) for each row in `X`."""
        if self._mean is None or self._precision is None:
            raise RuntimeError("MahalanobisOutlierGate must be fitted before compute_di().")
        return compute_di_values(X, self._mean, self._precision)

    def score_sample(self, x_live: npt.ArrayLike) -> tuple[float, bool]:
        """Return `(mahalanobis_di, epistemic_gate_passed)` for a single live feature vector."""
        di_arr = self.compute_di(np.asarray(x_live, dtype=np.float64).reshape(1, -1))
        d_m = float(di_arr[0])
        passed = bool(d_m <= self.di_threshold)
        return d_m, passed

    def passes_gate(self, X: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Return boolean mask `D_M(X) <= di_threshold`."""
        di_arr = self.compute_di(X)
        return np.asarray(di_arr <= self.di_threshold, dtype=np.bool_)


class LightGBMMetaLabelWorker:
    """Two-Stage Meta-Labelling Worker combining Stage 1 SVD, Stage 2 Calibrated LightGBM, and Mahalanobis Gate.

    Enforces `n_jobs=4` on LightGBM (and optional XGBoost ensemble) for Apple M1 Max
    P-core isolation.
    """

    def __init__(
        self,
        n_svd_components: int | None = None,
        di_threshold: float = 4.5,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        use_xgboost_ensemble: bool = False,
        xgb_ensemble_weight: float = 0.35,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 5,
        n_jobs: int = 4,
        random_state: int = 42,
        fee_schedule: FeeSchedule | None = None,
    ) -> None:
        if n_jobs < 1 or n_jobs > 4:
            raise ValueError(f"n_jobs must be in [1, 4] on M1 Max worker pool, got {n_jobs}")
        if not (0.0 <= xgb_ensemble_weight <= 1.0):
            raise ValueError(f"xgb_ensemble_weight must be in [0, 1], got {xgb_ensemble_weight}")

        self.n_svd_components = n_svd_components
        self.di_threshold = float(di_threshold)
        self.calibration_method = calibration_method
        self.use_xgboost_ensemble = use_xgboost_ensemble
        self.xgb_ensemble_weight = float(xgb_ensemble_weight)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.num_leaves = int(num_leaves)
        self.min_child_samples = int(min_child_samples)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.fee_schedule = fee_schedule or FeeSchedule.usdt_perp()

        self.svd_orthogonalizer = ContinuousSVDOrthogonalizer(
            n_components=n_svd_components,
            whiten=False,
        )
        self.outlier_gate = MahalanobisOutlierGate(di_threshold=di_threshold)

        self._lgbm_model: Any = None
        self._xgb_model: Any = None
        self._constant_prob: float | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return True if Stage 1 SVD, Mahalanobis Gate, and Stage 2 Meta-Model are fitted."""
        return self._is_fitted

    @staticmethod
    def _assemble_stage2_features(
        svd_factors: npt.NDArray[np.float64],
        liquidation_exhaustion_flag: npt.ArrayLike,
        sigma_bar: npt.ArrayLike,
        vpin: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Concatenate `[svd_factors, liquidation_exhaustion_flag, sigma_bar, vpin]`."""
        f_mat = np.asarray(svd_factors, dtype=np.float64)
        if f_mat.ndim == 1:
            f_mat = f_mat.reshape(1, -1)
        n_rows = f_mat.shape[0]

        flag_arr = np.broadcast_to(
            np.asarray(liquidation_exhaustion_flag, dtype=np.float64).reshape(-1),
            (n_rows,),
        )
        sigma_arr = np.broadcast_to(
            np.asarray(sigma_bar, dtype=np.float64).reshape(-1),
            (n_rows,),
        )
        vpin_arr = np.broadcast_to(
            np.asarray(vpin, dtype=np.float64).reshape(-1),
            (n_rows,),
        )

        if np.any((flag_arr != 0.0) & (flag_arr != 1.0)):
            raise ValueError("liquidation_exhaustion_flag entries must be binary in {0, 1}.")

        return np.column_stack([f_mat, flag_arr, sigma_arr, vpin_arr])

    def _build_base_lgbm(self, n_samples: int) -> Any:
        eff_min_child = max(2, min(self.min_child_samples, max(2, n_samples // 4)))
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=eff_min_child,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=-1,
            importance_type="gain",
        )

    def _build_base_xgb(self) -> Any:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )

    def fit(
        self,
        X_continuous_train: npt.ArrayLike,
        y_meta_train: Sequence[int] | npt.NDArray[np.int_],
        liquidation_exhaustion_flag_train: Sequence[int] | npt.NDArray[np.int_] | None = None,
        sigma_bar_train: Sequence[float] | npt.NDArray[np.float64] | None = None,
        vpin_train: Sequence[float] | npt.NDArray[np.float64] | None = None,
        feature_names: Sequence[str] | None = None,
        sample_weight: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> LightGBMMetaLabelWorker:
        """Fit Stage 1 SVD, Mahalanobis Epistemic Gate, and Stage 2 Calibrated LightGBM strictly on train split.

        ``sample_weight`` (optional, e.g. label uniqueness or recency) weights the Stage 2 classifier and its
        calibration; the SVD and the outlier gate are fitted unweighted.
        """
        # 1. Fit Stage 1 Continuous SVD Orthogonalizer (excluding binary liquidation flag)
        svd_train = self.svd_orthogonalizer.fit_transform(
            X_continuous_train,
            feature_names=feature_names,
        )
        n_samples = svd_train.shape[0]

        y_arr = np.asarray(y_meta_train, dtype=np.int64).reshape(-1)
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"Length mismatch: X_continuous_train has {n_samples} rows, y_meta_train has {y_arr.shape[0]}"
            )
        if np.any((y_arr != 0) & (y_arr != 1)):
            raise ValueError("y_meta_train labels must be binary in {0, 1}.")
        sw: npt.NDArray[np.float64] | None = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if sw.shape[0] != n_samples or np.any(~np.isfinite(sw)) or np.any(sw < 0.0):
                raise ValueError("sample_weight must be finite, non-negative and one per training row")

        # 2. Fit Mahalanobis Epistemic Outlier Gate on training continuous SVD factors
        self.outlier_gate.di_threshold = self.di_threshold
        self.outlier_gate.fit(svd_train)

        # 3. Assemble Stage 2 feature matrix
        flags = (
            np.zeros(n_samples, dtype=np.float64)
            if liquidation_exhaustion_flag_train is None
            else np.asarray(liquidation_exhaustion_flag_train, dtype=np.float64)
        )
        sigmas = (
            np.full(n_samples, 0.01, dtype=np.float64)
            if sigma_bar_train is None
            else np.asarray(sigma_bar_train, dtype=np.float64)
        )
        vpins = (
            np.full(n_samples, 0.25, dtype=np.float64)
            if vpin_train is None
            else np.asarray(vpin_train, dtype=np.float64)
        )

        x_stage2 = self._assemble_stage2_features(
            svd_factors=svd_train,
            liquidation_exhaustion_flag=flags,
            sigma_bar=sigmas,
            vpin=vpins,
        )

        # 4. Fit calibrated LightGBM (and optional calibrated XGBoost ensemble)
        unique_classes, class_counts = np.unique(y_arr, return_counts=True)
        if unique_classes.shape[0] < 2:
            # Degenerate single-class training split fallback
            only_cls = int(unique_classes[0])
            self._constant_prob = 0.95 if only_cls == 1 else 0.05
            self._lgbm_model = None
            self._xgb_model = None
            self._is_fitted = True
            return self

        self._constant_prob = None
        min_class_count = int(np.min(class_counts))
        base_lgbm = self._build_base_lgbm(n_samples=n_samples)

        if min_class_count >= 3:
            cv_folds = min(3, min_class_count)
            # Isotonic requires sufficient samples per fold to prevent step-function NaNs
            eff_method: Literal["sigmoid", "isotonic"] = (
                "isotonic"
                if (self.calibration_method == "isotonic" and min_class_count >= 15)
                else "sigmoid"
            )
            cal_lgbm = CalibratedClassifierCV(
                estimator=base_lgbm,
                method=eff_method,
                cv=cv_folds,
            )
            cal_lgbm.fit(x_stage2, y_arr, sample_weight=sw)
            self._lgbm_model = cal_lgbm
        else:
            base_lgbm.fit(x_stage2, y_arr, sample_weight=sw)
            self._lgbm_model = base_lgbm

        if self.use_xgboost_ensemble:
            base_xgb = self._build_base_xgb()
            if min_class_count >= 3:
                cv_folds = min(3, min_class_count)
                eff_method_xgb: Literal["sigmoid", "isotonic"] = (
                    "isotonic"
                    if (self.calibration_method == "isotonic" and min_class_count >= 15)
                    else "sigmoid"
                )
                cal_xgb = CalibratedClassifierCV(
                    estimator=base_xgb,
                    method=eff_method_xgb,
                    cv=cv_folds,
                )
                cal_xgb.fit(x_stage2, y_arr)
                self._xgb_model = cal_xgb
            else:
                base_xgb.fit(x_stage2, y_arr)
                self._xgb_model = base_xgb
        else:
            self._xgb_model = None

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X_continuous: npt.ArrayLike,
        liquidation_exhaustion_flag: Sequence[int] | npt.NDArray[np.int_] | int = 0,
        sigma_bar: Sequence[float] | npt.NDArray[np.float64] | float = 0.01,
        vpin: Sequence[float] | npt.NDArray[np.float64] | float = 0.25,
    ) -> npt.NDArray[np.float64]:
        """Predict calibrated meta-probabilities p_meta in [0, 1] for each sample."""
        if not self._is_fitted:
            raise RuntimeError("LightGBMMetaLabelWorker must be fitted before predict_proba().")

        svd_factors = self.svd_orthogonalizer.transform(X_continuous)
        if svd_factors.ndim == 1:
            svd_factors = svd_factors.reshape(1, -1)

        x_stage2 = self._assemble_stage2_features(
            svd_factors=svd_factors,
            liquidation_exhaustion_flag=liquidation_exhaustion_flag,
            sigma_bar=sigma_bar,
            vpin=vpin,
        )

        if self._constant_prob is not None or self._lgbm_model is None:
            return np.full(x_stage2.shape[0], self._constant_prob or 0.5, dtype=np.float64)

        probs_lgbm = np.asarray(self._lgbm_model.predict_proba(x_stage2)[:, 1], dtype=np.float64)

        if self.use_xgboost_ensemble and self._xgb_model is not None:
            probs_xgb = np.asarray(self._xgb_model.predict_proba(x_stage2)[:, 1], dtype=np.float64)
            w = self.xgb_ensemble_weight
            probs = (1.0 - w) * probs_lgbm + w * probs_xgb
        else:
            probs = probs_lgbm

        return np.clip(probs, 0.0, 1.0)

    def predict_single(
        self,
        instrument_id: str,
        ts_close_ns: int,
        x_continuous_row: npt.ArrayLike,
        liquidation_exhaustion_flag: int = 0,
        primary_direction: Literal[-1, 0, 1] | None = None,
        sigma_bar: float = 0.01,
        vpin: float = 0.25,
        target_profit_bps: float = 120.0,
        stop_loss_bps: float = 80.0,
    ) -> MetaPredictionOutput:
        """Run full Stage 1 SVD + Mahalanobis Gate + Stage 2 LightGBM inference for a single bar."""
        if not self._is_fitted:
            raise RuntimeError("LightGBMMetaLabelWorker must be fitted before predict_single().")
        if liquidation_exhaustion_flag not in (0, 1):
            raise ValueError(
                f"liquidation_exhaustion_flag must be in {{0, 1}}, got {liquidation_exhaustion_flag}"
            )

        row_arr = np.asarray(x_continuous_row, dtype=np.float64).reshape(-1)
        svd_row = self.svd_orthogonalizer.transform(row_arr).reshape(-1)

        if primary_direction is None:
            eff_primary = self.svd_orthogonalizer.emit_primary_direction(
                x_continuous_row=row_arr,
                liquidation_exhaustion_flag=liquidation_exhaustion_flag,
            )
        else:
            if primary_direction not in (-1, 0, 1):
                raise ValueError(f"primary_direction must be in {{-1, 0, 1}}, got {primary_direction}")
            # If liquidation_exhaustion_flag is active, apply the direct regime override
            if liquidation_exhaustion_flag == 1 and primary_direction == 0:
                eff_primary = self.svd_orthogonalizer.emit_primary_direction(
                    x_continuous_row=row_arr,
                    liquidation_exhaustion_flag=1,
                )
            else:
                eff_primary = primary_direction

        d_m, gate_passed = self.outlier_gate.score_sample(svd_row)

        p_meta_arr = self.predict_proba(
            X_continuous=row_arr.reshape(1, -1),
            liquidation_exhaustion_flag=liquidation_exhaustion_flag,
            sigma_bar=sigma_bar,
            vpin=vpin,
        )
        p_meta = float(p_meta_arr[0])

        fee_bps = self.fee_schedule.roundtrip_maker_plus_slippage_bps
        raw_expected_edge_bps = (
            p_meta * float(target_profit_bps)
            - (1.0 - p_meta) * float(stop_loss_bps)
            - fee_bps
        )
        expected_net_edge_bps = (
            float(raw_expected_edge_bps) if (gate_passed and eff_primary != 0) else 0.0
        )

        return MetaPredictionOutput(
            instrument_id=instrument_id,
            ts_close_ns=ts_close_ns,
            primary_direction=eff_primary,
            p_meta=p_meta,
            mahalanobis_di=d_m,
            di_threshold=self.outlier_gate.di_threshold,
            epistemic_gate_passed=gate_passed,
            svd_factors=tuple(float(v) for v in svd_row),
            liquidation_exhaustion_flag=int(liquidation_exhaustion_flag),
            expected_net_edge_bps=expected_net_edge_bps,
        )

    def predict_from_snapshot(
        self,
        snapshot: PositioningSignalSnapshot,
        sigma_bar: float = 0.01,
        vpin: float = 0.25,
        extended: bool = False,
    ) -> MetaPredictionOutput:
        """Run full two-stage inference directly from a `PositioningSignalSnapshot`."""
        x_cont, flags, primary_dirs = extract_continuous_and_binary_from_snapshots(
            [snapshot], extended=extended
        )
        raw_dir = int(primary_dirs[0])
        typed_dir: Literal[-1, 0, 1] = -1 if raw_dir == -1 else (1 if raw_dir == 1 else 0)
        return self.predict_single(
            instrument_id=snapshot.instrument_id,
            ts_close_ns=snapshot.ts_close_ns,
            x_continuous_row=x_cont[0],
            liquidation_exhaustion_flag=int(flags[0]),
            primary_direction=typed_dir,
            sigma_bar=sigma_bar,
            vpin=vpin,
        )

    def predict_batch(
        self,
        instrument_id: str,
        ts_close_ns_list: Sequence[int],
        X_continuous: npt.ArrayLike,
        liquidation_exhaustion_flags: Sequence[int] | npt.NDArray[np.int_] | None = None,
        primary_directions: Sequence[int] | npt.NDArray[np.int_] | None = None,
        sigma_bars: Sequence[float] | npt.NDArray[np.float64] | float = 0.01,
        vpins: Sequence[float] | npt.NDArray[np.float64] | float = 0.25,
    ) -> list[MetaPredictionOutput]:
        """Run vectorized two-stage inference across a batch of bars."""
        x_mat = np.asarray(X_continuous, dtype=np.float64)
        if x_mat.ndim == 1:
            x_mat = x_mat.reshape(1, -1)
        n_rows = x_mat.shape[0]

        if len(ts_close_ns_list) != n_rows:
            raise ValueError(
                f"Length mismatch: len(ts_close_ns_list)={len(ts_close_ns_list)} != {n_rows}"
            )

        flags = (
            np.zeros(n_rows, dtype=np.int64)
            if liquidation_exhaustion_flags is None
            else np.asarray(liquidation_exhaustion_flags, dtype=np.int64).reshape(-1)
        )
        sigmas = np.broadcast_to(np.asarray(sigma_bars, dtype=np.float64).reshape(-1), (n_rows,))
        vpin_arr = np.broadcast_to(np.asarray(vpins, dtype=np.float64).reshape(-1), (n_rows,))

        outputs: list[MetaPredictionOutput] = []
        for i in range(n_rows):
            if primary_directions is None:
                p_dir: Literal[-1, 0, 1] | None = None
            else:
                d_val = int(primary_directions[i])
                p_dir = -1 if d_val == -1 else (1 if d_val == 1 else 0)

            outputs.append(
                self.predict_single(
                    instrument_id=instrument_id,
                    ts_close_ns=int(ts_close_ns_list[i]),
                    x_continuous_row=x_mat[i],
                    liquidation_exhaustion_flag=int(flags[i]),
                    primary_direction=p_dir,
                    sigma_bar=float(sigmas[i]),
                    vpin=float(vpin_arr[i]),
                )
            )
        return outputs


__all__ = [
    "CONTINUOUS_SIGNAL_COLUMNS",
    "EXCLUDED_BINARY_COLUMNS",
    "EXTENDED_CONTINUOUS_COLUMNS",
    "ContinuousSVDOrthogonalizer",
    "DI_values",
    "LightGBMMetaLabelWorker",
    "MahalanobisOutlierGate",
    "compute_di_values",
    "extract_continuous_and_binary_from_snapshots",
]
