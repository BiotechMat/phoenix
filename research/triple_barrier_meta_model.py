"""Triple-Barrier Meta-Model Bridge for Project Phoenix.

Integrates:
1. Causal volatility estimation and fee-adjusted Triple-Barrier labeling (core/models/triple_barrier.py).
2. Continuous SVD orthogonalization and noise filtering (ContinuousSVDOrthogonalizer).
3. Ledoit-Wolf Mahalanobis epistemic outlier gating (MahalanobisOutlierGate).
4. Calibrated LightGBM (and optional XGBoost) meta-modeling (LightGBMMetaLabelWorker).
5. Bridge to 45-split CPCV StrategyTournament (research/cpcv_gates.py).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from core.contracts import (
    DollarBar,
    FeeSchedule,
    PositioningRegime,
    PositioningSignalSnapshot,
    TripleBarrierConfig,
    TripleBarrierLabel,
    compute_fractional_kelly,
    compute_signed_target_weights,
)
from core.features import (
    FrozenFFDTransformer,
    apply_ffd_weights_numba,
    calibrate_optimal_d,
    classify_positioning_regime,
    compute_ffd_weights,
    compute_positioning_snapshots,
)
from core.models import (
    CONTINUOUS_SIGNAL_COLUMNS,
    EXCLUDED_BINARY_COLUMNS,
    EXTENDED_CONTINUOUS_COLUMNS,
    ContinuousSVDOrthogonalizer,
    FeeAdjustedTripleBarrierLabeler,
    LightGBMMetaLabelWorker,
    MahalanobisOutlierGate,
    compute_causal_bar_volatility,
    compute_effective_barriers,
    extract_continuous_and_binary_from_snapshots,
    label_single_bar_triple_barrier,
    label_triple_barrier_series,
)
from research.cpcv_gates import (
    CPCVSplit,
    CombinatorialPurgedCV,
    StrategyCandidate,
    StrategyTournament,
    TournamentResult,
    compute_net_of_fee_returns,
)

logger = logging.getLogger("phoenix.research.triple_barrier_meta_model")

__all__ = [
    "TripleBarrierDataset",
    "TripleBarrierMetaModelBridge",
    "build_triple_barrier_dataset",
    "create_meta_model_tournament_candidate",
]


@dataclass(frozen=True)
class TripleBarrierDataset:
    """Aligned dataset container coupling DollarBars, labels, and orthogonalized features."""

    bars: list[DollarBar]
    volatilities: npt.NDArray[np.float64]
    primary_directions: npt.NDArray[np.int64]
    labels: list[TripleBarrierLabel]
    continuous_features: npt.NDArray[np.float64]
    feature_names: tuple[str, ...]
    exhaustion_flags: npt.NDArray[np.int64]
    target_meta_labels: npt.NDArray[np.int64]
    vpin: npt.NDArray[np.float64]
    gross_returns: npt.NDArray[np.float64]
    net_returns: npt.NDArray[np.float64]

    @property
    def n_samples(self) -> int:
        """Total number of aligned bars in dataset."""
        return len(self.bars)


def _compute_simple_vpin(bars: Sequence[DollarBar], window: int = 20) -> npt.NDArray[np.float64]:
    """Estimate volume-synchronized probability of toxicity (VPIN) proxy from buy/sell dollar volumes."""
    n = len(bars)
    vpin = np.full(n, 0.25, dtype=np.float64)
    if n < window:
        return vpin

    buy_vols = np.array([b.buy_dollar_volume for b in bars], dtype=np.float64)
    sell_vols = np.array([b.sell_dollar_volume for b in bars], dtype=np.float64)
    tot_vols = np.maximum(buy_vols + sell_vols, 1.0)

    # Imbalance absolute difference |V_buy - V_sell| / V_total
    imbalances = np.abs(buy_vols - sell_vols) / tot_vols

    # Rolling average over trailing window
    for t in range(window - 1, n):
        vpin[t] = float(np.mean(imbalances[t - window + 1 : t + 1]))

    return np.clip(vpin, 0.05, 0.95)


def build_triple_barrier_dataset(
    bars: Sequence[DollarBar],
    primary_signals: Sequence[int] | npt.NDArray[np.int_] | None = None,
    barrier_config: TripleBarrierConfig | None = None,
    extended_features: bool = True,
    vol_window: int = 20,
) -> TripleBarrierDataset:
    """Construct an end-to-end dataset with causal volatility, triple-barrier labels, and features.

    Parameters
    ----------
    bars:
        Input sequence of medium-frequency DollarBar instances.
    primary_signals:
        Optional pre-computed primary directional signals y_t in {-1, 0, +1}.
        If None, generates an EMA trend momentum primary signal.
    barrier_config:
        Triple barrier parameter configuration (defaults to Hyperliquid fee schedule).
    extended_features:
        If True, extracts 7 continuous features instead of 3.
    vol_window:
        Window size for causal realized volatility estimation.
    """
    bar_list = list(bars)
    n = len(bar_list)
    if n == 0:
        raise ValueError("Cannot construct TripleBarrierDataset from empty bars sequence")

    cfg = barrier_config or TripleBarrierConfig(
        fee_schedule=FeeSchedule.hyperliquid(),
        vertical_barrier_bars=16,
    )

    # 1. Strictly causal bar volatility sigma_bar
    volatilities = compute_causal_bar_volatility(bar_list, window=vol_window)

    # 2. Primary directional signals y_primary in {-1, 0, +1}
    if primary_signals is not None:
        p_arr = np.asarray(primary_signals, dtype=np.int64).ravel()
        if p_arr.shape[0] != n:
            raise ValueError(f"primary_signals length ({p_arr.shape[0]}) does not match bars count ({n})")
        primary_dirs = p_arr
    else:
        # Default causal trend momentum primary signal
        closes = np.array([b.close for b in bar_list], dtype=np.float64)
        primary_dirs = np.zeros(n, dtype=np.int64)
        if n >= 24:
            alpha_fast = 2.0 / 9.0
            alpha_slow = 2.0 / 25.0
            ema_f = closes[0]
            ema_s = closes[0]
            for t in range(n):
                ema_f = alpha_fast * closes[t] + (1.0 - alpha_fast) * ema_f
                ema_s = alpha_slow * closes[t] + (1.0 - alpha_slow) * ema_s
                if t >= 24:
                    spread = (ema_f - ema_s) / ema_s
                    if spread > 0.0004:
                        primary_dirs[t] = 1
                    elif spread < -0.0004:
                        primary_dirs[t] = -1
                    else:
                        primary_dirs[t] = 0

    # 3. Triple-Barrier Labeling
    # label_single_bar_triple_barrier takes Literal[-1, 0, 1]
    labels: list[TripleBarrierLabel] = []
    for t in range(n):
        dir_t = int(primary_dirs[t])
        lit_dir: Literal[-1, 0, 1] = 1 if dir_t > 0 else (-1 if dir_t < 0 else 0)
        lbl = label_single_bar_triple_barrier(
            bars=bar_list,
            bar_index=t,
            primary_direction=lit_dir,
            sigma_bar=float(volatilities[t]),
            config=cfg,
        )
        labels.append(lbl)

    target_meta_labels = np.array([lbl.meta_label for lbl in labels], dtype=np.int64)
    gross_returns = np.array([lbl.gross_return for lbl in labels], dtype=np.float64)
    net_returns = np.array([lbl.net_return for lbl in labels], dtype=np.float64)

    # 4. Stationary feature extraction
    closes = np.array([b.close for b in bar_list], dtype=np.float64)
    open_interests = np.array([b.open_interest for b in bar_list], dtype=np.float64)
    fundings = np.array([b.funding_rate_8h for b in bar_list], dtype=np.float64)
    liquidations = np.array([b.liquidation_volume_usd for b in bar_list], dtype=np.float64)
    vpin = _compute_simple_vpin(bar_list)

    # Compute continuous features
    log_closes = np.log(np.maximum(closes, 1.0))
    log_oi = np.log(np.maximum(open_interests, 1.0))

    # Rolling difference / z-scores
    ret_1 = np.zeros(n, dtype=np.float64)
    oi_ret_1 = np.zeros(n, dtype=np.float64)
    if n > 1:
        ret_1[1:] = np.diff(log_closes)
        oi_ret_1[1:] = np.diff(log_oi)

    # Rolling 20-bar z-scores
    z_close = np.zeros(n, dtype=np.float64)
    z_oi = np.zeros(n, dtype=np.float64)
    z_basis = np.zeros(n, dtype=np.float64)
    z_liq = np.zeros(n, dtype=np.float64)
    basis_pct = np.array([b.basis for b in bar_list], dtype=np.float64)

    for t in range(n):
        w_start = max(0, t - 19)
        slice_ret = ret_1[w_start : t + 1]
        slice_oi = oi_ret_1[w_start : t + 1]
        slice_liq = liquidations[w_start : t + 1]

        s_r = float(np.std(slice_ret, ddof=1)) if slice_ret.size > 2 else 0.01
        s_oi = float(np.std(slice_oi, ddof=1)) if slice_oi.size > 2 else 0.01
        s_liq = float(np.std(slice_liq, ddof=1)) if slice_liq.size > 2 else 1000.0

        z_close[t] = float((ret_1[t] - np.mean(slice_ret)) / max(s_r, 1e-6))
        z_oi[t] = float((oi_ret_1[t] - np.mean(slice_oi)) / max(s_oi, 1e-6))
        z_basis[t] = float(fundings[t] * 1e4)  # funding in basis points as basis z-score proxy
        z_liq[t] = float((liquidations[t] - np.mean(slice_liq)) / max(s_liq, 1e-6))

    # Sparse binary 99th percentile liquidation exhaustion flag
    liq_99 = float(np.percentile(liquidations, 99.0)) if n >= 50 else float(np.max(liquidations))
    exhaustion_flags = np.array([1 if liq >= liq_99 and liq > 1e4 else 0 for liq in liquidations], dtype=np.int64)

    if extended_features:
        feature_names = EXTENDED_CONTINUOUS_COLUMNS
        cont_features = np.column_stack([
            z_close,
            z_oi,
            z_basis,
            basis_pct,
            fundings,
            oi_ret_1,
            z_liq,
        ])
    else:
        feature_names = CONTINUOUS_SIGNAL_COLUMNS
        cont_features = np.column_stack([
            z_close,
            z_oi,
            z_basis,
        ])

    return TripleBarrierDataset(
        bars=bar_list,
        volatilities=volatilities,
        primary_directions=primary_dirs,
        labels=labels,
        continuous_features=cont_features,
        feature_names=feature_names,
        exhaustion_flags=exhaustion_flags,
        target_meta_labels=target_meta_labels,
        vpin=vpin,
        gross_returns=gross_returns,
        net_returns=net_returns,
    )


class TripleBarrierMetaModelBridge:
    """Unified Research Bridge connecting Triple-Barrier labeling, SVD noise filtering, and LightGBM meta-modeling.

    Pipeline Stages:
    1. Labeling: Fee-adjusted triple-barrier labeling with minimum profit hurdles and 8h funding accrual.
    2. Feature Preprocessing: Thin SVD decomposition on continuous features (V_j dominant sign > 0)
       excluding sparse binary liquidation spikes.
    3. Epistemic Guardrail: Ledoit-Wolf shrinkage Mahalanobis distance D_M(x) blocking trades in outlier regimes.
    4. Meta-Classification: Calibrated LightGBM predicting meta-probability p_meta in [0, 1].
    5. Bet Sizing: Fractional Kelly betting conditioned on p_meta and epistemic pass status.
    6. Tournament Interface: Evaluates candidates across the 45-split CPCV gate without lookahead bias.
    """

    def __init__(
        self,
        barrier_config: TripleBarrierConfig | None = None,
        fee_schedule: FeeSchedule | None = None,
        n_svd_components: int | None = None,
        di_threshold: float = 4.5,
        calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
        use_xgboost_ensemble: bool = False,
        p_meta_threshold: float = 0.52,
        kelly_fraction: float = 0.50,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.barrier_config = barrier_config or TripleBarrierConfig(
            fee_schedule=fee_schedule or FeeSchedule.hyperliquid(),
            vertical_barrier_bars=16,
        )
        self.p_meta_threshold = float(p_meta_threshold)
        self.kelly_fraction = float(kelly_fraction)

        self.worker = LightGBMMetaLabelWorker(
            n_svd_components=n_svd_components,
            di_threshold=di_threshold,
            calibration_method=calibration_method,
            use_xgboost_ensemble=use_xgboost_ensemble,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            fee_schedule=self.barrier_config.fee_schedule,
        )
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return True if the underlying SVD, Mahalanobis gate, and LightGBM models are fitted."""
        return self._is_fitted

    def fit(
        self,
        dataset: TripleBarrierDataset,
        train_indices: npt.NDArray[np.int64] | None = None,
    ) -> TripleBarrierMetaModelBridge:
        """Fit Stage 1 SVD, Mahalanobis gate, and Stage 2 LightGBM meta-model.

        Enforces zero lookahead when train_indices is supplied (CPCV splitting).
        """
        if train_indices is not None:
            idx = np.asarray(train_indices, dtype=np.int64).ravel()
            x_cont = dataset.continuous_features[idx]
            y_meta = dataset.target_meta_labels[idx]
            flags = dataset.exhaustion_flags[idx]
            sigmas = dataset.volatilities[idx]
            vpin_arr = dataset.vpin[idx]
        else:
            x_cont = dataset.continuous_features
            y_meta = dataset.target_meta_labels
            flags = dataset.exhaustion_flags
            sigmas = dataset.volatilities
            vpin_arr = dataset.vpin

        self.worker.fit(
            X_continuous_train=x_cont,
            y_meta_train=y_meta,
            liquidation_exhaustion_flag_train=flags,
            sigma_bar_train=sigmas,
            vpin_train=vpin_arr,
            feature_names=dataset.feature_names,
        )
        self._is_fitted = True
        return self

    def predict_meta_probabilities(
        self,
        continuous_features: npt.ArrayLike,
        exhaustion_flags: npt.ArrayLike | None = None,
        sigmas: npt.ArrayLike | None = None,
        vpins: npt.ArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Predict calibrated meta-probabilities and evaluate the Mahalanobis outlier gate.

        Returns
        -------
        p_meta:
            Calibrated meta-probabilities p_meta in [0, 1].
        gate_passed:
            Boolean array indicating whether each sample passes the Mahalanobis epistemic gate.
        """
        if not self._is_fitted:
            raise RuntimeError("TripleBarrierMetaModelBridge must be fitted before predict_meta_probabilities().")

        x_cont = np.asarray(continuous_features, dtype=np.float64)
        n_samples = x_cont.shape[0] if x_cont.ndim > 1 else 1

        flags = (
            np.asarray(exhaustion_flags, dtype=np.int64)
            if exhaustion_flags is not None
            else np.zeros(n_samples, dtype=np.int64)
        )
        sig_arr = (
            np.asarray(sigmas, dtype=np.float64)
            if sigmas is not None
            else np.full(n_samples, 0.01, dtype=np.float64)
        )
        v_arr = (
            np.asarray(vpins, dtype=np.float64)
            if vpins is not None
            else np.full(n_samples, 0.25, dtype=np.float64)
        )

        p_meta = self.worker.predict_proba(
            X_continuous=x_cont,
            liquidation_exhaustion_flag=flags,
            sigma_bar=sig_arr,
            vpin=v_arr,
        )

        # Mahalanobis epistemic gate on SVD factors
        svd_factors = self.worker.svd_orthogonalizer.transform(x_cont)
        gate_passed = self.worker.outlier_gate.passes_gate(svd_factors)

        return p_meta, gate_passed

    def predict_positions(
        self,
        bars: Sequence[DollarBar],
        primary_signals: Sequence[int] | npt.NDArray[np.int_] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Generate final execution positions pos_t in [-1, 1] conditioned on meta-probabilities.

        Decision Rule:
        - If primary_direction == 0: pos = 0.0.
        - If Mahalanobis outlier gate rejects the sample: pos = 0.0 (regime anomaly).
        - If p_meta < p_meta_threshold: pos = 0.0 (unfavorable bet, meta-model blocks trade).
        - Otherwise: pos = primary_direction * clamp(2 * (p_meta - 0.5), 0.1, 1.0) * (2 * kelly_fraction).
        """
        dataset = build_triple_barrier_dataset(
            bars=bars,
            primary_signals=primary_signals,
            barrier_config=self.barrier_config,
            extended_features=(
                self.worker.svd_orthogonalizer._feature_names is None
                or len(self.worker.svd_orthogonalizer._feature_names) == len(EXTENDED_CONTINUOUS_COLUMNS)
            ),
        )

        if not self._is_fitted:
            # Auto-fit if not already fitted
            self.fit(dataset)

        p_meta, gate_passed = self.predict_meta_probabilities(
            continuous_features=dataset.continuous_features,
            exhaustion_flags=dataset.exhaustion_flags,
            sigmas=dataset.volatilities,
            vpins=dataset.vpin,
        )

        n = dataset.n_samples
        positions = np.zeros(n, dtype=np.float64)

        for t in range(n):
            dir_t = int(dataset.primary_directions[t])
            if dir_t == 0:
                continue

            # Epistemic outlier gate check
            if not gate_passed[t]:
                continue

            # Meta-model hurdle check
            prob = float(p_meta[t])
            if prob < self.p_meta_threshold:
                continue

            # Scale sizing by probability confidence and Fractional Kelly
            edge = max(0.1, min(1.0, 2.0 * (prob - 0.5)))
            size = edge * (self.kelly_fraction * 2.0)
            positions[t] = float(np.clip(dir_t * size, -1.0, 1.0))

        return positions

    def evaluate_cpcv_split(
        self,
        split: CPCVSplit,
        dataset: TripleBarrierDataset,
    ) -> tuple[float, float, npt.NDArray[np.float64]]:
        """Evaluate a single CPCV split strictly enforcing zero lookahead leakage.

        1. Fits SVD, Mahalanobis gate, and LightGBM strictly on split.train_indices.
        2. Generates out-of-sample positions on split.test_indices.
        3. Computes net-of-fee returns and returns (is_sharpe, oos_sharpe, test_returns).
        """
        # Create fresh clone for the split
        split_bridge = TripleBarrierMetaModelBridge(
            barrier_config=self.barrier_config,
            n_svd_components=self.worker.n_svd_components,
            di_threshold=self.worker.di_threshold,
            calibration_method=self.worker.calibration_method,
            use_xgboost_ensemble=self.worker.use_xgboost_ensemble,
            p_meta_threshold=self.p_meta_threshold,
            kelly_fraction=self.kelly_fraction,
            n_estimators=self.worker.n_estimators,
            learning_rate=self.worker.learning_rate,
            random_state=self.worker.random_state,
        )
        split_bridge.fit(dataset, train_indices=split.train_indices)

        # IS Evaluation
        p_is, gate_is = split_bridge.predict_meta_probabilities(
            continuous_features=dataset.continuous_features[split.train_indices],
            exhaustion_flags=dataset.exhaustion_flags[split.train_indices],
            sigmas=dataset.volatilities[split.train_indices],
            vpins=dataset.vpin[split.train_indices],
        )
        is_pos = np.zeros(len(split.train_indices), dtype=np.float64)
        for i, t_idx in enumerate(split.train_indices):
            dir_t = int(dataset.primary_directions[t_idx])
            if dir_t != 0 and gate_is[i] and p_is[i] >= self.p_meta_threshold:
                edge = max(0.1, min(1.0, 2.0 * (p_is[i] - 0.5)))
                is_pos[i] = dir_t * edge * (self.kelly_fraction * 2.0)

        # OOS Evaluation
        p_oos, gate_oos = split_bridge.predict_meta_probabilities(
            continuous_features=dataset.continuous_features[split.test_indices],
            exhaustion_flags=dataset.exhaustion_flags[split.test_indices],
            sigmas=dataset.volatilities[split.test_indices],
            vpins=dataset.vpin[split.test_indices],
        )
        oos_pos = np.zeros(len(split.test_indices), dtype=np.float64)
        for j, te_idx in enumerate(split.test_indices):
            dir_te = int(dataset.primary_directions[te_idx])
            if dir_te != 0 and gate_oos[j] and p_oos[j] >= self.p_meta_threshold:
                edge = max(0.1, min(1.0, 2.0 * (p_oos[j] - 0.5)))
                oos_pos[j] = dir_te * edge * (self.kelly_fraction * 2.0)

        # Compute net returns
        closes = np.array([b.close for b in dataset.bars], dtype=np.float64)
        r_all = np.zeros(len(closes), dtype=np.float64)
        if len(closes) > 1:
            r_all[1:] = np.diff(closes) / closes[:-1]

        r_is = r_all[split.train_indices]
        r_oos = r_all[split.test_indices]
        fund_oos = np.array([dataset.bars[i].funding_rate_8h for i in split.test_indices], dtype=np.float64)

        bd_is = compute_net_of_fee_returns(
            gross_asset_returns=r_is,
            positions=is_pos,
            fee_schedule=self.barrier_config.fee_schedule,
        )
        bd_oos = compute_net_of_fee_returns(
            gross_asset_returns=r_oos,
            positions=oos_pos,
            funding_rates_8h=fund_oos,
            fee_schedule=self.barrier_config.fee_schedule,
        )

        std_is = float(np.std(bd_is.net_returns, ddof=1)) if len(bd_is.net_returns) > 2 else 0.0
        std_oos = float(np.std(bd_oos.net_returns, ddof=1)) if len(bd_oos.net_returns) > 2 else 0.0

        is_sr = float(np.mean(bd_is.net_returns) / std_is) if std_is > 1e-12 else 0.0
        oos_sr = float(np.mean(bd_oos.net_returns) / std_oos) if std_oos > 1e-12 else 0.0

        return is_sr, oos_sr, bd_oos.net_returns


def create_meta_model_tournament_candidate(
    strategy_id: str = "svd_lgbm_triple_barrier_meta",
    bridge: TripleBarrierMetaModelBridge | None = None,
    description: str | None = None,
    p_meta_threshold: float = 0.52,
    kelly_fraction: float = 0.50,
) -> StrategyCandidate:
    """Create a tournament strategy candidate backed by the TripleBarrierMetaModelBridge."""
    active_bridge = bridge or TripleBarrierMetaModelBridge(
        p_meta_threshold=p_meta_threshold,
        kelly_fraction=kelly_fraction,
    )

    def _generator(bars: Sequence[DollarBar]) -> npt.NDArray[np.float64]:
        return active_bridge.predict_positions(bars)

    desc = description or (
        f"Two-Stage SVD + Calibrated LightGBM Meta-Model "
        f"(thresh={p_meta_threshold:.2f}, kelly={kelly_fraction:.2f})"
    )

    return StrategyCandidate(
        strategy_id=strategy_id,
        strategy_type="triple_barrier_meta_model",
        description=desc,
        parameters={
            "p_meta_threshold": p_meta_threshold,
            "kelly_fraction": kelly_fraction,
            "di_threshold": active_bridge.worker.di_threshold,
        },
        position_generator=_generator,
    )
