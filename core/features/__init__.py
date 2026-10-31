"""Medium-Frequency Derivatives Alpha & Stationary Features package."""

from __future__ import annotations

from core.features.derivatives_actor import (
    HORIZON_BARS_PER_DAY,
    DerivativesActorConfig,
    DerivativesFeatureActor,
    DynamicDollarBarAggregator,
    aggregate_dollar_bars_polars,
)
from core.features.ffd import (
    FrozenFFDTransformer,
    apply_ffd_weights_numba,
    calibrate_optimal_d,
    compute_adf_pvalue,
    compute_ffd_weights,
    compute_level_correlation,
)
from core.features.positioning import (
    ZSCORE_99TH_PERCENTILE,
    PositioningFeatureEngine,
    causal_rolling_percentile_numba,
    causal_rolling_zscore_numba,
    classify_positioning_regime,
    compute_positioning_snapshots,
)

__all__ = [
    "HORIZON_BARS_PER_DAY",
    "ZSCORE_99TH_PERCENTILE",
    "DerivativesActorConfig",
    "DerivativesFeatureActor",
    "DynamicDollarBarAggregator",
    "FrozenFFDTransformer",
    "PositioningFeatureEngine",
    "aggregate_dollar_bars_polars",
    "apply_ffd_weights_numba",
    "calibrate_optimal_d",
    "causal_rolling_percentile_numba",
    "causal_rolling_zscore_numba",
    "classify_positioning_regime",
    "compute_adf_pvalue",
    "compute_ffd_weights",
    "compute_level_correlation",
    "compute_positioning_snapshots",
]
