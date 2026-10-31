"""CPCV check of the online residual layer and abstention (REINFORCEMENT_PLAN task P5.3 acceptance).

In every CPCV split the meta-labeller is fitted on the training rows, the residual ridge on its training
residuals, and both are scored on the purged test rows: out-of-fold log-loss with and without the residual
layer, and the abstention rate (test rows whose Mahalanobis distance exceeds the 99th percentile of the
training distances).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from core.contracts import DollarBar, FeeSchedule, TripleBarrierConfig
from core.models import LightGBMMetaLabelWorker
from core.models.online_residual import OnlineResidualRidge, abstention_threshold
from research.feature_eval import _log_loss, prepare_bars
from research.validation.cpcv_gatekeeper import CombinatorialPurgedCV


class ResidualEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    folds: int
    oof_logloss_base: float
    oof_logloss_residual: float
    folds_improved: float
    abstention_rate: float
    improves: bool


def evaluate_residual_layer(bars: Sequence[DollarBar], n_groups: int = 6, k_test: int = 2, l2: float = 10.0, forgetting: float = 0.999) -> ResidualEvalResult:
    prep = prepare_bars(bars)
    n = prep.n_labelled
    x, y, flags = prep.x_base[:n], prep.labels[:n], prep.flags[:n]
    cv = CombinatorialPurgedCV(vertical_barrier_bars=TripleBarrierConfig().vertical_barrier_bars, n_groups=n_groups, k_test_groups=k_test, embargo_pct=0.01)
    base_ll: list[float] = []
    res_ll: list[float] = []
    abstain: list[float] = []
    for split in cv.generate_splits(n):
        tr = np.asarray(split.train_indices)
        te = np.asarray(split.test_indices)
        if tr.size < 50 or te.size < 10 or np.unique(y[tr]).size < 2:
            continue
        w = LightGBMMetaLabelWorker(di_threshold=3.5, n_jobs=1, fee_schedule=FeeSchedule.hyperliquid())
        w.fit(x[tr], y[tr], liquidation_exhaustion_flag_train=flags[tr])
        p_tr = np.asarray(w.predict_proba(x[tr], liquidation_exhaustion_flag=flags[tr]))
        p_te = np.asarray(w.predict_proba(x[te], liquidation_exhaustion_flag=flags[te]))
        ridge = OnlineResidualRidge(x.shape[1], l2=l2, forgetting=forgetting).fit(x[tr], y[tr] - p_tr)
        base_ll.append(_log_loss(y[te], p_te))
        res_ll.append(_log_loss(y[te], ridge.adjust(p_te, x[te])))
        train_di = w.outlier_gate.compute_di(w.svd_orthogonalizer.transform(x[tr]))
        test_di = w.outlier_gate.compute_di(w.svd_orthogonalizer.transform(x[te]))
        abstain.append(float(np.mean(test_di > abstention_threshold(train_di))))
    b = float(np.mean(base_ll)) if base_ll else float("nan")
    r = float(np.mean(res_ll)) if res_ll else float("nan")
    improved = float(np.mean([rr < bb for bb, rr in zip(base_ll, res_ll)])) if base_ll else 0.0
    return ResidualEvalResult(folds=len(base_ll), oof_logloss_base=b, oof_logloss_residual=r, folds_improved=improved, abstention_rate=float(np.mean(abstain)) if abstain else 0.0, improves=bool(r < b and improved >= 0.6))


__all__ = ["ResidualEvalResult", "evaluate_residual_layer"]
