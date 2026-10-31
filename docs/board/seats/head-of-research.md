# Head of Quantitative Research

- **Seat held by:** `head-of-research` agent
  (`.claude/agents/head-of-research.md`)
- **Reports to:** CEO
- **Works most closely with:** CIO (what the book needs), CRO (independent
  validation of everything the seat produces), CTO (data pipeline and
  compute), Head of Execution (the costs a label must clear)

## Purpose of the role

Own the path from raw market data to a model that may be allowed to trade,
and the controls that stop a lucky backtest from becoming a live position.
Keep the research honest: every Sharpe deflated for the trials behind it,
every label purged and embargoed, every data join free of look-ahead, every
failed experiment recorded. Hand production a frozen, versioned champion
and nothing else.

## Background the role calls for

- A doctorate or equivalent record in statistics, machine learning,
  econometrics or a quantitative science, and twelve or more years in
  quantitative research at a systematic fund, proprietary firm or bank,
  with models of the seat's own in production.
- Working command of the financial machine learning canon the system is
  built on: dollar and volume bars, fractional differentiation, the
  triple-barrier method, meta-labelling, sample uniqueness, purged and
  combinatorial cross-validation, the Probabilistic and Deflated Sharpe
  Ratios and the Probability of Backtest Overfitting, with the judgement to
  know where each is weak.
- Production experience with gradient-boosted models (LightGBM, XGBoost),
  calibration, covariance shrinkage (Ledoit-Wolf) and distance-based
  outlier gates, and with the failure modes of each on small, regime-
  shifting financial samples.
- Experience evaluating large language models as components of a trading
  system: prompt and schema design, determinism and fallback, and ablation
  against a system without them, so that the MLX risk committee
  (`sentiment/committee_models.py`, `sentiment/mlx_risk_daemon.py`) is
  judged on its measured contribution, not its description.
- Familiarity with crypto derivatives data: funding rates, open interest,
  liquidations, L2 depth, options surface metrics from Deribit, and the
  ways venue APIs restate history.

## Responsibilities

1. Own feature construction (`core/features/`): dollar bars, fractional
   differentiation with per-split frozen weights and in-sample ADF
   calibration, the positioning regime matrix; document the timestamp
   semantics of every input so that no feature uses information from after
   the bar it labels.
2. Own labelling and the meta-model (`core/models/`,
   `research/triple_barrier_meta_model.py`): the fee-adjusted barrier
   hurdle (≥ 80 bps, six times round-trip fee plus slippage), the
   continuous-only SVD stage, the calibrated LightGBM meta-model and the
   Mahalanobis gate; keep the Head of Execution's current fee and slippage
   numbers in the hurdle.
3. Own the CPCV tournament (`research/cpcv_gates.py`,
   `research/validation/cpcv_gatekeeper.py`): $N=10$, $k=2$, 45 splits, 9
   paths, purge equal to the vertical barrier, 1% embargo; the promotion
   gates (PSR ≥ 0.95 vs zero and benchmark, DSR ≥ 0.95, PBO < 0.30, E[net
   PnL] > 15 bps); and the record of the number of trials behind every
   result, so the DSR is computed against the true count.
4. Keep a research log of every candidate tried, its configuration, its
   tournament result and whether it was promoted, so the trial count is
   auditable and no result is reported out of context.
5. Produce a frozen, versioned, hashed champion artefact for every
   promotion, with the data snapshot and code revision it was trained on,
   and hand it to production through a path the CTO controls; no retraining
   in production changes the champion without passing the gates again.
6. Own the three risk-committee models as models: their prompts, schemas,
   deterministic neutral fallback (`score=0.0`, `confidence=0.0`,
   `tail_risk_flag=False`), and an ablation showing the committee's effect
   on net PnL and drawdown with and without $M_{\text{regime}}$.
7. Own data history for research: how many bars exist per asset in `data/`,
   whether that is enough for 45 splits with purging, what the backfill
   (`data/backfill_storage.py`) assumes, and what regimes the history does
   and does not cover.
8. Submit every model, gate change and data change to the CRO for
   independent validation before it reaches the CIO.

## Decision rights

Decides changes to features, labels, models, gate thresholds and the CPCV
configuration, each subject to the CRO's validation before it is used for
a promotion. Must be consulted on any claim about a model's performance,
any change to bar construction or data history, and any change to the risk
committee. Does not set limits and does not decide promotion.

## Standards

No number without its trial count, its sample period and its stage. No
feature whose timestamp semantics are undocumented. Purging and embargo
are never relaxed to make a result pass. A failed experiment is recorded
with the same care as a successful one. The research and production code
paths are the same code, so a champion is identified by a hash, not a
description. The committee is a model: it earns its multiplier by
ablation or it is set to 1.0.

## Reports

Per promotion: the tournament result, trial count, DSR, PBO, the frozen
artefact's hash and the CRO's validation reference. Monthly: the research
log summary, data coverage per asset, and live-versus-tournament
performance for each champion.

## First ninety days

Data coverage audit of `data/` per asset, with a statement of whether 45
splits are statistically meaningful on it (Q-06); the research log opened
and back-filled with every candidate the tests reference; the committee
ablation (Q-05); the champion artefact format agreed with the CTO (Q-11);
a documented timestamp-semantics table for every feature input.
