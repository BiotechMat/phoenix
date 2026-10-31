# Phoenix Reinforcement Plan

Date: 2026-10-31
Companion to: `docs/QUANT_STRATEGY_RESEARCH.md` (evidence and sources live there; this document is the work plan).

## How to use this document

- Work is organised into six phases. Phases 0 and 1 are prerequisites; nothing in Phases 2 to 5 can be measured until they are done.
- Every task has an ID, the files it touches, concrete steps, an acceptance test, an effort estimate and its dependencies. Effort: **S** = up to 1 day, **M** = 2 to 4 days, **L** = 1 to 2 weeks, for one engineer.
- A phase is complete when its gate passes. Gates are listed at the end of each phase.
- Standing rules for every task:
  1. `pytest tests -q` and `mypy core sleeves risk sentiment research ui --strict` pass before merge.
  2. No synthetic or random data in any live code path. If a feed is unavailable, the field is `None` and the dependent feature is flagged unavailable.
  3. Every strategy or parameter change is a logged trial (task P1.6) and must pass the CPCV gate (PSR ≥ 0.95, DSR ≥ 0.95, PBO < 0.30, E[net] > 15 bps) before it reaches the live loop.
  4. New sleeves ship behind a config flag, default off, and run in shadow mode first (task P5.1).

## Timeline at a glance

| Phase | Weeks | Theme | Gate |
|-------|-------|-------|------|
| 0 | 1 to 2 | Make the live loop run the real strategy stack | Zero-signal replay produces zero weights; ledger counts closed trades only |
| 1 | 2 to 4 | Data feeds, backfill, replay harness, trial registry | 12 months of 1h bars for BTC/ETH/SOL with funding, OI, liquidations and trades; replay of Oct 2025 runs end to end |
| 2 | 4 to 7 | Carry v2, trend sleeve, feature expansion, derivatives fix | At least one new sleeve or feature set passes CPCV with a logged trial count |
| 3 | 6 to 9 | Allocation, circuit breakers, regime model, cascade controls | Oct 2025 and Feb 2025 replays show lower drawdown without disabling the calm month before |
| 4 | 8 to 11 | Execution: lead signal, markouts, queue model, fee routing | Median 30-second markout improves in paper replay |
| 5 | ongoing | Shadow runs, decay monitoring, online layer, promotion workflow | First model promoted through the full workflow |

Phases 2, 3 and 4 overlap deliberately; they touch different files.

---

## Phase 0: Integrity of the live loop (weeks 1 to 2)

Goal: `ControlCenterStateStore.step_simulation_tick` in `ui/api_server.py` evaluates the real sleeves, allocates from real inputs, manages positions, and records only real outcomes.

### P0.1 Wire the sleeves into the tick
- Files: `ui/api_server.py`, `sleeves/sleeve_3_funding_carry.py`, `sleeves/sleeve_1_derivatives_alpha.py`
- Steps: for each of BTC, ETH, SOL call `FundingCarrySleeve.evaluate_from_snapshot` and `DerivativesAlphaSleeve.generate_alpha_signal`; build `AssetAllocationInput` from the returned `SleeveSignalOutput` (direction, meta probability, expected return, fee schedule); delete the hard-coded ETH long and SOL short; make `sleeve_telemetry` return the latest real evaluations.
- Acceptance: a unit test feeds a neutral snapshot (zero funding, empty options surface) and asserts every weight is 0.0; a second test feeds +10 bps/8h funding on BTC and asserts a carry evaluation is actionable and appears in telemetry.
- Effort: M. Depends on: nothing.

### P0.2 Remove the probability floor and direction defaults
- Files: `ui/api_server.py`
- Steps: delete `p_eff = max(0.58, new_meta.p_meta)`; delete both `else 1` fallbacks on direction; pass the real `primary_direction` (including 0) to `predict_single`.
- Acceptance: test asserts that `p_meta = 0.50` or direction 0 yields a zero weight and `HOLD_CASH`.
- Effort: S. Depends on: nothing.

### P0.3 Real HRP weights
- Files: `ui/api_server.py`, `risk/portfolio_allocator.py`
- Steps: keep a rolling matrix of 1h log returns for the three assets (96 rows minimum); call `compute_hrp_cvar_weight_dict` once per hour; fall back to equal weights with a logged warning when fewer than 96 rows exist.
- Acceptance: test shows weights change when the return matrix changes and sum to 1.
- Effort: S. Depends on: nothing.

### P0.4 Real bar fields
- Files: `ui/api_server.py`, `core/data_ingestion.py`
- Steps: replace random high/low/volume/buy fraction with Hyperliquid `candleSnapshot` (15m) and the trades stream from P1.1; replace the random `liquidation_volume_usd` with the liquidation feed from P1.3 or `0.0` plus an `unavailable` flag until it exists; compute `sigma_bar` as the EWMA of 15m log returns (span 96) instead of the constant 0.0045.
- Acceptance: a test runs one tick with a mocked snapshot and asserts no `numpy.random` call occurs in the tick path (patch the generator to raise).
- Effort: M. Depends on: P1.1 for the trade stream (interim: candles only).

### P0.5 Position lifecycle
- Files: new `core/execution/position_manager.py`, `ui/api_server.py`, `core/contracts.py`
- Steps: add a `Position` record (asset, side, entry price, entry bar, size, upper/lower barrier, vertical bar, sleeve id); open on a non-zero target weight when flat; exit on barrier touch using live high/low, on vertical expiry, on regime veto, or on a flipped target; realise PnL and fees on close; keep per-sleeve attribution. Win/loss counters increment only on close.
- Acceptance: deterministic replay test with a scripted price path hits upper, lower and vertical barriers in sequence and the ledger shows exactly three closed trades with the expected PnL.
- Effort: L. Depends on: P0.1, P0.4.

### P0.6 Signed, conditional carry PnL and real fees
- Files: `ui/api_server.py`, `core/execution/position_manager.py`
- Steps: accrue funding only on open perp legs, with sign (short pays when funding is negative); charge maker or taker fee on actual notional traded; remove the fixed 8% turnover and the derived "fee savings".
- Acceptance: test with an open spot-long/perp-short carry and negative funding shows a funding debit.
- Effort: S. Depends on: P0.5.

### P0.7 Honest retraining
- Files: `ui/api_server.py` (`_fit_worker_on_bars`), `core/models/meta_labeling_worker.py`
- Steps: when labels are single-class, skip the retrain and log; train on the full Parquet history with purging of the last `vertical_barrier_bars`; require at least 500 labelled bars; retrain on a schedule (daily) rather than every 20 ticks.
- Acceptance: test asserts no label mutation when labels are single-class and that the previous worker is retained.
- Effort: S. Depends on: nothing.

### P0.8 Live inputs for the risk committee
- Files: `ui/api_server.py`, `sentiment/committee_models.py`, `core/data_ingestion.py`
- Steps: fetch Deribit DVOL (`public/get_volatility_index_data`), stablecoin marks for USDC/USDT/USDe from at least two venues, per-asset 1h funding for ETH and SOL; pass them to the sentinel and breadth models.
- Acceptance: test asserts the sentinel's `stablecoin_depeg_bps` comes from the fetched marks, not a constant.
- Effort: M. Depends on: P1.4.

**Phase 0 gate**: an end-to-end replay with zero signals produces an unchanged ledger; a replay with scripted signals produces the expected trades; `tests/test_e2e_phoenix.py` extended with the above tests passes; the dashboard sleeve panel shows live values.

### Phase 0 implementation status (2026-10-31)

Implemented on branch `claude/intelligent-sagan-q8qknt`. Acceptance tests live in `tests/test_phase0_integrity.py` (25 tests, offline, no random data); the gate tests are `test_phase0_gate_zero_signal_replay_leaves_ledger_unchanged` and `test_phase0_gate_scripted_signals_produce_expected_trades_and_live_sleeve_panel`.

| Task | Status | Where |
|------|--------|-------|
| P0.1 | Done. Both sleeves are evaluated for BTC, ETH and SOL on every bar; `AssetAllocationInput` is built from each `SleeveSignalOutput`; `sleeve_telemetry` returns the latest real evaluations (per-asset carry evaluations, derivatives signals, HRP weights, availability, per-sleeve attribution). The hard-coded ETH long and SOL short are gone. | `ui/api_server.py` (`_evaluate_sleeves`, `_build_targets`, `sleeve_telemetry`) |
| P0.2 | Done. No probability floor, no direction default; `predict_single` receives the real positioning direction including 0. With no trained model the meta sleeve emits p = 0.5, direction 0, gate closed. | `ui/api_server.py` |
| P0.3 | Done. `RollingHourlyReturnMatrix` holds 1h log returns for the three assets (seeded from Hyperliquid 1h candles, appended live); `HRPWeightScheduler` recomputes HRP-CVaR at most hourly and falls back to equal weights with a logged warning below 96 rows. | `risk/portfolio_allocator.py` |
| P0.4 | Done (interim, candles only). Bars are Hyperliquid `candleSnapshot` 15m candles (real OHLCV and trade count); funding from `fundingHistory` for the seed and the live snapshot for new bars; taker buy/sell split is unavailable and reported as such (even split); `liquidation_volume_usd = 0.0` with `DataAvailability.liquidations = False`; `sigma_bar` is the EWMA of 15m log returns with span 96. No `numpy.random` call exists in the tick path. | `core/data_ingestion.py` (`HyperliquidCandleBarFeed`, `ScriptedBarFeed`), `core/models/triple_barrier.py` (`EWMAVolatilityEstimator`) |
| P0.5 | Done. `PositionManager` opens on a non-zero target when flat; exits on barrier touch against the bar high/low (same-bar double touch resolves to the stop), vertical expiry, regime veto, or a flipped target; hedged carry exits when the sleeve clears or flips; PnL and fees realised on close; per-sleeve attribution; win/loss counters move on close only. Open positions are persisted in the ledger and restored on restart. | `core/execution/position_manager.py`, `core/contracts.py` (`Position`, `ClosedTradeRecord`, `SleeveAttribution`) |
| P0.6 | Done. Funding accrues per bar on open perp legs only, signed (long pays positive funding; a carry perp-short is debited when funding is negative); fees are charged on notional actually traded per leg (maker on entries and passive exits, taker on stop-outs and vetoes). The fixed 8% turnover and the derived "fee savings" are removed (reported as 0). | `core/execution/position_manager.py`, `ui/api_server.py` |
| P0.7 | Done. Single-class labels skip the fit and keep the previous worker; training uses the Parquet store of real-feed bars with the last `vertical_barrier_bars` purged and requires at least 500 labelled bars; retraining runs on a daily schedule in bar time. | `ui/api_server.py` (`_fit_worker_on_bars`, `_maybe_retrain`) |
| P0.8 | Not started (depends on P1.4). The macro sentinel still receives fixed depeg and DVOL inputs; the breadth model now receives real per-asset 1h funding. | |

Consequences worth knowing:

- One tick now processes one closed 15-minute candle. A tick between candle closes returns `waiting_for_bar` and changes nothing; the dashboard poller runs every 10 seconds.
- Hyperliquid has no public open-interest history, so seeded bars carry the current OI and are not written to the training store. The meta-labeller therefore has no model until roughly 500 live bars with real OI have been recorded (about five days), and the dashboard says so. P1.2 (Bybit OI history backfill) removes this wait.
- Reported equity is base capital plus realised PnL; unrealised PnL of open positions is reported separately (`unrealized_pnl_usd`, `marked_equity_usd`).
- The cross-venue funding spread has no second-venue feed yet and is reported as `null`.
- Paper PnL will look worse than before. The previous numbers measured a forced trade every bar with a probability floor, not a strategy.

Existing tests changed: `test_account_ledger_restart_durability` and `test_trade_forensics_feature_attribution_endpoint` (`tests/test_architecture_verification.py`) now replay scripted bars on an isolated ledger, assert that trades are closed positions and that open positions survive a restart, and accept the two new feature-attribution keys; `test_r5_f21_persistent_account_ledger_initialization_5000_usd` (`tests/test_e2e_phoenix.py`) uses an isolated store so the reset no longer rewrites `data/account_ledger.json`.

---

## Phase 1: Data and research infrastructure (weeks 2 to 4)

### P1.1 Trade stream and quarter-hour boundary imbalance
- Files: `core/data_ingestion.py`, new `core/features/boundary_imbalance.py`
- Steps: subscribe to Hyperliquid websocket `trades` (and Bybit `publicTrade` as failover); persist to Parquet; compute signed taker volume in the first 10, 30 and 60 seconds after each :00/:15/:30/:45 boundary, normalised by trailing 24h average per-window volume.
- Acceptance: unit test on a synthetic trade tape; a live smoke test records at least one boundary per asset.
- Effort: M. Depends on: nothing.

### P1.2 Historical backfill, 12 months minimum
- Files: `data/backfill_storage.py`, new `scripts/backfill.py`
- Steps: Bybit 1h and 15m klines, funding history and OI history for BTC/ETH/SOL back to 2022; Hyperliquid `fundingHistory` and `candleSnapshot` for as far as available; Hyperliquid public L2 archive for the 10 Oct 2025 window; write one Parquet per asset per horizon with the existing 18-column schema plus `taker_buy_volume`.
- Acceptance: row counts and date coverage asserted by a test; no gaps longer than 2 bars.
- Effort: M. Depends on: nothing.

### P1.3 Liquidation feed
- Files: `core/data_ingestion.py`
- Steps: Bybit websocket `allLiquidation` per symbol aggregated to bar level; Hyperliquid liquidation fills from `userFillsByTime` on the HLP liquidator vault or from on-chain fill dumps; store as `liquidation_volume_usd` with side.
- Acceptance: the feature flag `liquidation_available` flips to true in live state and the positioning regime test for `CASCADE_EXHAUSTION_REVERSAL` runs on real data.
- Effort: M. Depends on: nothing.

### P1.4 Hyperliquid and Deribit endpoint coverage
- Files: `core/data_ingestion.py`, `sleeves/sleeve_1_derivatives_alpha.py`
- Steps: add `predictedFundings`, `perpsAtOpenInterestCap`, `perpDexLimits`, `candleSnapshot`, `fundingHistory`; add Deribit `get_instruments`, `ticker` (mark IV, delta, greeks), `get_volatility_index_data`; cache with TTLs; expose a `DataAvailability` record listing which feeds are live.
- Acceptance: each endpoint has a parser test on a recorded fixture; the dashboard shows the availability record.
- Effort: M. Depends on: nothing.

### P1.5 Lead-venue mid feed
- Files: `core/data_ingestion.py`
- Steps: Binance and Bybit book-ticker websockets for BTC/ETH/SOL; keep the last 2 seconds of mids with exchange timestamps; expose `lead_mid_change_bps(window_ms)`.
- Acceptance: a test replays a recorded 10-second window and asserts the computed lead move.
- Effort: S. Depends on: nothing.

### P1.6 Trial registry
- Files: new `research/trial_registry.py`, `research/validation/cpcv_gatekeeper.py`
- Steps: every CPCV run appends a JSON line (sleeve, features, parameters, window, seed, metrics); the gatekeeper reads the count of trials per sleeve family and uses it as the DSR's number of trials; add the MinBTL check and refuse promotion when the sample is shorter than the minimum backtest length.
- Acceptance: running the same configuration twice increments the count and lowers the DSR.
- Effort: S. Depends on: nothing.

### P1.7 Replay harness
- Files: new `research/replay.py`, `tests/test_replay.py`
- Steps: drive `step_simulation_tick` from Parquet bars with the same code path as live (no network), with pluggable sleeves and allocator; output per-trade records, equity curve, turnover, fees, markouts where available; named scenarios: `calm_2025_09`, `cascade_2025_10_10`, `tariff_2025_02_03`, `carry_unwind_2024_08_05`.
- Acceptance: each scenario runs in under 2 minutes and produces a report file; used by all later gates.
- Effort: L. Depends on: P0.5, P1.2.

### P1.8 Feature evaluation harness
- Files: new `research/feature_eval.py`
- Steps: for any candidate feature compute IC by horizon (15m, 1h, 4h, 12h, 24h), IC half-life, out-of-fold calibrated log-loss delta when added to the meta-labeller under CPCV, and fold-stability (importance variance across folds); write results to the trial registry.
- Acceptance: run on the existing three features and on a random-noise feature; the noise feature must fail.
- Effort: M. Depends on: P1.6.

**Phase 1 gate**: backfill covers 12 months; all four replay scenarios run; the trial registry is non-empty; the availability record shows trades, candles, funding, OI and lead mids live.

---

## Phase 2: Strategy reinforcement (weeks 4 to 7)

### P2.1 Funding carry v2
- Files: `sleeves/sleeve_3_funding_carry.py`, `core/contracts.py`
- Steps:
  1. Add `compute_deviation_rho(premium, oracle, r, r_prime, kappa=8760)` and a cost band from `FeeSchedule` plus impact.
  2. Entry when ρ exceeds the upper band and the 3-day mean funding is positive; exit when ρ first crosses zero, or 3-day mean funding < 0, or instantaneous funding < −1%/day, or ρ < 0 for 6 consecutive hours.
  3. Crowding rule: annualised funding > 25% and OI above its 90th rolling percentile, or asset listed in `perpsAtOpenInterestCap`, halves carry notional and sets a `crowded_long` flag consumed by the derivatives sleeve.
  4. Replace the heuristic meta probability with a monotone map fitted on historical realised carry outcomes (isotonic regression on predicted net edge).
  5. Cross-venue: use `predictedFundings` normalised per day; spread stop at −0.1%/day; take-profit when cumulative funding plus basis exceeds 1% of notional; sample 2 hours after settlement.
  6. Fee tier from the account's 14-day volume; use it in the hurdle.
- Acceptance: CPCV on hourly history 2022 to 2026; E[net] > 15 bps per round trip; drawdown in the Aug 2024 and Nov 2022 scenarios within the Ethena 11-day inversion bound; trial logged.
- Effort: L. Depends on: P1.2, P1.4, P1.6, P1.7.

### P2.2 Trend sleeve
- Files: new `sleeves/sleeve_2_trend.py`, `sleeves/__init__.py`, `config/config_m1max_64gb.json`
- Steps: continuous signal per lookback (24h, 168h, 720h on 1h bars) as tanh(return / realised vol); equal-weight across lookbacks; inverse-vol sizing with a 25-day EWMA and a forecast cap; correlation adjustment by 1/√(1 + (N−1)·ρ̄) over 30 days; `SleeveSignalOutput` with `sleeve_id = "trend"`; default off.
- Acceptance: CPCV on 1h bars; PnL correlation with the derivatives sleeve below 0.7; positive PnL in at least two of the three crisis scenarios; trial logged.
- Effort: M. Depends on: P1.2, P1.7.

### P2.3 Meta-labeller feature expansion
- Files: `core/features/`, `core/models/meta_labeling_worker.py`, `core/features/positioning.py`
- Steps: add, one at a time through P1.8: boundary imbalance (P1.1), basket-residualised 1h return, 1-week momentum, hour-of-day and NYSE-open flag, mark-oracle divergence and its 1h change, 48-bar variance of taker-flow imbalance, realised Amihud, 30-day funding z-score with clamp flag; add uniqueness sample weights and recency weights to `fit`; add fold-stability pruning; add a trend-scanning label as a second target.
- Acceptance: each feature individually improves out-of-fold calibrated log-loss or IC in CPCV and survives fold-stability; features that fail are recorded in the registry and not added.
- Effort: L. Depends on: P1.1, P1.8.

### P2.4 Derivatives sleeve fix
- Files: `sleeves/sleeve_1_derivatives_alpha.py`, `core/contracts.py`
- Steps: compute 25-delta risk reversal at 30 and 90 days from per-instrument delta and mark IV (Deribit `ticker`); term structure from expiry timestamps; GEX from taker-side flow per strike times gamma; OTM taker-flow imbalance feature; HLP marks at mark price; feed the components into the meta-labeller as features (preferred) or combine by IC weights shrunk 50% toward equal with sigmoid calibration.
- Acceptance: component ICs by horizon reported; the composite beats its best single component out of fold; trial logged.
- Effort: M. Depends on: P1.4, P1.8.

**Phase 2 gate**: at least one of P2.1 to P2.4 passes CPCV and is enabled in shadow mode (P5.1); the others are either passing, in progress, or recorded as failed trials.

---

## Phase 3: Allocation, risk and regimes (weeks 6 to 9)

### P3.1 Partial adjustment and impact
- Files: `risk/portfolio_allocator.py`, `core/contracts.py`
- Steps: `compute_signed_target_weights` returns an aim; a new `PartialAdjuster` moves the live weight a fraction λ per bar with λ = 1 − exp(−Δt / half-life) per sleeve, half-lives from P1.8; subtract impact I = k·σ·√(Q/V) from expected return with k = 0.5 initially and V = 24h venue volume; Kelly fraction γ = 0.5·min(1, edge / (2·SE(edge))); discretise weights to 0.05 steps; log the transfer coefficient between raw scores and executed weights.
- Acceptance: replay shows lower turnover at equal or better net PnL versus the threshold policy on the calm scenario; transfer coefficient reported in `/api/state`.
- Effort: M. Depends on: P0.5, P1.7, P1.8.

### P3.2 Circuit breakers
- Files: new `risk/circuit_breakers.py`, `core/contracts.py` (`ChampionChallengerState`), `ui/api_server.py`
- Steps: per-sleeve soft limit (−5% of sleeve allocation halves gross) and hard limit (−7.5% disables until re-promotion); per-sleeve per-asset StoplossGuard (3 stop-outs in 48 bars locks 24 bars), CooldownPeriod (4 bars after exit), LowProfit (20-trade rolling net < 0 disables the pair); daily loss budget per sleeve; smoothed Grossman–Zhou portfolio multiplier with a 1%/day decayed running max; operational lockdowns (stale feed > 30s, reject storm, unknown position state) that cancel resting orders and block new entries.
- Acceptance: unit tests per breaker; cascade replay shows the breakers fire in the expected order; calm replay shows no false lockdowns.
- Effort: M. Depends on: P0.5, P1.7.

### P3.3 Jump-model regime state
- Files: new `risk/regime_jump_model.py`, `risk/portfolio_allocator.py`, `sentiment/mlx_risk_daemon.py`
- Steps: features = realised vol (24h), funding (3-day mean), basis, DVOL, OI 24h change, average pairwise correlation (30 days); penalised k-means with a jump penalty chosen on validation Sharpe over 3 to 4 states; weekly refit, online inference; the state selects the HRP covariance (state-conditional), stop-width multiplier (tight in trend states, wide in range states), carry/directional split, and Kelly shrink in the crisis state; the LLM committee keeps `tail_risk_flag` and emergency flatten as a veto only.
- Acceptance: state persistence (mean dwell time) above 24 hours; at least one sleeve's out-of-fold net PnL improves when conditioned on the state; trial logged.
- Effort: L. Depends on: P1.2, P1.7.

### P3.4 Cascade and ADL controls
- Files: `risk/`, `core/data_ingestion.py`, `sleeves/sleeve_3_funding_carry.py`
- Steps: fragility score from taker-flow variance compression, OI percentile > 90, funding > 25% annualised, 10 bps depth falling > 50% in 1h; above threshold halve directional gross, de-lever the winning leg of any hedge, widen or pause quotes; ADL-rank estimate for open short legs as (mark/entry) × (notional/equity); collateral marks as the median of three venues with a 10% haircut on disagreement > 1%; a CVaR scenario with 90% top-of-book depth loss and 20%-per-30s forced liquidation.
- Acceptance: the Oct 2025 replay drawdown falls versus baseline while the preceding calm month keeps at least 90% of its trades.
- Effort: M. Depends on: P1.3, P1.4, P1.7, P3.2.

**Phase 3 gate**: cascade and tariff replays show lower drawdown than Phase 0 baseline; calm replay unchanged within 10% of trades; all breakers covered by tests.

---

## Phase 4: Execution (weeks 8 to 11)

### P4.1 Lead-venue quote pull
- Files: `core/execution/defensive_maker.py`, `core/data_ingestion.py`
- Steps: if the Binance or Bybit mid moved more than k × HL spread (k = 0.5) within the last 700 ms against a resting order, cancel or re-price it; count these events separately from VPIN/Hawkes cancels.
- Acceptance: replay on recorded tick data shows fewer fills immediately followed by adverse moves.
- Effort: S. Depends on: P1.5.

### P4.2 Markout ladder
- Files: new `core/execution/markouts.py`, `ui/api_server.py`
- Steps: for every fill record mid-to-mid markout at 1s, 5s, 10s, 30s, 60s, 5m; store with size bucket, minutes-to-funding, lead/lag state and sleeve; expose medians and percentiles in `/api/forensics`; feed the 30s median back into the toxicity threshold with a slow EWMA.
- Acceptance: the ladder populates in paper mode; a test computes markouts from a fixture.
- Effort: S. Depends on: P0.5, P1.5.

### P4.3 Queue position and maker/taker switch
- Files: `core/execution/defensive_maker.py`, new `core/execution/fill_model.py`
- Steps: estimate queue position from L2 size-ahead deltas; do not cancel/replace while in the front third unless P4.1 fires; train a censoring-aware fill-time model on HL L2 plus own ALO orders; switch to taker when (half-spread − taker fee) > expected adverse move given Hawkes intensity and queue state; post-fill cooldown of 30 to 60 s; inventory-skewed reservation price for carry hedge legs with funding-aware skew.
- Acceptance: replay shows higher net-of-fee PnL per fill than the current re-peg policy.
- Effort: L. Depends on: P4.1, P4.2.

### P4.4 Fee-tier and venue routing
- Files: `core/contracts.py` (`FeeSchedule`), `sleeves/sleeve_3_funding_carry.py`, `core/execution/defensive_maker.py`
- Steps: live fee tier from 14-day volume and staking discount; route the carry hedge leg to the lowest all-in cost venue from `predictedFundings` plus fees; parent-order / child-slice abstraction with passive-first slices for entries above 2% of 1-minute volume.
- Acceptance: the hurdle in the carry sleeve uses the live tier; a test shows a trade rejected at tier 0 and accepted at tier 3.
- Effort: M. Depends on: P1.4.

**Phase 4 gate**: median 30s markout improves versus Phase 0 baseline in paper replay; no increase in reject storms.

---

## Phase 5: Process (from week 4, ongoing)

### P5.1 Shadow mode and promotion workflow
- Files: `research/validation/cpcv_gatekeeper.py`, `ui/api_server.py`, `core/contracts.py`
- Steps: a promoted model runs at zero size for 200 trades; per-trade reconciliation of realised versus modelled slippage and markouts; kill thresholds pre-registered from the CPCV path distribution (live Sharpe below 80% of the median path, or drawdown beyond the 5th percentile path); only then size up.
- Acceptance: the first Phase 2 sleeve goes through the full workflow and the dashboard shows its shadow statistics.
- Effort: M. Depends on: P1.7.

### P5.2 Decay dashboard and retrain triggers
- Files: `research/feature_eval.py`, `ui/dashboard_app.py`
- Steps: rolling IC by lag per feature; retrain cadence equal to the shortest half-life in the feature set; retrain trigger on IC breach; CPCV re-run before redeploy.
- Acceptance: a panel shows IC half-lives; a retrain is triggered in a test when IC is forced below threshold.
- Effort: S. Depends on: P1.8.

### P5.3 Online residual layer and abstention
- Files: `core/models/meta_labeling_worker.py`
- Steps: a daily warm-started ridge on the residuals of the static LightGBM; Dissimilarity-Index abstention (no prediction when the live vector is beyond the 99th percentile of training distances); outlier-protection guard that skips a retrain when more than 30% of rows are flagged.
- Acceptance: out-of-fold log-loss improves with the residual layer; abstention rate reported and below 10% in calm replay.
- Effort: M. Depends on: P2.3.

### P5.4 Tournament report additions
- Files: `research/cpcv_gates.py`
- Steps: Monte Carlo trade-order shuffling, MAE/MFE by regime, transfer coefficient, turnover and impact cost per sleeve in the tournament report.
- Acceptance: report renders for the baseline.
- Effort: S. Depends on: P1.7.

---

## Dependency graph (critical path in bold)

**P0.1 → P0.5 → P1.7 → P2.1 / P2.2 / P3.1 / P3.2 → P5.1**

P1.1 → P2.3; P1.4 → P2.1, P2.4, P3.4, P4.4; P1.5 → P4.1 → P4.3; P1.6 → P1.8 → P2.3, P2.4, P3.1, P5.2; P1.3 → P3.4; P3.2 → P3.4; P2.3 → P5.3.

## First two weeks, day by day

| Day | Tasks |
|-----|-------|
| 1 | P0.2, P0.7 (small, unblock integrity); start P0.1 |
| 2 to 3 | P0.1, P0.3 |
| 4 to 5 | P1.4 (endpoints) and P1.6 (trial registry) in parallel with P0.4 |
| 6 to 8 | P0.5 position lifecycle |
| 9 | P0.6, P0.8 |
| 10 | Phase 0 gate tests; start P1.2 backfill script (runs overnight) |

## Deferred backlog (not before Phase 3 completes)

- Volatility-risk-premium sleeve (delta-hedged short vol gated to the low-vol regime).
- PCA/OU residual reversion across a wider Hyperliquid perp universe.
- Cross-sectional funding carry across 20+ perps.
- Deribit tail hedge sized to the cascade scenario.
- Statistical-arbitrage pairs beyond the three majors.
- HIP-3 markets as data or venue.

## Risks to the plan

| Risk | Mitigation |
|------|-----------|
| Historical trade-level data for Hyperliquid is thin before 2025 | Use Bybit tapes for feature research; validate HL-specific features on 2025 to 2026 only and say so in the trial registry |
| Feature expansion overfits a short crypto history | One feature at a time through P1.8; fold-stability pruning; MinBTL check in the gatekeeper |
| Phase 0 reveals the existing modules disagree with each other once wired | Treat as findings; fix contracts before adding strategies |
| Single engineer, overlapping phases | Critical path first; Phase 4 can slip without blocking Phases 2 and 3 |
| Paper PnL looks worse after Phase 0 | Expected; the current numbers do not measure a strategy |
