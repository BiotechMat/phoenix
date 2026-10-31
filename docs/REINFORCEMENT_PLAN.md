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
| P0.8 | Done. The macro sentinel receives the largest peg deviation among stablecoins quoted by at least two venues (median of the venue marks) and Deribit DVOL with its 30-day z-score, both through `LiveMacroInputsFeed`; the breadth model receives real per-asset 1h funding. A missing feed contributes no stress (0 bps, the sentinel's neutral DVOL default) and is flagged in `DataAvailability`, never replaced by a constant. Deribit DVOL is BTC-only for now. | `core/market_feeds.py`, `ui/api_server.py` |

Consequences worth knowing:

- One tick now processes one closed 15-minute candle. A tick between candle closes returns `waiting_for_bar` and changes nothing; the dashboard poller runs every 10 seconds.
- Hyperliquid has no public open-interest history, so seeded bars carry the current OI and are not written to the training store. The meta-labeller therefore has no model until roughly 500 live bars with real OI have been recorded (about five days), and the dashboard says so. Running `scripts/backfill.py` (P1.2, done) fills the training store with real open interest and removes this wait.
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

**Phase 1 gate status**: the tooling for every gate item exists and is tested. What remains is operational and has to run on a machine that can reach Bybit: `scripts/backfill.py --start 2024-07-01` (Bybit supplies the multi-year open-interest history; OKX keeps only about 45 days, so the 2024 and early-2025 scenario windows have no OI from this sandbox), then the four `research.replay --scenario` runs, then `research.feature_eval` to populate `data/trial_registry.jsonl`. With the server running, the availability record shows trades, candles, funding, OI (once the store holds 500 real rows with OI) and lead mids live.

### Phase 1 implementation status (2026-10-31)

| Task | Status | Where |
|------|--------|-------|
| P1.2 | Done for candles, mark/index prices, funding and open interest. `scripts/backfill.py` writes one Parquet per asset per horizon (`data/{coin}usdt_perp_bars.parquet` for 15m, which is also the live training store, and `..._1h.parquet`) with the 18-column schema plus `funding_rate_1h`, `taker_buy_volume` (null until P1.1), `bar_source`, `oi_source` and `fields_unavailable`. Candles and mark/index prices come from Bybit (primary) or OKX (years of history on both); funding from Hyperliquid `fundingHistory` (multi-year); open interest from Bybit (multi-year) or OKX (recent weeks only). Rows without open interest carry `null` and are excluded from training. Coverage and gap checks are asserted by `tests/test_backfill.py`; the CLI exits non-zero on any gap longer than two bars. Not yet backfilled: liquidation volume (P1.3), taker buy volume (P1.1), and the Hyperliquid L2 archive for 10 Oct 2025. | `data/backfill_storage.py`, `scripts/backfill.py` |
| P1.7 | Done. `research/replay.py` drives `step_simulation_tick` through `ScriptedBarFeed` on Parquet bars with no network, pluggable sleeves, causal model training (only rows closing before the scenario start reach the training store), and a JSON report with per-trade records, equity curve, turnover, fees, funding, drawdown and per-sleeve attribution. Markouts are reported unavailable until P4.2. The four named scenarios are registered; they need the 2024 to 2025 windows backfilled first (`scripts/backfill.py --start 2024-07-01`). A 19-bar scripted scenario replays in well under a second; two days of real bars with model training replays in seconds. | `research/replay.py`, `tests/test_replay.py` |
| P1.4 | Done for the endpoints a feed exists for. `core/market_feeds.py` adds Hyperliquid `predictedFundings` (normalised per hour and per day), `perpsAtOpenInterestCap` and `perpDexLimits`; Deribit `get_volatility_index_data` (DVOL with a 30-day hourly z-score), `get_instruments` (with a tenor selector for the 30 and 90 day expiries) and `ticker` (mark IV, delta, gamma, vega, theta); stablecoin marks for USDT, USDC and USDe from Coinbase, Kraken, Bitstamp, Gemini, OKX and KuCoin. Every parser has a fixture test; fetchers sit behind a 5-minute TTL cache that serves the last good value on failure. `DataAvailability` now lists every feed and the dashboard renders it as a pill row. `candleSnapshot` and `fundingHistory` were already covered in Phase 0. The 25-delta risk reversal and GEX from per-instrument greeks are P2.4. | `core/market_feeds.py`, `core/contracts.py`, `ui/dashboard_app.py` |
| P1.6 | Done. `research/trial_registry.py` appends one JSON line per CPCV run (sleeve family, model, features, parameters, window, seed, metrics); `run_registered_validation` passes the family's trial count to the DSR, so repeating a configuration lowers its DSR; `ChampionChallengerGatekeeper.evaluate_challenger(sample_bars=...)` refuses promotion when the sample is shorter than the MinBTL bound for the trial count. The tournament in `research/cpcv_gates.py` still counts only its own candidates; routing it through the registry is a one-line change when the first Phase 2 sleeve is validated. | `research/trial_registry.py`, `research/validation/cpcv_gatekeeper.py` |
| P1.1 | Done. `core/streams.py` subscribes to the Hyperliquid websocket `trades` for BTC, ETH and SOL through a reconnecting `StreamHub` (started by the app lifespan), keeps a 25-hour `TradeTape` with per-coin connection coverage, flushes part files to `data/trades/{coin}/`, and computes signed taker volume in the first 10, 30 and 60 seconds after every quarter-hour boundary, normalised by the trailing per-window volume (up to 24 hours; `trailing_hours_used` says how much). Bars processed while the stream covered the whole bar get real taker buy/sell dollar volume. Bybit `publicTrade` failover is not implemented. Verified live: BTC, ETH and SOL boundaries recorded across a real :45 boundary. | `core/streams.py` |
| P1.3 | Done for Bybit and OKX. Bybit websocket `allLiquidation.{SYMBOL}` (primary) and OKX REST `public/liquidation-orders` polling (fallback, contract sizes from `public/instruments`) feed a `LiquidationBook`; each covered bar gets its real `liquidation_volume_usd` with long/short split, and `DataAvailability.liquidations` flips true. The `CASCADE_EXHAUSTION_REVERSAL` regime now fires from streamed liquidations (test on scripted events; it will run on live events as they occur). Hyperliquid liquidation fills (HLP liquidator `userFillsByTime`) are not implemented. | `core/streams.py`, `ui/api_server.py` |
| P1.5 | Done. Binance futures `bookTicker` and Bybit `orderbook.1` top of book, with exchange timestamps, in a 2-second `LeadMidBuffer`; `lead_mid_change_bps(coin, window_ms)` returns the per-venue or median move. Tested on a real 10-second recorded window (`tests/fixtures/stream_recording_btc.json`). Binance and Bybit refuse REST from the development sandbox but their websockets connect. | `core/streams.py` |
| P1.8 | Done. `research/feature_eval.py` reports IC by horizon (15m to 24h, overlap-adjusted t), IC half-life, out-of-fold calibrated log-loss delta under CPCV with the live labeller and model, fold-level t of the delta, permutation-importance stability, and IC sign consistency; results go to the trial registry. On 12 days of real BTC bars none of the three live features passes and the noise control fails; a deliberately leaked future return passes on every fold. `python -m research.feature_eval` runs it on the stored bars. | `research/feature_eval.py` |

Sandbox note: the Bybit paths are exercised on recorded fixtures only, because Bybit and Binance refuse requests from the development sandbox; OKX and Hyperliquid were verified live. On a machine with Bybit access, `--venues bybit,okx` (the default) prefers Bybit for candles and multi-year open interest.

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

### Phase 2 implementation status (2026-10-31)

| Task | Status | Where |
|------|--------|-------|
| P2.2 | Implemented and evaluated; **does not pass the CPCV gate, ships off**. `sleeves/sleeve_2_trend.py` computes tanh(return / realised vol) over 24h, 168h and 720h, equal-weighted, sized by inverse 25-day EWMA vol with the 1/sqrt(1 + (N-1)ρ̄) adjustment, a per-asset cap and a 5% rebalance buffer; positions use a new "target" exit policy (no price barriers, exit when the sleeve clears or flips). `research/trend_eval.py` validated it on 3.4 years of real hourly bars (June 2023 to October 2026, OKX candles, Hyperliquid funding) net of maker fees, slippage and funding: portfolio return +20.2%, Sharpe 1.25, max drawdown 4.3%, positive in 2 of the 3 crisis windows (carry unwind +0.63%, tariff +0.16%, October 2025 cascade −0.15%). Per asset it fails the gate mainly on E[net PnL per trade] > 15 bps (BTC 1.7, ETH 4.9, SOL 14.3 bps) and on DSR for BTC (0.89) and SOL (0.88); three trials logged. The correlation with the derivatives sleeve cannot be measured because that sleeve's history cannot be reconstructed. Parameters were not tuned to pass. | `sleeves/sleeve_2_trend.py`, `research/trend_eval.py` |
| P2.4 | Computation fixed; **validation pending data, sleeve does not allocate**. `core/options_surface.py` computes Black-Scholes delta and gamma from Deribit mark IV per instrument (agrees with Deribit's own greeks to about 0.002 delta), the 25-delta risk reversal at 30 and 90 days, ATM term structure from real expiry timestamps, and dealer GEX per 1% move from gamma × OI. The previous parser averaged call minus put IV across all strikes and split near/far by response order. HLP positioning is now read from the vault's child accounts at mark value, per coin (the parent vault holds no positions, so the old reading was always empty). Surface components are appended to `data/options_surface/{coin}.parquet` each fetch so their ICs can be evaluated with P1.8 once history accrues; Deribit's public API has no historical surface. Because the corrected skew is usually negative in crypto, the unchanged ±1.5 vol-point rule would short BTC most of the time; `DerivativesAlphaConfig.allocate` (default off) keeps the sleeve evaluating and reporting without sizing until it is validated. Taker-flow-per-strike GEX and the OTM taker imbalance need Deribit trades and are not done. | `core/options_surface.py`, `sleeves/sleeve_1_derivatives_alpha.py` |
| P2.1 | Deviation rule implemented and evaluated; **v2 as specified fails, v1 retained**. `carry_v2_decide` in `sleeves/sleeve_3_funding_carry.py` implements ρ = κ·premium − (r − r′), the cost band κ·log(1 + C), the 3-day mean-funding entry filter, the hard exits, and the crowding cut (funding part; no historical OI). κ defaults to 1095, not the plan's 8760: Hyperliquid pays one-eighth of an 8-hour premium each hour, so the premium is paid 1095 times a year; 8760 would put the fair premium eight times too low. The plan's "ρ < 0 for 6 hours" exit is implied by "exit when ρ first crosses zero". `research/carry_eval.py` simulates v1 and v2 settlement by settlement on Hyperliquid's own history from 8 June 2023 (start of hourly settlement) with funding, premium convergence and two-leg costs, and validates each through CPCV and the trial registry. Result: v1 passes on all three assets (net per round trip 72, 68 and 65 bps; max drawdown under 0.5%); v2 loses on BTC and ETH and earns 2.8 bps per trip on SOL, because the crowding cut re-evaluated every hour flips size between 1.0 and 0.5 about 950 times per asset while in a position; with the cut disabled (a logged diagnostic trial) v2 passes at 35 to 39 bps per trip but is still dominated by v1. **Caveat**: the basis leg uses Hyperliquid's smoothed premium index, which understates executable perp-spot basis noise, so the simulated Sharpe ratios (about 10) are not realistic and the live loop's paper results are the honest check. Not done: isotonic meta-probability, cross-venue predicted-funding legs, live fee tier. | `sleeves/sleeve_3_funding_carry.py`, `research/carry_eval.py`, `data/backfill_storage.py` (`fetch_hyperliquid_settlements`) |
| P2.3 | Not started. It needs months of 15m bars with open interest (Bybit, or OKX beyond its 45-day window) to evaluate candidates through P1.8; from this sandbox only 12 days are available. | |

Data note: Hyperliquid's funding history settles every 8 hours in its early period (for example 0.0001 per 8 hours in June 2023) and every hour later (0.0000125 per hour). The backfill now divides each settlement by its interval, so stored `funding_rate_1h` is a true hourly rate throughout; earlier backfills should be refreshed with `scripts/backfill.py --funding-only`. The hourly return matrix used by HRP was also corrected to form returns from completed-hour closes; it previously used the first price of the new hour.

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

### Phase 3 implementation status (2026-10-31)

| Task | Status | Where |
|------|--------|-------|
| P3.2 | Done. `risk/circuit_breakers.py`: per-sleeve soft limit (5% of the sleeve's allocation halves its entry size) and hard limit (7.5% disables it until `re_enable`), per-sleeve daily loss budget (3%, resets at the UTC day), per-pair StoplossGuard (3 stop-outs in 48 bars lock 24 bars), CooldownPeriod (4 bars after any exit) and LowProfit (last 20 trades net negative disables the pair), a Grossman-Zhou portfolio multiplier on a running maximum that decays 1% per day (20% maximum drawdown, EWMA-smoothed over 4 bars), and an operational lockdown (feed older than 30 s, reject storm, unknown position state) that blocks entries and asks for resting orders to be cancelled. Breakers only scale or refuse entries, never exits. On by default in live mode; replays and tests pass them explicitly (`run_replay(circuit_breakers=...)`). Every breaker has a unit test; a scripted stop storm shows cooldown, then the StoplossGuard lock, in that order; a calm replay shows no lockdowns. The `ChampionChallengerState` contract change listed in the plan was not needed: breaker state lives in the breaker book and is reported in the sleeve telemetry and on the dashboard. The cascade and tariff replays need the 2024 to 2025 backfill with open interest. | `risk/circuit_breakers.py`, `ui/api_server.py`, `research/replay.py` |
| P3.1, P3.3, P3.4 | Not started. | |

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
