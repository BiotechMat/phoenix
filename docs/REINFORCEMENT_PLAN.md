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

**Phase 1 gate status: met.** The Binance public data archive (`data.binance.vision`, venue `binance`) is plain file hosting and answers from the sandbox where the Bybit and Binance APIs refuse. It supplies multi-year 15m/1h klines with taker buy volume, mark and index klines, and 5-minute open interest. `scripts/backfill.py --start 2023-06-01 --venues binance,okx` filled BTC, ETH and SOL from 1 June 2023 to 31 October 2026 at both horizons: 119,901 15m bars and 29,975 1h bars per asset, no gap, open interest on every archive row. The archive publishes the newest day or two late, so 28 to 31 October 2026 came from OKX without open interest; open interest from two venues is never mixed in one series. The four scenario replays and the feature evaluation then ran on these bars (P1.7, P1.8 below). With the server running, the availability record shows trades, candles, funding, OI and lead mids live.

### Phase 1 implementation status (2026-10-31)

| Task | Status | Where |
|------|--------|-------|
| P1.2 | Done for candles, mark/index prices, funding and open interest. `scripts/backfill.py` writes one Parquet per asset per horizon (`data/{coin}usdt_perp_bars.parquet` for 15m, which is also the live training store, and `..._1h.parquet`) with the 18-column schema plus `funding_rate_1h`, `taker_buy_volume` (null until P1.1), `bar_source`, `oi_source` and `fields_unavailable`. Candles and mark/index prices come from Bybit (primary) or OKX (years of history on both); funding from Hyperliquid `fundingHistory` (multi-year); open interest from Bybit (multi-year) or OKX (recent weeks only). Rows without open interest carry `null` and are excluded from training. Coverage and gap checks are asserted by `tests/test_backfill.py`; the CLI exits non-zero on any gap longer than two bars. Archive bars carry real taker buy/sell dollar volume (`taker_buy_volume` set, `taker_flow` dropped from `fields_unavailable`). Not backfilled: liquidation volume (P1.3; the archive's `liquidationSnapshot` file for June 2023 does not exist) and the Hyperliquid L2 archive for 10 Oct 2025. **Venue caveat**: the stored history uses Binance open interest, while the live loop reads Hyperliquid's, which is several times smaller; the OI z-score sees that level change as a move for about a month after live bars start. | `data/backfill_storage.py`, `scripts/backfill.py` |
| P1.7 | Done. `research/replay.py` drives `step_simulation_tick` through `ScriptedBarFeed` on Parquet bars with no network, pluggable sleeves, causal model training (only rows closing before the scenario start reach the training store), and a JSON report with per-trade records, equity curve, turnover, fees, funding, drawdown and per-sleeve attribution. Markouts are reported unavailable until P4.2. The four named scenarios run on the archive backfill; results are under P3.2/P3.4 and the Phase 3 gate. `--controls none|phase3|breakers|cascade` selects the Phase 0 baseline, both Phase 3 controls, or one of them. When every bar in the window carries venue taker volume, the cascade controller reads it from the bars (the live loop reads it from the trade stream instead). A 19-bar scripted scenario replays in well under a second; two days of real bars with model training replays in seconds. | `research/replay.py`, `tests/test_replay.py` |
| P1.4 | Done for the endpoints a feed exists for. `core/market_feeds.py` adds Hyperliquid `predictedFundings` (normalised per hour and per day), `perpsAtOpenInterestCap` and `perpDexLimits`; Deribit `get_volatility_index_data` (DVOL with a 30-day hourly z-score), `get_instruments` (with a tenor selector for the 30 and 90 day expiries) and `ticker` (mark IV, delta, gamma, vega, theta); stablecoin marks for USDT, USDC and USDe from Coinbase, Kraken, Bitstamp, Gemini, OKX and KuCoin. Every parser has a fixture test; fetchers sit behind a 5-minute TTL cache that serves the last good value on failure. `DataAvailability` now lists every feed and the dashboard renders it as a pill row. `candleSnapshot` and `fundingHistory` were already covered in Phase 0. The 25-delta risk reversal and GEX from per-instrument greeks are P2.4. | `core/market_feeds.py`, `core/contracts.py`, `ui/dashboard_app.py` |
| P1.6 | Done. `research/trial_registry.py` appends one JSON line per CPCV run (sleeve family, model, features, parameters, window, seed, metrics); `run_registered_validation` passes the family's trial count to the DSR, so repeating a configuration lowers its DSR; `ChampionChallengerGatekeeper.evaluate_challenger(sample_bars=...)` refuses promotion when the sample is shorter than the MinBTL bound for the trial count. The tournament in `research/cpcv_gates.py` still counts only its own candidates; routing it through the registry is a one-line change when the first Phase 2 sleeve is validated. | `research/trial_registry.py`, `research/validation/cpcv_gatekeeper.py` |
| P1.1 | Done. `core/streams.py` subscribes to the Hyperliquid websocket `trades` for BTC, ETH and SOL through a reconnecting `StreamHub` (started by the app lifespan), keeps a 25-hour `TradeTape` with per-coin connection coverage, flushes part files to `data/trades/{coin}/`, and computes signed taker volume in the first 10, 30 and 60 seconds after every quarter-hour boundary, normalised by the trailing per-window volume (up to 24 hours; `trailing_hours_used` says how much). Bars processed while the stream covered the whole bar get real taker buy/sell dollar volume. Bybit `publicTrade` failover is not implemented. Verified live: BTC, ETH and SOL boundaries recorded across a real :45 boundary. | `core/streams.py` |
| P1.3 | Done for Bybit and OKX. Bybit websocket `allLiquidation.{SYMBOL}` (primary) and OKX REST `public/liquidation-orders` polling (fallback, contract sizes from `public/instruments`) feed a `LiquidationBook`; each covered bar gets its real `liquidation_volume_usd` with long/short split, and `DataAvailability.liquidations` flips true. The `CASCADE_EXHAUSTION_REVERSAL` regime now fires from streamed liquidations (test on scripted events; it will run on live events as they occur). Hyperliquid liquidation fills (HLP liquidator `userFillsByTime`) are not implemented. | `core/streams.py`, `ui/api_server.py` |
| P1.5 | Done. Binance futures `bookTicker` and Bybit `orderbook.1` top of book, with exchange timestamps, in a 2-second `LeadMidBuffer`; `lead_mid_change_bps(coin, window_ms)` returns the per-venue or median move. Tested on a real 10-second recorded window (`tests/fixtures/stream_recording_btc.json`). Binance and Bybit refuse REST from the development sandbox but their websockets connect. | `core/streams.py` |
| P1.8 | Done. `research/feature_eval.py` reports IC by horizon (15m to 24h, overlap-adjusted t), IC half-life, out-of-fold calibrated log-loss delta under CPCV with the live labeller and model, fold-level t of the delta, permutation-importance stability, and IC sign consistency; results go to the trial registry. On 12 days of real BTC bars none of the three live features passed. On one year (35,040 BTC bars to 31 October 2026), `ffd_oi_zscore` passes on log-loss (−0.0132, better on 100% of folds, importance stable on 100%), `ffd_close_zscore` passes on IC only (−0.045 at its best horizon, t −8.35, a mean-reversion sign) while worsening log-loss, and `oi_funding_basis_zscore` fails. The seeded noise column fails on importance stability (27%) but lowers log-loss by about 0.001 on 93% of folds: adding any column helps this model a little, so later set tests use a noise-padded reference. A deliberately leaked future return passes on every fold. `python -m research.feature_eval --last 35040` runs it. | `research/feature_eval.py` |

Sandbox note: the Bybit paths are exercised on recorded fixtures only, because the Bybit and Binance APIs refuse requests from the development sandbox; the Binance archive, OKX and Hyperliquid were verified live. The default `--venues bybit,binance,okx` prefers Bybit where it answers.

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
| P2.1 | Deviation rule implemented and evaluated; **v2 passes with the full crowding rule but v1 dominates, v1 retained**. `carry_v2_decide` in `sleeves/sleeve_3_funding_carry.py` implements ρ = κ·premium − (r − r′), the cost band κ·log(1 + C), the 3-day mean-funding entry filter, the hard exits, and the crowding cut (annualised funding above 25% with open interest above its trailing 30-day 90th percentile, or the OI cap; funding alone when no OI percentile is known). κ defaults to 1095, not the plan's 8760: Hyperliquid pays one-eighth of an 8-hour premium each hour, so the premium is paid 1095 times a year; 8760 would put the fair premium eight times too low. The plan's "ρ < 0 for 6 hours" exit is implied by "exit when ρ first crosses zero". `research/carry_eval.py` simulates v1 and v2 settlement by settlement on Hyperliquid's own history from 8 June 2023 (start of hourly settlement) with funding, premium convergence and two-leg costs, and validates each through CPCV and the trial registry. Result: v1 passes on all three assets (net per round trip 72, 68 and 65 bps; max drawdown under 0.5%); v2 loses on BTC and ETH and earns 2.8 bps per trip on SOL, because the crowding cut re-evaluated every hour flips size between 1.0 and 0.5 about 950 times per asset while in a position; with the cut disabled (a logged diagnostic trial) v2 passes at 35 to 39 bps per trip but is still dominated by v1. Rerun with the plan's full crowding rule, using hourly open interest from the Binance archive: v2 now passes on all three assets at 24.0, 20.6 and 26.7 bps per round trip (88 to 106 round trips; max drawdown under 1%), against v1's unchanged 71.8, 67.7 and 64.8 bps. The OI condition removes most of the size flipping, but v2's tighter exits make 43 to 63% more round trips for a third to two-fifths of v1's edge per trip, so v1 stays live. **Caveat**: the basis leg uses Hyperliquid's smoothed premium index, which understates executable perp-spot basis noise, so the simulated Sharpe ratios (about 10) are not realistic and the live loop's paper results are the honest check. Not done: isotonic meta-probability, cross-venue predicted-funding legs, live fee tier. | `sleeves/sleeve_3_funding_carry.py`, `research/carry_eval.py`, `data/backfill_storage.py` (`fetch_hyperliquid_settlements`) |
| P2.3 | Evaluated; **three bar-only features run in a shadow challenger model, the stronger taker-flow features recorded but not wired**. `core/features/candidates.py` computes each candidate causally: 1-hour return net of the equal-weight basket (30-day beta), 1-week momentum, hour of day, NYSE first-hour flag (09:30 to 10:30 New York, weekdays, holidays not excluded), mark-index divergence and its 1-hour change, 48-bar variance of taker imbalance, Amihud illiquidity, 30-day funding z-score and funding-clamp flag, and the signed taker share of the first one-minute kline of each bar (the P1.1 boundary imbalance, from archive 1m klines). `research/p23_eval.py` runs each through P1.8 on BTC, then repeats the passing ones on the preceding non-overlapping year, then tests sets against the live model padded with the same number of seeded noise columns. Passed on both years: taker-imbalance variance (log-loss −0.0069 and −0.0055), first-minute taker imbalance (−0.0020, −0.0028), basket residual (−0.0017, −0.0023), divergence change (−0.0015, −0.0018), NYSE flag (−0.0010, −0.0016) and 1-week momentum (−0.0025, −0.0014). Passed in one year only: hour-of-day sine and divergence level. Failed: hour-of-day cosine, Amihud, funding z-score, clamp flag. Against the noise-padded reference, the six together lower log-loss by 0.0065 and 0.0042 (better on every fold); the bar-only three (momentum, divergence change, NYSE flag) by 0.0019 and 0.0008 (80% of folds). The bar-only three are `DEFAULT_META_CHALLENGER_FEATURES`: the store fits a second meta-labeller with them on the same rows and labels at every retrain, scores every bar, and routes its BTC target (sleeve `meta_labeller_challenger`) to the shadow book, so it trades at zero size under P5.1 while the live model keeps its three features (`DEFAULT_META_EXTRA_FEATURES` is empty). Promotion after 200 shadow trades within the kill thresholds is a configuration change. Columns are built by `bar_feature_matrix`; seeded candles without a venue mark count as unknown. The taker-flow features and the basket residual are not wired: live taker flow exists only for bars the trade stream covered, and the store keeps no 15m history for ETH and SOL, so live values would not match training. Training changes: label-uniqueness weights (log-loss 0.31300 to 0.31255, better on 67% of folds) and 30-day recency weights (worse on 87%) fail; a trend-scanning target scores a lower AUC on the triple-barrier label than training on that label (0.6506 against 0.6521). `LightGBMMetaLabelWorker.fit` gained an optional `sample_weight`. All runs are in the trial registry. | `core/features/candidates.py`, `research/p23_eval.py`, `ui/api_server.py`, `core/models/meta_labeling_worker.py` |

Data note: Hyperliquid's funding history settles every 8 hours in its early period (for example 0.0001 per 8 hours in June 2023) and every hour later (0.0000125 per hour). The backfill now divides each settlement by its interval, so stored `funding_rate_1h` is a true hourly rate throughout; earlier backfills should be refreshed with `scripts/backfill.py --funding-only`. The hourly return matrix used by HRP was also corrected to form returns from completed-hour closes; it previously used the first price of the new hour.

**Phase 2 gate status: met through P2.3.** The three P2.3 features that passed on two years run in a challenger meta-labeller that trades only in the shadow book (P5.1); the live meta-labeller sleeve is unchanged until the challenger completes its shadow run. Carry v2 passes CPCV but is dominated by v1, so it is recorded and not enabled. The trend sleeve (P2.2) failed its gate and is off. P2.4 awaits surface history.

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

**Phase 3 gate status: met, with one adverse scenario.** Replays on the archive backfill (`research.replay --scenario ... --controls none|phase3`), $5,000 paper account, model trained only on bars before each window. Every trade in these windows came from the carry sleeve; the derivatives sleeve does not allocate and trend is off.

| Scenario | Baseline: trades, net, max drawdown | Phase 3 controls: trades, net, max drawdown |
|---|---|---|
| Cascade, 8 to 13 Oct 2025 | 70, −$6.63, $49.07 | 19, +$19.87, $11.76 |
| Tariff shock, 1 to 6 Feb 2025 | 6, +$9.64, $3.08 | 3, +$9.19, $2.83 |
| Calm, September 2025 | 6, −$0.62, $0.70 | 6, −$0.57, $0.61 |
| Carry unwind, 3 to 8 Aug 2024 | 6, +$34.18, $3.49 | 4, +$19.88, $6.62 |

Drawdown is lower in the cascade and tariff windows and the calm month keeps the same trade count, so the gate as written passes. With only the breakers on, the cascade replay gives 20 trades, +$19.89 and $11.87, so the breakers (one StoplossGuard lock and the post-exit cooldowns) produce the improvement; the cascade controller alone changes nothing (see P3.4). In the August 2024 unwind the cooldown skipped two quick re-entries that would both have won, so drawdown rose from $3.49 to $6.62. Each window has few trades, so these are checks of behaviour, not estimates of edge.

### Phase 3 implementation status (2026-10-31)

| Task | Status | Where |
|------|--------|-------|
| P3.2 | Done. `risk/circuit_breakers.py`: per-sleeve soft limit (5% of the sleeve's allocation halves its entry size) and hard limit (7.5% disables it until `re_enable`), per-sleeve daily loss budget (3%, resets at the UTC day), per-pair StoplossGuard (3 stop-outs in 48 bars lock 24 bars), CooldownPeriod (4 bars after any exit) and LowProfit (last 20 trades net negative disables the pair), a Grossman-Zhou portfolio multiplier on a running maximum that decays 1% per day (20% maximum drawdown, EWMA-smoothed over 4 bars), and an operational lockdown (feed older than 30 s, reject storm, unknown position state) that blocks entries and asks for resting orders to be cancelled. Breakers only scale or refuse entries, never exits. On by default in live mode; replays and tests pass them explicitly (`run_replay(circuit_breakers=...)`). Every breaker has a unit test; a scripted stop storm shows cooldown, then the StoplossGuard lock, in that order; a calm replay shows no lockdowns. The `ChampionChallengerState` contract change listed in the plan was not needed: breaker state lives in the breaker book and is reported in the sleeve telemetry and on the dashboard. Scenario replays on the archive backfill are in the Phase 3 gate below: the breakers account for the whole improvement in the October 2025 cascade. | `risk/circuit_breakers.py`, `ui/api_server.py`, `research/replay.py` |
| P3.1 | Components built; **partial adjustment rejected for the trend sleeve**. `risk/partial_adjustment.py` provides the Gârleanu-Pedersen adjuster (λ = 1 − exp(−Δt / half-life) per sleeve), a 0.05 weight grid (rounding to nearest; rounding toward zero was tried first and held weights below their aims), square-root impact k·σ·√(Q/V) with k = 0.5, the uncertainty-scaled Kelly γ, and the transfer coefficient. Impact is now deducted from the meta-labeller's and derivatives sleeve's expected returns (negligible at a $5,000 account, material at size), and the transfer coefficient is in `/api/state`. Acceptance on real hourly data (trend sleeve, June 2023 to October 2026, half-life from P1.8's IC half-life of the forecast, about 1 hour): partial adjustment returned +17.0% net at 75.8 turnover and Sharpe 0.93, against the existing 0.05 threshold buffer's +20.2% at 51.3 turnover and Sharpe 1.25; in the calm September 2025 window −0.01% at 3.00 turnover against +0.36% at 2.18. It does not lower turnover at equal PnL, so the buffer stays. The IC half-life measured on next-hour returns is a poor clock for a signal built on 24-hour to 30-day lookbacks; a half-life measured at the sleeve's holding horizon is the next thing to try. Position resizing in the live position manager was not built because no sleeve has earned it. | `risk/partial_adjustment.py`, `ui/api_server.py` |
| P3.3 | Built and evaluated; **persistent but does not improve any sleeve, not wired live**. `risk/regime_jump_model.py` fits the statistical jump model (squared distance to centroids plus λ per state switch) by alternating a numba-compiled Viterbi assignment with centroid updates over seeded k-means++ starts, and infers states with a causal forward filter. `research/regime_eval.py` builds hourly features from real data (24h realised volatility, 3-day funding, 24h premium, Deribit DVOL, 30-day average correlation; open-interest change omitted for lack of history) and runs a walk-forward test: 12-month expanding training window, quarterly test blocks, K ∈ {3, 4} and λ ∈ {10, 50, 200} chosen on a 3-month validation Sharpe, sleeves traded only in states with a positive training mean. On June 2024 to October 2026 out-of-fold: mean dwell 426 hours (trend) and 290 hours (carry), well above the 24-hour requirement; trend +10.1% unconditioned against +8.8% conditioned; carry identical (every state was profitable). Two trials logged. The state therefore does not select covariance, stop widths, the carry/directional split or Kelly shrink in the live loop; the committee's veto is unchanged. | `risk/regime_jump_model.py`, `research/regime_eval.py` |
| P3.4 | Controls built and wired; **no effect in the October 2025 replay**. `risk/cascade_controls.py` scores fragility per asset from four conditions, each counted only when its input exists: 48-bar taker-flow variance below half its trailing median (needs the trade stream), open interest above its 90th rolling percentile (mid-rank, so a flat series is not "at highs"), funding above 25% annualised, and 10 bps depth down more than 50% in an hour (from the live L2 snapshot). Two conditions make an asset fragile: directional entries in it are halved, new carry entries are blocked, and quoting is flagged to pause (one condition: widen). The plan's "de-lever the winning leg of a hedge" needs leg resizing, which the paper book does not do, so it is reported rather than executed. Also built: the ADL-rank estimate (mark/entry × notional/equity) for open short legs, USDC collateral marked at the venue median with a 10% sizing haircut when venues disagree by more than 1%, and a liquidation stress (90% depth loss, 20% force-closed every 30 s) reported in USD and as a share of equity. In the October 2025 cascade replay with only this controller on, the result is the same as the baseline (69 trades against 70, maximum drawdown $49.07 both). Fed the window's bars directly, it finds taker-flow compression often (98 to 130 of 480 bars per asset) and funding above 25% only on 4 bars; BTC open interest was above its 90th percentile on 39 bars. Two conditions coincided only briefly, on BTC alone. Depth history is unavailable in replay, so one of the four conditions can never fire there. Thresholds were not tuned to the event. | `risk/cascade_controls.py`, `ui/api_server.py` |

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

### Phase 4 implementation status (2026-10-31)

Evaluated on a 25-minute recording of Hyperliquid BBO and trades with Binance and Bybit top of book (805,530 events, BTC, ETH and SOL, October 2026). A six-minute BTC slice of it is a test fixture.

| Task | Status | Where |
|------|--------|-------|
| P4.1 | Done; **acceptance passes on recorded ticks**. `should_pull_for_lead_move` cancels a resting quote when the Binance/Bybit median mid moved against it by more than 0.5 x the Hyperliquid spread within 700 ms; the defensive maker emits `CANCEL_LEAD_VENUE_PULL`, counted apart from VPIN/Hawkes cancels. `research/quote_sim.py` replays the recording with a join-the-touch quoter and scores each fill as adverse when the 5 s mid moved through it. Adverse-fill rate without and with the pull: BTC 25.8% to 19.5% (1,169 to 1,006 fills, 246 pulls), ETH 22.2% to 15.0%, SOL 32.2% to 21.5%. The median 5 s markout is unchanged. The six-minute BTC fixture (23.6% to 21.5%) is the acceptance test. | `core/execution/defensive_maker.py`, `research/quote_sim.py`, `tests/test_lead_pull.py` |
| P4.2 | Done. `core/execution/markouts.py` records every paper fill with size bucket, minutes to funding, lead/lag state and sleeve, and resolves mid-to-mid markouts at 1 s, 5 s, 10 s, 30 s, 60 s and 5 min against the Hyperliquid BBO mids the stream hub now carries. Medians and percentiles are in `/api/forensics`; the slow EWMA of the 30 s median tightens the VPIN threshold when fills are being picked off. A fixture test computes the ladder. | `core/execution/markouts.py`, `ui/api_server.py` |
| P4.3 | Queue model and switch rule built; **acceptance fails**. `research/queue_sim.py` places a new order at the back of the displayed size, advances it on trades and pro-rata cancels ahead, fills on trade-through, holds orders in the front third instead of re-pegging (unless P4.1 fires), applies a 30 s post-fill cooldown, and switches to taker when half-spread minus taker fee exceeds the expected adverse move. Net per fill (30 s markout minus the 1.5 bp maker fee), re-peg / queue-aware / queue-aware with lead pull: BTC −2.52 / −2.91 / −2.41 bp, ETH −2.97 / −4.01 / −1.05 bp, SOL −2.75 / −3.75 / −3.25 bp. Queue-awareness alone is worse on every coin; with the lead pull it beats re-peg on BTC and ETH and loses on SOL. The taker switch never fired. Every policy loses after the maker fee at tier 0, so passive quoting is not profitable at this fee level. The censoring-aware fill-time model (`core/execution/fill_model.py`) needs our own resting orders and was not built. | `research/queue_sim.py`, `tests/test_queue_sim.py` |
| P4.4 | Done. `core/fee_tiers.py` reads the fee table from Hyperliquid `userFees`, derives the tier from 14-day volume and the staking discount, and feeds the store's fee schedule, so the carry hurdle uses the live tier; a test rejects a carry trade at tier 0 and accepts it at tier 3. The carry hedge venue is chosen net of fees from `predictedFundings`, and parents above 2% of one-minute volume are sliced passive-first. | `core/fee_tiers.py`, `ui/api_server.py`, `tests/test_fee_tiers.py` |

**Phase 4 gate: not met as worded.** On the recording the median markout is the same with and without the lead pull at every horizon from 1 s to 30 s (BTC +0.06 bp, ETH +0.22 bp, SOL +0.05 bp), because one-tick spreads pin the middle fill at half a tick. The mean 30 s markout improves on every coin (BTC −0.21 to −0.07 bp, ETH −0.49 to −0.08 bp, SOL −0.19 to +0.19 bp), as does the 25th percentile (BTC −1.23 to −0.97, ETH −1.55 to −0.67, SOL −2.40 to −1.28 bp). The median is the wrong statistic for this gate at one-tick spreads; the mean and lower tail show the improvement. The recording is 25 minutes; no paper replay of the live loop has fills yet. No reject storms were observed, but the paper book does not exercise a live order gateway.

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

### Phase 5 implementation status (2026-10-31)

| Task | Status | Where |
|------|--------|-------|
| P5.1 | Workflow built; **no sleeve has completed it**. `research/validation/shadow.py` pre-registers kill thresholds from the CPCV path distribution (live Sharpe below 80% of the median path, drawdown beyond the 5th-percentile path) and reconciles realised against modelled slippage and markouts per shadow trade. Unvalidated or newly enabled sleeves (the derivatives sleeve with allocation off, the trend sleeve with shadow on, now the default) trade in a separate shadow book that never reaches the ledger. Shadow statistics are in the sleeve telemetry. No Phase 2 sleeve passed its gates. The first entrant is the P2.3 challenger meta-labeller (`meta_labeller_challenger`), which starts its 200-trade shadow run with the next live session; no kill thresholds are pre-registered for it yet, because the meta-labeller has no CPCV path distribution of trade returns, so its shadow book reports statistics without a kill rule until they are. | `research/validation/shadow.py`, `ui/api_server.py` |
| P5.2 | Done. `research/decay_monitor.py` computes rolling sign-aligned IC and half-lives per live feature and recommends a cadence equal to the shortest half-life. A breach triggers an early retrain under the P0.7 rules; a test forces one. Half-lives and breaches are in the sleeve telemetry. | `research/decay_monitor.py`, `ui/api_server.py`, `tests/test_decay_monitor.py` |
| P5.3 | Done; **acceptance passes narrowly**. `core/models/online_residual.py` fits a ridge on the meta-labeller's training residuals with exponential forgetting (0.999 per row), which can be warm-started without a refit. The store abstains (probability 0.5, gate closed) when the live Mahalanobis distance exceeds the 99th percentile of the training distances, and skips a retrain when more than 30% of the latest 96 rows are beyond the current gate. It is wired in the store's fit and predict path rather than inside the worker, so the worker's contract is unchanged. `research/residual_eval.py` on real BTC bars over CPCV: out-of-fold log-loss 0.23558 without and 0.23316 with the residual layer, lower in 60% of folds (the threshold), abstention rate 1.2%. | `core/models/online_residual.py`, `research/residual_eval.py`, `ui/api_server.py` |
| P5.4 | Done. `research/tournament_report.py` adds, per candidate, Monte Carlo trade-order shuffling (500 seeded permutations of per-trade net returns, 5th/50th/95th-percentile maximum drawdown), MAE/MFE by positioning regime at entry, transfer coefficient, turnover and square-root impact cost, and renders a text table. On the baseline candidates the realised maximum drawdown exceeds the shuffled 95th percentile for most of them, so their losses cluster in time more than chance would give. | `research/tournament_report.py`, `tests/test_tournament_report.py` |

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
