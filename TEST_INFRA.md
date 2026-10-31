# Project Phoenix Test Infrastructure (`TEST_INFRA.md`)

This document outlines the testing infrastructure, architectural design, execution environments, fixtures, and coverage mapping for Project Phoenix's end-to-end (E2E) test suite (`tests/test_e2e_phoenix.py`).

---

## 1. Executive Summary

Project Phoenix employs a rigorous, 4-tier opaque-box test framework ensuring institutional-grade reliability, mathematical precision, and fault tolerance across the entire multi-asset crypto trading architecture.

- **Primary Test Suite**: `tests/test_e2e_phoenix.py`
- **Total Test Cases**: 64 comprehensive tests
- **Pass Rate**: 100% (64 passed, 0 failures, 0 warnings)
- **Type Checking**: 100% compliance with `mypy --strict` (0 type errors)
- **Frameworks**: `pytest`, `pytest-asyncio`, `hypothesis`, `FastAPI TestClient`, `pyarrow`, `polars`, `mlx`

---

## 2. 4-Tier Testing Methodology

The test suite is structured into four distinct verification tiers in alignment with the testing specifications:

### Tier 1: Feature Coverage (25 Tests)
Exercises the primary happy paths and expected domain contracts for all core requirements:
- **R1: Market Data & Ingestion (5 tests)**
  - `test_r1_data_ingestion_multi_asset_sampling_tick_to_dollar_bars`: Verifies $500k VWAP dollar bar compilation on BTC, ETH, and SOL.
  - `test_r1_fast_fractional_differentiation_stationarity_memory`: Validates FFD stationarity (ADF p-value < 0.05) and memory retention (correlation > 0.85).
  - `test_r1_parquet_storage_schema_and_column_pruning`: Confirms 19-column Apache Parquet storage and partition pruning by date and instrument.
  - `test_r1_synthetic_order_book_bbo_l2_depth_generation`: Verifies L2 top-10 bid/ask depth, tight spreads, and cumulative volume.
  - `test_r1_high_throughput_arrow_zero_copy_streaming`: Confirms PyArrow Table conversions and zero-copy slicing for real-time telemetry.

- **R2: Trading Sleeves & ALO Execution (5 tests)**
  - `test_r2_basis_momentum_and_funding_rate_signals`: Validates 8h funding velocity and positive carry basis capture.
  - `test_r2_adaptive_liquidity_harvesting_maker_intent`: Verifies Add-Liquidity-Only (ALO) resting limit orders at bid/ask with zero taker fees.
  - `test_r2_toxic_flow_vpin_hawkes_sweep_cancellation`: Tests millisecond cancellation triggers when VPIN > 0.60 or Hawkes $\eta > 0.70$.
  - `test_r2_fractional_kelly_and_gross_leverage_bounds`: Confirms half-Kelly sizing capped at 3.0x gross leverage and single-asset caps of 0.40.
  - `test_r2_positioning_regime_classification_five_states`: Validates full classification across all 5 positioning regimes.

- **R3: Macro & Sentiment Multi-Agent Committee (5 tests)**
  - `test_r3_macro_stablecoin_sentinel_risk_scoring`: Validates ASRI scoring, depeg warnings, and tail-risk flags.
  - `test_r3_cross_sectional_breadth_analyst_evaluation`: Confirms PCA dispersion, market breadth ratios, and funding dispersion.
  - `test_r3_adversarial_red_team_critic_stress_veto`: Verifies stress scoring and emergency deleveraging recommendations.
  - `test_r3_risk_committee_consensus_reconciliation`: Tests unanimous and majority reconciliation into unified risk multipliers.
  - `test_r3_mlx_neural_scorer_apple_silicon_execution`: Verifies Apple Silicon Metal GPU tensor execution via MLX.

- **R4: Quant Research & Validation CPCV Gates (5 tests)**
  - `test_r4_fee_adjusted_triple_barrier_labeling`: Verifies upper profit, lower stop, and vertical time barrier labeling with taker fee drag.
  - `test_r4_two_stage_svd_lightgbm_meta_labeler`: Validates continuous SVD orthogonalization, Mahalanobis outlier gating, and LightGBM meta-probability.
  - `test_r4_combinatorial_purged_cross_validation_45_splits`: Confirms exactly $\binom{10}{2} = 45$ purged and embargoed evaluation splits.
  - `test_r4_probabilistic_sharpe_ratio_validation`: Validates PSR calculation under non-normal skewness and kurtosis.
  - `test_r4_strategy_tournament_champion_challenger_promotion`: Evaluates tournament ranking and production candidate promotion.

- **R5: Web UI & Operator Dashboard (5 tests)**
  - `test_r5_fastapi_rest_state_and_pnl_telemetry`: Verifies `/api/state` returns valid PnL, cash balance, and engine metrics.
  - `test_r5_plain_english_forensic_tooltips_registry`: Validates plain-English tooltip explanations for complex quantitative metrics.
  - `test_r5_real_time_alert_log_event_stream`: Verifies high-priority alert generation for circuit breakers and tail-risk events.
  - `test_r5_interactive_parameter_override_circuit_breaker`: Tests runtime overrides (`/api/control/halt`, `/api/control/resume`).
  - `test_r5_html_dashboard_asset_serving`: Verifies dashboard frontend HTML and UI component rendering.

---

### Tier 2: Boundary & Corner Cases (25 Tests)
Stress-tests boundary conditions, edge cases, and unexpected inputs:
- **R1 Boundaries**: Zero-volume bars, millisecond clock jumps / out-of-order ticks, empty order books, extreme price spikes, zero-trade FFD sequences.
- **R2 Boundaries**: Extreme VPIN boundary transitions ($VPIN = 0.599 \to 0.601$), zero/negative portfolio equity handling, single-asset allocation clamping, extreme funding rates, zero spread execution.
- **R3 Boundaries**: Complete stablecoin depeg crash scenario, zero-variance flat price breadth, extreme $100M liquidation cascade, conflicting agent split decisions, MLX high-dimensional tensor stress.
- **R4 Boundaries**: Zero-volatility flat price triple-barrier labeling, high-dimension collinear feature SVD orthogonalization, small-sample CPCV sample boundaries, extreme kurtosis Deflated Sharpe Ratio, tournament rejection of zero-alpha strategies.
- **R5 Boundaries**: Invalid HTTP override payloads, rapid multi-state WebSocket updates, empty alert log state, large JSON payload serialization, missing state persistence recovery.

---

### Tier 3: Cross-Feature Combinations (8 Pairwise Tests)
Evaluates interactions across subsystem boundaries:
- **R1 + R2**: Live market data feed directly generating dollar bars, feeding positioning signals, and issuing ALO orders.
- **R2 + R3**: High adversarial stress / stablecoin depeg dynamically suppressing trading sleeve leverage multiplier to 0.0.
- **R2 + R4**: Purged CPCV candidate strategy providing signals directly executed through defensive ALO maker logic.
- **R3 + R5**: Risk committee tail-risk alert broadcasting immediately through UI REST and event logging streams.
- **R4 + R5**: CPCV strategy tournament results exposed to dashboard API and displayed with plain-English tooltips.
- **R1 + R4**: Synthetic dollar bar stream labeled with fee-adjusted triple barriers and validated through CPCV splits.
- **R1 + R3**: Synthetic order book sweeps driving VPIN spikes that trigger Red Team Critic stress vetoes.
- **R2 + R5**: ALO trade execution updating persistent ledger and attaching forensics reasons to the trade log.

---

### Tier 4: Real-World Scenarios (6 Realistic Workflows)
Simulates realistic, multi-step market regimes and operational conditions:
- **Scenario 1: End-to-End Baseline Regime Trading Workflow**
  Full pipeline execution: live data ingestion $\to$ feature computation $\to$ meta-labeling $\to$ consensus risk check $\to$ ALO execution $\to$ REST state updates.
- **Scenario 2: Market Flash Crash + USDT Depeg Crisis Response**
  Simulates a sudden 750 bps USDT depeg and market plunge, verifying tail-risk flags, leverage scaling to 0.0, and toxic order cancellations.
- **Scenario 3: Server Restart Durability Across Active Trading Sessions**
  Verifies ledger state preservation ($5,000 capital, trade history, realized PnL) across unexpected server halts and restarts.
- **Scenario 4: CPCV Strategy Tournament Lifecycle & Promotion**
  Executes a full 45-split tournament between Trend, Carry, Mean-Reversion, and Breakout candidates, validating PSR, DSR, PBO, and gatekeeper promotion.
- **Scenario 5: Operator Emergency Kill-Switch & Systematic Resume**
  Tests emergency operator intervention (`POST /api/control/halt`), zeroing leverage and order rejections, followed by state recovery on resume.
- **Scenario 6: Toxic Order Flow VPIN & Hawkes Sweep Protection**
  Simulates institutional aggressive sweeping across the book, validating immediate ALO cancellation and toxic shield defense.

---

## 3. Environment & Hardware Prerequisites

The test infrastructure is designed for high-performance execution on macOS Apple Silicon and POSIX environments:

| Component | Specification | Purpose in Test Infrastructure |
|---|---|---|
| **Python** | 3.12+ | Core runtime environment |
| **Metal GPU / MLX** | Apple Silicon M1 Max (`mlx.core.default_device()`) | Neural risk scoring & committee inference |
| **Numba** | JIT with `@njit(fastmath=True)` | Ultra-fast FFD weight application & microstructure calculations |
| **PyArrow** | 19.0.1+ | Parquet storage, column pruning, zero-copy serialization |
| **Polars** | 1.27.1+ | High-performance columnar bar processing |
| **FastAPI** | `TestClient` (`httpx`) | In-process HTTP REST API and state testing without live networking |
| **Pytest** | 9.1.1+ (`pytest-asyncio`) | Test execution, parameterization, and lifecycle fixtures |
| **Mypy** | Strict Mode (`--strict`) | Static typing enforcement across all test definitions |

---

## 4. Test Execution Instructions

### Running the Full Phoenix E2E Test Suite
```bash
# Run all 64 E2E tests with verbose output
.venv/bin/pytest tests/test_e2e_phoenix.py -v
```

### Running All Project Tests
```bash
# Run both architecture verification and Phoenix E2E tests (74 tests total)
.venv/bin/pytest tests/ -v
```

### Running Specific Tiers
```bash
# Tier 1: Feature Coverage (25 tests)
.venv/bin/pytest tests/test_e2e_phoenix.py -k "TestTier1" -v

# Tier 2: Boundary & Corner Cases (25 tests)
.venv/bin/pytest tests/test_e2e_phoenix.py -k "TestTier2" -v

# Tier 3: Cross-Feature Combinations (8 tests)
.venv/bin/pytest tests/test_e2e_phoenix.py -k "TestTier3" -v

# Tier 4: Real-World Scenarios (6 tests)
.venv/bin/pytest tests/test_e2e_phoenix.py -k "TestTier4" -v
```

### Running Strict Type Checking
```bash
.venv/bin/mypy tests/test_e2e_phoenix.py --strict
```

---

## 5. Isolation, Fixtures, and Non-Interference

1. **Self-Contained File System Operations**:
   All disk operations (parquet writes, JSON ledger persistence, database creation) use pytest's `tmp_path` fixture. No persistent artifacts pollute the working directory or git tree.
2. **Deterministic Seed Control**:
   Stochastic processes, synthetic price generations, and neural model initializations utilize fixed seeds (`np.random.default_rng(42)`) to ensure 100% reproducibility.
3. **No Network Dependencies**:
   All external systems (Hyperliquid L1/L2 feeds, DefiLlama, Deribit DVOL, Binance) are exercised using in-memory mock engines and synchronous test clients. The test suite runs entirely offline and hermetically.
4. **Zero Facade Testing**:
   Every test exercises actual mathematical implementations:
   - SVD orthogonalization computes real eigenvalues and condition numbers.
   - Triple-barrier labeling evaluates real causal volatility horizons.
   - CPCV generates real combinatorial split indices.
   - Defensive maker logic executes real price backoff algorithms.

---

## 6. Traceability Matrix (Requirements to Tests)

| Requirement | Description | Tier 1 Tests | Tier 2 Tests | Tier 3 Tests | Tier 4 Scenarios |
|---|---|---|---|---|---|
| **R1** | Market Data & Ingestion | 5 tests | 5 tests | 3 tests | Scenario 1, 6 |
| **R2** | Trading Sleeves & ALO Execution | 5 tests | 5 tests | 4 tests | Scenario 1, 2, 5, 6 |
| **R3** | Macro & Sentiment Multi-Agent | 5 tests | 5 tests | 3 tests | Scenario 1, 2 |
| **R4** | Quant Research & CPCV Gates | 5 tests | 5 tests | 3 tests | Scenario 1, 4 |
| **R5** | Web UI & Operator Dashboard | 5 tests | 5 tests | 3 tests | Scenario 1, 3, 5 |
