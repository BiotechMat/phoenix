# Project Phoenix: Institutional Crypto-Derivatives Architecture for Hyperliquid L1

## Architecture Overview
Project Phoenix is an institutional-grade algorithmic trading system engineered for Apple Silicon M1 Max (64GB Unified Memory) operating 100% on-chain on the Hyperliquid L1 decentralized perpetual DEX. The architecture connects sub-second market data ingestion, quantitative trading sleeves (delta-neutral funding carry and derivatives alpha), a 3-model Metal GPU-accelerated MLX risk committee, a 45-split CPCV research validation tournament, and a persistent paper-trading ledger ($5,000 base capital) with trade forensics ("Why Did It Trade?") on a FastAPI/Uvicorn web dashboard at `http://127.0.0.1:8080`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Real-time Hyperliquid L1 Ingestion | Simultaneous BTC, ETH, SOL mid, mark, oracle, OI, 1h funding rates via `metaAndAssetCtxs` | M1 | R1 |
| 2 | Hyperliquid L2 Order Book Depth | Top-20 level L2 order book streaming via `l2Book` endpoint | M1 | R1 |
| 3 | Bybit V5 Failover Ingestion | Multi-asset failover to Bybit V5 linear ticker when Hyperliquid is degraded | M1 | R1 |
| 4 | Simulated Paper Execution on Real Data | Live order matching against real-time market data in `ui/api_server.py` | M1 | R1 |
| 5 | Hyperliquid 1-Hour Funding Carry | Delta-neutral carry sleeve capturing 1h settlements and annualizing APR correctly | M2 | R2 |
| 6 | Cross-Venue Funding Spread Capture | Arbitrage funding rate differentials between Hyperliquid and Bybit/OKX | M2 | R2 |
| 7 | ALO Post-Only Maker Routing | Add-Liquidity-Only execution avoiding taker fees and securing 1.5 bps maker rebate/fee | M2 | R2 |
| 8 | Hyperliquid Fee Schedule Definition | Standardized 1.5 bps Maker / 4.5 bps Taker fee schedule in `core/contracts.py` | M2 | R2 |
| 9 | Deribit Options Surface Metrics | Ingestion of 25-delta skew, term structure slope, and dealer net GEX | M2 | R2 |
| 10 | Hyperliquid Whale & Vault Positioning | HLP vault (`0xdfc24b...`) tracking, OI skew, and counterparty positioning metrics | M2 | R2 |
| 11 | Derivatives Alpha Sleeve Signals | Alpha sleeve emitting directional signals ($y_{\text{primary}}$) and meta-probabilities ($p_{\text{meta}}$) | M2 | R2 |
| 12 | Apple Silicon MLX GPU Acceleration | Metal GPU acceleration on M1 Max Unified Memory (`Device(gpu, 0)`) | M3 | R3 |
| 13 | Macro / Stablecoin Sentinel Model | First risk committee model evaluating macro risk, peg stability, and systemic stress | M3 | R3 |
| 14 | Cross-Sectional Breadth Analyst Model | Second risk committee model evaluating multi-asset momentum, dispersion, and correlations | M3 | R3 |
| 15 | Adversarial Red-Team Critic Model | Third risk committee model challenging portfolio allocations and detecting liquidation cascades | M3 | R3 |
| 16 | Risk Committee Consensus Engine | Tri-model consensus mechanism computing unified portfolio regime multiplier $M_{\text{regime}}$ | M3 | R3 |
| 17 | Multi-Asset Historical Parquet Backfill | Storage in `data/` for historical 15m bars across BTC, ETH, and SOL with 18 columns | M4 | R4 |
| 18 | 45-Split CPCV Gatekeeper | Combinatorial Purged Cross-Validation with 9 paths, zero label overlap, and purging/embargo | M4 | R4 |
| 19 | Deflated Sharpe Ratio & PBO Verification | DSR calculation penalizing multi-trial selection bias and estimating PBO | M4 | R4 |
| 20 | Automated Strategy Tournament Runner | Backtest tournament evaluating candidate models across 45 splits prior to promotion | M4 | R4 |
| 21 | Triple Barrier Meta-Modeling Bridge | Integration of triple-barrier labeling, SVD filtering, and LightGBM meta-labeling | M4 | R4 |
| 22 | Persistent Account Ledger (`account_ledger.json`) | Disk persistence preserving $5,000 base capital, accumulated PnL, trades, and win/loss across restarts | M5 | R5 |
| 23 | Atomic Ledger State Management | Atomic read/write operations in `ControlCenterStateStore` to eliminate state corruption | M5 | R5 |
| 24 | Feature Attribution & Trade Forensics ("Why Did It Trade?") | Explanatory metrics (regime, meta-prob, Mahalanobis distance, VPIN/Hawkes, Kelly size) | M5 | R5 |
| 25 | UI Telemetry & Forensics Dashboard | Web interface on `http://127.0.0.1:8080` displaying multi-asset feeds, sleeves, risk, forensics | M5 | R5 |
| 26 | Architecture Verification Test Suite | 0 failures in `pytest tests/test_architecture_verification.py` verifying all requirements | M6 | AC |
| 27 | Clean Static Type Analysis | 0 errors in `mypy core sleeves risk sentiment research ui tests --strict` | M6 | AC |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Live Multi-Asset Market Data Ingestion | `core/data_ingestion.py`, `ui/api_server.py` | none | DONE |
| M2 | Quantitative Trading Sleeves | `sleeves/sleeve_3_funding_carry.py`, `sleeves/sleeve_1_derivatives_alpha.py`, `core/execution/defensive_maker.py`, `core/contracts.py` | M1 contracts | DONE |
| M3 | Apple Silicon (MLX) Risk Committee | `sentiment/mlx_risk_daemon.py`, `risk/` | none | DONE |
| M4 | Parquet Backfill & CPCV Tournament | `data/`, `research/cpcv_gates.py`, `research/triple_barrier_meta_model.py` | none | DONE |
| M5 | Persistent Ledger & UI Forensics Dashboard | `data/account_ledger.json`, `ui/api_server.py`, `ui/dashboard_app.py` | M1, M2, M3 | DONE |
| M6 | Final Acceptance & E2E Test Suite | `tests/`, full architecture verification & mypy strict pass | M1, M2, M3, M4, M5 | DONE |

## Interface Contracts

### M1 Ingestion ↔ System Consumers
- `MultiAssetMarketSnapshot`:
  - `timestamp_ns: int`
  - `assets: dict[str, AssetMarketSnapshot]` where assets include "BTC", "ETH", "SOL"
  - Each `AssetMarketSnapshot` contains: `mid_price`, `mark_price`, `oracle_price`, `open_interest`, `funding_rate_1h`, `funding_rate_8h_equiv`, `day_volume_usd`, `l2_book: L2OrderBookSnapshot`
- `L2OrderBookSnapshot`:
  - `bids: list[tuple[float, float]]` (price, size)
  - `asks: list[tuple[float, float]]` (price, size)
  - `spread_bps: float`
  - `depth_10bps_usd: float`

### M2 Sleeves ↔ Risk Allocator
- `SleeveSignalOutput`:
  - `sleeve_id: str` ("funding_carry", "derivatives_alpha")
  - `target_asset: str` ("BTC", "ETH", "SOL")
  - `direction: int` (-1, 0, 1)
  - `meta_probability: float` (0.50 to 1.00)
  - `raw_expected_return: float`
  - `recommended_leverage: float`
  - `routing_tag: str` ("ALO_POST_ONLY")
  - `forensic_metadata: dict[str, Any]`

### M3 MLX Risk Committee ↔ Portfolio Allocator & UI
- `RiskCommitteeOutput`:
  - `macro_sentinel_score: float` (0 to 100)
  - `breadth_analyst_score: float` (0 to 100)
  - `red_team_critic_score: float` (0 to 100)
  - `consensus_risk_score: float` (0 to 100)
  - `regime_multiplier: float` ($M_{\text{regime}} \in [0.0, 1.0]$)
  - `tail_risk_flag: bool`
  - `committee_rationale: str`

### M5 Ledger State ↔ API Server
- `AccountLedgerState`:
  - `base_capital_usd: float = 5000.0`
  - `current_cash_usd: float`
  - `total_realized_pnl: float`
  - `win_count: int`
  - `loss_count: int`
  - `trade_history: list[TradeRecord]`
  - `last_updated_iso: str`

## Code Layout
- `core/`:
  - `data_ingestion.py`: Real-time ingestion engine (Hyperliquid primary, Bybit fallback)
  - `contracts.py`: Dataclasses, schemas, FeeSchedule.hyperliquid()
  - `execution/defensive_maker.py`: ALO post-only maker routing
- `sleeves/`:
  - `sleeve_3_funding_carry.py`: 1h settlement delta-neutral carry
  - `sleeve_1_derivatives_alpha.py`: Deribit options surface + Hyperliquid vault metrics
  - `derivatives/funding_carry.py`: Backwards-compatibility alias to `sleeve_3_funding_carry.py`
- `sentiment/`:
  - `mlx_risk_daemon.py`: Metal GPU 3-model risk committee daemon
- `research/`:
  - `cpcv_gates.py`: 45-split CPCV gate & tournament evaluator
  - `triple_barrier_meta_model.py`: Triple-barrier labeling & LightGBM meta-model bridge
- `data/`:
  - `account_ledger.json`: Persistent $5,000 account state
  - `btcusdt_perp_bars.parquet`, `ethusdt_perp_bars.parquet`, `solusdt_perp_bars.parquet`
- `ui/`:
  - `api_server.py`: FastAPI server streaming multi-asset telemetry, paper engine, ledger persistence
  - `dashboard_app.py`: Web dashboard with trade forensics and multi-model risk view
- `tests/`:
  - `test_architecture_verification.py`: Verification test suite
  - `test_e2e_phoenix.py`: Comprehensive E2E test suite (Tiers 1-4)
