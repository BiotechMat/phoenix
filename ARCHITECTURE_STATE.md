# ARCHITECTURE STATE — GEMINI ARGON 4 QUANTITATIVE DERIVATIVES ENGINE

**Target Hardware:** Apple MacBook Pro M1 Max (10-Core CPU [8 P-Cores, 2 E-Cores], 32-Core GPU, 64GB Unified Memory, 400GB/s Memory Bandwidth)
**Execution Core:** Native NautilusTrader (`LiveNode`, `BacktestNode`, `DataActor`, `Strategy`, `ExecutionAlgorithm`)
**Operating Horizon:** Medium-Frequency Dollar/Volume Bars (15m–4h horizons, 80–350 bps target moves) + Structural Delta-Neutral Funding Carry

---

## 1. Memory Partitioning & Thread Pinning (64GB Unified Memory)

| Subsystem Process | Memory Cap | Compute Hardware | Isolation & Threading |
| :--- | :--- | :--- | :--- |
| **OS Kernel, Network & Redis/Arrow IPC** | `10.0 GB` | Unwired RAM | System Managed |
| **NautilusTrader Rust Core & L1-L20 Book** | `8.0 GB` | Pinned CPU RAM | Single-threaded deterministic Rust event loop |
| **Out-of-Process LightGBM/XGBoost & Polars Worker** | `12.0 GB` | CPU P-Cores | `OMP_NUM_THREADS=4`, `POLARS_MAX_THREADS=4`, `n_jobs=4`, `max_workers=2` |
| **PyTorch / Metal Auxiliary Compute Pool** | `10.0 GB` | Metal MPS | `LOW_WATERMARK=0.5`, `HIGH_WATERMARK=0.7`, `FAST_MATH=1` |
| **Persistent `mlx-lm` Sentiment Daemon (`Qwen3-8B-4bit`)** | `10.0 GB` | Wired GPU Pool | Singleton `BatchGenerator` (`iogpu.wired_limit_mb=40960` ceiling) |
| **Free Safety Headroom** | `14.0 GB` | Unallocated | Prevents macOS swap compression |

*All process entry points enforce `multiprocessing.set_start_method("spawn", force=True)` prior to numerical library imports.*

---

## 2. Subsystem Modules & Contract Registry

1. **Core Contracts & Hardware Bootstrap (`core/contracts.py`, `core/hardware_bootstrap.py`)**
   - Strictly typed Pydantic v2 schemas for Dollar Bars, FFD Stationary State, Positioning Regimes, Fee-Adjusted Triple-Barrier Labels, Two-Stage Meta-Labels, Systemic Risk (`SystemicRiskOutput`), Execution State, CPCV Promotion Gates, and Signed `HRP-CVaR` Portfolio Allocation (`compute_signed_target_weights`).
2. **Medium-Frequency Derivatives Alpha & Stationary Features (`core/features/`, `sleeves/derivatives/`)**
   - `DerivativesFeatureActor`: Native NautilusTrader `DataActor` aggregating exchange ticks into Dynamic Dollar Bars (~15m, 1h, 4h).
   - Frozen Per-Split Fractional Differentiation (`FFD`) with $\tau = 10^{-5}$ weight truncation and in-sample ADF calibration ($p_{\text{ADF}} \le 0.01$).
   - Perpetual Futures Positioning & Squeeze Regime Matrix (`CROWDED_LONG_VULNERABILITY`, `CROWDED_SHORT_SQUEEZE`, `GENUINE_TREND_CONVICTION`, `CASCADE_EXHAUSTION_REVERSAL`).
   - `FundingCarrySleeve`: Delta-neutral spot-long / perpetual-short funding carry harvester.
3. **Fee-Adjusted Triple-Barrier & Two-Stage Meta-Labelling (`core/models/`)**
   - Minimum profit hurdle $\text{Barrier}_{\text{min}} = \max(k_{\text{pt}} \cdot \sigma_{\text{bar}}, 6 \times (\text{Fee}_{\text{roundtrip}} + \text{Slippage}_{\text{est}})) \ge 80\text{ bps}$.
   - Stage 1 Continuous-Only SVD (excluding sparse binary 99th-percentile Liquidation Exhaustion flag from SVD rotation).
   - Stage 2 Calibrated `LightGBM` (`n_jobs=4`) Meta-Model predicting $p_{\text{meta}, i} \in [0, 1]$ + Ledoit-Wolf Mahalanobis outlier gate (`DI_values`).
4. **Persistent `mlx-lm` Macro & Systemic Risk Daemon (`sentiment/mlx_risk_daemon.py`)**
   - Singleton `BatchGenerator` with `mlx-community/Qwen3-8B-4bit`.
   - Strict Pydantic v2 validation (`SystemicRiskOutput`) with deterministic neutral fallback (`score=0.0`, `confidence=0.0`, `tail_risk_flag=False`).
   - Emits top-down `ASRI` ($[0, 100]$) and portfolio risk multiplier $M_{\text{regime}} \in [0.0, 1.0]$.
5. **Defensive Maker-First Execution & $O(1)$ Hawkes Shield (`core/execution/`)**
   - `DefensiveMakerExecutionAlgorithm`: BVC VPIN (50-bucket volume clock) + `@numba.njit` $O(1)$ recursive Markovian Hawkes intensity update ($< 5\text{ ms}$ for 10,000 updates) with asynchronous 30-second L-BFGS-B parameter refit.
   - Post-Only (`GTX`) pegging with dual re-peg hysteresis ($\ge 1.5\text{ bps}$ BBO shift AND $\ge 500\text{ ms}$ cooldown) and 3-strike rejection backoff ($1\text{ tick}$ deeper + $1,000\text{ ms}$ wait).
   - Toxic Sweep Shield ($\text{VPIN} > 0.60$ or $\eta_{\text{Hawkes}} > 0.70$) cancels resting entry orders in $< 5\text{ ms}$.
6. **Net-of-Fee CPCV Gatekeeper, Fallback State Machine & Signed `HRP-CVaR` (`risk/`, `research/validation/`)**
   - 45-Split Combinatorial Purged Cross-Validation ($N=10, k=2$, `purge_bars = vertical_barrier_bars`, `1% embargo`).
   - Hard Promotion Gates: $\text{PSR} \ge 0.95$ (vs. $0$ and equal-weight benchmark), $\text{DSR} \ge 0.95$, $\text{PBO} < 0.30$, $\text{E}[\text{Net PnL}] > 15\text{ bps}$.
   - Champion/Challenger Fallback State Machine transitioning to 100% Cash / Delta-Neutral Carry if live 30d $\text{PSR} < 0.80$ or 3 consecutive retrains fail.
   - Governing Allocation Equation: $w_{\text{final}, i} = y_{\text{primary}, i} \cdot w_{\text{HRP}, i} \cdot f_{\text{Kelly}}(p_{\text{meta}, i}) \cdot M_{\text{regime}}$ with $2.5\text{x}$ gross leverage cap.
7. **Interactive Control Center, Telegram Bot & Layman's Documentation (`ui/`, `docs/`)**
   - FastAPI + Interactive HTML/Plotly Control Center (`http://localhost:8080`) with plain-English tooltips, one-click operational controls, Telegram bot commands, and 4 comprehensive layman's manuals in `docs/`.
