# Troubleshooting & Safety Guide: Fallbacks, 64GB M1 Max Memory Caps & Outage Recovery

This guide explains the built-in automatic safety nets of the **Gemini Argon 4** engine and gives you clear, step-by-step runbooks for handling model validation failures, macOS memory alerts, local AI timeouts, and exchange connection drops.

---

## 1. What Happens When an AI Model Fails Validation? (Automatic Fallback Protocol)

You never have to worry about a freshly retrained AI model going rogue after a bad training run. Every model is policed by the **Champion / Challenger Gatekeeper (`ChampionChallengerState`)** and the **45-Split CPCV Exam Board (`CPCVValidationResult`)**.

### Scenario A: A New "Challenger" Model Fails Any of the 5 Promotion Gates
Suppose the scheduled retraining job builds a new Challenger model, and across the 45 historical CPCV splits it scores `PSR = 0.91` (below the `0.95` minimum) or `PBO = 0.34` (above the `0.30` overfitting limit).
* **What the Bot Does Automatically:**
  1. `passed_all_gates` evaluates to `False`.
  2. The Challenger model is **quarantined and discarded**.
  3. The proven **Champion** model remains in charge without interruption, and `consecutive_failed_retrains` increments by `1`.
* **What You Need to Do:** **Nothing.** You can inspect the exact gate scores on **Tab 2 (Portfolio & Validation)** or by sending `/gates` on Telegram.

### Scenario B: 3 Consecutive Retrains Fail OR Live 30-Day Performance Drops
What if the market undergoes a structural shift where *even the active Champion* starts losing its edge?
* **Automatic Trigger Conditions:**
  - The active Champion's rolling 30-day live Probabilistic Sharpe Ratio drops below `0.80` (`champion_live_30d_psr < 0.80`), **OR**
  - Three consecutive retraining cycles fail to produce a passing model (`consecutive_failed_retrains >= 3`).
* **What the Bot Does Automatically:**
  1. The state machine immediately flips `regime_mode` from `DIRECTIONAL_PLUS_CARRY` to **`CARRY_ONLY_FALLBACK`**.
  2. All directional Long/Short predictions are zeroed out.
  3. Your capital is automatically parked in **100% Cash + Delta-Neutral Funding Carry (`FundingCarrySleeve`)**—collecting 8-hour funding payments with zero directional price risk while waiting for a new Challenger model to pass all five 45-split CPCV gates.
* **What You Need to Do:** Review the **Champion vs. Challenger Gate Chart** in Tab 2. The system will remain safely in `CARRY_ONLY_FALLBACK` until a newly trained model passes every statistical hurdle (`PSR >= 0.95`, `DSR >= 0.95`, `PBO < 0.30`, `Net Edge > 15 bps`).

---

## 2. How Memory Caps Work on Your 64GB Apple M1 Max

Your MacBook Pro M1 Max has **64 GB of Unified Memory** shared between the CPU cores and the 32-core GPU. In quantitative trading, the biggest hardware danger on macOS is **SSD Swap Compression**: if Python, Rust, PyTorch, and the local LLM jointly request 62 GB of RAM, macOS starts paging memory to disk, causing multi-second latency spikes.

To guarantee **zero swap lag**, `EnvironmentConfig` (`core/contracts.py`) and `core/hardware_bootstrap.py` enforce a strict **50.0 GB Active Memory Ceiling** and leave **14.0 GB of permanent free headroom**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                   APPLE M1 MAX 64.0 GB UNIFIED MEMORY                │
├───────────────────────────────────────────────────┬──────────────────┤
│             50.0 GB HARD ACTIVE BUDGET            │ 14.0 GB HEADROOM │
├──────────┬──────────┬──────────┬──────────┬───────┴──────────────────┤
│ OS & IPC │ Nautilus │ LightGBM │ PyTorch  │ MLX Qwen3-8B             │
│  10.0 GB │  8.0 GB  │ 12.0 GB  │ 10.0 GB  │   10.0 GB                │
└──────────┴──────────┴──────────┴──────────┴──────────────────────────┘
```

### Key Hardware Safeguards Built Into the Code
1. **GPU Wired Memory Ceiling (`wired_limit_mb = 40960`):** Prevents Apple Metal/MLX GPU tasks from ever locking more than 40 GB of unified RAM, ensuring macOS networking and the NautilusTrader Rust core always have breathing room.
2. **CPU Thread Pinning (`OMP_NUM_THREADS=4`, `POLARS_MAX_THREADS=4`, `lgbm_n_jobs=4`, `cpcv_max_workers=2`):** Keeps heavy AI training pinned to 4 Performance Cores so the live order-book event loop never starves for CPU cycles.
3. **Clean Process Spawning (`spawn` mode):** Every multi-process worker uses `multiprocessing.set_start_method("spawn", force=True)` before loading math libraries, preventing Apple Accelerate / OpenMP deadlocks.

### Troubleshooting Memory Pressure
* **Symptom:** Activity Monitor shows "Memory Pressure" in Yellow because external apps (e.g., 50 Chrome tabs, Docker Desktop, or Xcode) are consuming more than 14 GB of RAM alongside the trading engine.
* **Fix:**
  1. Close non-essential desktop applications (especially unused browser windows or virtual machines) to restore the 14 GB safety cushion.
  2. If you want to verify that the trading engine's internal budget validator is intact, run:
     ```bash
     python -c "from core.contracts import EnvironmentConfig; print(EnvironmentConfig(profile_name='prod_m1max_64gb', is_mock=False, wired_limit_mb=40960))"
     ```

---

## 3. What Happens If the Local Macro AI (`Qwen3-8B-4bit`) Times Out?

The local `mlx-lm` daemon (`sentiment/mlx_risk_daemon.py`) analyzes macro and systemic news on your GPU. What if a malformed news feed causes the LLM to output invalid JSON or take too long?
* **Automatic Safeguard:** Every output is validated through `SystemicRiskOutput` (`core/contracts.py`). If validation fails or times out, the daemon immediately invokes `SystemicRiskOutput.neutral_fallback()`, returning a deterministic safe state (`score=0.0`, `confidence=0.0`, `tail_risk_flag=False`) with a clear explanation on your dashboard banner (`"Macro risk monitor in neutral fallback mode: ..."`).
* **Result:** The trading engine never crashes or hangs waiting for the AI.

---

## 4. What to Do If an Exchange Connection Drops

Cryptocurrency exchanges occasionally drop WebSocket streams, experience cloudflare hiccups, or undergo scheduled maintenance. Here is how the bot protects you and what steps to follow:

### Built-In Automatic Exchange Protections
1. **Post-Only (`GTX`) Order Protection:** Because resting entry orders are posted as `GTX` (Post-Only Maker), they never cross the spread accidentally on reconnect.
2. **3-Strike Rejection Backoff (`max_gtx_rejections = 3`):** If the order book is jittering during an exchange micro-outage and rejects a `GTX` order 3 times in a row, `DefensiveMakerExecutionAlgorithm` automatically switches to `BACKOFF_DEEPER_GTX`—stepping 1 tick deeper into the book and pausing 1,000 ms (`rejection_backoff_ms = 1000`) instead of spamming the exchange API.
3. **Millisecond Toxic Shield (`CANCEL_ALL_TOXIC_SHIELD`):** When a connection resumes after a sudden volatility burst, if `VPIN > 0.60` or `Hawkes Eta > 0.70`, resting entry orders are canceled immediately.

### Step-by-Step Operator Runbook for Exchange Disconnections

* **Step 1 — Pause New Entries Immediately (< 5 Seconds):**
  - Click **⏸️ Pause Entries** on `http://localhost:8080` **OR** send `/pause` to the Telegram Bot.
  - This switches the engine to `PAUSED_ENTRIES`, stopping any new orders from being queued while the exchange connection stabilizes.
* **Step 2 — Check Exchange Status & Open Positions:**
  - Look at the top status bar on `http://localhost:8080` or send `/status` on Telegram.
  - If the exchange outage is minor (< 2 minutes) and reconnects cleanly, verify that `VPIN` is below `0.60` and `ASRI` is normal, then click **🟢 Start Dry-Run** (or resume active mode).
* **Step 3 — If the Exchange Outage Is Severe or Prolonged (> 5 Minutes):**
  - Click **🚨 Emergency Flatten All** on `http://localhost:8080` **OR** send `/flat` on Telegram.
  - This issues `EMERGENCY_TAKER_EXIT` across all active sleeves, closing open positions to 100% cash (`EMERGENCY_FLATTENED`).
  - Wait until the exchange's official status page reports 100% normal WebSocket and REST API latency for at least 15 minutes before resuming.
