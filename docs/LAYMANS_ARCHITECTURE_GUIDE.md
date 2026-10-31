# Layman's Architecture Guide: How the Gemini Argon 4 Trading Engine Works

Welcome to the **Gemini Argon 4 Quantitative Derivatives Engine** (`nautilus_quant_m1max`). You do not need a PhD in mathematics, physics, or quantitative finance to understand, operate, and supervise this system. This guide explains every moving part of the trading engine in plain, everyday English—using intuitive real-world analogies right alongside the actual code module names so you always know what is happening under the hood.

---

## 1. The Big Picture: Why This Bot Exists (and How It Makes Money Safely)

Most amateur trading bots fail for one simple reason: **they trade too fast and get eaten alive by exchange fees.** If a bot tries to predict where Bitcoin will be in 30 seconds, the tiny price move it captures (say, 0.03% or 3 basis points) is smaller than the toll the exchange charges to enter and exit the trade (often 0.04% to 0.11%).

Our system is built on the exact opposite philosophy: **Patience, Structural Imbalances, and Maker-First Execution.**

1. **Medium-Frequency Horizon (15 Minutes to 4 Hours):** Instead of chasing microsecond noise, the bot waits for large, multi-hour structural setups where crowded traders are trapped and forced to liquidate. It only targets moves of **0.80% to 3.50% (80 to 350 basis points)**—moves that are **10 to 30 times larger** than our trading fees.
2. **Two Engines Working Together (Directional Alpha + Delta-Neutral Carry):**
   - **Engine A — Directional Squeeze & Trend Alpha (`sleeves/derivatives/`):** Rides strong trends or profits when over-leveraged traders get squeezed out of their positions.
   - **Engine B — Delta-Neutral Funding Carry (`FundingCarrySleeve`):** Acts like a "crypto landlord." When bullish traders pay high interest ("funding rates") every 8 hours to hold leveraged futures, the bot buys the real coin on the Spot market and sells an equal amount of Perpetual Futures. Because the two positions cancel out price risk (if Bitcoin drops 10%, the futures gain what the spot loses), the bot simply collects the 8-hour interest checks with near-zero directional risk.

---

## 2. The Four Brains of the Bot

Think of the trading system as an investment firm run by **Four Specialized Brains** and guarded by **One Reflexive Bodyguard**. All five communicate through strict, tamper-proof digital contracts defined in `core/contracts.py`.

```
[Exchange Ticks & Order Book]
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ BRAIN 1: The Market Seismograph (core/features/ & derivatives/)   │
│ Groups raw trades into Dynamic Dollar Bars, cleans price history  │
│ (FFD), and detects Crowded Long/Short Squeeze Regimes.            │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ BRAIN 2: The Sniper Quality Inspector (core/models/)              │
│ Scores every setup with LightGBM (p_meta) after subtracting all   │
│ fees, and checks the Mahalanobis "Uncharted Territory" Alarm.     │
└───────────────────────────────────────────────────────────────────┘
        │
        ├───────────────────────────────────────────────────────────┐
        ▼                                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────┐
│ BRAIN 3: The Global Weather Forecaster│   │ BRAIN 4: The Treasurer & Exam │
│ (sentiment/mlx_risk_daemon.py)        │   │ Board (risk/ & validation/)   │
│ Local Qwen3-8B AI reads macro news    │──▶│ Runs 45-split CPCV exams and  │
│ & sets the ASRI Risk Dial (M_regime). │   │ sizes positions (HRP-CVaR).   │
└───────────────────────────────────────┘   └───────────────────────────────┘
                                                            │
                                                            ▼
                                            ┌───────────────────────────────┐
                                            │ THE BODYGUARD: Maker Shield   │
                                            │ (core/execution/)             │
                                            │ Places low-fee Post-Only GTX  │
                                            │ orders & pulls them in <5ms   │
                                            │ if VPIN/Hawkes detects sharks.│
                                            └───────────────────────────────┘
```

---

### Brain 1: The Market Seismograph & Crowd-Detector (`core/features/` & `sleeves/derivatives/`)

#### A. Dynamic Dollar Bars (`DollarBar` in `DerivativesFeatureActor`)
* **The Problem with Clocks:** Ordinary charts draw a new bar every 15 minutes on the wall clock. At 3:00 AM on a Sunday, nothing happens for 15 minutes; during a Federal Reserve announcement, $500 million trades in two minutes. Time-based charts stretch out dead periods and compress crashes.
* **Our Solution (Analogy: A Rain Gauge):** Instead of checking a bucket every 15 minutes regardless of weather, our `DerivativesFeatureActor` uses a **Dollar Volume Gauge**. It only draws a new bar when a fixed amount of real money (e.g., $25 million of BTC) has changed hands. Quiet markets produce fewer bars; chaotic markets produce more bars, giving the AI high-resolution vision right when it matters most.

#### B. Fractional Differentiation (`FFDWeightsConfig` / `ffd_stationarity`)
* **The Problem with Raw Prices:** Bitcoin was $10,000 a few years ago and $90,000 later. If an AI trains on raw dollar prices, it gets confused when prices reach new levels. Standard math fixes this by looking only at 1-bar percentage changes—but that throws away all memory of long-term support, resistance, and trend exhaustion.
* **Our Solution (Analogy: Tinted Sunglasses):** **Fractional Differentiation (`FFD`)** is like polarized sunglasses for price charts. It removes the blinding glare of the absolute price level (making the data statistically "stationary" so $90,000 looks mathematically comparable to $30,000) while preserving up to 90% of the long-term trend memory. Crucially, the FFD weights (`d_star`, truncated at $\tau = 10^{-5}$) are frozen per training split so future information never leaks into past tests.

#### C. The Four Perpetual Positioning & Squeeze Regimes (`PositioningRegime`)
In crypto perpetual futures, we can see three things that stock traders rarely see in real time: **Open Interest** (total active leveraged bets), **Funding Rates** (who is paying whom every 8 hours), and **Liquidation Streams** (forced margin calls). Brain 1 combines these into four plain-English regimes:
1. `CROWDED_LONG_VULNERABILITY` *(Too Many Bulls on Margin)*: Everyone is borrowing money to bet prices go up, paying sky-high funding fees, while price stalls. Like a crowded theater with a narrow exit, a small dip can trigger a chain-reaction crash ("long squeeze").
2. `CROWDED_SHORT_SQUEEZE` *(Bears Trapped in a Corner)*: Traders are heavily betting against the coin while smart money quietly buys Spot. If the price ticks up, short-sellers are forced to buy back at any price, rocketing the coin higher.
3. `GENUINE_TREND_CONVICTION` *(Healthy, Sustainable Trend)*: Price, Spot buying, and Open Interest are rising together without extreme funding distortion. Safe to ride the trend.
4. `CASCADE_EXHAUSTION_REVERSAL` *(Margin-Call Washout Complete)*: A massive wave of forced liquidations (`liquidation_exhaustion_flag = 1`, top 99th percentile spike) has just wiped out over-leveraged traders, leaving the market washed out and primed to snap back.

---

### Brain 2: The Sniper Quality Inspector (`core/models/`)

Brain 1 spots *which direction* a potential trade points (`primary_direction`: `+1` Long, `-1` Short, or `0` Neutral). Brain 2's job is to ask: **"After paying all exchange fees, slippage, and funding costs, what is the probability this specific trade actually wins?"**

#### A. Fee-Adjusted Triple-Barrier Labelling (`TripleBarrierConfig` & `TripleBarrierLabel`)
Imagine putting a trade inside a **Three-Walled Box**:
1. **Top Wall (Profit Target):** Set at $2.0\times$ recent volatility, but **never less than 0.80% (80 bps)** and never less than $6\times$ our round-trip fees plus slippage.
2. **Bottom Wall (Stop-Loss):** Set at $1.5\times$ recent volatility to cut losers quickly.
3. **Right Wall (Time Limit / Vertical Barrier):** Set at 16 bars. If the trade meanders sideways and hits the time limit without reaching the profit goal, the trade expires.

When training the AI, a historical trade is only marked a "True Winner" (`meta_label = 1`) if it hits the profit target **net of all Maker/Taker fees and funding payments**.

#### B. Two-Stage Meta-Labelling (`MetaPredictionOutput` / `p_meta`)
* **Stage 1 (Continuous SVD Compression):** Cleans and compresses continuous market indicators into independent factors so correlated signals don't double-count, while keeping the rare `liquidation_exhaustion_flag` separate and crisp.
* **Stage 2 (Calibrated LightGBM Meta-Model):** Outputs a single **Meta-Probability (`p_meta`)** between `0.00` (0%) and `1.00` (100%).
  - If `p_meta <= 0.50`: The AI says **"Pass."** Our position size is automatically zero.
  - If `p_meta = 0.75`: High confidence; the bot allocates a meaningful position.

#### C. The "Uncharted Territory" Alarm: Mahalanobis Distance (`mahalanobis_di`)
Machine learning models are great at patterns they have seen before, and terrible when something completely unprecedented happens (like an exchange hack or stablecoin de-peg).
* **Analogy:** Think of **Mahalanobis Distance (`DI_values`)** as a GPS fence around "normal market behavior." It measures how far today's combination of volatility, funding, and order flow is from everything in the training history.
* If `mahalanobis_di > di_threshold`, `epistemic_gate_passed` flips to `False` and **blocks the trade immediately**—even if the AI claims 95% confidence. In plain English: *"I have never seen a market shaped like this before, so I am keeping my hands in my pockets."*

---

### Brain 3: The Global Weather Forecaster (`sentiment/mlx_risk_daemon.py`)

Charts and order books cannot read breaking news headlines about an impending regulatory lawsuit, an exchange halting withdrawals, or a major stablecoin losing its $1.00 peg.

* **Local On-Device AI (`mlx-community/Qwen3-8B-4bit`):** Running natively on your M1 Max's 32-core Apple Silicon GPU inside a locked `10 GB` memory pool, this local AI reads macro/crypto risk feeds and outputs a strictly validated report (`SystemicRiskOutput`):
  1. **ASRI Score (`score` from `0` to `100`):** The **Aggregated Systemic Risk Index**—think of it as a "Crypto Hurricane Category Scale."
     - `0 to 40` (Clear Skies): Full normal trading (`M_regime = 1.0`).
     - `40 to 70` (Gathering Clouds): Automatically dials down all position sizes linearly from `100%` down to `50%`.
     - `70 to 95` (Severe Storm): Slashes position sizes from `50%` down to `0%`.
     - `95 to 100` or `tail_risk_flag = True` (Emergency Shelter): Immediately reduces directional risk multiplier `M_regime` to `0.0` (100% halt on directional bets).
  2. **Plain-English Summary (`plain_english_summary`):** Displayed right at the top of your dashboard so you can read in one sentence why the AI raised or lowered the risk dial.
  3. **Fail-Safe Neutral Fallback (`SystemicRiskOutput.neutral_fallback`):** If the AI ever takes too long or outputs malformed text, it never crashes the bot—it falls back deterministically to a safe baseline and logs the reason.

---

### Brain 4: The Portfolio Treasurer & Exam Board (`risk/` & `research/validation/`)

#### A. The 45-Split Exam Board: CPCV, PSR, DSR, and PBO (`CPCVValidationResult`)
Before any newly trained AI model ("Challenger") is allowed to touch a single dollar of your capital, it must beat the currently active model ("Champion") across **45 different historical time-travel tests** called **Combinatorial Purged Cross-Validation (`CPCV`)**.
To prevent the AI from "cheating" by memorizing past data, we delete (`purge_bars`) the bars right around test boundaries and add a `1% embargo` buffer. A model only passes (`passed_all_gates == True`) if it clears **all five hard hurdles simultaneously**:
1. **Probabilistic Sharpe Ratio (`psr_vs_zero >= 0.95`):** At least 95% statistical certainty that the model's risk-adjusted profit is genuinely above zero.
2. **Benchmark PSR (`psr_vs_benchmark >= 0.95`):** At least 95% certainty that it beats a simple equal-weight baseline.
3. **Deflated Sharpe Ratio (`dsr >= 0.95`):** Adjusts for "lucky guesses" across multiple trials (`total_trials`). Even after penalizing for how many models we tested, certainty must remain $\ge 95\%$.
4. **Probability of Backtest Overfitting (`pbo < 0.30`):** Less than a 30% chance that the model only looked good in backtesting by accident.
5. **Net Profit Expectancy (`expected_net_pnl_bps > 15.0`):** Must average more than 15 bps (0.15%) profit per trade **after all fees**.

#### B. Automatic Fallback to Cash / Delta-Neutral Carry (`ChampionChallengerState`)
What if the market enters a weird choppy phase and the new Challenger model fails the 45-split exam?
* **Rule 1:** The bot refuses to promote the failed Challenger and keeps running the proven Champion.
* **Rule 2:** If the live Champion's 30-day rolling score drops below `0.80` (`champion_live_30d_psr < 0.80`) OR 3 retraining attempts fail in a row (`consecutive_failed_retrains >= 3`), the bot **automatically shifts into `CARRY_ONLY_FALLBACK` mode**—turning off directional bets and parking capital safely in Cash and Delta-Neutral Funding Carry until a model passes every gate again.

#### C. The Governing Sizing Equation: Signed `HRP-CVaR` & Fractional Kelly (`compute_signed_target_weights`)
Once a trade passes every check, how much money goes into it? The Treasurer uses one master formula:

$$w_{\text{final}, i} = y_{\text{primary}, i} \times w_{\text{HRP}, i} \times f_{\text{Kelly}}(p_{\text{meta}, i}) \times M_{\text{regime}}$$

In plain English, every position size is the product of four common-sense numbers:
1. **Direction (`y_primary`):** `+1` for Long, `-1` for Short, `0` for Flat.
2. **Diversified Base Weight (`w_HRP` from `HRP-CVaR`):** Instead of putting all eggs in one basket (or splitting equally between correlated coins that crash together), **Hierarchical Risk Parity with Conditional Value-at-Risk (`HRP-CVaR`)** groups coins by how they move during crashes and gives smaller slices to volatile, crash-prone coins.
3. **Conviction Scalar (`f_Kelly`):** Scales from `0%` (when `p_meta <= 0.50`) up to a conservative Half-Kelly ceiling based on how confident Brain 2 is.
4. **Macro Weather Dial (`M_regime`):** Scales the entire portfolio down if Brain 3 detects macro danger.
5. **Hard Leverage Ceiling (`max_gross_leverage = 2.5x`):** Across all coins combined, total exposure is mathematically capped so it can never exceed `2.5x` your account equity.

---

## 3. The Reflexive Bodyguard: Defensive Maker Execution & Hawkes Shield (`core/execution/`)

Even a brilliant trade idea loses money if you execute it clumsily. When you click "Buy Market" (a **Taker** order), you cross the spread and pay high exchange fees (`5.5 bps` on USDT perps). Instead, `DefensiveMakerExecutionAlgorithm` uses **Post-Only Maker (`GTX`)** limit orders (`2.0 bps` on USDT perps, or `0.0 bps` on USDC perps)—cutting fee drag by more than 60%.

Wait—what if a whale starts dumping $100 million on the market while our limit buy order is sitting on the order book? You don't want to be the person catching a falling knife! That is why our Bodyguard runs two millisecond-level shields:

1. **VPIN Toxicity Meter (`vpin` — Volume-Synchronized Probability of Informed Trading):**
   * **Analogy:** Measures whether order flow is balanced two-way retail trading (`vpin < 0.35`, safe green water) or one-sided institutional dumping/sweeping (`vpin > 0.60`, shark-infested water).
2. **$O(1)$ Markovian Hawkes Reflexivity (`branching_ratio_eta`):**
   * **Analogy:** Measures whether trades are triggering *more* trades in a self-reinforcing avalanche (like snowballs starting an avalanche). Updated in **under 5 milliseconds** using compiled `@numba.njit` math.
3. **Instant Toxic Sweep Shield (`CANCEL_ALL_TOXIC_SHIELD`):**
   * The instant `vpin > 0.60` OR `branching_ratio_eta > 0.70`, `toxic_sweep_active` turns `True` and the Bodyguard **pulls all resting entry orders in under 5 milliseconds**—stepping out of the way until the avalanche passes.
4. **Anti-Spam Hysteresis (`HOLD_RESTING_HYSTERESIS` & `BACKOFF_DEEPER_GTX`):**
   * Exchanges penalize bots that cancel and replace orders every millisecond. Our Bodyguard only moves a resting order if the best price shifts by at least **1.5 bps** AND at least **500 milliseconds** have passed (`repeg_min_bps = 1.5`, `repeg_cooldown_ms = 500`). If the exchange rejects a Post-Only order 3 times in a row because the price is moving fast (`max_gtx_rejections = 3`), it steps back 1 tick deeper into the book and waits 1 full second (`1,000 ms`).

---

## 4. Hardware Harmony: How Everything Fits Inside Your 64GB M1 Max

Your Apple M1 Max has **64 GB of Unified Memory** shared between the CPU and the 32-core GPU. If five heavy programs grab memory at the same time without rules, macOS starts compressing memory onto the SSD ("swap"), causing micro-freezes.

`core/hardware_bootstrap.py` and `EnvironmentConfig` enforce a strict **50 GB active ceiling** leaving **14 GB of untouched safety headroom**:

| Component | Locked Memory Budget | What It Does |
| :--- | :--- | :--- |
| **macOS Kernel, Network & Redis/Arrow IPC** | `10.0 GB` | Fast shared-memory messaging between processes |
| **NautilusTrader Rust Core & Order Books** | `8.0 GB` | Ultra-fast live market data & order execution |
| **LightGBM / XGBoost & Polars Data Worker** | `12.0 GB` | Feature calculation & AI scoring (pinned to 4 CPU threads) |
| **PyTorch / Metal GPU Pool** | `10.0 GB` | Matrix math & covariance estimation on GPU |
| **Persistent `mlx-lm` `Qwen3-8B-4bit` Daemon** | `10.0 GB` | Local macro risk AI (`ASRI`) |
| **Untouched Safety Headroom** | **`14.0 GB`** | **Guarantees zero SSD swap lag during market crashes** |
