# UI & Operations Manual: Running the Control Center, Telegram Bot & Live Checklist

This step-by-step operator's manual walks you through launching the system, reading every chart and gauge on the **`http://localhost:8080`** Web Control Center, controlling the bot from your phone via Telegram, running historical backtests, and safely graduating from Paper Trading (`Dry-Run`) to Live Trading.

---

## 1. Quick-Start: Launching the Bot & Control Center

### Step 1: Open Your Terminal in the Project Folder
```bash
cd /Users/mbeale/.gemini/antigravity/scratch/nautilus_quant_m1max
source .venv/bin/activate
```

### Step 2: Run the Hardware & Self-Test Verification (Optional but Recommended)
Before starting a session, you can verify your M1 Max hardware caps and configuration:
```bash
python -m uvicorn ui.api_server:create_app --factory --host 127.0.0.1 --port 8080
```

### Step 3: Open the Web Control Center
Open any browser (Safari, Chrome, Brave) and navigate to:
👉 **`http://localhost:8080`**

Every single badge, chart title, metric, and button on the page has a **Plain-English Tooltip**. Simply hover your mouse cursor over any item with a dotted underline or `?` icon to read a plain-English explanation of what it means and what numbers are healthy.

---

## 2. Tour of the Three Dashboard Tabs (`http://localhost:8080`)

### Tab 1: Live Trading & Charts
This is your primary cockpit view. It shows what the market is doing right now and how the bot is reacting.

1. **Top Macro & Shield Status Bar:**
   - **Execution Mode Pill (`DIRECTIONAL_PLUS_CARRY`, `CARRY_ONLY_FALLBACK`, `PAUSED_ENTRIES`, `EMERGENCY_FLATTENED`):** Shows the current operating mode at a glance.
   - **ASRI Macro Weather Gauge (`0–100`):** Shows the local `Qwen3-8B-4bit` AI's systemic risk score, the resulting portfolio risk multiplier ($M_{\text{regime}}$), and the **Plain-English Summary** banner explaining current macro conditions.
   - **VPIN Toxicity & $O(1)$ Hawkes Shield Badge:**
     - **Green (`SAFE`):** `VPIN <= 0.60` and `Hawkes Eta <= 0.70`. Resting Post-Only (`GTX`) limit orders are safely collecting maker rebates/low fees.
     - **Red (`TOXIC SWEEP ACTIVE`):** `VPIN > 0.60` or `Hawkes Eta > 0.70`. The millisecond shield has automatically pulled resting entry orders out of harm's way.
2. **Interactive Dynamic Dollar-Bar Candlestick Chart (with Toggleable Overlays):**
   - Use the legend or overlay toggles at the top of the chart to turn layers on or off:
     - **Triple-Barrier Profit & Stop Bands:** Green dashed lines show the net-of-fee profit target ($\ge 80\text{ bps}$); red dashed lines show the volatility stop-loss.
     - **LightGBM Meta-Probability ($p_{\text{meta}}$):** Plotted on the secondary subplot from `0.00` to `1.00`. Trades are only taken when this line rises above the `0.50` threshold.
     - **VPIN & Hawkes Intensity Ribbon:** Shows real-time order-book toxicity and trade-avalanche reflexivity.
     - **Perpetual Positioning & Squeeze Regimes:** Color-coded markers indicating `CROWDED_LONG_VULNERABILITY`, `CROWDED_SHORT_SQUEEZE`, `GENUINE_TREND_CONVICTION`, or `CASCADE_EXHAUSTION_REVERSAL`.
     - **Mahalanobis `DI_values` Outlier Line:** If this spikes above the red threshold line, the market is in "uncharted territory" and new trades are automatically blocked.

---

### Tab 2: Portfolio & Validation
This tab shows your financial scorecard, fee efficiency, position sizing, and whether the AI models are passing their statistical exams.

1. **Live Net-of-Fee PnL & Fee Savings Scorecard:**
   - **Net-of-Fee PnL:** Your true bottom-line profit after subtracting every exchange fee, funding payment, and slippage cost.
   - **Maker vs. Taker Fill Ratio:** Shows what percentage of your trades executed as passive **Maker (`GTX`)** orders (`2.0 bps` on USDT perps / `0.0 bps` on USDC perps) vs. aggressive **Taker** orders (`5.5 bps`). Aim for **> 85% Maker fills**.
   - **Fee Savings (USD & bps):** Shows how much money the `DefensiveMakerExecutionAlgorithm` saved you compared to standard market orders.
2. **Signed `HRP-CVaR` Target Weights Chart ($w_{\text{final}, i}$):**
   - Horizontal bar chart showing the exact target allocation for each coin (e.g., `+0.45x` Long BTC, `-0.30x` Short SOL, plus Delta-Neutral Funding Carry sleeves).
   - Displays the decomposition of each weight: Base `HRP-CVaR` weight ($w_{\text{HRP}}$), Fractional Kelly conviction ($f_{\text{Kelly}}$), Macro Multiplier ($M_{\text{regime}}$), and total gross leverage (strictly capped at $\le 2.50\text{x}$).
3. **Champion vs. Challenger 45-Split CPCV Gatekeeper Chart:**
   - Side-by-side comparison of the active **Champion** model and the latest **Challenger** candidate across the five hard promotion gates:
     - **PSR vs. Zero:** Must be $\ge 0.95$ (95% confidence)
     - **PSR vs. Benchmark:** Must be $\ge 0.95$
     - **Deflated Sharpe Ratio (DSR):** Must be $\ge 0.95$
     - **Probability of Backtest Overfitting (PBO):** Must be $< 0.30$ (under 30% overfitting risk)
     - **Expected Net PnL per Trade:** Must be $> 15.0\text{ bps}$ after all fees

---

### Tab 3: How It Works (Plain-English Guide)
- Built directly into the browser so you never have to hunt for files on disk.
- Switch between all four official operator manuals (**Architecture Guide**, **UI & Operations Manual**, **Plain-English Glossary**, and **Troubleshooting & Safety**) right inside the dashboard.

---

## 3. The Four One-Click Operational Controls

Located in the top-right header of `http://localhost:8080`, these four buttons give you immediate manual override authority at any time:

| Button | Action ID | Plain-English Effect | When to Use It |
| :--- | :--- | :--- | :--- |
| **🟢 Start Dry-Run** | `start_dry_run` | Activates Paper Trading / Full Directional + Carry simulation mode (`DIRECTIONAL_PLUS_CARRY`) using live market data without risking real exchange capital. | When testing new configurations, verifying connectivity, or resuming normal operation. |
| **⏸️ Pause Entries** | `pause_entries` | Immediately cancels all unfilled entry orders and sets mode to `PAUSED_ENTRIES`. Existing open positions keep their protective stop-losses and profit targets. | Before a major speech, exchange maintenance window, or when you want to wind down exposure gracefully. |
| **🛡️ Switch to Carry-Only** | `switch_carry_only` | Sets mode to `CARRY_ONLY_FALLBACK`. Disables directional bets and restricts capital to low-risk **Delta-Neutral Funding Carry** (Spot Long + Perp Short) and Cash. | When markets are whipsawing without direction and you only want to collect 8-hour funding payments. |
| **🚨 Emergency Flatten All** | `emergency_flatten` | Sets mode to `EMERGENCY_FLATTENED`. Cancels every resting order and closes all open positions across all coins immediately using `EMERGENCY_TAKER_EXIT`. | In a true black-swan emergency when you want to be 100% in cash within seconds. |

---

## 4. Using the Telegram Mobile Bot (`ui/telegram_bot.py`)

You don't need to sit in front of your MacBook Pro all day. The Telegram companion sends **plain-English alerts** whenever an important decision occurs:
- **Trade Opened Alert:** Explains *which* coin was traded, *why* (e.g., *"Opened +0.42x Long on BTCUSDT-PERP because a Crowded Short Squeeze was detected with 74% Meta-Model confidence and +28 bps expected net edge after fees"*), and where the profit/stop barriers sit.
- **Toxic Shield Alert:** Explains *why* a resting order was pulled (e.g., *"Pulled resting BUY order on ETHUSDT-PERP in 1.8ms because VPIN Toxicity jumped to 0.68 [threshold 0.60], indicating a one-sided institutional sell sweep"*).

### Available Slash Commands on Your Phone
Send any of these commands to your bot in Telegram:
- `/status` — Returns a plain-English health check: current operating mode, ASRI macro weather score, VPIN/Hawkes shield status, and gross leverage.
- `/profit` — Shows today's and cumulative **Net-of-Fee PnL**, Maker vs. Taker fill percentage, fee savings in dollars, and active funding carry yield.
- `/gates` — Reports the 45-split CPCV exam scores (`PSR`, `DSR`, `PBO`, `Net Edge bps`) for the current Champion model and whether any Challenger was recently promoted or rejected.
- `/explain` — Gives a plain-English narrative of the most recent trade decision AND why the bot is currently sized the way it is (breaking down `HRP-CVaR`, `p_meta`, `Mahalanobis DI`, and `ASRI`).
- `/pause` — Remotely triggers **Pause Entries** (`PAUSED_ENTRIES`) from your phone.
- `/flat` — Remotely triggers **Emergency Flatten All** (`EMERGENCY_FLATTENED`) from your phone, closing all positions to 100% cash.

---

## 5. Safe Graduation Checklist: From `Dry-Run` (Paper Trading) to Live Trading

Never rush directly into live capital. Follow this 5-step graduation protocol:

1. **Stage 1 — Historical 45-Split CPCV Verification:**
   - Open **Tab 2 (Portfolio & Validation)** and confirm that `passed_all_gates` is `True` (`PSR >= 0.95`, `DSR >= 0.95`, `PBO < 0.30`, `Expected Net PnL > 15 bps`).
2. **Stage 2 — Minimum 7 Days in `Dry-Run` Mode:**
   - Click **Start Dry-Run** and let the system process live exchange WebSocket feeds for at least one full week (covering at least 21 eight-hour funding intervals).
3. **Stage 3 — Verify Execution Hygiene:**
   - Confirm that your **Maker Fill Ratio** is above **80%** and that the **Hawkes/VPIN Toxic Sweep Shield** triggers cleanly during high-volatility news candles without causing repeated `GTX` rejection loops.
4. **Stage 4 — Test Emergency Controls:**
   - Test `/pause` and `/flat` from Telegram (or the web dashboard) during `Dry-Run` mode to verify instant response.
5. **Stage 5 — Start at 10% Capital Allocation:**
   - When switching to live exchange API keys, fund the sub-account with only 10% of your intended capital for the first 14 days while monitoring live 30-day rolling PSR (`champion_live_30d_psr >= 0.80`).

## What changed in Phase 0 of the reinforcement plan

- **One bar is one real 15-minute Hyperliquid candle.** "▶ Start Dry-Run" polls every 10 seconds and processes a bar only when a new candle has closed. "⚡ Poll Next 15m Bar" checks once; if nothing has closed yet the engine says so and nothing changes.
- **Trades are positions, not bars.** A position opens when a sleeve has a real signal, and closes at its profit barrier, its stop barrier, after 16 bars, on a risk-committee veto, or when the signal flips. Wins and losses are counted only when a position closes. "Equity" is starting capital plus realised profit; open positions show separately as unrealised PnL.
- **Nothing is invented.** Inputs without a live feed (taker buy/sell split, liquidations, open-interest history, a second funding venue) are shown as unavailable. The AI win-probability model trains only once about five days of real bars have been recorded; until then the meta sleeve stays flat and the panel says "no model".
- **The sleeve panel is live.** Funding carry APR, breakeven hours, actionability, Deribit skew, dealer GEX and HLP crowding come from the sleeves' latest real evaluations on every bar.


## Backfilling history and replaying a window

- **Backfill**: `.venv/bin/python scripts/backfill.py --months 12` fills `data/*usdt_perp_bars.parquet` (15m) and `data/*usdt_perp_bars_1h.parquet` with real candles, mark and index prices, Hyperliquid funding and open interest for BTC, ETH and SOL. Bybit is preferred, then the Binance public data archive (data.binance.vision), then OKX. The archive is plain file hosting, so it works where the Bybit and Binance APIs refuse requests; it has multi-year 5-minute open interest and real buyer/seller volume, but publishes the newest day or two with a delay. Fill those days from OKX without open interest, for example `--start 2026-10-27 --venues okx --no-open-interest`, so open interest from two venues is never mixed in one series. The table it prints shows rows, gaps and which venue supplied what. Once the 15m store has about five hundred bars with open interest, the AI win-probability model trains at the next start.
- **Replay**: `.venv/bin/python -m research.replay --scenario calm_2025_09` (or `--start`/`--end` for any window) runs the same engine as the dashboard over stored bars, with no network, and writes a JSON report to `data/replays/` with every trade, the equity curve, fees, funding and per-sleeve results. Add `--controls phase3` to run with the circuit breakers and cascade controls on; the default `--controls none` is the Phase 0 baseline with both off.
- **Open interest from two venues**: the backfilled history uses Binance open interest, while the live loop reads Hyperliquid's, which is several times smaller. For the first month after live bars start, the open-interest z-score sees that level change as a move; treat model signals in that period with care.

## Live feed availability panel

The "Live Feed Availability" box lists every input the engine can use and whether a real feed is supplying it right now: candles, funding history, open-interest history, predicted fundings, open-interest caps, options surface, HLP vault, Deribit DVOL, stablecoin marks, taker flow, liquidations, trades and lead-venue mids. A grey pill means the feed is missing; the engine then treats that input as unknown (zero or ignored) and says so, rather than filling it with a guess. The line under the pills shows the current stablecoin peg deviation and DVOL that the macro sentinel is using.

## Streams and feature checks

- **Streams**: when the server runs it also listens to three websockets: Hyperliquid trades, Binance and Bybit top of book, and Bybit liquidations (with OKX as a backup for liquidations). Bars that close while these were connected carry real buyer/seller volume and real liquidation volume; otherwise those fields stay marked as unavailable. The availability panel shows the lead-venue price move over the last 0.7 seconds and the latest quarter-hour boundary imbalance.
- **Feature checks**: `.venv/bin/python -m research.feature_eval` tests whether each input actually predicts anything on stored bars, against a pure-noise control, and records the result. A feature that cannot beat noise is not added to the model.

## Sleeves that are off by default

- **Trend (Sleeve 2)** is built and tested but switched off. On 3.4 years of real hourly data it made money overall and in two of three crisis windows, but its average profit per trade is below the 15 bps the promotion gate requires. `python -m research.trend_eval` re-runs that check on your data.
- **Derivatives alpha (Sleeve 1)** now measures option skew and dealer gamma correctly and reads the HLP vault's real positions. Its trading rule has not been validated on the corrected numbers, so the dashboard shows its signal marked "SHADOW_NOT_VALIDATED" and it does not open positions.
