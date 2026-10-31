# Plain-English Glossary of Quantitative Terms

Every mathematical term in the **Gemini Argon 4** codebase is translated below into a **simple, 2-sentence plain-English definition**, followed by concrete examples of what **"Good / Healthy"** vs. **"Bad / Warning"** numbers look like on your dashboard.

---

## 1. Market Data & Feature Engineering Terms

### Dynamic Dollar Bars (`DollarBar`)
* **Plain-English Definition:** Instead of drawing a new chart candle every 15 minutes on the wall clock (even when the market is asleep), the bot draws a new candle only after a fixed amount of real money (e.g., $25 million) has traded. This gives the AI high-speed vision during volatile crashes and filters out dead, meaningless noise during quiet weekends.
* **Good / Healthy Numbers:** Consistent dollar volume per bar with balanced buy/sell participation (`buy_dollar_volume` roughly matching `sell_dollar_volume` during calm periods).
* **Bad / Warning Numbers:** Extreme one-sided volume spikes accompanied by massive forced liquidations (`liquidation_volume_usd` spiking into the 99th percentile).

### Fractional Differentiation (`FFD` / `ffd_stationarity`)
* **Plain-English Definition:** A mathematical filter that removes the raw dollar price level (so Bitcoin at $90,000 looks statistically comparable to Bitcoin at $30,000) without erasing the chart's long-term trend memory. It acts like polarized sunglasses that cut out blinding price glare while keeping the underlying terrain crystal clear for the AI.
* **Good / Healthy Numbers:** `d_star` between `0.15` and `0.45`, stationarity test `adf_pvalue <= 0.01` (99%+ confidence the series is stable), and `level_correlation >= 0.80` (retains 80%+ of trend memory).
* **Bad / Warning Numbers:** `adf_pvalue > 0.05` (meaning raw price drift is still polluting the indicator) or `d_star > 0.55` (meaning too much trend memory is being erased).

### Perpetual Positioning & Squeeze Regimes (`PositioningRegime` / `squeeze_regime`)
* **Plain-English Definition:** A crowd-detector that combines Open Interest (total leveraged bets), 8-hour Funding Rates, and Liquidation spikes to tell whether traders are dangerously overcrowded on one side of the boat. It classifies the market into four actionable setups—such as a `CROWDED_SHORT_SQUEEZE` (where bears are trapped and forced to buy back) or `CROWDED_LONG_VULNERABILITY` (where over-leveraged bulls risk a domino-effect crash).
* **Good / Healthy Numbers:** Entering a Long trade during `CROWDED_SHORT_SQUEEZE` or `GENUINE_TREND_CONVICTION`, or entering after a washed-out `CASCADE_EXHAUSTION_REVERSAL`.
* **Bad / Warning Numbers:** Buying into `CROWDED_LONG_VULNERABILITY` when `funding_rate_8h > 0.05%` (5 bps per 8h), which means you are paying expensive fees to stand in a crowded room.

---

## 2. AI Prediction & Safety Gate Terms

### Fee-Adjusted Triple-Barrier Labelling (`triple_barrier`)
* **Plain-English Definition:** A training method that places every historical trade inside a three-walled box—a profit target on top, a stop-loss on the bottom, and a 16-bar expiration timer on the right—to see which wall the price touches first. Crucially, the top wall requires at least an **80 bps (0.80%) move** and at least **6 times our round-trip trading fees and slippage**, ensuring the AI never learns to chase tiny, fee-losing scalps.
* **Good / Healthy Numbers:** Profit target barrier $\ge 80\text{ bps}$ (`0.80%`) with `barrier_touched == "PROFIT"` and positive `net_return` after subtracting all fees and funding.
* **Bad / Warning Numbers:** Any setup whose expected move is smaller than `80 bps` or times out repeatedly at the `VERTICAL` 16-bar wall without covering fees.

### Two-Stage Meta-Labeling (`meta_probability` / `p_meta`)
* **Plain-English Definition:** Instead of asking one model to predict both market direction and bet size at the same time, Stage 1 picks the direction (`+1` Long or `-1` Short) and Stage 2 (a calibrated LightGBM AI) acts as a "Sniper Quality Inspector" that outputs the probability (`p_meta` from `0.00` to `1.00`) that Stage 1's idea will actually be profitable after fees. If the inspector is 50% confident or less (`p_meta <= 0.50`), the trade is skipped completely (`0%` size).
* **Good / Healthy Numbers:** `p_meta` between `0.60` (`60%`) and `0.85` (`85%`), paired with `expected_net_edge_bps > 15.0 bps`.
* **Bad / Warning Numbers:** `p_meta <= 0.50` (coin-flip confidence—automatically blocked from trading).

### Mahalanobis Distance Outlier Gate (`mahalanobis_di` / `DI_values`)
* **Plain-English Definition:** An "Uncharted Territory" alarm that measures how strange today's combination of market indicators looks compared to everything the AI saw during training. If the distance score (`mahalanobis_di`) crosses the safety threshold (`di_threshold`), the bot admits *"I have never seen a market shaped like this before"* and blocks new trades (`epistemic_gate_passed = False`).
* **Good / Healthy Numbers:** `mahalanobis_di < di_threshold` (e.g., `DI = 1.8` when threshold is `3.5`), meaning current market behavior is familiar and inside the AI's training zone (`epistemic_gate_passed = True`).
* **Bad / Warning Numbers:** `mahalanobis_di >= di_threshold` (e.g., `DI = 5.2`), which turns the badge red and vetoes the trade regardless of how high `p_meta` looks.

### Aggregated Systemic Risk Index (`ASRI` / `asri_macro`)
* **Plain-English Definition:** A `0 to 100` "Crypto Weather & Hurricane Scale" generated locally on your M1 Max GPU by the `Qwen3-8B-4bit` AI daemon as it monitors macro, regulatory, exchange-solvency, and stablecoin news. When the score rises above `40`, it automatically turns down the master portfolio risk dial (`M_regime`) from `1.0` (`100%`) all the way to `0.0` (`0%` halt) if a crisis hits.
* **Good / Healthy Numbers:** `ASRI <= 40.0` and `tail_risk_flag = False`, which keeps `M_regime = 1.00` (clear skies, 100% normal position sizing).
* **Bad / Warning Numbers:** `ASRI > 70.0` (severe storm, cutting sizes by MORE than 50%) or `ASRI >= 95.0` / `tail_risk_flag = True` (emergency shelter, setting `M_regime = 0.0` to halt all directional risk).

---

## 3. Order Book Shield & Execution Terms

### VPIN Toxicity (`vpin_toxicity` / `vpin`)
* **Plain-English Definition:** Short for *Volume-Synchronized Probability of Informed Trading*, this `0.00 to 1.00` meter measures whether order flow is calm two-way trading or a one-sided institutional dump ("sharks in the water"). If VPIN climbs above `0.60`, our execution bodyguard immediately cancels resting limit orders so we don't get run over by a whale.
* **Good / Healthy Numbers:** `vpin` between `0.15` and `0.45` (balanced, safe order book for passive Maker orders).
* **Bad / Warning Numbers:** `vpin > 0.60` (toxic one-sided sweep active—resting entry orders are automatically canceled in under 5 milliseconds).

### $O(1)$ Markovian Hawkes Reflexivity (`hawkes_reflexivity` / `branching_ratio_eta`)
* **Plain-English Definition:** A real-time "Avalanche Meter" that measures what fraction of current market trades are being triggered by *previous* trades in a self-reinforcing chain reaction (like snowballs setting off an avalanche). Updated in microseconds (`O(1)` time), if the branching ratio (`eta`) exceeds `0.70`, the bot steps out of the book until the cascade cools down.
* **Good / Healthy Numbers:** `branching_ratio_eta` between `0.10` and `0.50` (calm, mean-reverting order flow where trades fade out naturally).
* **Bad / Warning Numbers:** `branching_ratio_eta > 0.70` (critical reflexivity—high danger of a flash crash or vertical squeeze cascade; triggers the Toxic Sweep Shield).

### Post-Only Maker Orders (`GTX` / `DefensiveMakerExecutionAlgorithm`)
* **Plain-English Definition:** Instead of paying expensive "Taker" fees (`5.5 bps`) to buy or sell immediately at market price, the bot posts resting "Maker" limit orders (`GTX`) on the order book that pay only `2.0 bps` on USDT perps (or `0.0 bps` on USDC perps). If a price jump would accidentally cause our order to cross the spread as an expensive Taker, the `GTX` tag forces the exchange to reject and re-peg it safely.
* **Good / Healthy Numbers:** Maker fill ratio **> 85%** and `consecutive_rejections < 3`.
* **Bad / Warning Numbers:** `consecutive_rejections >= 3`, which indicates a fast-moving book and triggers a 1-tick deeper backoff and 1-second cooldown.

---

## 4. Exam Board & Portfolio Sizing Terms

### Combinatorial Purged Cross-Validation (`CPCV` / `cpcv_gates`)
* **Plain-English Definition:** A rigorous **45-split time-travel exam** that tests every AI model across 45 different combinations of historical bull, bear, and sideways markets—while deleting ("purging" and "embargoing") overlapping data around test borders so the AI cannot cheat by memorizing the timeline. A new "Challenger" model is never allowed to trade live unless it passes all 45 exam splits net of fees.
* **Good / Healthy Numbers:** `passed_all_gates = True` across all 45 splits (`num_splits = 45`, `n_oos_paths = 9`) with `expected_net_pnl_bps > 15.0`.
* **Bad / Warning Numbers:** `passed_all_gates = False` or `expected_net_pnl_bps <= 15.0` (Challenger is rejected; bot stays on the proven Champion or falls back to Carry-Only).

### Probabilistic Sharpe Ratio (`PSR`), Deflated Sharpe Ratio (`DSR`) & Overfitting Probability (`PBO`)
* **Plain-English Definition:** Three statistical lie-detectors that grade a strategy's backtest: **PSR** measures the probability (`0% to 100%`) that true skill exceeds zero and beats the benchmark; **DSR** penalizes that score for how many models we tested so a "lucky fluke" out of 100 tries doesn't fool us; and **PBO** measures the chance that the model simply memorized historical noise. Together, they form an ironclad gatekeeper against fake backtests.
* **Good / Healthy Numbers:** `psr_vs_zero >= 0.95` ($\ge 95\%$), `psr_vs_benchmark >= 0.95` ($\ge 95\%$), `dsr >= 0.95` ($\ge 95\%$), and `pbo < 0.30` ($< 30\%$ overfitting risk).
* **Bad / Warning Numbers:** `psr < 0.95`, `dsr < 0.95`, or `pbo >= 0.30` (model fails promotion; if the live Champion's 30-day PSR drops below `0.80`, the bot automatically shifts to `CARRY_ONLY_FALLBACK`).

### Signed `HRP-CVaR` & Fractional Kelly Weights (`hrp_cvar_weights` / $w_{\text{final}, i}$)
* **Plain-English Definition:** Our portfolio sizing engine (`HRP-CVaR`) groups coins into a family tree based on how closely they crash together and gives smaller base weights (`w_HRP`) to assets with worse worst-case tail losses (`CVaR`), then multiplies by the AI's Half-Kelly confidence (`f_Kelly`) and the macro weather dial (`M_regime`). This ensures no single coin or correlated cluster can sink your account, while enforcing a hard **2.5x gross leverage cap**.
* **Good / Healthy Numbers:** Balanced single-asset weights (typically between `-0.60x` and `+0.60x`) and total `gross_leverage <= 2.50x`.
* **Bad / Warning Numbers:** Any trade whose expected return is smaller than `2x Maker Fee` (`rejected_for_fees`), which is automatically zeroed out (`w_final = 0.0`).

### Delta-Neutral Funding Carry (`funding_carry` / `FundingCarrySleeve`)
* **Plain-English Definition:** A market-neutral income strategy that buys $1 of a coin on the Spot market while simultaneously shorting $1 of the same coin's Perpetual Future, canceling out up-or-down price movement (`Delta = 0`). As long as bullish traders pay positive 8-hour funding rates to hold leveraged longs, our bot collects those regular interest payments like rent on an apartment building.
* **Good / Healthy Numbers:** Positive 8-hour funding rate (`funding_rate_8h >= 0.0001` or `1.0 bps` per 8h, which equals ~10.95% annualized yield) and tight Spot-to-Perp basis tracking.
* **Bad / Warning Numbers:** deeply negative funding rates (`funding_rate_8h < 0.0`), where shorts have to pay longs—at which point the Carry sleeve automatically steps aside into Cash.
