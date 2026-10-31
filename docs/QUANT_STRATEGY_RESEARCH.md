# Quant-Firm and Algo-Bot Strategy Research: What Phoenix Can Use

Date: 2026-10-31
Scope: BTC/ETH/SOL perpetuals on Hyperliquid (Bybit/OKX secondary), 15m to 4h horizon, $5k paper book.

This document does three things:

1. Audits what Phoenix actually does today, including where the live loop in `ui/api_server.py` diverges from the research-grade modules.
2. Summarises what is publicly known about the strategies of top quantitative firms, crypto-native prop/market-making firms, the best open-source bots, and ML-alpha competition winners. Every claim carries a source and an evidence grade.
3. Extracts a prioritised reinforcement plan mapped to Phoenix files, with concrete parameters and the validation gate each change must pass.

Evidence grades used throughout: **A** = peer-reviewed paper, primary filing, or primary code/docs. **B** = primary but self-reported (firm blog, competition write-up, vendor doc). **C** = secondary reporting, folklore, or anecdote. Folklore is labelled as such; nothing graded C should change a position-sizing rule on its own.

---

## 1. Executive summary

The research is unambiguous on one point: the firms that last do not win with one clever signal. They win with (a) many weak, low-correlation signals combined under an explicit cost model, (b) a position process that trades partially toward a target instead of flipping on thresholds, (c) risk controls that are independent of the alpha code, and (d) a research process that counts every trial. Phoenix already has unusually good scaffolding for (d) (CPCV, Deflated Sharpe, PBO) and for toxicity-aware execution. Its gaps are in (a), (b), and (c), and in the fact that the live paper loop does not yet run the strategy stack the documentation describes.

Top ten actions, in order of expected value per unit of work:

| # | Action | Where | Evidence | Section |
|---|--------|-------|----------|---------|
| 0 | Wire the real sleeves, HRP weights, and a position lifecycle into the live tick; remove hard-coded directions, probabilities and synthetic bar fields | `ui/api_server.py` | n/a (integrity) | 2 |
| 1 | Replace fixed-APR carry entry/exit with He–Manela–Ross–von Wachter band-entry / zero-exit on the funding-basis deviation, plus a 3-day funding average and a funding-flip hard exit | `sleeves/sleeve_3_funding_carry.py` | A | 3.2, 4.1 |
| 2 | Add a multi-horizon, vol-scaled, correlation-adjusted trend sleeve (1d / 1w / 1m lookbacks, continuous signal) | new `sleeves/sleeve_2_trend.py` | A | 3.3, 4.2 |
| 3 | Widen the meta-labeller feature set: quarter-hour boundary order imbalance, basket-residualised returns, 1-week TSMOM, hour-of-day, mark-oracle divergence, taker-flow variance, Amihud | `core/models/meta_labeling_worker.py`, `core/features/` | A/B | 3.4, 4.3 |
| 4 | Gârleanu–Pedersen partial adjustment toward an aim portfolio, with measured signal half-lives and an explicit square-root impact term | `risk/portfolio_allocator.py` | A | 3.1, 4.4 |
| 5 | Pod-style per-sleeve drawdown rules (soft −5% of sleeve allocation halves gross, hard −7.5% disables and forces re-promotion) plus FreqAI-style stop-out counters and cooldowns | new `risk/circuit_breakers.py` | B/A | 3.6, 4.5 |
| 6 | Binance/Bybit lead as a quote-pull signal and a per-fill markout ladder for the maker algo; fee-tier and rebate-aware routing | `core/execution/defensive_maker.py` | A/B | 3.5, 4.6 |
| 7 | Replace the hand-weighted derivatives composite with calibrated, IC-weighted (shrunk toward equal) combination; fix the Deribit parse (true 25-delta 30d/90d risk reversal, expiry-aware term structure, taker-flow GEX) | `sleeves/sleeve_1_derivatives_alpha.py` | A/B | 3.4, 4.7 |
| 8 | Statistical jump-model regime state with a persistence penalty, used to condition HRP covariance, stop widths and the carry/directional split; keep the LLM committee as a veto, not the primary multiplier | `sentiment/`, `risk/` | A | 3.6, 4.8 |
| 9 | Cascade and ADL controls: taker-flow variance compression, OI-cap flags, funding > 25% annualised crowding rule, ADL-rank monitor on the carry short leg, multi-venue collateral marks | `risk/`, `core/data_ingestion.py` | A/B | 3.7, 4.9 |
| 10 | Research-process hardening: trial registry feeding the DSR's N, fold-stability feature pruning, online warm-start residual layer, shadow-run promoted models for N trades with slippage reconciliation | `research/` | A/B | 3.8, 4.10 |

Things the research says **not** to adopt: grid/DCA/martingale bots, copy-trading leaderboard wallets, liquidation heatmaps as entry triggers, OI-times-price matrices as standalone alphas, Hurst/ADX regime gates, blog-grade HMM gating. See Section 5.

---

## 2. Audit: what Phoenix does today

### 2.1 Research-grade modules (verified by reading the code)

| Module | What it does | Key parameters |
|--------|--------------|----------------|
| `sleeves/sleeve_3_funding_carry.py` | Threshold carry: spot-long/perp-short when 8h funding ≥ 3 bps; perp-long when funding ≤ −5 bps and ASRI ≤ 40; cross-venue HL-vs-Bybit spread ≥ 7 bps/8h. Breakeven and net-edge hurdles. Hysteresis exit at 0.5 bps/8h. | 72h assumed hold, 48h max breakeven, 8 bps min net edge, ASRI caps 85/40, 1.0x gross cap |
| `sleeves/sleeve_1_derivatives_alpha.py` | Hand-weighted composite: 0.40 skew + 0.30 HLP crowding + 0.20 term structure + 0.10 GEX, basis bonus. Direction if |composite| > 0.15. `p_meta = 0.50 + 0.45·|composite|`. | Skew ±1.5%, term ±0.5%, GEX ±$10M, HLP crowding $2M |
| `core/features/positioning.py` | Four-state regime matrix from funding/OI/basis/liquidations with FFD z-scores (window 48, d = 0.35). | Crowding 5 bps/8h, OI flush −5%, liquidation z ≥ 2.33 |
| `core/models/meta_labeling_worker.py` | Stage-1 SVD on continuous features, Ledoit–Wolf Mahalanobis gate, Stage-2 LightGBM with `CalibratedClassifierCV` (sigmoid). | Only 3 continuous features by default (7 extended); DI 4.5 (3.5 in live), 100 trees |
| `core/contracts.py` (triple barrier, Kelly, regime) | PT 2σ, SL 1.5σ, 16-bar vertical, 80 bps hard floor; half-Kelly on `2p−1`; ASRI piecewise multiplier (40/70/95). | |
| `risk/portfolio_allocator.py` | HRP with Ledoit–Wolf, single linkage, inverse 95% CVaR bisection, signed by direction × Kelly × regime, 2.5x gross cap, round-trip fee hurdle. | |
| `research/validation/cpcv_gatekeeper.py` | 45-split CPCV, PSR ≥ 0.95, DSR ≥ 0.95, PBO < 0.30, E[net] > 15 bps, live 30d PSR < 0.80 fallback. | |
| `core/execution/defensive_maker.py`, `vpin_hawkes.py` | VPIN (50 buckets) + Hawkes intensity shield; GTX re-peg hysteresis 1.5 bps / 500 ms; 3-strike backoff. | VPIN 0.60, η 0.70 |
| `sentiment/committee_models.py` | Macro sentinel, breadth analyst, red-team critic, consensus multiplier. | |

### 2.2 Where the live loop diverges (this is the biggest "usable" finding)

`ControlCenterStateStore.step_simulation_tick` in `ui/api_server.py` is what produces the ledger, the win/loss counts and the forensics. Reading it line by line:

- **The two sleeves are instantiated but never called.** `sleeve_telemetry` returns constants (cross-venue spread 8.4 bps, skew 0.024, GEX $42.5M, breakeven 12h). The dashboard "sleeve" panel does not reflect live evaluation.
- **Directions are hard-coded for two of three assets.** ETH `y_primary = 1` and SOL `y_primary = −1` on every bar. BTC direction defaults to `+1` whenever the model emits 0 (both when calling `predict_single` and when building the allocation input).
- **The probability floor forces a trade every bar.** `p_eff = max(0.58, p_meta)` guarantees a positive Kelly fraction regardless of what LightGBM says.
- **HRP weights are constants** (0.45 / 0.35 / 0.20) and `expected_gross_return_bps` is constant (96 / 88 / 108), so the fee hurdle can never reject a trade.
- **Bar fields are synthetic.** High/low, volume, buy fraction and liquidation volume are drawn from a random generator; `sigma_bar` is fixed at 0.0045; the macro sentinel receives fixed inputs (depeg 1.5 bps, DVOL 51) and the breadth model receives placeholder funding rates for ETH/SOL.
- **No position lifecycle.** The portfolio is re-targeted every bar; there are no entries, exits, barriers or stops. Each bar's mark-to-market is counted as a "win" or "loss". The `TripleBarrierLabel` shown in the UI is built after the fact with `barrier_touched` set from the sign of that bar's PnL.
- **Carry PnL is always positive**: `abs(funding_8h)/32 × 15% notional`, independent of whether a carry position exists or which side it is on. Fees are a fixed 8% turnover assumption, and "fee savings" are derived from that.
- **Retraining**: every 20 ticks on the last 200 bars (two days of 15m bars) with a 16-bar vertical barrier; if labels come out single-class the code fabricates labels (first third = 1, second third = 0) rather than skipping the retrain.

None of this is a criticism of the module design, which is sound. It means the first "reinforcement" is to make the live loop consume the modules. Until that happens, paper results do not measure any strategy, and no item in Sections 3–4 can be validated live.

---

## 3. Research findings

### 3.1 Signal philosophy and combination (top quant firms)

- **Many weak alphas, combined under a risk and cost model.** WorldQuant's "101 Formulaic Alphas" (Kakushadze, arXiv 1601.00991, **A**) documents production-style alphas with mean pairwise correlation ≈ 16%, holding periods 0.6–6.4 days, and no significant dependence of returns on turnover. The operator set is small (`rank`, `ts_rank`, `delta`, `correlation`, `decay_linear`, `scale`). Community-reported BRAIN admission gates (**C**): Sharpe ≥ 1.25, fitness = Sharpe·√(|ret|/max(turnover,0.125)) ≥ 1, self-correlation < 0.7.
- **Breadth beats skill** (Grinold–Kahn; Clarke–de Silva–Thorley, **A**): IR ≈ TC · IC · √Breadth. The transfer coefficient captures how much signal is lost between raw score and executed position. With three assets, Phoenix's breadth has to come from independent horizons and signal families.
- **Forecast-combination puzzle** (Elliott; Chan–Pauwels, **A**): equal or lightly shrunk weights usually beat estimated optimal weights because weight-estimation error dominates. Qian–Hua–Sorensen weight by IC/σ(IC).
- **Alpha decay and turnover** (Gârleanu–Pedersen, JF 2013, **A**): the optimal policy under quadratic costs is to aim in front of the target and trade a fraction of the way each period; slow-decaying signals get more weight in the aim portfolio.
- **Renaissance (reported, B/C)**: ~50.75% hit rate on enormous trade counts, thousands of simultaneous positions, holding periods of days, HMM lineage, leverage via basket options (documented in the 2014 Senate PSI hearing, **A**), and obsessive cost measurement. Transferable lesson: expect 51–55% hit rates and build the cost model first.
- **Two Sigma regime modelling** (Botte & Bao 2021, **B**): a Gaussian mixture over 17 risk factors yielding four regimes with their own means, vols and correlations.
- **HRT / XTX (B/C)**: both describe prediction over seconds-to-hours horizons rather than pure speed; HRT states "market data contains very little signal, making it easy to overfit."
- **Jane Street (A/B)**: bond prospectuses and reporting show a firm that mainly fears gap drawdowns and buys short-dated puts against them; grinding losses are not covered by that hedge.

### 3.2 Carry and funding (crypto-native)

- **Deviation-from-no-arbitrage carry** (He, Manela, Ross, von Wachter, arXiv 2212.06888, **A**): define ρ = κ(f − s) − (r − r′). Enter when ρ leaves the cost band ±κ·log(1 ± C), exit when ρ first returns to zero rather than on a timer. Perp-to-spot convergence, not funding receipts, provides most of the PnL; the funding share is shrinking (~11%/yr). Reported Sharpe: BTC 3.5 (no fees) to 1.8 (high fees); ETH 4.8 to 2.6. Trailing 120-day returns positively predict ρ. Companion pricing theory: Ackerer, Hugonnier, Jermann (arXiv 2310.11771, **A**).
- **Structural positive bias** (BitMEX Q3-2025 report, **B**): funding positive >92% of the time, clustered at the 0.01%/8h interest anchor. Negative funding is the information-rich tail; model funding as anchor plus jumps.
- **Extreme-funding contrarian** (BitMEX 2018; Jarvis Labs; K33, all **B/C**): fading 1–2σ funding before settlement works; fading >2.5σ or clamp-pinned readings tends to blow up because the trend continues. Jarvis: long after negative funding, 48h hold, Sharpe 0.84, no out-of-sample.
- **Funding-aware market making** (arXiv 2605.06405, **A**, calibrated on Hyperliquid ETH/BTC/SOL): extending Avellaneda–Stoikov with funding as an OU-plus-jump state coupled to inventory beats classical AS on ETH/BTC in PnL and inventory RMS.
- **What kills carry** (Ethena docs, Chaos Labs, FTI, **A/B**): sustained negative funding (longest observed streak ≈ 13 days; worst week ≈ −3%), exchange counterparty failure, collateral depeg (USDe marked at ~$0.65 on Binance for 40 minutes on 10 Oct 2025), rebalancing slippage, and ADL removing the short leg of a hedge mid-cascade. Ethena's drawdowns are 1–2 week regime flips, not hourly noise.
- **Cross-venue data**: Hyperliquid's `predictedFundings` returns predicted next funding for HL, Binance and Bybit in one call (**A**). An MDPI 2026 study of 9.1M hourly observations finds funding persistence is episodic and cross-venue spreads peak ~2h after settlement (**A**).
- **When to flip carry to directional** (FTI 10/10 post-mortem, **B**): funding > 25–30% annualised with OI at highs preceded the October 2025 cascade.
- **Carry across asset classes** (Koijen et al., JFE 2018, **A**): carry predicts returns everywhere including options; carry strategies crash together in recessions. Crypto spot-long/perp-short carry: ~8% mean at 0.8% vol (Schmeling–Schrimpf–Todorov, arXiv 2510.14435, **A**).

### 3.3 Trend, momentum, reversal, seasonality

- **Time-series momentum** (Moskowitz–Ooi–Pedersen 2012; Hurst–Ooi–Pedersen "A Century of Evidence", **A**): sign of past 1/3/12-month return, positions scaled to constant ex-ante vol (EWMA, 60-day centre of mass), equal-weighted across lookbacks. Kim–Tse–Wald show much of the Sharpe comes from the vol scaling itself. Baltas–Kosowski (**A**): a continuous signal in [−1, 1] plus a pairwise-correlation leverage adjustment cuts turnover and matters when correlations are high, as they are among BTC/ETH/SOL.
- **Crypto trend**: Liu & Tsyvinski (NBER w24877, **A**): 1–4 week time-series momentum in BTC/ETH; BTC top-quintile prior-week return gives 11.2%/week next week, weaker for ETH, in-sample. Liu–Tsyvinski–Wu (JF 2022, **A**): 1–4 week cross-sectional momentum only in above-median-size coins. Zarattini–Pagani–Barbon (**B**): Donchian ensemble across lookbacks with vol sizing on top-20 coins, net Sharpe > 1.5. Caution (arXiv 2009.12155, **A**): optimised SMA/EMA parameters on BTC do not predict next-year Sharpe.
- **Reversal**: daily reversal exists across 3,600 coins but is an illiquidity effect; the largest coins show daily continuation, not reversal (IRFA 2021 "Up or down?", **A**). Nagel (RFS 2012, **A**): reversal returns are liquidity-provision returns and rise with VIX, so condition mean-reversion sleeves on DVOL state.
- **Intraday structure**: Kim & Hansen (arXiv 2607.09426, **A**, six Binance perps 2021–2024): order imbalance in the first seconds after each quarter-hour boundary predicts 4–12h cumulative returns (public component rising to ~17 bps IQR effect at 12h); out-of-sample R² for quarter-hour opening returns 3.4% combined, directional accuracy 57%. Removing funding bars does not change the result. Volatility is highest in the hour after 16:00 UTC and lowest near 05:00 UTC (arXiv 2109.12142, **A**). Day-of-week return effects have faded; only the volatility effect remains (**A**). Weekend/CME-gap trades are fading now that CME trades 24/7 (**C**).
- **Crisis alpha** (Man AHL, **B**): slow trend hedges equity drawdowns and inflation shocks; over ~1-week windows it does not have time to adapt. Do not expect 15m–4h signals to provide crisis alpha.

### 3.4 Microstructure, options and on-chain features

- **Lead-lag**: Binance led Hyperliquid on 29/29 assets by 600–700 ms (Arrakis, modified Hayashi–Yoshida, Feb 2026, **B**); HL's lag is explained by two-block maker/taker sequencing. CME micro futures' information share ≈ 0.56 vs Binance spot 0.44 at 55 ms (arXiv 2506.08718, **A**). This lead is both the main adverse-selection source for HL makers and an exploitable quote-skew signal.
- **Mark–oracle divergence**: HL mark = median of (oracle + 150s EMA of HL-mid minus oracle; HL BBO/last median; weighted median of Binance/OKX/Bybit/Gate/MEXC mids), updated ~3s (HL docs, **A**). Divergence is a direct read of HL-local pressure vs CEX consensus.
- **Options**: net buying pressure in OTM Deribit options predicts next-interval spot returns; ATM flow is vol-trader driven (arXiv 2109.02776, **A**). A 2026 SMU dissertation (**B**, unreviewed) finds the 25-delta 90-day risk reversal significant with Newey–West controls, more so than 30-day. Glassnode's GEX (**B**) infers dealer inventory from taker side, not OI sign, because the equity heuristic fails when retail buys calls. BTC volatility risk premium is larger than equities' and highest in low-vol regimes (Almeida et al., arXiv 2410.15195, **A**).
- **Order flow**: 10-second BTC OFI has limited predictive power (arXiv 2308.14235, **A**); OI-times-price matrices and long/short ratios have no academic support (**C**).
- **Liquidation cascades** (Garcia Seuma, arXiv 2607.27070, **A**, seven BTC cascades 2022–2025 incl. 10/10): no variable is event-invariant; the only population-level precursor is compression of taker order-flow variance. Heatmaps are OI plus assumed leverage, not observed liquidation prices (**C**).
- **On-chain**: USDT net inflows to exchanges predict BTC/ETH returns at 1–2h and vol at 6h (arXiv 2411.06327, **A**). Artemis cross-chain stablecoin flow factor for L1s: long-short Sharpe 1.67, long leg dominant (**B**).
- **Hyperliquid-specific data (A)**: `perpsAtOpenInterestCap` (crowding flag; HLP exempt), `perpDexLimits`, `metaAndAssetCtxs` premium/impact prices, on-chain TWAP sub-orders every 30s (visible, fadeable flow), builder-code fill dumps (retail-vs-pro segmentation), `vaultDetails` for HLP. Using HLP net inventory as a contrarian signal is plausible but untested (**C**).
- **Market impact** (Donier–Bonart on 1M+ Bitcoin metaorders, **A**): square-root law I = k·σ·√(Q/V) holds over four decades of size.

### 3.5 Market making and execution

- **Markout ladders** (industry practice, **B**): per-fill mid-to-mid markouts at 1s/5s/10s/30s/60s/5m, medians and percentiles, split by size bucket, time-to-funding and lead/lag state. Persistently negative markouts mean the quotes are being picked off regardless of spread capture.
- **Queue position has value** (Moallemi–Yuan, **A**): of the same order as the half-spread in large-tick names. Albers et al. (arXiv 2502.18625, **A**) document a negative correlation between maker fill probability and post-fill return. Rule: cross to taker when (half-spread − taker fee) > E[adverse move | fill].
- **Fill-probability models** (arXiv 2307.04863, **A**): survival analysis with censoring-aware loss on microstructure features; calibrating only on own fills is selection-biased.
- **Avellaneda–Stoikov in practice** (Hummingbot docs, hftbacktest GLFT recipe, **A**): reservation price r = s − qγσ²(T−t); κ estimated from book-snapshot deltas; `filled_order_delay` as an explicit post-fill cooldown; HL SDK `basic_adding.py` quotes ALO 0.3% from BBO and pauses quoting while position state is unknown.
- **Hyperliquid fees 2026 (A)**: perps taker 4.5 → 2.4 bps and maker 1.5 → 0 bps across tiers 0–6 on 14-day volume; maker rebates −0.1/−0.2/−0.3 bps at > 0.5%/1.5%/3% maker share; staking discounts 5–40%. Fee-tier awareness changes which carry trades clear the hurdle.
- **Virtu (S-1 and 10-K, A)**: one losing day in 1,238; the control stack is per-strategy lockdown, central strategy monitoring, aggregate exposure monitoring and operational controls, all independent of the alpha code.
- **RL execution** (arXiv 2511.07434, **A**, thin): beats TWAP by 2–3 bps at 30 min and 23 bps at 2h on one asset; Almgren–Chriss with a passive-first slice captures most of it at Phoenix's size.

### 3.6 Portfolio construction, regimes and risk

- **Pod-shop drawdown rules** (WSJ/Bloomberg reporting, **C** but consistent): at −5% of allocation capital is halved; at −7.5% the pod is usually cut. Losses are measured against allocated capital, so the rule doubles as a capital-allocation signal.
- **Downside correlation** (Ang–Chen 2002, **A**): downside correlations exceed normal-implied by ~12 points; regime-switching captures this best. HRP on a single covariance understates crash co-movement among BTC/ETH/SOL and between the carry and directional sleeves.
- **Volatility targeting** (Harvey et al. 2018; Moreira–Muir 2017, **A**): raises Sharpe for risk assets, reduces tail risk and vol-of-vol everywhere; Cederburg et al. find no systematic out-of-sample gain across 103 strategies after costs. Keep vol scaling at the position layer; be sceptical of per-signal scaling.
- **Fractional Kelly** (MacLean–Thorp–Ziemba, **A**): the fraction exists to absorb estimation error, which is largest in the mean; fractional Kelly with true parameters equals full Kelly on shrunk estimates. Size the fraction off the standard error of the edge.
- **Regime models**: statistical jump models (Nystrup–Kolm–Lindström; Shu–Yu–Mulvey arXiv 2402.05272, **A**) beat HMMs on persistence and out-of-sample Sharpe with costs; jump penalty chosen on validation Sharpe. Blog-grade HMM results and Hurst/ADX gates have no credible out-of-sample evidence (**C**). BOCPD on order-flow beats ARMA for forecasting but has no PnL backtest (**A**).
- **Stops** (Kaminski–Lo, **A**): under a random walk stops lower expected return; under momentum or regime switching they help. This justifies regime-conditional stop widths.
- **Drawdown control** (Grossman–Zhou, **A**): exposure ∝ (wealth − floor × running max); loses optimality in discrete time and can go negative-EV near the limit with proportional costs, so use a smoothed version.
- **Tail hedging** (Jane Street reporting, **B**): short-dated puts protect against gap crashes that stops cannot fill through; size to liquidation-cascade scenarios.
- **Covariance hygiene** (López de Prado MLAM, **A**): Marcenko–Pastur denoising, detoning the market eigenvector, NCO; NCO-CRISP beats baseline HRP in a crypto study (CEMA WP 928, **A**). With 3 assets this matters less than detoning the sleeve-return covariance.

### 3.7 Failure cases worth encoding

| Event | Mechanism | Encodable control |
|-------|-----------|-------------------|
| 12 Mar 2020 | BTC-collateral longs double-leveraged; cross-exchange arb broke on chain congestion; exchange outage halted the spiral | Pre-positioned capital per venue; deposit-latency assumption in cross-venue sizing |
| Luna/UST, 3AC, Alameda | "Market-neutral" books with illiquid collateral and venue concentration were directional | Collateral haircuts; per-venue exposure caps; no non-USDC collateral without multi-venue marks |
| JELLY (26 Mar 2025) | Thin-spot oracle manipulation forced a loss into HLP; validators settled at pre-manipulation price | OI/exposure caps as crowding flags; expect "validator put" discretion |
| 10 Oct 2025 | ~$19B liquidated, ~$10B on HL; OI fell ~60%; first cross-margin ADL on HL in 2+ years; top-of-book depth fell > 90%; USDe marked at $0.65 on Binance for 40 min | Depth-shock scenario (90% top-of-book loss) in CVaR sizing; ADL-rank monitor on hedged short legs; collateral marks from multiple venues |
| Ethena Aug 2024 | sUSDe APY 19% → 4% in 11 days on funding inversion | Carry exit horizon of 1–2 weeks, not hourly; funding-flip hard exit |
| Drift (1 Apr 2026) | ~$200–285M lost via compromised admin key | Admin/venue risk dominates strategy risk on secondary venues; hard per-venue caps |

HL engine facts to encode (HL docs, **A**): maintenance margin = half of initial at max leverage; book liquidation first, HLP backstop below 2/3 maintenance; positions > 100k USDC liquidated 20% at a time with 30s cooldown; ADL ranks counterparties by (mark/entry) × (notional/account value) and closes at previous mark.

### 3.8 ML practice and research process

- **Competitions** (G-Research Crypto, Jane Street 2021/2024, Optiver, Ubiquant, DRW, **B**): LightGBM dominates when features are hand-engineered and samples are few; NNs win on raw high-dimensional data. Universal wins: residualised targets (asset minus basket mean), purged grouped time CV, dropping fold-unstable features, cross-sectional means per timestamp as features, daily online fine-tuning layered on a static ensemble, regime-expert ensembles, final fit on the most recent data. G-Research's host: "feature engineering had a far greater impact than model development."
- **FreqAI (A, docs)**: Dissimilarity Index abstention (discard predictions far from the training manifold), outlier-protection guard (skip the retrain if > 30% of points are flagged), recency-weighted training rows, PCA to 0.999 variance, continual learning flagged as overfit-prone.
- **Freqtrade protections (A)**: StoplossGuard, MaxDrawdown, LowProfitPairs, CooldownPeriod: cheap, model-agnostic per-symbol circuit breakers.
- **Meta-labelling extensions** (Meyer–Barziy–Joubert 2023, **A**): probability calibration materially improves fixed sizing rules such as CDF-of-z sizing; average active bets; discretise sizes to reduce churn. Trend-scanning labels as a second label.
- **Multiple testing** (Harvey–Liu 2015; Bailey et al. MinBTL, **A**): t = 3 can be insignificant after multiple testing; ~45 configurations on a 5-year sample yield in-sample SR ≈ 1 with out-of-sample ≈ 0. The DSR is only as honest as its trial count.
- **Model decay** (QuantBench arXiv 2504.18600, **A**): more frequent updates slow alpha decay; retrain cadence should track the shortest signal half-life, with an IC-breach trigger.
- **Incubation** (practitioner consensus, **C**): backtest → paper → small-size live for N trades (not N days) → full size; reconcile realised vs modelled slippage per fill; pre-register kill thresholds from the CPCV path distribution (e.g. live Sharpe ≥ 80% of backtest).
- **Grid/DCA/martingale** (Passivbot docs, exchange disclosures, **A/C**): negative skew, no edge; bankruptcy distance ≈ 1/wallet-exposure; "human intervention is the final backstop." Only the loss-allowance pacing budget is transferable.

---

## 4. Extraction: the reinforcement plan for Phoenix

Each item lists what to change, where, suggested parameters, the evidence grade, and the gate it must pass before promotion. All promotions go through the existing CPCV gatekeeper (PSR ≥ 0.95, DSR ≥ 0.95, PBO < 0.30, E[net] > 15 bps) **with the trial registry of 4.10 feeding the DSR's N**.

### Tier 0: make the live loop run the strategy (prerequisite)

File: `ui/api_server.py` (`step_simulation_tick`, `_fit_worker_on_bars`, `sleeve_telemetry`).

1. Call `FundingCarrySleeve.evaluate_from_snapshot` and `DerivativesAlphaSleeve.generate_alpha_signal` each bar for all three assets; feed their `SleeveSignalOutput` into `AssetAllocationInput` instead of constants.
2. Compute `w_hrp` from a rolling return matrix via `compute_hrp_cvar_weight_dict` (e.g. 96 bars of 15m returns, refreshed hourly).
3. Remove `p_eff = max(0.58, p_meta)` and the `else 1` direction defaults. A 0 direction must produce a 0 weight.
4. Replace synthetic high/low/volume/liquidation fields with real data: Bybit 15m klines already exist in `_fetch_initial_bybit_15m_history`; add HL `candleSnapshot` for live bars, and a liquidation feed (Bybit `allLiquidation` websocket or HL on-chain liquidation fills). Until a real feed exists, set `liquidation_volume_usd` to 0 and mark the feature as unavailable rather than random.
5. Introduce a position lifecycle: entry at signal, exit on triple-barrier touch (using live `sigma_bar` from realised vol, not 0.0045), vertical barrier, or regime veto. Count wins and losses per closed trade. Charge fees on actual turnover.
6. Carry PnL must be signed and conditional on an open carry position; the spot leg's funding does not exist, the perp leg's funding does.
7. In `_fit_worker_on_bars`, skip the retrain when labels are single-class instead of fabricating labels. Retrain on the full Parquet history with purging, not the last 200 bars.
8. Feed real inputs to the macro sentinel (Deribit DVOL endpoint, USDC/USDT/USDe multi-venue marks) and breadth analyst (per-asset 1h funding).

Gate: the E2E test suite should include a replay test where known synthetic signals produce the expected weights, and a test asserting zero direction yields zero weight.

### 4.1 Funding carry sleeve

File: `sleeves/sleeve_3_funding_carry.py`. Evidence: **A** (He et al.), **A/B** (Ethena, FTI, HL docs).

- **Deviation signal**: compute ρ = κ·(mark − oracle)/oracle − (r − r′) per hour using HL `premium` from `metaAndAssetCtxs` (κ = 8760 for hourly funding). Enter when ρ > +κ·log(1 + C) where C = round-trip cost from `FeeSchedule` plus impact; exit when ρ first crosses zero. Keep the current breakeven/net-edge checks as a secondary filter.
- **Smoothing**: use a 3-day rolling mean of funding (Paradigm practice) for entries; keep the instantaneous rate for exits.
- **Hard exits**: 3-day average funding < 0, or instantaneous funding < −1%/day, or ρ < 0 for 6 consecutive hours.
- **Crowding cut**: if annualised funding > 25% and OI is above its 90th rolling percentile (or `perpsAtOpenInterestCap` lists the asset), halve carry notional and permit a short-bias overlay in the derivatives sleeve.
- **Conditioning**: add trailing 120-day return as a regime input to expected carry (positive past returns predict higher ρ).
- **Cross-venue**: replace the fixed 7 bps spread rule with `predictedFundings` across HL/Binance/Bybit, normalised to per-day, with Hummingbot's exit triggers (spread stop at −0.1%/day, take-profit on cumulative funding plus basis). Sample the spread ~2h after settlement, when it peaks.
- **Replace heuristic `meta_prob`** (currently 0.70 + 0.25·edge/100) with the calibrated probability from a small model trained on realised carry outcomes, or at minimum a monotone map validated on history.
- **ADL monitor**: compute own (mark/entry) × (notional/equity) for the short leg and de-lever when it ranks high relative to the HL ADL queue during cascade indicators (4.9).
- **Fee tiers**: pull the live fee tier and staking discount; at tier ≥ 3 the maker leg is near 0 bps and many currently rejected carries clear.

Gate: CPCV on hourly funding history 2022–2026 for BTC/ETH/SOL; require E[net] > 15 bps per round trip and a max drawdown below the Ethena 11-day inversion scenario.

### 4.2 New trend sleeve

New file: `sleeves/sleeve_2_trend.py`. Evidence: **A**.

- Signals: continuous time-series momentum over three lookbacks scaled to Phoenix's clock (1 day, 1 week, 1 month on 1h bars), each in [−1, 1] via a tanh of return / realised vol; equal-weighted across lookbacks (Hurst–Ooi–Pedersen), optionally a Donchian ensemble as a second opinion (Zarattini et al.).
- Sizing: per-instrument inverse-vol scaling with a 25-day EWMA (Carver), capped forecast, and the Baltas–Kosowski correlation adjustment: scale gross by 1/√(1 + (N−1)·ρ̄) where ρ̄ is the average pairwise correlation over the last 30 days.
- Horizon: 1–4 week holding is where the crypto evidence sits. This sleeve is the slow hedge for the carry sleeve; it is not a 15m signal.
- Output: `SleeveSignalOutput` with `sleeve_id = "trend"`, direction = sign, `meta_probability` from the calibrated LightGBM meta-labeller trained on trend-sleeve entries only.

Gate: CPCV on 1h bars; require the sleeve's PnL correlation with the derivatives sleeve to be below 0.7 (WorldQuant admission rule) and positive crisis-period PnL in the Aug 2024, Feb 2025 and Oct 2025 windows.

### 4.3 Feature expansion for the meta-labeller

Files: `core/features/`, `core/models/meta_labeling_worker.py` (extend `EXTENDED_CONTINUOUS_COLUMNS`). Evidence: **A/B** as noted.

Add, in this priority:

1. Quarter-hour boundary order imbalance: signed taker volume in the first 10–60 s after each :00/:15/:30/:45 (Kim & Hansen, **A**). Requires a trade stream (HL `trades` websocket or Bybit `publicTrade`).
2. Basket-residualised return: asset 1h return minus the BTC/ETH/SOL mean (G-Research 3rd place, **B**); and residualised target for training.
3. 1-week time-series momentum and its sign (Liu & Tsyvinski, **A**).
4. Hour-of-day and NYSE-open flag as conditioning features, not standalone alphas (**A** for the vol pattern).
5. Mark–oracle divergence in bps and its 1h change (HL docs, **A**).
6. Rolling variance of taker-flow imbalance over 24–48 bars; compression is cascade fragility (Garcia Seuma, **A**).
7. Realised Amihud (|return| / dollar volume) and Kyle-λ regression slope over 96 bars (Deng–Zhou, **A/B**).
8. USDT exchange net inflow at 1–4h, if a data source is available (arXiv 2411.06327, **A**).
9. Funding z-score over 30 days with a clamp flag (BitMEX, **B**).
10. Deribit OTM taker-flow imbalance and 25-delta 90-day risk reversal (see 4.7).

Training changes: sample weights by label uniqueness, recency weighting (FreqAI `weight_factor`), fold-stability pruning (drop features whose importance is strong in one CPCV fold and weak in others, DRW 1st place), and a trend-scanning label as a second target for the 1–4h horizon.

Gate: each feature must raise out-of-fold calibrated log-loss or IC in CPCV with its own trial logged; batch additions are not accepted.

### 4.4 Allocation: partial adjustment, impact and calibration

File: `risk/portfolio_allocator.py`, `core/contracts.py`. Evidence: **A**.

- **Aim portfolio**: compute the target w* as today, then move w_t = w_{t−1} + λ·(aim − w_{t−1}) with λ set per sleeve from measured signal half-lives (fit IC vs lag; start with λ = 1 − exp(−Δt / half-life)). The aim overweights slow signals relative to fast ones in proportion to half-life / cost.
- **Impact term**: subtract I = k·σ·√(Q/V) from expected return in `compute_signed_target_weights`, with k calibrated from own fills (start k = 0.5) and V = HL 24h volume for the asset.
- **Kelly fraction**: replace the constant γ = 0.5 with γ = 0.5 · min(1, edge / (2·SE(edge))), so that low-confidence edges shrink toward quarter-Kelly. Average active bets across open signals and discretise to 0.05 steps to reduce churn.
- **Sleeve combination**: when more than one sleeve has a view on an asset, combine by IC-weighted average shrunk 50% toward equal weights, not by a stacker.
- **Covariance**: feed HRP a regime-conditional covariance (downside or jump-model state, 4.8) and detone the sleeve-return covariance before clustering.
- **Transfer coefficient**: log the correlation between raw sleeve scores and executed weights; alert if it drops below 0.5.

Gate: turnover and net PnL comparison in CPCV against the current threshold policy; promotion requires lower turnover at equal or better net PnL.

### 4.5 Circuit breakers and drawdown governance

New file: `risk/circuit_breakers.py`; wire into `ChampionChallengerState`. Evidence: **B/A**.

- Per-sleeve soft limit: −5% of the sleeve's allocated capital halves its gross; hard limit −7.5% disables the sleeve and requires re-promotion through CPCV.
- Per-sleeve per-asset: StoplossGuard (lock after 3 stop-outs in 48 bars), CooldownPeriod (4 bars after any exit), LowProfit (disable the pair if 20-trade rolling net PnL < 0).
- Loss-allowance pacing: a daily realised-loss budget per sleeve (Passivbot pattern) that halts new entries when exhausted.
- Portfolio: smoothed Grossman–Zhou multiplier m = clip((E − 0.85·M_decayed)/(0.15·M_decayed), 0, 1) where M_decayed is the running max decayed at 1%/day, applied on top of the regime multiplier.
- Operational (Virtu): per-strategy lockdown on anomaly (stale feed > 30s, order reject storm, position state unknown), aggregate exposure monitor across venues, cancel orphan orders on a timer, never quote while position state is unknown.

Gate: unit tests for each breaker; replay tests on the Oct 2025 and Feb 2025 windows.

### 4.6 Execution upgrades

File: `core/execution/defensive_maker.py`, `core/execution/vpin_hawkes.py`, `core/data_ingestion.py`. Evidence: **A/B**.

- Subscribe to Binance and Bybit mid for the same asset; pull or re-price HL quotes when the leading mid moves more than k × HL spread (start k = 0.5) within the last 700 ms. This is the single largest adverse-selection reducer available on HL.
- Markout ladder per fill at 1s/5s/10s/30s/60s/5m, split by size bucket, minutes-to-funding and lead/lag state; feed the ladder back into the toxicity shield's threshold.
- Queue-position estimate from L2 deltas; do not cancel/replace when queue position is in the front third unless the lead signal fires.
- Maker/taker switch: cross when (half-spread − taker fee) > expected adverse move given Hawkes intensity and queue state; train a censoring-aware fill-time model on HL L2 plus own ALO orders.
- Post-fill cooldown (`filled_order_delay` ≈ 30–60 s) and inventory-skewed reservation price for carry hedge legs; funding-aware skew leaning inventory toward the side being paid.
- Fee-tier and rebate awareness: track 14-day volume tier and maker share; route the carry leg to the venue with the lowest all-in cost from `predictedFundings` plus fees.
- Larger entries through a parent-order / child-slice abstraction (Nautilus TWAP pattern) with passive-first slices.

Gate: markout distribution must improve (median 30s markout less negative) in paper replay before any threshold change ships.

### 4.7 Derivatives alpha sleeve

File: `sleeves/sleeve_1_derivatives_alpha.py`. Evidence: **A/B**.

- Fix the Deribit parse: compute a true 25-delta risk reversal from per-instrument delta (`get_book_summary_by_currency` lacks greeks; use `get_instruments` plus `ticker` or `get_order_book` for mark IV and delta), at 30-day and 90-day tenors; compute the term structure from expiry dates, not list order; compute GEX from taker-side flow (buy minus sell volume per strike times gamma), not OI sign.
- Use the HLP `vaultDetails` position marks at mark price, not entry price, and treat HLP inventory as an untested feature until CPCV says otherwise.
- Add OTM taker-flow imbalance as a feature (arXiv 2109.02776).
- Replace the 0.40/0.30/0.20/0.10 weights and the `0.50 + 0.45·|composite|` probability with: (a) each component as a feature in the LightGBM meta-labeller, or (b) if kept as a composite, IC-weighted combination shrunk toward equal weights with sigmoid calibration on realised outcomes.
- Gate the volatility-risk-premium use to the low-vol regime (Almeida et al.); consider a small delta-hedged short-vol sleeve later, only after the regime model (4.8) exists.

Gate: component-level IC by horizon on 2021–2026 Deribit history before combination; the composite must beat its best single component out of fold.

### 4.8 Regime layer

Files: `sentiment/`, `risk/`. Evidence: **A**.

- Fit a statistical jump model (penalised k-means on realised vol, funding, basis, DVOL, OI change, average pairwise correlation; jump penalty tuned on validation Sharpe) producing 3–4 persistent states. Retrain weekly; infer online.
- Use the state to (a) select the HRP covariance, (b) set stop widths (tight in trend states, wide or none in range states per Kaminski–Lo), (c) set the carry/directional split, (d) shrink Kelly in the crisis state.
- Keep the LLM risk committee as a veto (`tail_risk_flag`, emergency flatten) rather than the primary continuous multiplier; its inputs must be live (Tier 0).
- BOCPD run-length on Hawkes intensity as a reset trigger for the toxicity shield's baseline.

Gate: the jump-model state must improve out-of-fold net PnL of at least one sleeve versus the unconditional policy.

### 4.9 Cascade and ADL controls

Files: `risk/`, `core/data_ingestion.py`. Evidence: **A/B**.

- Ingest `perpsAtOpenInterestCap`, `perpDexLimits`, HL liquidation fills and on-chain TWAP sub-orders; expose OI percentile, funding annualised, taker-flow variance ratio.
- Fragility score = f(taker-flow variance compression, OI percentile > 90, funding > 25% annualised, depth at 10 bps falling > 50% over 1h). Above threshold: halve directional gross, de-lever the winning leg of any hedge (ADL risk), widen maker quotes or pause.
- Collateral: mark any non-USDC collateral at the median of three venues; haircut 10% when venue marks disagree by more than 1%.
- CVaR sizing scenario: 90% top-of-book depth loss with 20%-per-30s forced liquidation mechanics (HL docs).
- Optional tail hedge: Deribit OTM puts sized to the liquidation-cascade scenario for the carry book, re-evaluated weekly.

Gate: replay the 10 Oct 2025 window from the HL L2 archive; the controls must cut the simulated drawdown without disabling trading in the preceding calm month.

### 4.10 Research process hardening

Files: `research/`, `research/validation/cpcv_gatekeeper.py`. Evidence: **A/B**.

- Trial registry: every backtest configuration per sleeve is logged (features, parameters, window) and its count feeds the DSR's number of trials; add the MinBTL check for the short crypto history.
- Fold-stability pruning and basket-residualised targets as standard pipeline steps.
- Online adaptation layer: a daily warm-started ridge or small GBDT on the residuals of the static LightGBM (Jane Street 2024 / Optiver Close pattern); Dissimilarity-Index abstention and the outlier-protection guard before any prediction.
- Half-life dashboard: rolling IC by lag per feature; retrain cadence tied to the shortest half-life; retrain trigger on IC breach.
- Shadow run: every promoted model runs at zero or minimum size for N = 200 trades, with per-fill reconciliation of realised versus modelled slippage and markouts; kill thresholds pre-registered from the CPCV path distribution (e.g. live Sharpe < 80% of the CPCV median path, or drawdown beyond the 5th percentile path).
- Monte Carlo trade-order shuffling (Jesse pattern) and MAE/MFE by regime added to the tournament report.

Gate: none needed; these are process controls.

---

## 5. What not to adopt, and why

| Idea | Verdict | Reason |
|------|---------|--------|
| Grid / DCA / martingale bots (Passivbot, 3Commas, Pionex) | Reject | Negative skew with no edge; bankruptcy distance ≈ 1/exposure; vendors admit trending markets cause large losses |
| Copy-trading Hyperliquid leaderboard wallets | Reject as a strategy | ~14% of HL traders net profitable; hedges on other venues invisible; copy flow moves price before the poller sees it. "Profitable-cohort net positioning vs crowd" is acceptable as a feature to test |
| Liquidation heatmaps as entry triggers | Reject | They are OI plus assumed leverage, not observed prices; no event-invariant precursor exists |
| OI-times-price direction matrices, long/short ratio | Reject as standalone | No academic support; OI is two-sided by construction. Acceptable only as a conditioning feature with funding sign |
| Hurst / ADX / variance-ratio regime gates | Reject | Biased on short windows; ~50% hit rate in tests; no out-of-sample evidence |
| Blog-grade HMM gating | Reject | Transition-probability drift and init sensitivity; use jump models instead |
| Weekend / CME-gap trades | Reject | CME trades 24/7; structural basis gone |
| Per-signal volatility scaling | Deprioritise | No systematic out-of-sample gain after costs across 103 strategies; scale at the position layer |
| Stacked meta-learners for sleeve blending | Deprioritise | Forecast-combination puzzle; use equal or IC-shrunk weights unless CPCV validates the stacker |
| Hummingbot indicator controllers, NostalgiaForInfinity templates | Reject | Naive indicators; no published live performance |
| RL execution | Deprioritise | Gains only material at ≥ 1h child-order horizons; Almgren–Chriss with passive-first slices suffices at Phoenix's size |

---

## 6. Open research gaps (no credible evidence either way)

Basis momentum in crypto; crypto momentum crashes; independent validation of GEX effects in BTC; HLP net inventory as a signal; funding-settlement (00/08/16 UTC) return effects; cross-sectional momentum with only three assets; HIP-3 markets as a signal or venue (thin books, 2x fees outside growth mode, deployer-oracle risk).

---

## 7. Sources

Quant-firm practice
- Kakushadze, "101 Formulaic Alphas": https://arxiv.org/abs/1601.00991
- Clarke, de Silva, Thorley, fundamental law critique: https://newfrontieradvisors.com/media/1744/fundamental-law-september-2019.pdf
- Gârleanu & Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs": https://www.nber.org/papers/w15205
- Senate PSI hearing on Renaissance basket options (2014): https://www.hsgac.senate.gov/wp-content/uploads/imo/media/doc/STMT%20-%20Renaissance%20(July%2022%202014).pdf
- Two Sigma, "A Machine Learning Approach to Regime Modeling": https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling
- Hudson River Trading, machine learning page: https://www.hudsonrivertrading.com/machine-learning/
- Virtu S-1 (2014): https://www.sec.gov/Archives/edgar/data/0001592386/000104746914002070/a2218589zs-1.htm
- Jane Street July 2025 loss reporting: https://www.gulf-times.com/article/731123/business/jane-street-takes-15bn-hit-in-july-tied-to-ai-sell-off/amp
- Millennium drawdown rules (reporting): https://www.businessinsider.in/stock-market/news/the-strategy-a-69-billion-hedge-fund-uses-to-make-sure-it-never-loses-money-in-the-stock-market/amp_articleshow/113885877.cms
- Elliott, forecast combination: https://econweb.ucsd.edu/~gelliott/Combination.pdf

Trend, momentum, carry, reversal
- Moskowitz, Ooi, Pedersen, "Time Series Momentum": https://econpapers.repec.org/RePEc:eee:jfinec:v:104:y:2012:i:2:p:228-250
- Hurst, Ooi, Pedersen, "A Century of Evidence on Trend-Following": https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing
- Baltas & Kosowski, correlation-adjusted trend: https://papers.ssrn.com/abstract=2140091
- Liu & Tsyvinski, "Risks and Returns of Cryptocurrency": https://www.nber.org/system/files/working_papers/w24877/w24877.pdf
- Liu, Tsyvinski, Wu, "Common Risk Factors in Cryptocurrency": https://www.nber.org/papers/w25882.pdf
- Zarattini, Pagani, Barbon, "Catching Crypto Trends": https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/
- SMA/EMA parameter instability on BTC: https://ar5iv.labs.arxiv.org/html/2009.12155
- Koijen, Moskowitz, Pedersen, Vrugt, "Carry": https://econpapers.repec.org/RePEc:eee:jfinec:v:127:y:2018:i:2:p:197-225
- Nagel, "Evaporating Liquidity": https://www.nber.org/papers/w17653
- Large-coin daily momentum vs small-coin reversal (IRFA 2021): https://www.scinapse.io/papers/3204386317
- Intraday volatility pattern in crypto: https://arxiv.org/pdf/2109.12142v2.pdf
- Man AHL, trend in crises: https://www.man.com/insights/trend-following-generate-alpha

Funding, basis, perps
- He, Manela, Ross, von Wachter, "Fundamentals of Perpetual Futures": https://arxiv.org/html/2212.06888v5
- Ackerer, Hugonnier, Jermann, "Perpetual Futures Pricing": https://arxiv.org/abs/2310.11771
- Schmeling, Schrimpf, Todorov, crypto carry: https://arxiv.org/pdf/2510.14435
- BitMEX Q3-2025 derivatives report: https://www.bitmex.com/blog/2025q3-derivatives-report
- BitMEX funding mean-reversion study (2018): https://www.bitmex.com/blog/funding-mean-reversions-2018
- Funding-aware market making on Hyperliquid: https://arxiv.org/abs/2605.06405
- Ethena funding risk docs: https://docs.ethena.fi/protocol-overview/risks/funding-risk.md
- Chaos Labs Ethena risk assessment: https://chaoslabs.xyz/resources/chaos_Labs_ethena_perpetual_assessment_risk_report.pdf
- Funding persistence across 26 exchanges (MDPI 2026): https://www.mdpi.com/2227-7072/14/5/103/htm
- Paradigm, basis trading guide: https://paradigm.co/blog/death-of-the-degen-trading-basis-for-yield
- FTI, October 2025 crash: https://www.fticonsulting.com/insights/articles/crypto-crash-october-2025-leverage-met-liquidity
- Hummingbot funding-rate arbitrage script: https://github.com/hummingbot/hummingbot/blob/development/scripts/v2_funding_rate_arb.py

Microstructure, options, on-chain
- Kim & Hansen, quarter-hour effect in crypto perps: https://arxiv.org/html/2607.09426v2
- Short-horizon OFI in BTC: https://arxiv.org/pdf/2308.14235
- Deribit OTM net buying pressure: https://arxiv.org/abs/2109.02776
- 25-delta risk reversal predictability (SMU dissertation): https://ink.library.smu.edu.sg/etd_coll/887
- Glassnode gamma exposure methodology: https://research.glassnode.com/gamma-exposure
- Almeida et al., BTC volatility risk premium: https://arxiv.org/abs/2410.15195
- Garcia Seuma, liquidation cascade precursors: https://arxiv.org/abs/2607.27070
- Stablecoin exchange inflows and returns: https://arxiv.org/pdf/2411.06327
- Artemis crypto factor model: https://research.artemisanalytics.com/p/crypto-factor-model-analysis-launching-ae0
- Arrakis, Binance leads Hyperliquid by 600–700 ms: https://arrakis.finance/blog/crypto-price-discovery
- CME vs Binance price discovery: https://arxiv.org/abs/2506.08718
- Donier & Bonart, square-root impact in Bitcoin: https://www.arxiv.org/abs/1412.4503
- Deng & Zhou, Amihud illiquidity in crypto: https://arxiv.org/pdf/2306.15807

Hyperliquid documentation
- Perpetuals info endpoint (predictedFundings, perpsAtOpenInterestCap, perpDexLimits): https://hyperliquid.gitbook.io/Hyperliquid-docs/for-developers/api/info-endpoint/perpetuals
- Robust price indices (mark/oracle): https://hyperliquid.gitbook.io/Hyperliquid-docs/trading/robust-price-indices
- Liquidations: https://hyperliquid.gitbook.io/Hyperliquid-docs/trading/liquidations
- Auto-deleveraging: https://hyperliquid.gitbook.io/Hyperliquid-docs/trading/auto-deleveraging
- Fees: https://hyperliquid.gitbook.io/Hyperliquid-docs/trading/fees
- Builder codes: https://hyperliquid.gitbook.io/Hyperliquid-docs/trading/builder-codes
- HIP-3: https://hyperliquid.gitbook.io/Hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals
- Official SDK `basic_adding.py`: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_adding.py
- JELLY incident (Arkham): https://info.arkm.com/research/jellyjelly-exploit-on-hyperliquid
- 10/10 autopsy (The Defiant): https://thedefiant.io/newsletter/defi-daily/the-ultimate-10-10-crash-autopsy
- ADL debate summary: https://thogiti.github.io/2025/12/11/Autodeleveraging-Hyperliquid-653M-debate.html

Execution and market making
- Hummingbot Avellaneda–Stoikov docs: https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/
- hftbacktest GLFT recipe: https://hftbacktest.readthedocs.io/en/latest/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html
- Moallemi & Yuan, queue value: https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/25461/queue-value-2016.pdf
- Albers et al., maker fills and post-fill returns: https://arxiv.org/pdf/2502.18625
- Fill-probability survival models: https://arxiv.org/pdf/2307.04863v2
- Markout definition: https://kx.com/glossary/what-is-markout-and-how-is-it-calculated/
- RL-Exec: https://arxiv.org/pdf/2511.07434.pdf
- NautilusTrader execution concepts: https://nautilustrader.io/docs/latest/concepts/execution

Portfolio, regimes, risk
- Ang & Chen, asymmetric correlations: https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/1516/corr.pdf
- Harvey et al., impact of volatility targeting: https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf
- Moreira & Muir, volatility-managed portfolios: https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf
- Cederburg et al., out-of-sample vol management: https://econpapers.repec.org/RePEc:eee:jfinec:v:138:y:2020:i:1:p:95-117
- MacLean, Thorp, Ziemba, "Good and Bad Properties of the Kelly Criterion": https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
- Shu, Yu, Mulvey, statistical jump models: https://arxiv.org/abs/2402.05272v1
- Nystrup, Kolm, Lindström, online persistent state classification: https://orbit.dtu.dk/en/publications/greedy-online-classification-of-persistent-market-states-using-re/
- Kaminski & Lo, "When Do Stop-Loss Rules Stop Losses?": https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf
- Grossman & Zhou, drawdown control: https://ideas.repec.org/a/bla/mathfi/v3y1993i3p241-276.html
- López de Prado, "Machine Learning for Asset Managers" (denoising, NCO): https://papers.ssrn.com/abstract=3469961
- NCO vs HRP in crypto (CEMA WP 928): https://ideas.repec.org/p/cem/doctra/928.html

ML practice and research process
- G-Research crypto competition wrap-up: https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/
- Jane Street 2024 8th-place write-up (online fine-tuning): https://www.kaggle.com/c/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution
- DRW crypto 1st place (fold-stability pruning): https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st
- Ubiquant 1st place: https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy
- Numerai scoring and neutralisation: https://docs.numer.ai/numerai-tournament/scoring
- FreqAI parameter table: https://www.freqtrade.io/en/latest/freqai-parameter-table/
- Freqtrade protections: https://www.freqtrade.io/en/latest/includes/protections/
- Meta-labeling and calibration (Meyer, Barziy, Joubert): https://en.wikipedia.org/wiki/Meta-Labeling
- Harvey & Liu, "Backtesting": https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF
- Bailey, Borwein, López de Prado, Zhu, backtest overfitting: https://papers.ssrn.com/abstract=2326253
- QuantBench, alpha decay vs update frequency: https://arxiv.org/pdf/2504.18600
- Passivbot risk management: https://github.com/enarjord/passivbot/blob/master/docs/risk_management.md
- Hyperdash profitability stat (via BeInCrypto): https://beincrypto.com/hyperliquid-traders-profitability
