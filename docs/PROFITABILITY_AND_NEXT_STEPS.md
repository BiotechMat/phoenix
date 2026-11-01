# Profitability and next steps: research memo

Date: 1 November 2026
Basis: branch `claude/intelligent-sagan-q8qknt` (all six reinforcement phases implemented, results recorded in `docs/REINFORCEMENT_PLAN.md`), the board charter and open questions on `origin/main`, three external research passes, and a direct pull of Hyperliquid's funding history on 1 November 2026.
Stage labels: every figure is marked **backtest**, **paper**, **live** or **external**. Nothing in Phoenix has traded live. Nothing here is a profitability claim; under the charter that needs the CFO's reconciliation of a live ledger.

---

## 1. The answer in one page

**Where the system stands.** The reinforcement programme did what it was meant to do: the live loop now runs the real strategy stack, every candidate was tested honestly, and most candidates failed their gates. One sleeve is live on paper, funding carry v1. The paper ledger holds one trade. The derivatives sleeve reports but does not size. The trend sleeve passed on portfolio statistics but failed its per-trade gate and ships off. The directional meta-labelled model never trades under the live sizing rule and loses slightly under a corrected one.

**The profitability finding.** The carry sleeve's backtest edge (65 to 72 bps net per round trip, June 2023 to October 2026) is an average dominated by 2024. On Hyperliquid's own hourly funding history, which I re-pulled and verified on 1 November 2026:

| Share of hours with funding at or above the sleeve's 3 bps per 8h entry threshold | BTC | ETH | SOL |
|---|---|---|---|
| 2024 | 21.7% | 23.4% | 26.8% |
| 2025 | 3.7% | 3.3% | 3.4% |
| 2026 to date | 0.08% | 0.05% | 0.08% |
| Longest spell below the threshold, 2026 | 229 days | 233 days | 233 days |

Funding captured while the condition held was 15 to 21% of notional in 2024, 1.7 to 2.0% in 2025 and 0.02 to 0.04% in 2026 (external, venue API, my computation). An independent simulation of the sleeve's rule on the same data earned roughly 2% on pair notional in 2025 and nothing in 2026. **The only sleeve that passes its gates has had no entry condition to act on for most of this year.**

**The economics at $5,000.** Idle USDC earns about 4% risk-free (3-month T-bill 4.00% on 29 October 2026) and 4 to 5% in USDC lending on Hyperliquid's own HyperCore, HyperLend or Morpho (30-day means, external). An always-on short-perp carry on BTC/ETH at 2026 funding levels grosses 5.5 to 6% on perp notional, which on this book is $100 to $200 a year net after rebalances and with half the capital idle in the spot leg. That is below the passive floor on the same cash. The trend backtest's own realised rate is 5.6% a year, about $280 on $5,000, with a one-year standard error around $500. The cash run rate is about $1,116 a year (CFO estimate). No evidenced yield covers it below roughly $11,000 of capital at Sharpe 1, or $28,000 to $56,000 at realistic carry yields; the all-in hurdle including the founder's time needs on the order of $100,000 or more.

**What a paper record can show.** A 30-day record has about 9% power to distinguish Sharpe 1 from Sharpe 0; 90 days has 12 to 13%; one year has 26%. The charter's live kill rule (30-day PSR below 0.80) would kill a true Sharpe-1 strategy about 71% of the time. Paper trading validates plumbing, fills, fees and reconciliation. It cannot validate edge on any horizon this venture can afford.

**Recommendation.** Treat the $5,000 book as a research and operations budget, not a business, and say so in the decision register. Stop optimising the directional model at this scale. Re-base the carry sleeve on a hurdle (expected funding must beat the lending yield plus costs) rather than a fixed 3 bps threshold, and record its 2025 and 2026 results separately. Re-gate the trend sleeve as a pooled, per-unit-time strategy, extend its breadth, and keep it in shadow. Build the live order gateway and kill path now, because operational learning is the one thing a small live tranche can buy. Set the return target, the paper-record purpose and the kill box as decisions (Section 8). The largest structural opportunity visible in the data, the 2 to 5.5 point funding premium Hyperliquid pays shorts over Binance, is a cross-venue trade the board has deferred on venue and jurisdiction grounds; it needs capital on two venues and a CCO review before it can be researched further.

---

## 2. What the programme established (verified on the branch)

| Component | Status on branch | Evidence (stage) |
|---|---|---|
| Live loop integrity (Phase 0) | Done. Real candles, real sleeves, HRP from returns, position lifecycle, signed funding, notional fees, no random data, honest retraining | `tests/test_phase0_integrity.py`, 25 tests |
| Data and research infra (Phase 1) | Done. Binance archive backfill June 2023 to October 2026 with OI, trade/liquidation/lead-mid streams, trial registry feeding DSR, replay harness, feature evaluation | Phase 1 gate met |
| Carry v1 | **Live on paper.** 72 / 68 / 65 bps net per round trip BTC/ETH/SOL, max drawdown under 0.5% | backtest, settlement-level, June 2023 to Oct 2026; basis leg uses the smoothed premium index, Sharpe overstated |
| Carry v2 (deviation band) | Passes CPCV at 24 / 21 / 27 bps per trip; dominated by v1; off | backtest |
| Trend sleeve | +20.2% total, Sharpe 1.25, max drawdown 4.3% over 3.4 years; fails E[net per trade] (1.7 / 4.9 / 14.3 bps) and DSR on BTC and SOL (0.88 to 0.89); off, shadow-capable | backtest, 3 trials logged |
| Directional meta-labeller | No trades under even-odds Kelly; 28 trades per CPCV path at median per-trade Sharpe −0.033 with payoff-aware Kelly; challenger −0.098; off | backtest, 9 CPCV paths, last year of BTC |
| Derivatives sleeve inputs | Fixed (true 25-delta risk reversal, expiry-based term structure, gamma-based GEX, HLP child-account marks); awaits surface history; allocation off | n/a |
| Jump-model regime | Persistent states; no out-of-fold improvement for trend or carry; not wired | backtest |
| Partial adjustment | Worse than a 5% buffer for trend (Sharpe 0.93 vs 1.25); buffer kept | backtest |
| Circuit breakers, cascade controls | Built and tested; breakers alone cut the October 2025 replay drawdown from $49 to $12; cascade controller changed nothing in that window | replay on backfill, few trades |
| Execution | Lead-venue pull cuts adverse fills (BTC 25.8% to 19.5%); every passive quoting policy loses after the tier-0 maker fee; queue-aware policy not better than re-pegging | 25-minute tick recording |
| Shadow workflow, decay monitor, online residual layer | Built; no sleeve has entered the 200-trade shadow run; residual layer passes narrowly (log-loss 0.2356 to 0.2332) | backtest |
| Paper ledger | 1 trade, net −$0.08 | paper, 31 October 2026 |

Two housekeeping facts matter for what follows. The implementation branch does not contain the board constitution (`CLAUDE.md`, `docs/board/`), which lives on `origin/main`; the two lines need merging before the board can review the code it governs. And the live-loop retrain gate and the Phase 0 fixes answer most of the board's first-meeting findings (probability floor, direction default, fabricated labels and bar fields, absolute-value carry, constant fees), so the proposed decision D-0002 should be re-checked against this branch rather than the state it described.

---

## 3. The funding regime (external, verified)

Hyperliquid's funding is the premium plus a clamp toward a 0.01% per 8 hours interest term, paid hourly. That term is a floor of 10.95% annualised when the premium is near zero. The data show the market has sat on that floor for most of 2025 and 2026.

| Hyperliquid hourly funding | Mean annualised | Hours positive | Hours at or above 3 bps/8h | Longest negative run |
|---|---|---|---|---|
| BTC 2024 / 2025 / 2026 YTD | 24.1% / 10.6% / 5.4% | 96% / 91% / 79% | 21.7% / 3.7% / 0.08% | 11 h / 31 h / 45 h |
| ETH 2024 / 2025 / 2026 YTD | 22.0% / 8.5% / 5.9% | 93% / 83% / 82% | 23.4% / 3.3% / 0.05% | 25 h / 38 h / 64 h |
| SOL 2024 / 2025 / 2026 YTD | 28.1% / 5.3% / 0.7% | 93% / 76% / 64% | 26.8% / 3.4% / 0.08% | 28 h / 75 h / 125 h |

Source: `fundingHistory` on `api.hyperliquid.xyz/info`, pulled 1 November 2026 (24,849 hourly rows per asset from 1 January 2024). The research agent's independent pull matched to the decimal.

Within 2026 there were two regimes. January to April was low or negative (BTC monthly means 7.3%, 2.3%, 0.9%, −0.8%; SOL −17.6% in February and −7.8% for the whole first quarter). July to October reverted to the floor (BTC 8.3 to 9.5%, ETH 6.8 to 10.2%, SOL 5.9 to 8.7%), with the rate at exactly the floor 55 to 75% of hours and almost no hours above the sleeve's threshold.

Three consequences for the carry sleeve:

1. **The 3 bps entry threshold is calibrated to 2024.** In the current regime it selects nothing. The sleeve is not broken; its opportunity set is empty.
2. **An always-on carry at the floor is the realistic version of this trade**, and at 5.5 to 9% gross on perp notional it competes directly with 4 to 5% USDC lending on the same venue, with leg, custody and funding-flip risk that lending does not carry.
3. **Hyperliquid pays shorts more than Binance.** Over 2025 to 2026 Hyperliquid's mean funding exceeded Binance's by 2 to 5.5 percentage points on BTC and ETH (external: Binance public archive vs Hyperliquid API, agent computation; BitMEX Research reports a 7.2 point average premium on BTC 2023 to 2026, secondary). This is the structural edge in the data. Capturing it means short Hyperliquid, long Binance or Bybit, which the board deferred (CCO stop on second venues, Q-03 and Q-13 open) and which requires capital on both venues.

---

## 4. The economics of a $5,000 book (external and backtest, estimates)

**Passive floor on the same cash.**

| Alternative | Yield (dated) | Lockup, risk |
|---|---|---|
| 3-month US T-bill | 4.00% (29 October 2026) | None |
| Aave v3 USDC | 4.1% now, 3.6% one-year mean | Smart contract |
| HyperCore Borrow and Lend USDC | 9.0% spot, 4.9% 30-day mean (1 November 2026) | Utilisation, new product (September 2026) |
| HyperLend, Morpho on HyperEVM | 7 to 8% spot, 4.8 to 5.3% 30-day mean | Smart contract, small pools |
| Ethena sUSDe (institutional funding carry) | 4 to 5% through 2026; 0.13% for a week after 10 October 2025 | 7-day cooldown, protocol risk |
| Hyperliquid HLP (the venue's house book) | about 1 to 3% in 2026 outside two liquidation windfalls; on-chain APR field −0.12% on 1 November 2026 | 4-day lock, socialised losses |
| Market-neutral crypto hedge funds | +6% to +12% in 2025 to 2026 (index data, vendor) | n/a |

**What each Phoenix sleeve plausibly earns on $5,000, net (estimates).**

| Sleeve | Basis | Expected per year | Comment |
|---|---|---|---|
| Carry v1 as configured | 2025 and 2026 venue data | $30 to $60 in a 2025-type year; about $0 in 2026 | In position 9 to 16% of 2025, 0 to 2% of 2026 |
| Always-on floor carry, BTC/ETH | 2026 funding 5.5 to 6% gross on perp notional, pair notional about $2,500, maker fees $1.38 per two-leg round trip | $100 to $200 | Below USDC lending on the same cash |
| Trend at backtest realised vol (about 4 to 5%) | +20.2% over 3.4 years | about $280, standard error about $500 | Backtest, 3 assets, at the optimistic end of the dated literature |
| Trend scaled to 10% vol by leverage | Sharpe 1.0 to 1.25 | $500 to $625, standard error about $500 | Needs 2x leverage on backtest evidence alone |
| Directional meta-labeller | CPCV paths | negative | Net breakeven precision 0.33 to 0.36 vs calibrated p rarely above 0.30 |

**Break-even capital** (annual cost divided by expected net yield).

| Net yield on total capital | Capital to cover $1,116 cash costs | Capital to cover $13,116 all-in |
|---|---|---|
| 2% (thresholded carry, 2025-type year) | $55,800 | $655,800 |
| 4% (always-on carry or T-bill) | $27,900 | $327,900 |
| 6% (trend at realised vol) | $18,600 | $218,600 |
| 10% (Sharpe 1 at 10% vol) | $11,160 | $131,160 |
| 12.5% (Sharpe 1.25 at 10% vol) | $8,930 | $104,900 |

**Fees are not the constraint; scale is.** An all-maker two-leg carry round trip at tier 0 costs 11 bps of one leg ($1.38 on $1,250 per leg) and pays back in about four days at the funding floor. Tier 1 starts at $5 million of 14-day volume, out of reach. HYPE staking for a 10% discount needs about $8,800 of HYPE to save $0.14 per round trip; the free 4% referral discount is worth more. The address rate limit (10,000 requests, then one per USDC traded) is the hidden constraint for a small account and rules out a high-frequency requoter on this capital.

**The retail evidence is one-directional.** On the full on-chain Hyperliquid record (March 2025 to June 2026, 729,540 wallets, peer-reviewed), front-end traders lost $1.11 billion net, 65% of it to fees, 20% to liquidations; the only profitable cohort were makers paying 0.16 bps in fees. Phoenix's maker-only, low-frequency, never-liquidated design removes the three loss channels. That is necessary, not evidence of positive expectancy.

---

## 5. Why the gates and the sizing rule produced these results (external, methodology)

**The per-trade hurdle is the wrong unit for trend.** Skill is measured per unit time (Sharpe, PSR, minimum track record length on return observations). A sleeve that rebalances in 5% increments makes many small adjustments by design; E[net per trade] penalises exactly the feature that makes it cheap. Recommended replacement for trend: pooled three-asset daily net Sharpe at least 1.0 with PSR(0) at least 0.95, DSR at least 0.90 with the trial count taken from the full configuration grid (clustered to an effective N), break-even cost at least 3x realised round-trip cost, Sharpe at least 0.7 at 2x assumed costs, positive in at least two of three crisis windows. The minimum track record for PSR 0.95 is about 990 days at Sharpe 1.0 and 634 days at Sharpe 1.25; the 1,220-day backtest passes at 1.25 with little slack per asset, which is what the 0.88 to 0.89 DSR on BTC and SOL reflects.

**The directional sleeve's problem is precision, not the sizer.** With an 80 bps target and a 30 bps stop the gross breakeven probability is 0.27; net of 6 bps round trip it is 0.33, net of 9 bps (taker both legs) 0.36. Calibrated p rarely exceeds 0.30, so the sleeve is negative expectancy net of fees under any sizer. Two design facts drive the low p: the 80 bps floor is about 4 sigma on 15-minute bars, so most events time out at the vertical barrier; and the label is a barrier touch rather than the sign of the net return. Fixes to test, in order: label on net return sign; symmetric k-sigma barriers for the horizon with a floor of 4 to 6x cost; payoff-aware Kelly (f = p − q/b) with b a design parameter. None of these is expected to turn a 0.30-precision signal into a profitable one at $5,000; the realistic outcome of this experiment is to retire the sleeve with a recorded reason.

**Regime conditioning failing is the expected outcome.** The literature finds regimes help risk management more than forecasting; a 290 to 426 hour dwell with identification lag leaves little of each regime to act on, and vol scaling already captures most of the usable information. The defensible minimum is vol scaling, a single pre-registered crisis veto, and regime as a reporting dimension.

**Partial adjustment losing to a buffer is also expected.** Gârleanu–Pedersen assumes quadratic costs; a small book faces proportional fees plus a tick, for which the optimal policy is a no-trade band. The 5% buffer should be checked for robustness across 3 to 8%, not replaced.

**The kill rule needs replacing before it is relied on.** A 30-day PSR below 0.80 requires an observed annualised Sharpe near 3 to survive; it kills a true Sharpe-1 strategy about 71% of the time and a Sharpe-2 strategy 61% of the time. Statistically coherent alternatives: a drawdown or time-under-water stop from Bailey and López de Prado's formulas (at 10% vol and Sharpe 1.25 the 95% maximum quantile loss is 5.4% of capital, about 158 days to the bottom, 1.7 years under water; a kill at 6 to 7% drawdown or 18 months under water is the 5% false-kill analogue), a CUSUM on standardised daily returns with good = backtest Sharpe and bad = 0, or an anytime-valid e-process that promotes when evidence favours the backtest Sharpe and kills when it favours zero.

---

## 6. Venue facts that change the plan (external, Hyperliquid docs and API, 1 November 2026)

- **The spot leg is feasible on-venue.** UBTC/USDC, UETH/USDC and USOL/USDC have 0.13, 0.44 and 0.31 bps spreads and $68,000 to $637,000 within 10 bps of mid; a $1,250 ALO order is 0.2 to 1.8% of that depth. Prefer UBTC, then UETH; USOL is thin. Dated futures do not exist on Hyperliquid. Portfolio margin (spot offsets a perp short) requires account value above $10,000, so a $5,000 account posts full margin on both legs.
- **Unit, the issuer of UBTC/UETH/USOL, names the United Kingdom as a prohibited jurisdiction** in its terms (revised 16 April 2026). Buying UBTC on the spot book does not touch Unit's interface; minting or redeeming to native BTC would. If the company is UK-based this closes the exit route for the spot leg and is a question for counsel.
- **Hyperliquid's terms** (15 June 2026 version, read via a secondary quotation because the primary page returned 403) restrict US persons and Ontario; the UK is not named. The FCA listed Hyperliquid as an unauthorised firm on 21 May 2026. The FCA's retail crypto-derivatives ban (PS20/10) is firm-side and does not bind a company trading its own capital; own-account dealing is likely excluded from the regulated perimeter under RAO article 15. HMRC treats a company's crypto derivatives under Part 7 CTA 2009 (income), with the spot leg taxed separately. All for counsel.
- **Gateway constraints**: one agent wallet per process (180-day expiry, trade-only to be verified on testnet), master address for all queries, `scheduleCancel` dead-man's switch re-armed inside its window (10 triggers per day), ALO batches separate from IOC, explicit prices on every order, persistent fill storage because only the last 10,000 fills are retrievable, and a self-built websocket reconnect because the SDK's manager does not reconnect or resubscribe.
- **Risk constants**: maintenance margin 1.25% / 2.0% / 2.5% (BTC/ETH/SOL), max leverage 40x / 25x / 20x, hourly funding cap 4%, backstop liquidation below two-thirds of maintenance, ADL ranked by (mark/entry) times leverage. None of BTC, ETH or SOL was at its OI cap on 1 November 2026.

---

## 7. Next steps

### 7.1 Decisions the CEO should record (with recommended values)

| Question | Recommendation | Reasoning |
|---|---|---|
| Q-01 return target and drawdown | Target: T-bill plus 4 to 8 points net (8 to 12% absolute) at 10% volatility or less; hard stop at 10% peak-to-trough; board review at 7%. Anything above 15% net is a regime outcome, not a plan | Market-neutral crypto funds returned 6 to 12% in 2025 to 2026; the passive floor is 4 to 5% |
| Q-09 profitability test | Period: 12 months from the first live tranche. Costs: $1,116 cash plus the founder's hours at a rate the CEO sets. Test: cumulative net live PnL above cash costs and the live record inside the backtest's Monte Carlo 95% drawdown band. Also record that at $5,000 the test cannot be passed under any evidenced yield | Section 4 |
| Q-04 paper record | 90 days, with its purpose recorded as operational (fills, realised basis noise, settlement timing, reconciliation, uptime), not statistical. Plumbing gate: 100% reconciled fills, realised slippage within 1.5x modelled, fee and funding accruals matching venue statements, no unreconciled restart | 9 to 13% power to detect Sharpe 1 |
| Q-02 first tranche | Either no live capital on strategy grounds, or a minimum operational tranche ($500 to $1,000, loss limit 20% of tranche) whose stated purpose is to exercise the gateway, kill path and reconciliation at real size. Do not describe it as a return-seeking tranche | Operational learning is the only thing a small live tranche buys |
| Live kill rule | Replace 30-day PSR below 0.80 with a drawdown and time-under-water stop calibrated from the backtest (Section 5), plus a CUSUM monitor | 71% false-kill rate at Sharpe 1 |
| Q-03 jurisdiction | Give the CCO the five facts (tax residence, trading location, entity, capital ownership, KYB name). The Unit prohibition and the FCA warning are both UK-specific findings | Section 6 |
| Capital strategy | Record explicitly that $5,000 is a research and operations budget; a return-seeking book needs $30,000 to $60,000 at market-neutral yields or $11,000 and a proven Sharpe-1 strategy to cover cash costs | Section 4 |

### 7.2 Strategy work, in priority order

1. **Carry re-based on a hurdle, not a threshold (S).** Enter when the trailing 3-day mean funding, annualised, exceeds the current USDC lending yield plus the amortised round-trip cost; exit when the 7-day mean falls below the lending yield or on the existing funding-flip rules. Report 2024, 2025 and 2026 separately, with the funding-only and basis components split. Expected result: a small positive excess over lending in floor regimes, negative in Q1-2026-type regimes, and a clear statement of what the sleeve is worth.
2. **Cross-venue funding premium study (M, blocked by governance).** Quantify the Hyperliquid-over-Binance and over-Bybit funding spread on BTC/ETH from 2023 with both legs' costs, capital split and transfer latency. This is the largest structural edge in the data. It cannot proceed to paper without the CCO's venue review and a decision on second-venue capital; the research can proceed now and should.
3. **Trend sleeve re-gated and widened (M).** Pooled three-asset gate per Section 5; cost-sensitivity curve at 1x, 1.5x, 2x, 3x fees; buffer robustness across 3 to 8%; breadth extension to the 10 to 15 most liquid Hyperliquid perps, where the dated evidence is strongest. Keep in shadow until the pooled gate and 200 shadow trades are met. Do not lever to 10% vol on backtest evidence.
4. **Directional sleeve: one pre-registered experiment, then retire or keep (M).** Net-return-sign labels, symmetric k-sigma barriers with a 4 to 6x cost floor, payoff-aware Kelly, at least 300 out-of-sample trades per CPCV path. Acceptance: 95% confidence interval of (p − 1/(1 + b_net)) above zero. If it fails, retire the sleeve in the register and stop spending on it.
5. **Kill-rule calibration (S).** Compute the maximum quantile loss and time under water from each sleeve's backtest moments, Monte Carlo the false-kill rate, and register the new thresholds in `data/shadow_thresholds.json`.
6. **Derivatives sleeve (hold).** Start recording the corrected surface components now; evaluate component ICs after six months of history. No allocation before then.

### 7.3 Operations work, in priority order

1. **Merge the implementation branch with `origin/main`** so the board documents govern the code that exists, then re-check D-0002's six conditions against the merged head and record which are met.
2. **Order gateway with reconciliation (L).** Agent wallet per process, boot-time reconciliation from `clearinghouseState`, open orders and `userFills` with a flatten-only mode on any unexplained difference, persistent fill journal, self-built websocket reconnect, `scheduleCancel` re-arm, reduce-only flatten, cancel-all. Testnet first.
3. **Kill-path drills (M).** Ten forced-kill and network-cut drills on testnet, then one at one lot on mainnet, each ending flat with zero unexplained orders. Remote flatten through the existing Telegram bot with an allowlist.
4. **CI (S).** GitHub Actions running `pytest`, `mypy --strict` and a secret scan on every push, with Metal-only tests behind a marker; a replay job that drives a recorded tick file through the execution layer and asserts reconciliation invariants.
5. **Shadow-order slippage study (M).** Two weeks of `l2Book` snapshots and paper orders on $500 to $5,000 clips; acceptance: realised all-in taker cost at or below 6 bps per side.
6. **Monthly venue re-pull (S).** Funding history, HLP vault details, fee schedule and lending yields pulled from the venue monthly as the primary source; the carry hurdle in 7.2.1 reads the lending yield from the same pull.

### 7.4 What not to do

- Do not start the 30-day paper clock as a Sharpe validation. It cannot be one.
- Do not lever the trend sleeve to a 10% vol target to make the break-even arithmetic work.
- Do not stake HYPE for fee discounts at this scale.
- Do not run a high-frequency requoter on a $5,000 account; the address rate limit and the tier-0 maker fee both say no.
- Do not treat the 65 to 72 bps per trip carry figure as current. It is a 2024 artefact until the by-year report says otherwise.

---

## 8. Draft decision for the register (proposed, not written)

> **D-000x: Capital purpose, return target and paper-record purpose**
>
> The $5,000 paper base and any first live tranche are a research and operations budget. The company's return target for a return-seeking book is T-bill plus 4 to 8 points net at 10% volatility or less, with a 10% hard drawdown stop and board review at 7%. The 90-day paper record is an operational qualification (fills, slippage, accruals, reconciliation, uptime) and is recorded as unable to validate edge. The live kill rule is a drawdown and time-under-water stop calibrated from each sleeve's backtest, replacing the 30-day PSR floor. The carry sleeve's entry rule is re-based on the venue lending yield. Cross-venue carry research may proceed; cross-venue paper or live trading awaits the CCO's venue review and a second-venue capital decision. The directional sleeve gets one pre-registered experiment and is retired if it fails.

---

## 9. Verification notes and sources

**Verified in this session.** Hyperliquid `fundingHistory` for BTC, ETH and SOL from 1 January 2024 to 1 November 2026 (script in the session, not committed; numbers in Sections 1 and 3). The research agent's independent pull of the same endpoint agreed exactly. The branch state in Section 2 was read from the code, tests and `docs/REINFORCEMENT_PLAN.md` on `claude/intelligent-sagan-q8qknt`.

**Not verified.** Hyperliquid's terms page (403 from this environment; quoted from a secondary source dated 15 June 2026). Binance archive funding statistics (agent computation from `data.binance.vision`). All passive-yield figures (DefiLlama, Aave, FRED, dated 1 November 2026 unless stated). The CFO's run-rate figures are estimates from the first board memo.

**Primary sources read 1 November 2026**
- Hyperliquid docs: fees, funding, margin tiers, liquidations, auto-deleveraging, portfolio margin, order types, exchange endpoint, nonces and API wallets, rate limits, websocket subscriptions, tick and lot size, referrals, builder codes. https://hyperliquid.gitbook.io/hyperliquid-docs/
- Hyperliquid API: `fundingHistory`, `spotMeta`, `l2Book`, `vaultDetails`, `perpsAtOpenInterestCap`, `allMids`. https://api.hyperliquid.xyz/info
- Unit docs and terms of service (revised 16 April 2026). https://docs.hyperunit.xyz/
- FCA warning on Hyperliquid (21 May 2026). https://www.fca.org.uk/news/warnings/hyperliquid
- FCA PS20/10. HMRC CRYPTO10150, CRYPTO40150, CFM50070. RAO article 15. https://www.legislation.gov.uk/uksi/2001/544/article/15
- FRED DTB3. https://fred.stlouisfed.org/series/DTB3
- Jia, Moallemi, Wang, Zeng, "Who Loses and Why?", AFT 2026. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.AFT.2026.36
- Bailey and López de Prado, "Drawdown-based stop-outs and the triple penance rule", Journal of Risk 2015. https://escholarship.org/content/qt758637dr/qt758637dr.pdf
- Bailey and López de Prado, Deflated Sharpe Ratio. https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Harvey and Liu, "Backtesting" (2015) and "Evaluating Trading Strategies" (2014). https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF
- Lo, "The Statistics of Sharpe Ratios" (2002). https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf
- Thorp, "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market". https://www.gwern.net/doc/statistics/decision/2006-thorp.pdf
- Chague, De-Losso, Giovannetti, "Day Trading for a Living?" https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101
- Harvey et al., "The Impact of Volatility Targeting", JPM 2018. https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf
- Philips, Yashchin, Stein, CUSUM for active managers, JPM 2003. https://researcher.ibm.com/publications/using-statistical-process-control-to-monitor-active-managers
- AdaptiveTrend (arXiv 2602.11708), vol-scaled TSMOM benchmark on Binance perps. https://arxiv.org/html/2602.11708v1
- Man AHL, "In Crypto We Trend". https://www.man.com/insights/in-crypto-we-trend
- Arrakis, Binance leads Hyperliquid by 600 to 800 ms. https://arrakis.finance/blog/crypto-price-discovery
- FIA Guide to Automated Trading Systems (kill switch design). https://www.fia.org/sites/default/files/2019-09/FIA-Guide-to-the-Development-and-Operation-of-Automated-Trading-Systems.pdf
- NautilusTrader live reconciliation docs. https://nautilustrader.io/docs/nightly/concepts/live
- hyperliquid-python-sdk examples. https://github.com/hyperliquid-dex/hyperliquid-python-sdk/tree/master/examples
- Ethena sUSDe yield series (DefiLlama). https://yields.llama.fi/chart/66985a81-9c51-46ca-9977-42b4fe7bc6df
- BitMEX Research Q3 2025 derivatives report. https://www.bitmex.com/blog/2025q3-derivatives-report
- Galaxy VisionTrack indices; Presto crypto investor survey (December 2025). https://visiontrack.galaxy.com/indices/
