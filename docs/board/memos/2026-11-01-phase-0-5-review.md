# Board memo: review of the phase 0-5 work, 1 November 2026

- **Question as asked:** "What does the board think of the recent phase 0-5 work done"
- **Work reviewed:**
  - Branch `claude/intelligent-sagan-q8qknt`, head `edb8b32`: 19 commits, 82 files, about +16,900 / -950 lines.
  - Built from `274e4cb`, before the board existed (D-0001). Not merged.
  - Plan: `docs/REINFORCEMENT_PLAN.md`.
- **Seats convened:** all seven. The branch touches limits, contracts, fees, execution parameters, sleeves, models, data sources and published documents.
- **Decision taken during the meeting:** D-0002. The CEO overrode the CCO's stop on models trained on Binance archive data. The CCO's objection on Binance data terms is therefore recorded but not carried in this memo's recommendation.
- **Draft decision proposed:** below, status Proposed. The CEO has not decided.

## Question

Is the phase 0-5 work fit to merge? What does it change, and what does it show about edge and profitability?

## Recommendation

No seat recommends merging the branch as it stands, and no seat recommends discarding it.

The board recommends merging it in reviewed pieces after named amendments:
- keep the integrity, data, accounting and measurement work;
- turn the online residual layer off;
- record the new breaker values as a limits decision;
- run CI on the exact head.

No figure on the branch is cited as performance.

The CTO and CRO carry the merge conditions. The Head of Research and CFO carry the evidence conditions. The CIO and Head of Execution concur, with amendments of their own.

## Where the officers agree

- **The branch is a real step forward on integrity.** The CIO, HoR and CFO each verified these 31 October findings as fixed in code:
  - **Model overrides gone:**
    - The p_meta floor and default-to-long are removed. The paper loop runs the real sleeves and HRP (`ui/api_server.py:1816-1823`). Zero signal gives HOLD_CASH (`tests/test_phase0_integrity.py:117`).
    - A single-class label set now skips the retrain instead of fabricating labels (`:860-866`).
  - **Real data:**
    - Random bar fields are gone. Bars come from real Hyperliquid 15m candles, one closed candle per tick.
    - Trading is refused on simulated fallback (`:2107-2135`).
    - Liquidation volume is 0.0 and flagged unless a stream covered the bar.
  - **Accounting fixed:**
    - PnL is booked on closed fills as `ClosedTradeRecord` (`:2357-2377`).
    - Fees are charged on notional (`core/execution/position_manager.py:261-263, 436-442`).
    - Funding is signed, `-side × rate × notional` (`:332-338`). The CRO accepts this as fixing the carry `abs()` booking.
    - `fee_savings` is set to 0.
- **The negative results are honest and correctly acted on.**
  - Trend fails the gate and is off.
  - Regime conditioning and partial adjustment are rejected.
  - The main meta-labeller has no demonstrated edge along CPCV paths.
  - The P2.3 challenger is off.
  - Payoff-aware Kelly runs only in shadow, which is off by default (CIO, HoR, CRO, CFO).
- **No figure on the branch is performance.** Every number is backtest or simulation, and none is reconciled (CFO, CIO, HoR).
- **The online residual layer must be off.**
  - `OnlineResidualRidge` refits at every retrain and overwrites live `p_meta`, which sizes ledger positions (`ui/api_server.py:897, 1382, 2244`).
  - Its only evidence is an N=6 run not written to the registry (HoR, CRO).
- **Retraining still swaps the model with no gate**, and no frozen, hashed champion exists (`api_server.py:980-986`) (HoR, CRO).
- **Ledger faults remain** (CFO, CTO):
  - history is truncated at 200 (`:813-816`);
  - a write failure is only logged (`:826`);
  - totals are running accumulators, not recomputed from history;
  - a failed load silently starts a fresh $5,000 book (`:740-747`);
  - the legacy 31 October tick record (net -$0.08) is loaded as the opening balance.
- **The spot leg is still charged the perp maker rate** (`position_manager.py:119`; `sleeves/sleeve_3_funding_carry.py:241-245`; `research/carry_eval.py:16`) (CIO, HoE, CFO).
- **Still undone:**
  - There is still no venue kill switch, venue client, cancel-all or reconciliation (CRO, CTO).
  - `CARRY_ONLY_FALLBACK` still trades carry (`api_server.py:1772-1776`) (CIO).
- **Too big to review as one unit.** `api_server.py` is now 2,873 lines in one class. Review it per phase (CTO).

## Where they disagree

- **What carry v1's backtest shows.** Backtest, net: 71.8 / 67.7 / 64.8 bps per round trip for BTC / ETH / SOL, since June 2023 (`docs/REINFORCEMENT_PLAN.md:224`).
  - **CIO:** after the roughly 5 bps spot-fee gap, v1 is estimated at about 60-67 bps per round trip, still above the 15 bps gate. That is the CIO's unverified estimate. The honest line-up is carry v1 only, not champion until rerun and validated by the CRO.
  - **HoR:** not accepted. CPCV runs with `zero_fee` on a stream already net of costs (`carry_eval.py:215`). The basis comes from Hyperliquid's smoothed premium index, which the commit itself says overstates the result, giving a Sharpe of about 10. The trial count cannot be checked: the registry is git-ignored, and carry v1 was probably trial 1, so DSR equals PSR.
  - **HoE:** the cost model leaves out adverse selection. On a 25-minute recording, passive fills lost 1.05 to 4.01 bp net per fill after the tier-0 maker fee (simulation, not paper; `REINFORCEMENT_PLAN.md:319`).
  - **CFO:** even at 70 bps, about 19 round trips a year per asset on about $1,600 notional is roughly $650 a year. That is below the $1,116 a year cash run rate, before any founder time. Rough and unverified arithmetic.
- **What to merge.**
  - CIO and HoR: merge the integrity and data work and the negative results after amendments.
  - HoE: accept the measurement tooling (markouts, lead-pull, queue simulator) once amended. Reject the P4.4 "live tier" claim and the adaptive VPIN threshold.
  - CRO and CTO: do not merge as it stands, and list conditions.
  - The difference is over sequencing. The CTO wants the branch split and reviewed per phase. No seat asks for it to be discarded.
- **The loosened test.** `tests/test_e2e_phoenix.py:659` changed `in (18, 19)` to `>= 18`, citing the new provenance columns. The CTO accepts that the product change is cited, but requires the check to name the exact set of columns. CLAUDE.md forbids loosening a test; the CEO decides whether this counts.

## Stops and red lines raised

- **CRO, all four stops stand:**
  - **First live capital:** no venue kill switch, no tranche or per-asset limit, no PSR trip wired, no reconciliation on restart.
  - **Promotion of the online-retrained model:** stands and is **breached in substance**, because the residual layer sizes ledger positions unvalidated. It lifts when the layer is off, or has passed CPCV at N=10 as a frozen, hashed artefact with a CRO validation memo.
  - **Limit increases and cross-venue carry:** stands, not triggered. Moving payoff-aware Kelly out of shadow would trigger it, because with b > 1 it sizes above even-odds Kelly (`core/contracts.py:628`).
  - **Trading while tripped:** stands. A restart clears `EMERGENCY_FLATTENED` and all breaker state (`api_server.py:509`; `risk/circuit_breakers.py:117-127`).
- **CRO, limits set without a decision.** About 20 values in `risk/circuit_breakers.py:42-58` and `risk/cascade_controls.py:36-45`. None is an increase. These are limits, so they need a decision. The CRO also found these defects:
  - The Grossman-Zhou peak decays 1% a day, so the 20% floor re-arms after about 22 days at a flat loss (CRO estimate, untested).
  - Sleeve fractions sum to 133%, and an unlisted sleeve defaults to 100% of equity (`:43, 183`).
  - Two lockdown triggers are hard-coded never to fire (`api_server.py:1591`).
  - Breakers act on entries only.
  - Breaker state is not persisted.
- **CCO stops:**
  - **New today:** Binance, and any other venue that `best_hedge_venue` ranks (`api_server.py:1449-1461`), as an execution venue.
  - **Standing:** first live capital; Bybit, Coinbase, OKX and Deribit as execution venues.
  - **Public performance claims:** now also covers the backtest figures published on the public branch (`REINFORCEMENT_PLAN.md:222, 224, 277`) and the still-committed ledger.
  - The Binance data stop is overridden by D-0002.
- **HoE red lines:**
  - No merge while P4.4 claims the fee store uses the live tier: no code outside the tests passes `fee_table`.
  - No self-moving VPIN threshold (the floor of 0.30) without TCA and CRO sign-off.
  - The fee fixture must carry a read date and be labelled as another account at tier 3.
  - Spot fees on the spot leg.
  - `prefer_taker` fixed before any "never cross" conclusion. It is negative by construction and uses look-ahead (`research/queue_sim.py:28-30`).
  - Every paper figure carries the matcher's assumptions.
- **HoR red lines:**
  - No change to live `p_meta` by a layer that has not passed CPCV at N=10.
  - No model swap without the gatekeeper and a hashed artefact.
  - No performance claim without a committed registry.
- **CFO red lines:**
  - No branch figure cited as performance.
  - The legacy tick record stays out of the record of record.
  - The 30-day clock does not start until history is complete, a write failure halts trading, and PnL recomputes from history.
- **CTO red lines:**
  - Dependencies used but not declared.
  - The data source's remaining gate items (see below).
  - A ledger write failure that is only logged.
- **CIO red lines:**
  - No sleeve enabled, even in shadow, without a written thesis and retirement condition. None exists.
  - `CARRY_ONLY_FALLBACK` must not trade carry.

## Unverified items the answer depends on

- **CTO, tests and types.** Run on Linux, Python 3.13, with `mlx-cpu==0.32.4` added off the lock, on 1 November 2026:
  - pytest: 312 collected, 309 passed, 3 failed.
    - One needs Metal.
    - Two need git-ignored Parquet files.
    - One of those two also has a hard-coded `/Users/mbeale/` path (`tests/test_stress_empirical_challenger2.py:367`).
  - `mypy --strict` on the gated packages: clean on 60 files. `data/` and `scripts/` are also clean. `mypy --strict tests`: 18 errors, not gated.
  - Yesterday's 6 errors in `committee_models.py` did not reproduce. The cause is unresolved; the CTO judges it depends on the environment.
  - `run.sh:10` still runs only the architecture tests. There is no CI.
  - Mac behaviour is unverified. The CRO, HoR, HoE and CFO ran no tests.
- **Backtest figures.** All from `REINFORCEMENT_PLAN.md`, none reproducible from the repository:
  - **Trend:** +20.2% net, Sharpe 1.25, max drawdown 4.3%, June 2023 to October 2026. 3 trials; DSR 0.89 (BTC) and 0.88 (SOL); PBO absent. Fails the gate.
  - **Carry v2:** 24.0 / 20.6 / 26.7 bps net per round trip, 88-106 trips. Trial count, DSR and PBO absent. Not enabled.
  - **Meta-labeller:** median per-trade Sharpe -0.033 (payoff-aware) and -0.098 on 9 paths, last year of BTC. Net or gross not stated.
  - **P2.3 features:** log-loss gains of 0.001-0.007, run at N=6, below the mandated N=10.
  - **Phase 3 replays:** 3-70 trades per window, e.g. cascade +$19.87 against baseline -$6.63. The document itself says these are not edge.
  - The 25-minute recording behind the queue-simulation figures is not in the repository.
- **Data.**
  - 119,901 15m bars per asset, June 2023 to October 2026, claimed and not on disk here (HoR).
  - Binance `create_time` is assumed to mark the end of the window.
  - Mark and index timestamps are unverified for look-ahead (HoR, CTO).
  - Archive downloads are not checked against Binance's checksums (CTO).
  - Training is on Binance USDT-perp bars while live is on Hyperliquid. The argument that z-scores cancel the difference rests on 48 bars (HoR).
- **Fees.**
  - The fixture `tests/fixtures/hyperliquid_user_fees.json` is another account at tier 3, undated and with no recorded source (HoE, CCO).
  - Its base rates match the HoE's read of 31 October 2026: perp 1.5/4.5 bps, spot 4.0/7.0 bps.
  - Bybit is unverified.
- **Venue terms:** Hyperliquid, Deribit, Bybit, OKX and Binance API terms are unverified. The CEO's jurisdiction is unknown (Q-03).
- **Personal data:** more than 100 third-party Hyperliquid wallet addresses are in `tests/fixtures/stream_recording_btc.json`. The CTO's scan found no keys, tokens or seeds anywhere in the diff or fixtures.

## Needs between seats

- **CRO →**
  - HoR: validate the residual layer or confirm it is off; a view on the breaker values as gates.
  - CTO: persist breaker and regime state; a venue cancel-all and reduce-only path.
  - HoE: real reject-count and position-state inputs for the lockdown.
  - CFO: confirm signed funding reconciles.
- **HoR →**
  - CRO: validate the DSR and PBO rewrites.
  - HoE: costs for carry and trend.
  - CTO: where the registry and the data-snapshot hash are versioned.
  - CFO: no profitability language.
- **CIO →**
  - HoR: trial counts, DSR and PBO for carry v1 and v2.
  - HoE: rerun carry with the spot fee and adverse selection.
  - CRO: rule on per-sleeve 2.5x caps with no book-level cap.
  - CFO: per-sleeve attribution.
- **HoE →**
  - CRO: review the adaptive VPIN floor as a shield change.
  - CTO: review the `CANCEL_LEAD_VENUE_PULL` contract change.
  - HoR: label the recording results as "simulation, 25 min, n=1 recording".
  - CFO: use perp 1.5/4.5 bps and spot 4.0/7.0 bps, not the tier-3 rates.
- **CFO →**
  - CTO: append-only history, fail-closed write and load, a clean book, a recompute check.
  - HoE: a dated spot schedule; slippage on fills.
  - HoR: PBO and trial counts.
  - CRO: are the breakers the limits of record?
- **CTO →**
  - HoR: the Kelly change, the registry, CPCV, archive timestamps.
  - CRO: the Kelly change.
  - HoE: the `defensive_maker` changes.
- **CCO →**
  - CTO: repository private; hash the wallet addresses in the stream recording; record where the fee fixture came from; an inventory of data sources.
  - HoE: confirm one resting order per side and the cancel-to-fill rate on the live venue.

## Draft decision (proposed; not written to the register)

> # D-NNNN: Merge conditions for the phase 0-5 work, and the start of the paper record of record
>
> - **Date:** 1 November 2026
> - **Decided by:** CEO
> - **Status:** Proposed
> - **Consulted:** CIO, Head of Research, CRO, CTO, Head of Execution, CFO, CCO
> - **Kind:** Strategy
>
> ## Decision
>
> Branch `claude/intelligent-sagan-q8qknt` (`edb8b32`) is not merged as one change. It is rebased onto `main` and merged in reviewed pieces, per phase, each green under CI. Before the piece that carries `ui/api_server.py` merges, the following hold:
>
> 1. The online residual layer is off by default.
> 2. Payoff-aware Kelly and the P2.3 challenger stay in shadow and off.
> 3. `CARRY_ONLY_FALLBACK` holds 100% cash.
> 4. The adaptive VPIN threshold feeds telemetry only.
> 5. The spot leg is charged the spot schedule, 4.0/7.0 bps, read 31 October 2026.
> 6. The P4.4 live-tier claim is corrected or wired.
> 7. The fee fixture is dated and labelled as another account at tier 3.
> 8. A ledger write or load failure halts trading.
> 9. Trade history is complete and append-only, and PnL recomputes from it under a test.
> 10. Breaker and regime state persist across restarts under a test.
> 11. `websockets` and `pyarrow` are declared and `uv.lock` is regenerated.
> 12. Binance archive downloads are checked against their checksums.
> 13. `TTLCache` has a maximum age, and a silent socket counts as disconnected.
> 14. `run.sh` runs the full suite.
> 15. CI runs `pytest` and `mypy --strict`, including `data/` and `scripts/`, and a secret scan.
> 16. The `/Users/mbeale/` path is removed.
> 17. The wallet addresses in `tests/fixtures/stream_recording_btc.json` are hashed.
> 18. `tests/test_e2e_phoenix.py:659` checks the exact column set.
>
> The trial registry, or a hash of its snapshot, is committed, and failed experiments, including `meta_shadow_paths` and the residual layer, are registered in it.
>
> The breaker and cascade values in `risk/circuit_breakers.py` and `risk/cascade_controls.py` stay paper defaults. The CRO brings them as a separate limits decision with corrections: a bounded drawdown peak; sleeve fractions summing to at most 100%; the strictest limit as the default for an unlisted sleeve; real lockdown inputs.
>
> The 30-day paper record of record (Q-04) starts on the merged head once the CFO, CTO and HoE confirm conditions 1-15 in writing. The legacy record of 31 October is set aside, not carried as opening balance. No figure on the branch is cited as performance.
>
> This replaces the unadopted remediation draft in `docs/board/memos/2026-10-31-first-board-meeting.md`, most of whose conditions the branch meets or this list carries.
>
> ## Context
>
> Board review of 1 November 2026, this memo. The branch fixes the overrides, fabricated data and accounting faults found on 31 October. It adds limits without a decision, an unvalidated layer that sizes positions, and undeclared dependencies. Tests run by the CTO on 1 November 2026: 309 of 312 passed, with the 3 failures down to environment; `mypy --strict` is clean on the gated packages. Every result on the branch is backtest or simulation.
>
> ## Options considered
>
> 1. Merge as is. This breaches the CRO's model-promotion stop in substance, sets limits without a decision, and leaves the ledger failing open.
> 2. Merge in pieces after amendments. This keeps the integrity and measurement work, and the CTO estimates 4-6 developer days. (Recommended.)
> 3. Discard and rebuild. This loses verified fixes; no seat proposed it.
>
> ## Reasoning
>
> Verified in code by the seats: the fixes and defects cited in the memo. Estimates: the effort, and the CIO's and CFO's carry arithmetic. This decision overrides no stop. All CRO and CCO stops stand, apart from the Binance data stop overridden by D-0002.
>
> ## Dissent
>
> None recorded on the decision. The CIO and HoR differ on what carry v1's backtest shows; this decision does not depend on it.
>
> ## Consequences
>
> - CTO: rebase, CI and the engineering items.
> - HoE: fees, fixture, P4.4.
> - HoR: registry, residual off, N=10 for any gate.
> - CRO: the limits decision on breaker values.
> - CIO: a written thesis and retirement condition for carry before it is enabled as the sleeve of record.
> - CFO: a clean book and a recompute test.
> - CCO: repository visibility and wallet addresses.
> - The next board review checks conditions 1-18 on the merged head and records the start date of the paper record.
