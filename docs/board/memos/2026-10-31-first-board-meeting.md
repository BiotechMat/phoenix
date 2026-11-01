# Board memo: first board meeting, 31 October 2026

- **Question as asked:** "have a first board meeting, discuss current platform and planned upgrades and profitability"
- **Seats convened:** all seven (CIO, Head of Research, CRO, CTO, Head of Execution, CFO, CCO). The question is strategic: it covers the platform, the roadmap, the profitability test and the path to first live capital.
- **Draft decision proposed:** the remediation programme below, status Proposed. The CEO has not decided. It was drafted as D-0002; that number was taken on 1 November 2026 by the Binance override, so the draft takes the next free number when decided.

## Question

What is the true state of the Phoenix platform, which planned upgrades come first, and what would it take to show that the system is profitable after every cost?

## Recommendation

All seven seats agree that Phoenix is a paper research harness. It is not yet a trading system, and nothing in the repository is evidence of edge, gross or net. The board recommends:

1. A remediation programme to make the paper record honest before the 30-day paper clock (Q-04) starts.
2. Deferring every expansion: new venues, cross-venue carry, the alpha sleeve's capital, the LLM committee's multiplier and online retraining.
3. Settling the Q-01 return target and drawdown, and the Q-03 jurisdiction facts, as the CEO's next decisions.

The CFO, CRO, Head of Research and CTO carry the recommendation; the CIO, Head of Execution and CCO concur.

## Where the officers agree

- **The paper record is not evidence.** The ledger holds one paper record: net -$0.08, gross -$0.02, fees $0.06, -0.2 bps of the $5,000 base, to 31 October 2026 02:53 UTC (`data/account_ledger.json`). Every seat says this is no sample. The CFO and CTO add that it is not a fill either: it is a per-bar mark-to-market tick (`ui/api_server.py:998-1001, 1174-1216`).
- **The paper loop does not run the system the documents describe.**
  - p_meta is floored at 0.58 and direction defaults to long (`ui/api_server.py:1116, 1120`). The CIO and HoR note that the one trade went in at p_meta 0.21.
  - Asset weights and edges are hard-coded (`:1128-1149`).
  - The allocator and the sleeves are never called (CIO).
  - The committee's inputs are hard-coded and the Qwen daemon is not wired in (HoR).
- **The market data is partly fabricated.**
  - Bar highs, lows and volume, and liquidation volume, are random (`:911-917, 940`) (HoR, CTO, CRO).
  - RNG noise is added to returns and prices (`:219-220, 245-247, 970-973`) (CFO, CTO).
  - Each bar is stamped as 15 minutes but produced every 4 s (`:1436`), so volatility, funding accrual and barriers are off by about 225x (HoR, CTO).
  - The ETH and SOL Parquet files are synthetic (`data/backfill_storage.py:136-171`) (HoR).
  - The feed falls back to simulated data silently (`core/data_ingestion.py:476-511`) (CTO).
- **Labels are fabricated when one class appears** (`ui/api_server.py:639-642`). The CRO and HoR each call this a red line.
- **Carry is booked on `abs(funding_8h)`** (`:995, 1002`), so negative funding shows as income (CRO, HoE, CFO).
- **Fees are a constant**, 5,000 × 0.08 × 1.5 bps = $0.06 per bar, unrelated to notional (`:1004-1005`). "Fee savings" is fee × 2 (`:1020`), a counterfactual and not income (HoE, CFO, CIO).
- **No execution path to any venue exists.** There is no order gateway, no reconciliation on restart, and no kill switch that touches a venue: `emergency_flatten` only changes a label (`ui/api_server.py:1321-1326`) (CRO, CTO, HoE).
- **Only one limit exists in code**, the 2.5x gross cap (`risk/portfolio_allocator.py:39, 328`; `core/contracts.py:679-685`; tested). The drawdown, position, concentration, tranche-loss and liquidation-distance limits do not exist (CRO).
- **No champion artefact exists.** Online refits run every 20 bars on 200 bars with no gate. A grep finds no hashing or serialisation of models (HoR, CRO; answers Q-11 in the negative).
- **Gate statistics are miscounted.**
  - DSR uses 45, the number of CPCV splits, as its trial count (`research/validation/cpcv_gatekeeper.py:296, 405`).
  - Single-strategy PBO is a heuristic (`:364-377, 452-456`).
  - The tournament counts only the current pool (`research/cpcv_gates.py:511`) (HoR).
  - The tournament PSR seeds the live PSR (`research/cpcv_gates.py:693`) (CIO).
- **Defer:** cross-venue carry, Bybit, OKX and Deribit as execution venues, any committee multiplier other than 1.0, the alpha sleeve's capital, and online retraining (all seats that addressed them).
- **Paper figures support qualification for live, never a profitability claim.** A claim needs live PnL reconciled by the CFO (CFO, HoR, HoE).

## Where they disagree

- **Cost of a carry round trip.**
  - CIO: 7.5 bps, from the code's parameters: perp 4.0 + spot 3.5 (`sleeves/sleeve_3_funding_carry.py:237-250`). That gives break-even after 2.5 funding epochs at the 3 bps/8h entry.
  - HoE: the code charges the spot leg at the perp maker rate (`:243-246`). Hyperliquid tier-0 spot, read 31 October 2026, is 4.0 bps maker and 7.0 bps taker, so an all-maker carry round trip is 12.5 bps.
  - The HoE's figure rests on the venue page; the CIO's on the code. The CIO's break-even arithmetic moves with whichever figure holds.
- **Cost hurdle for the directional model.** The paper labeler uses 4 bps (`ui/api_server.py:1074-1076`). The HoE asks the board to plan on about 10 bps net per directional round trip: their judgement, unmeasured, assuming some taker legs and 1-3 bps of adverse selection. The HoR asks that the barrier hurdle move to the HoE's costs.
- **Length of the paper record.** The charter gate is at least 30 days (charter §6). The CIO wants 90 days for carry, so that it covers more than one funding regime. The CFO and CTO want the clock not to start at all until fills, wall-clock time and signed funding are in the ledger.
- **Fallback destination.** The code's `CARRY_ONLY_FALLBACK` covers both cash and carry (`cpcv_gatekeeper.py:805`). The CIO sets the destination as 100% cash until carry has its own reconciled paper record. No seat opposed this.
- **Return target against the cost base.**
  - CIO: maximum drawdown 15% of the tranche, board review at 10%, and a target of at least 15% a year net, live. That implies a net live Sharpe of at least 1.0, and 1.0x gross directional plus 1.0x carry inside the 2.5x cap.
  - CFO arithmetic (estimates): $750 a year at 15% on $5,000. The cash run rate is about $1,116 a year, and the full hurdle about $13,116 a year with the founder's time at an illustrative $50/h × 20 h/month. Covering that at 20% net needs about $66,000 of capital.
  - The CIO accepts that $5,000 is for validation, not profit (R-11).
  - The CRO has given no numbers and asks the CEO for the appetite.

## Stops and red lines raised

- **CRO stop: first live capital.** It lifts when all of the following are done:
  - drawdown, per-asset, per-sleeve and tranche-loss limits are in `risk/` with tests;
  - automatic trips (PSR, drawdown, operator) are wired;
  - the kill switch has been exercised end to end, cancelling and flattening on the venue at minimum size;
  - reconciliation on restart is in place;
  - a watchdog and a remote flatten path exist;
  - 30 days of paper record are reconciled after the carry fix.
- **CRO stop: promotion of the online-retrained model** to paper of record or live. It lifts when the model passes the CPCV gates as a frozen, hashed artefact without fabricated labels, and the CRO has written a validation memo.
- **CRO stop: any limit increase, including above 2.5x, and cross-venue carry.** It lifts with a CRO stress memo at the new limit and a recorded decision.
- **CRO stop: trading while a kill switch is tripped.** It lifts when a restart cannot clear `EMERGENCY_FLATTENED` without a recorded operator reset, and a test proves it.
- **CCO stop: first live capital.** It lifts when the CEO answers Q-03, counsel's view is written into the position, and Hyperliquid's terms are read from the primary source and dated.
- **CCO stop: Bybit, Coinbase, OKX and Deribit as execution venues.** Each lifts when that venue's terms are read, dated and cleared for the CEO's jurisdiction. This includes Bybit failover and any spot leg on a second venue.
- **CCO stop: any public performance claim.** The CCO read `github.com/BiotechMat/phoenix` as Public on 31 October 2026. The repository publishes the method and a committed paper ledger. It also contains promotional wording:
  - "zero directional price risk" (`docs/TROUBLESHOOTING_AND_SAFETY.md:27`);
  - "Institutional" (`pyproject.toml:8`);
  - "institutional-grade" (`TEST_INFRA.md:9`);
  - "ironclad gatekeeper" (`docs/PLAIN_ENGLISH_GLOSSARY.md:77`).

  The stop lifts when the repository is private, or when the ledger and the wording are removed and the CCO has reviewed what remains.
- **CCO: no outside capital or clients.** None exists.
- **Red lines (not stops):**
  - CIO: no record built while the p_meta floor and default-to-long stand; no capital to the alpha sleeve before the gates pass; the CIO will not ask for leverage above 2.5x.
  - HoR: no fabricated labels or bar fields in any training path; no ungated retraining while the model sizes positions; no DSR computed with the split count; no committee multiplier other than 1.0 without an ablation.
  - CFO: the 30-day clock does not start until the ledger records fills, real time and signed funding; no current ledger figure may be cited as performance, paper or otherwise.
  - CTO: no paper record produced on simulated fallback or synthetic timestamps counts towards the gate; no live order before a tested venue gateway with reconciliation on restart; "tests and types green" holds only on CI.
  - HoE: no live order before a venue-side kill-switch exercise; no paper PnL or fee-savings figure cited while the matcher's assumptions are unlabelled.

## Unverified items the answer depends on

- **CTO: tests and types.** Run in a Linux container on 31 October 2026, from a copy of the repository, with `mlx[cpu]==0.32.4` added off the lockfile:
  - `pytest tests/`: 135 collected, 132 passed, 3 failed.
    - The Metal test fails because Metal is unavailable on Linux.
    - The Parquet test fails because it needs git-ignored files (`tests/test_e2e_phoenix.py:650-653`).
    - The tournament test reads a hard-coded `/Users/mbeale/...` path (`tests/test_stress_empirical_challenger2.py:366`).
  - mypy with the project config: clean on 37 files.
  - Explicit `mypy --strict`: 6 errors in `sentiment/committee_models.py:242-269, 747`, because `pyproject.toml:53-63` relaxes two strict checks.
  - `run.sh:10` runs only the 15 architecture tests.
  - Not run: the macOS/Metal results in `TEST_READY.md`. The CRO and HoR did not run the tests.
- **HoE: fees.** Hyperliquid tier-0 perp 1.5/4.5 bps and spot 4.0/7.0 bps, read from the venue page 31 October 2026. Bybit `usdt_perp` 2.0/5.5 bps is unverified (the page returned 503). Slippage of 1.0 bp is a code assumption. The 10 bps directional planning figure is judgement.
- **HoE: paper matcher assumptions.** Fills at bar close, no queue, partials, latency or adverse selection (`ui/api_server.py:1213`). `maker_fill_ratio` is hard-coded at 0.914 (`:835`); spread and breakeven are constants (`:755-757`); the "rebate" of 1.5 bps is actually a fee (`:753`). Every paper figure is inflated by these.
- **CFO: run rate, all estimates.**
  - M1 Max $3,000/36 months = $83/month.
  - Power about $9.5/month; tariff unverified.
  - Data, connectivity, LLM compute and Telegram $0 incremental.
  - Founder's rate and hours to be set by the CEO.
  - The ledger's `timestamp_ns` decodes to 27 October 2024 while its ISO time says 2026, and history is truncated at 200 records (`ui/api_server.py:618`).
- **HoR: data.** Parquet depth unknown, as the files are git-ignored (Q-06). The DI scale is unverified. The impact of the label overwrite is inferred, not measured.
- **CIO: carry.** How often funding exceeds 3 bps/8h, by asset, is unmeasured. Hyperliquid spot depth is unverified (Q-10).
- **CRO:** Hyperliquid and Bybit margin and liquidation rules not read; no stress loss in dollars yet.
- **CCO: jurisdiction and venues.**
  - Jurisdiction and company form unknown.
  - Hyperliquid terms: the pages returned 404 or rendered empty. Bybit terms returned 503. Coinbase, OKX and Deribit not read. All are unverified.
  - Secondary sources suggest Hyperliquid restricts US persons and Ontario, and that Bybit left the UK in 2021 over the derivatives ban. Both are secondary and unverified.
  - Licences: Qwen3-8B-4bit is Apache-2.0 and nautilus-trader 1.231.0 is LGPL-3.0-or-later, both read 31 October 2026. The rest of `uv.lock` is not reviewed. The repository has no LICENSE file.

## Needs between seats

- **CRO →**
  - CTO: a venue client with cancel-all and reduce-only flatten; reconciliation on restart; a watchdog and remote flatten; CI; key custody.
  - HoR: remove the label fabrication; gate refits; a frozen artefact; the committee ablation.
  - HoE: margin and liquidation rules, read and dated.
  - CFO: the carry `abs()` fixed before any paper figure is cited.
  - CEO: Q-01 and Q-02.
- **HoR →**
  - HoE: dated fees, slippage and rebate (Q-16).
  - CTO: a trade-level data source, an artefact store and load path, and a working pytest environment.
  - CRO: independent validation of the DSR and PBO fixes; treat existing `CPCVValidationResult`s as invalid.
  - CFO: no paper figure cited as edge.
- **CIO →**
  - CRO: drawdown and per-sleeve limits in code, and the boundary between allocation parameters and limits.
  - HoR: a funding-regime study; CPCV for alpha; the PSR seeding.
  - HoE: spot fees and depth.
  - CTO: wire the sleeves and allocator into the loop; fix the timestamp.
  - CFO: run rate; rule that fee savings are not income.
- **CTO →**
  - HoE: matcher assumptions under the synthetic cadence.
  - CFO: a cost sheet, including CI minutes and about 20 developer days.
  - HoR: Q-06 and Q-11.
  - CRO: Q-08 and Q-17.
  - CCO: Q-15.
- **HoE →**
  - CTO: approval of the `core/contracts.py` fee changes; execution plumbing in the paper loop.
  - HoR: hurdle at verified costs.
  - CFO: drop fee savings; notional-based fees.
  - CRO: the venue kill-switch test.
  - CIO: the Q-10 decision.
- **CFO →**
  - CTO: real fills, a real clock, signed funding, no RNG in PnL, append-only full history, persisted market data.
  - HoE: one dated fee schedule.
  - CRO: the tranche loss limit.
  - CCO: tax framing.
  - CEO: the founder's rate and hours; the M1 Max cost.
- **CCO →**
  - CTO: make the repository private, stop committing the ledger, run a full licence scan, draft a custody policy.
  - HoE: order-to-trade and cancel-rate TCA; the `evaluate_order_action` cooldown gap (`core/execution/defensive_maker.py:747-757`); an audit log of order-action reasons.
  - CFO: per-fill and funding records.
  - CRO: the counterparty review.
  - CEO: the five Q-03 questions: tax residence and trading location; trading through a company or personally; who owns the capital, machine and code; KYB in whose name; any outside money promised.

## Draft decision (proposed; not written to the register)

> # D-NNNN: Paper-stage remediation before the paper record of record begins
>
> - **Date:** 31 October 2026
> - **Decided by:** CEO
> - **Status:** Proposed
> - **Consulted:** CIO, Head of Research, CRO, CTO, Head of Execution, CFO, CCO
> - **Kind:** Strategy
>
> ## Decision
>
> No paper output produced before this decision counts towards the "first live capital" gate. The 30-day paper record of record (charter §6, Q-04) starts only when the CFO, CTO and HoE confirm in writing that each of the following holds on the exact head, under CI:
>
> 1. The paper loop routes through the sleeves and `SignedHRPCVaRAllocator`, with the p_meta floor, the default-to-long and the hard-coded weights and edges removed (`ui/api_server.py:1116, 1120, 1128-1149`).
> 2. No fabricated labels, bar fields, liquidation volume or RNG in prices or PnL (`:219-220, 245-247, 639-642, 911-917, 940, 970-973`).
> 3. Wall-clock timestamps; trading blocked while the feed is on simulated fallback; each record tagged with its feed source.
> 4. Funding booked signed; fees charged on actual notional at a fee schedule dated against the venue's page; `fee_savings_usd` removed from income.
> 5. Ledger history append-only and complete; a ledger write failure halts trading.
> 6. CI runs `pytest` and `mypy --strict` and a secret scan on every push, with Metal-only tests behind a marker.
>
> Until the Head of Research's ablation (Q-05) is done, the committee multiplier $M_{\text{regime}}$ is fixed at 1.0. Until the derivatives alpha sleeve passes the CPCV gates as a frozen, hashed artefact with a CRO validation memo, it carries zero allocation. The fallback destination is 100% cash. The following are deferred: cross-venue carry; Bybit, OKX and Deribit as execution venues; online retraining in the paper path; real Qwen inference. No first live capital is scheduled.
>
> ## Context
>
> At the first board meeting (memo `docs/board/memos/2026-10-31-first-board-meeting.md`) all seven seats found the paper loop overrides the model, fabricates market data and labels, books carry on absolute funding and charges a constant fee. The ledger holds one paper record, net -$0.08 (paper, 31 October 2026), which is not a fill. CPCV DSR is computed with the split count as the trial count. No venue order path or venue-side kill switch exists. Tests run on Linux by the CTO on 31 October 2026: 132 of 135 passed; explicit `mypy --strict` finds 6 errors.
>
> ## Options considered
>
> 1. Start the 30-day paper clock now on the current loop. The record would be optimistic by construction (CRO, HoE, CFO) and would not count towards the gate (CTO, CFO).
> 2. Remediate first, then start the clock, with expansion deferred. This delays live capital but makes the paper record auditable. (Recommended by all seats.)
> 3. Build the Hyperliquid order gateway in parallel with remediation. Possible, but every seat that ranked work put remediation and CI first. The gateway is estimated at 10-15 developer days (CTO estimate).
>
> ## Reasoning
>
> The paper stage exists to qualify the system for live capital. A record built on the current loop cannot do that. Verified: the code citations above and the CTO's test run. Estimates: effort, the HoE's cost planning figures and the CFO's run rate. This decision overrides no stop. The CRO and CCO stops on first live capital stand.
>
> ## Dissent
>
> None recorded on the decision. The CIO asks for 90 days of paper record for carry, not 30. That is not settled here.
>
> ## Consequences
>
> - CTO: CI (Q-12), feed and ledger fail-closed, wall-clock time.
> - HoE: fee schedule dated and spot fee corrected in `core/contracts.py` (Q-16, with CTO), defensive maker connected to the paper loop, notional fees, trade-through fills with a queue haircut, minimal TCA.
> - HoR: label and bar fabrication removed, DSR on a cumulative trial log, multi-strategy PBO only.
> - CFO: ledger recomputation from fills.
> - CRO: limits in `risk/` per Q-17 once the CEO sets Q-01 and Q-02.
> - The register's R-01 and R-09 are raised.
> - The next board review checks the six conditions above and, if they hold, records the start date of the paper record.
>
> The CEO's separate decisions follow: Q-01 (return target and drawdown; a limit decision), Q-03 (jurisdiction facts to the CCO), the repository's visibility (CCO) and the founder's hourly rate (CFO).
