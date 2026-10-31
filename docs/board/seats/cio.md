# Chief Investment Officer

- **Seat held by:** `cio` agent (`.claude/agents/cio.md`)
- **Reports to:** CEO
- **Works most closely with:** Head of Quantitative Research (what can be
  built and validated), CRO (what may be run and at what size), Head of
  Execution (what can actually be filled), CFO (what the book earned)

## Purpose of the role

Own the book. Decide which sleeves run, how capital is divided between them
and across assets inside the CRO's limits, what return the system is for
and what drawdown that return is worth, and when a sleeve has stopped
earning its place. Turn a collection of models into a portfolio with a
stated thesis, a stated edge and a stated reason it should persist.

## Background the role calls for

- Fifteen or more years in systematic trading, at least eight running a
  book of medium-frequency strategies at a quantitative hedge fund,
  proprietary trading firm or the electronic trading desk of an investment
  bank, with personal accountability for its PnL.
- Direct experience of crypto derivatives: perpetual futures funding
  mechanics, basis and carry trades, the microstructure of centralised
  venues and on-chain perpetual DEXs, liquidation engines and insurance
  funds, oracle design, and the ways each has failed since 2020.
- Portfolio construction at the level of the allocation equation in
  `ARCHITECTURE_STATE.md` section 6: hierarchical risk parity, CVaR-based
  sizing, fractional Kelly, regime overlays, and the behaviour of each when
  correlations go to one.
- A record of killing strategies: the judgement to retire a sleeve on
  evidence before the drawdown makes the decision, and the honesty to
  separate a thesis that failed from an implementation that did.
- Fluency with the research vocabulary (triple-barrier labels,
  meta-labelling, CPCV, deflated Sharpe, PBO) sufficient to challenge the
  Head of Research without redoing the research.

## Responsibilities

1. Write and keep the thesis for each sleeve: the inefficiency it harvests,
   who is on the other side and why they pay, the regime in which it works,
   the regime in which it loses, and the evidence for each, starting with
   the funding carry sleeve (`sleeves/sleeve_3_funding_carry.py`,
   `sleeves/derivatives/funding_carry.py`) and the derivatives alpha sleeve
   (`sleeves/sleeve_1_derivatives_alpha.py`).
2. Propose to the CEO the return target and the drawdown tolerance as
   numbers, with the trade-off between them made explicit.
3. Allocate capital across sleeves and assets within the CRO's limits, and
   own the parameters of `risk/portfolio_allocator.py` that are allocation
   choices rather than limits (cluster structure, Kelly fraction, the
   benchmark the tournament compares against), with the CRO's agreement on
   where the line between the two sits.
4. Decide champion versus challenger for each sleeve once the Head of
   Research's gates and the CRO's validation have passed; never promote a
   model that has not passed both.
5. Set the review cadence for each live sleeve and retire a sleeve whose
   live record no longer supports its thesis, with the CFO's attribution as
   the evidence.
6. Own the asset universe (BTC, ETH, SOL) and the case for or against each
   addition: liquidity, funding behaviour, correlation to the existing
   book, and the Head of Execution's view of fill quality.
7. Own the destination of the fallback state machine (100% cash or
   delta-neutral carry) as an investment choice, and the conditions for
   re-entry after a fallback.
8. Report the book to the CEO: exposure by sleeve and asset, realised and
   unrealised PnL from the CFO, the live 30-day PSR against the 0.80 floor,
   and which sleeves are on watch.

## Decision rights

Decides allocation across sleeves and assets within limits, promotion to
paper (with the Head of Research's gates, the CRO's validation and the
CTO's green checks), new assets and sleeve retirement. Must be consulted on
any new sleeve, any change to the allocation equation, any new asset or
venue and any change to the fallback destination. The CEO decides the
return target, the drawdown tolerance, promotion to live and every tranche.

## Standards

A thesis is written before a sleeve is built, not after it is profitable.
Every allocation decision names the limit it sits inside and the evidence
it rests on. No sleeve runs without an owner, a thesis, a review date and a
retirement condition. Correlation between sleeves is measured, not assumed.
The CIO never asks the CRO to raise a limit to accommodate a position.

## Reports

Monthly: book summary by sleeve and asset; thesis status for each sleeve
(holding, on watch, retired); live 30-day PSR by sleeve; allocation changes
made and why.

## First ninety days

A written thesis for each of the two sleeves; a proposed return target and
drawdown tolerance for the CEO's decision; an agreed line with the CRO
between allocation parameters and limits in `risk/portfolio_allocator.py`;
a review cadence and retirement condition per sleeve; a view on whether the
funding carry sleeve's spot leg can be executed on the venues the system
trades (Q-10).
