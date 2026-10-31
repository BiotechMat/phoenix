# D-0001: Phoenix has an executive board modelled on a quantitative trading firm

- **Date:** 31 October 2026
- **Decided by:** CEO
- **Status:** Decided
- **Consulted:** None (the seats did not exist before this decision)
- **Kind:** Governance

## Decision

Project Phoenix has an executive board. The CEO decides; seven officer
agents advise within fixed remits: Chief Investment Officer (`cio`), Head
of Quantitative Research (`head-of-research`), Chief Risk Officer (`cro`),
Chief Technology Officer (`cto`), Head of Execution and Trading
(`head-of-execution`), Chief Financial Officer (`cfo`) and Chief
Compliance Officer and General Counsel (`cco`). The charter is
`docs/board/charter.md`; job descriptions are in `docs/board/seats/`;
agents are in `.claude/agents/`; `/board <question>` convenes the seats a
question touches. Decisions are recorded in this register and never
edited; a limit, a tranche, a venue, a promotion to live and an override of
a stop are always decisions. The CRO and the CCO each hold a stop on the
first live capital; the CRO on any limit increase; the CCO on any new
venue. All seats are active from this date.

## Context

The CEO instructed on 31 October 2026 that Phoenix should have a board
working the same way as the Drybench executive board (that repository's
`docs/board/charter.md`, adopted 5 October 2026), with seats drawn from the
structure of a highly profitable quantitative trading firm or the markets
division of an investment bank, and with the aim of a profitable internal
algorithmic trading platform.

State of the repository on the same date, verified: six milestones are
marked done in `PROJECT.md`; the system runs paper trading on live data
against a $5,000 base (`data/account_ledger.json`, account
`hyperliquid_paper_01`, one trade recorded, net PnL −$0.08); the only
limit in code is the 2.5x gross leverage cap
(`risk/portfolio_allocator.py`); the promotion gates and fallback
thresholds are constants in `research/validation/cpcv_gatekeeper.py`;
there is no continuous integration (no `.github/` directory); the test
results in `TEST_READY.md` are self-reported; no live key exists in the
repository and no custody policy is written.

## Options considered

1. Copy the Drybench seats (CTO, CSO, CFO, GC, CPO, CMO, CDO). Consequence:
   seats for product, marketing and design that an internal trading system
   does not need, and no seat for the book, for risk or for execution.
2. A front-office, risk and control structure from a trading firm, cut to
   the seats a one-person company needs: CIO, Head of Research, CRO, CTO,
   Head of Execution, CFO, CCO. Chosen.
3. The same structure with the CCO paused until go-live, as Drybench paused
   its General Counsel (Drybench D-0010). Not chosen: the move from paper
   to live is itself the legal event, and the CCO's position on venues and
   jurisdiction has to exist before it, not after.

## Reasoning

Recorded from the CEO's instruction. The seats follow the three lines a
trading firm keeps apart: the front office that takes risk (CIO, Head of
Execution), the research that proposes what to trade (Head of Research),
and the independent control functions that limit, count and permit it
(CRO, CFO, CCO), with the CTO owning the machine all of them run on. The
independence of the CRO from the CIO, and of validation from research, is
written into the charter (section 1) because the system's single operator
is also its developer, so the separation has to live in the process rather
than in the organisation chart. Seats considered and not created are
listed in charter section 2.

## Dissent

None recorded. No officer existed to be consulted.

## Consequences

- `CLAUDE.md` carries the governance section and points here.
- `docs/board/open-questions.md` opens with the questions and risks each
  seat owns, drawn from the repository's state on 31 October 2026.
- Each seat's first-ninety-days list in `docs/board/seats/` is its initial
  work.
- The standing gates in charter section 6 apply from this date; the first
  to bite is "any merge" (tests and `mypy --strict` green, no secret, no
  limit change without a decision), which the CTO is to make a CI check.
- No limit, tranche, venue or promotion is changed by this decision; each
  will be its own decision.
