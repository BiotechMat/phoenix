# Phoenix

Project Phoenix is an internal, proprietary algorithmic trading platform
for crypto perpetual futures, built for one Apple M1 Max machine and run on
paper against live market data today (`PROJECT.md`,
`ARCHITECTURE_STATE.md`). It trades the company's own capital and nobody
else's: no outside money, no clients, no public performance claims. Its one
aim is a trading system that is profitable after every cost, and provably
so.

## Read first

- `PROJECT.md`: the architecture overview, the feature inventory by
  milestone, the milestones and the interface contracts between them.
- `ARCHITECTURE_STATE.md`: the memory partition and thread pinning on the
  M1 Max, and the subsystem modules with their parameters, thresholds and
  the governing allocation equation.
- `TEST_INFRA.md` and `TEST_READY.md`: the test suite's design and its last
  self-reported result. Treat the result as a claim about one local run
  until CI exists (Q-12).
- `docs/LAYMANS_ARCHITECTURE_GUIDE.md`, `docs/PLAIN_ENGLISH_GLOSSARY.md`,
  `docs/UI_AND_OPERATIONS_MANUAL.md`, `docs/TROUBLESHOOTING_AND_SAFETY.md`:
  the operator's manuals.
- `docs/board/`: the governance (below).

## Governance

The project has an executive board, modelled on the front-office, risk and
control structure of a quantitative trading firm (D-0001, 31 October 2026).
The CEO (the user) decides; seven officer agents in `.claude/agents/`
advise within fixed remits: `cio` (Chief Investment Officer),
`head-of-research` (Head of Quantitative Research), `cro` (Chief Risk
Officer), `cto` (Chief Technology Officer), `head-of-execution` (Head of
Execution and Trading), `cfo` (Chief Financial Officer) and `cco` (Chief
Compliance Officer and General Counsel). The charter is
`docs/board/charter.md`, job descriptions are in `docs/board/seats/`,
decisions in `docs/board/decisions/`, memos in `docs/board/memos/`, open
questions and risks with owners in `docs/board/open-questions.md`.

- Use `/board <question>` to convene the officers a question touches and
  return one memo. Consult one officer directly for a question inside one
  remit.
- The CRO and the CCO each hold a stop on the first live capital; the CRO
  on any increase to any limit and on any promotion its validation has not
  passed; the CCO on any new venue, any outside capital or client, and any
  public performance claim. The CEO may override a stop; the override is
  recorded as a decision with the dissent in full.
- A limit, a capital tranche, a venue, a promotion to live and an override
  of a stop are always decisions. Record decisions the CEO makes in the
  register with the template; a new decision takes the next free number in
  `docs/board/decisions/README.md` on `origin/main` at the moment you
  commit; never edit a past decision, supersede it.
- Consult the seats before you change their ground: the CRO before any
  limit, kill-switch rule, fallback threshold or anything that raises
  exposure; the Head of Research before any feature, label, model, gate
  threshold or CPCV change, and before any change to the risk committee;
  the Head of Execution before any execution parameter, shield threshold,
  fee constant or change to the paper matcher; the CTO before any new
  venue, data source, engine or dependency, any contract in
  `core/contracts.py`, and anything touching the ledger's persistence; the
  CFO before any recurring cost or any claim of profitability; the CCO
  before any venue, custody change, public statement or anything with
  legal exposure.
- The standing gates in charter section 6 are checked before the event
  they name, against the exact code and data that will run.

## Rules the board holds

These apply to any code or document added here. The seats' job
descriptions say who checks each.

- **Every number is net and labelled by stage.** Any return, Sharpe, PnL
  or hit rate is net of fees, funding, slippage and running costs, states
  its period and sample, and is labelled backtest, tournament, paper or
  live. A backtest states its trial count, DSR and PBO. Gross is labelled
  gross.
- **Limits are code.** A limit lives in `risk/` (or the module the CRO
  names), is enforced at run time and has a test. A limit in prose does not
  exist. Changing a limit is a decision.
- **Paper before live; the kill switch before paper is trusted.** No live
  order before the gates in charter section 6 pass. A kill-switch trip
  must flatten and cancel on the venue, not only in the ledger.
- **No secret, ever.** No private key, seed, API key, token or venue
  credential in the repository, in a log, in the ledger or in a test
  fixture. Live keys carry trade permission only, never withdraw.
- **Research is reproducible and production runs what was validated.**
  A promoted model is a frozen, versioned, hashed artefact trained on a
  named data snapshot and code revision. Purging and embargo are never
  relaxed to pass a gate. Failed experiments are recorded.
- **Costs are verified against the venue.** Fee schedules, funding
  mechanics and margin rules in code carry the date they were read from the
  venue's page.
- **The ledger is the book of record and reconciles.** `data/account_ledger.json`
  is written atomically; its PnL is recomputed from trade history and,
  once live, reconciled to venue statements. No profitability claim
  without the CFO's reconciliation.
- **Tests and types gate every merge.** `pytest` green and `mypy --strict`
  clean on the exact head (`run.sh` runs both before the server starts).
  A failing test is fixed in the product, or in its expectation with the
  product change cited; never deleted, skipped or loosened.
- **Proprietary capital only.** No outside money, no clients, no public
  performance claims, without the CCO's review and a decision.

## Where things live

- `core/`: contracts (`contracts.py`), ingestion (`data_ingestion.py`),
  features (`features/`), labels and the meta-model (`models/`), execution
  (`execution/`), hardware bootstrap.
- `sleeves/`: the funding carry and derivatives alpha sleeves.
- `risk/`: the portfolio allocator and the limits.
- `research/`: the CPCV gates, the triple-barrier meta-model and the
  validation gatekeeper.
- `sentiment/`: the MLX risk committee and its daemon.
- `data/`: the account ledger and Parquet storage (Parquet files are
  git-ignored).
- `ui/`: the control centre API, dashboard and Telegram bot.
- `tests/`: the architecture verification, end-to-end and stress suites.
- `config/`: the M1 Max and CI-mock configurations. `scripts/`: kernel
  tuning. `phoenix` and `run.sh`: the operator's entry points.
