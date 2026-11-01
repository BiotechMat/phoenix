# D-0002: Override of a stop: the CCO's stop on models trained on Binance archive data

- **Date:** 1 November 2026
- **Decided by:** CEO
- **Status:** Decided
- **Consulted:** CCO (holds the stop). The other seats reviewed the phase 0-5 branch the same day but were not asked about this decision.
- **Kind:** Override of a stop

## Decision

The CEO deems Phoenix's use of the Binance Vision public archive (data.binance.vision) acceptable under its dataset terms. The CEO's words: "ignore binance data terms for now, CEO deems our useage acceptable".

The CCO's stop of 1 November 2026 is overridden. That stop barred any model trained on Binance archive data from trading as paper of record or live until counsel's view was written. Binance archive data may therefore be used in Phoenix's backfill, research, backtests and models, as on branch `claude/intelligent-sagan-q8qknt` at `edb8b32` (`scripts/backfill.py`, `data/backfill_storage.py`, `docs/REINFORCEMENT_PLAN.md:104, 224, 225`). The CEO set no expiry; "for now" is the CEO's wording.

This decision does not lift any other stop:
- the CRO and CCO stops on first live capital;
- the CCO stop on Binance, Bybit, Coinbase, OKX and Deribit as execution venues;
- the CCO stop on any public performance claim.

It does not decide the geo-restriction question below.

## Context

The board reviewed the phase 0-5 branch on 1 November 2026. The CCO read the Binance Vision Dataset Terms that day (https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md, v1.0, last updated 26 August 2026). The read went through a tool that summarises pages, so the CCO flagged that the wording needs a verbatim check. As read, the terms say:
- the data is licensed CC BY-NC-SA 4.0 (cl. 3.1);
- commercial licensing is excluded (cl. 3.4);
- backtesting is allowed for "purely personal non-production research" (cl. 4.1);
- use for "live proprietary trading execution, automated commercial order generation" is prohibited (cl. 4.2);
- redistribution must credit Binance Vision under the same licence (cl. 4.5).

The branch uses the archive for the training store, carry v2 open interest and the P2.3 features. It commits four Binance-derived CSV fixtures (`tests/fixtures/binance_vision_*.csv`) without attribution, in a repository the CCO read as publicly readable on 1 November 2026.

## Options considered

1. Keep the stop until counsel's view is written. Phase 0-5 research that rests on the archive cannot reach paper of record.
2. Override the stop on the CEO's own reading of the terms. The work proceeds; the legal exposure the CCO identified is accepted by the CEO.

## Reasoning

The CEO's judgement is that the usage is acceptable. Verified: the CCO's summary-tool reading of the terms, dated 1 November 2026. Not verified: the verbatim wording; whether proprietary research counts as "commercial"; whether a trained model is a derivative work. No counsel's view exists. This decision overrides one CCO stop, recorded below.

## Dissent

The CCO's stop, recorded in full as given on 1 November 2026:

> Do not merge this branch, and do not train any model that will trade on the Binance archive data, until counsel has read Binance's dataset terms. [...] The archive fills the training store [...] Its open interest drives the carry v2 validation [...] Its 1m klines feed the P2.3 features [...]. That is a production use by a proprietary trader, which on the face of it is outside cl. 4.1. Whether a model trained on this data is a "derivative work" is a question for counsel. Four Binance-derived CSVs are committed [...]. They carry no credit to Binance and the repository has no LICENSE file. That breaches cl. 4.5. [...] Stop, new today: any model trained on Binance archive data trading paper of record or live, until counsel's view is written down.

## Consequences

- The Binance-data part of the CCO's merge objection is withdrawn by this decision. Every other seat's pre-merge requirements on the phase 0-5 branch stand.
- Not decided here, and still open with the CCO:
  - **Geo-restriction.** Binance and Bybit refuse REST from the development sandbox while their websockets connect (`docs/REINFORCEMENT_PLAN.md:177, 180`). This is a question of venue access and jurisdiction (Q-03, Q-13), not data terms.
  - **Personal identifiers.** More than 100 third-party Hyperliquid wallet addresses are in `tests/fixtures/stream_recording_btc.json`.
  - **Repository visibility.**
- Open-questions register: risk R-13 added (exposure under the Binance Vision terms, owner CCO). Q-15 notes this decision.
- Review: the CCO raises this again before the "first live capital" gate. Any change in the CEO's view is recorded by a decision superseding this one.
