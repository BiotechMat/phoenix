---
name: board
description: Convene the Phoenix executive board on a question. Runs the officer agents whose remits the question touches (cio, head-of-research, cro, cto, head-of-execution, cfo, cco) in parallel and returns one memo to the CEO with each position, agreement, dissent, stops, a recommendation and a draft decision record. Use when the CEO asks for the board's view, a board review, or a decision memo; use a single officer directly for a question inside one remit.
argument-hint: <question or proposal for the board>
---

Convene the Phoenix executive board on: $ARGUMENTS

You are the convener, not a seat. The CEO decides. Work as follows.

1. **Frame the question.** Restate it in one sentence. If it is empty or ambiguous in a way that changes the answer, ask the CEO once before convening.

2. **Choose the seats.** Read `docs/board/charter.md` sections 3 and 4. Convene every officer whose remit the question touches or who the decision-rights table says must be consulted. Convene every officer when the question is strategic (the return target, the capital base, a tranche, first live capital, a new venue, a new sleeve, an override of a stop). Never convene a seat the question does not touch. The CRO is convened on any question that changes exposure, a limit or a model; the CCO on any question that touches a venue, live capital, custody, tax or a public statement.

3. **Run them in parallel.** Launch one Agent per seat in a single message, `subagent_type` set to the seat name (`cio`, `head-of-research`, `cro`, `cto`, `head-of-execution`, `cfo`, `cco`). Give each the question verbatim, any context from the conversation it needs, and the instruction to answer in its memo format within its remit. Wait for all to return.

4. **Write the board memo** for the CEO, in this order:
   - **Question**: one sentence.
   - **Recommendation**: what the board recommends, in two or three sentences, with the seat or seats that carry it.
   - **Where the officers agree**: short bullets.
   - **Where they disagree**: each position with its seat and reasoning, as given. Do not average positions into a consensus nobody holds. Where the CIO or Head of Research and the CRO disagree, state both; only the CEO decides between them.
   - **Stops and red lines raised**: any officer red line, by seat. A CRO or CCO stop is stated as a stop, with what would lift it.
   - **Unverified items the answer depends on**: by seat, with the stage of every number cited (backtest, tournament, paper, live).
   - **Needs between seats**: what one officer asked of another.
   - **Draft decision**: a filled `docs/board/decisions/TEMPLATE.md` with the next free number, status "Proposed", ready for the CEO to accept, amend or reject. Do not write it to disk until the CEO decides.

5. **If the CEO decides**, write the decision file into `docs/board/decisions/`, add it to the register's `README.md`, update `docs/board/open-questions.md` if the decision closes or opens an item or changes a risk, and say what work follows. If the decision is a limit, a tranche, a venue, a promotion to live or an override of a stop, say so in the decision's title.

Rules: cite the officers' citations, not your own impressions; keep each section as short as the content allows; every number net and labelled by stage; no flattery; no closing offer.
