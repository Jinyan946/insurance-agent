# Insurance Claims SOP Agent

A conversational claims-support agent that **follows a fixed business workflow** —
`VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS` — while still talking like a person.

The core idea: **code owns the workflow, the LLM owns the language.**
The model never decides a phase transition, never sees data that a gate hasn't unlocked,
and never answers from anything but the facts the controller hands it.

## Quick start

```bash
cp .env.example .env            # put ANTHROPIC_API_KEY (or OPENAI_API_KEY) in .env
docker compose up --build       # → http://localhost:8000
```

Without Docker:

```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-ant-... DEMO_TODAY=2026-03-05 uvicorn app.main:app --port 8000
```

**API token** can be supplied three ways: env var (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`),
`.env` file, or per-session in the UI's **Settings** dialog (kept in server memory only).
With no key, the app runs in **mock mode** (regex extraction + deterministic templates) so the
whole SOP is still demoable and testable offline.

Tests (offline, deterministic):

```bash
python -m pytest -q        # or: docker compose run --rm agent pytest -q
```

## Test UI

Left: chat as the caller (sample-utterance chips included).
Right: live SOP inspector — gates (verified / consent / authorized), cross-phase memory,
the controller's plan for the current turn, extracted signals, an audit log of every tool call
and transition, and the email outbox.

## Architecture

```
caller msg ─► extract.py ──► sop.py (SOPController) ──► agent.py ──► guard ─► reply
              LLM + regex     deterministic state       LLM phrases     blocks leaks
              signals (any    machine, gates, tools,    the TurnPlan    before
              phase)          builds TurnPlan                           authorization
```

| Layer | File | Responsibility |
|---|---|---|
| Signal extraction | `app/extract.py` | Every turn, pull **all** useful signals (PII, rep info, intent, claim hints, emotion, refusal, off-topic, yes/no) regardless of phase. LLM tool-call with a JSON schema, merged with regex; everything normalized/validated. |
| SOP controller | `app/sop.py` | Deterministic state machine. Writes signals to memory, applies global policies, runs the phase step, chains steps when gates pass, emits a `TurnPlan` = *instructions* + *the only facts allowed* + *safe fallback text*. |
| Tools | `app/tools.py` | Bounded, deterministic: `verify_identity`, `check_representative`, `ConsentRequest`, `find_claims`, `claim_context` (grounded guidance + follow-up topic matching), `send_email_summary`. |
| Responder | `app/agent.py` | LLM turns the plan into natural speech. Phase-scoped system prompt. |
| Output guard | `app/agent.py::guard` | Defense in depth: before authorization, any reply containing a case ID, denial reason, required document, or `$` amount is replaced by the safe fallback. |

### Freedom per phase

| Phase | Control | What the LLM may do |
|---|---|---|
| **VERIFY_ID** | Strict | Converse, clarify, empathize, explain *why* verification matters. Has **no claim data in context** — it structurally cannot leak it. Can't confirm/deny which identity field was wrong. |
| **RESOLVE_INTENT** | Flexible | Interpret messy language; disambiguate among the caller's claims (list provided). |
| **PROCESS_CASE** | Flexible, grounded | Answer follow-ups from `claim_context` only (claim record, field meanings, doc guidance, matched follow-up topics, deadline status). Unknown → say so and offer a human. |
| **POST_PROCESS** | Semi-strict | Offer the email summary; caller chooses send or skip. Summary is generated from transcript + grounded facts and sent via tool. |

### Gates & policies (all in code)

- **Identity**: ≥3 of {full name, DOB, phone, email, SSN/national-ID last 4} must match **one** record with **no contradicting field**. Aliases (`name_aliases`, `email_aliases`, …) count. Policy number is a *locator*, not a factor. Mismatch → generic "doesn't match" (no field revealed), slots cleared; 3 failures → human handoff.
- **Representatives**: if the caller is acting for someone, the policyholder's identity is verified, the caller must be on `representatives.json`, then a **consent request** is sent to the policyholder (simulated via `consent_scenarios.json`: `default` approves on the 2nd check, `timeout` never does). No claim data until approved.
- **Cross-phase memory**: claim hints (type, status, month, year, case ID) and intent are stored whenever they're said. "My denied healthcare claim from January" during verification → after verification, `find_claims` resolves straight to `CL-2048` with no re-asking. Ambiguity (e.g. two January healthcare claims) → ask the caller to choose.
- **Scope guard**: off-topic → polite decline + steer back; 3rd off-topic → offer a human; continued → transfer.
- **Emotion & recovery**: frustration / anger / anxiety / confusion → empathy first, then the workflow. Refused fields → reassure + offer remaining fields. After 3 turns of pushback at a gate → stop persuading, offer verification *or* a human. Explicit "talk to a person" → immediate handoff.

## Demo scripts

1. **Spec test case** — click *Demo test case*. Verifies in one turn, uses the remembered hint, lands on CL-2048 with the grounded denial reason. Then ask *"How long does it take after I submit?"*, *"I can't get the pathology report"*, *"no that's all"*, *"yes"*.
2. **Angry caller** — *"My name is Margaret Chen."* → *Angry caller* chip → *Refuse SSN* chip → *"DOB March 15 1985, email margaret@email.com"*.
3. **Representative** — *Representative* chip → *"Her DOB is 1985-03-15, phone 650-521-2836, SSN last four 4472"* → *"she approved it"*. Switch Settings → consent `timeout` to see the expiry path.
4. **Scope** — *Off-topic* chip three times.

## Notes / assumptions

- `DEMO_TODAY` pins the clock (fixture deadlines are Mar/Apr 2026). Unset → real date, and the agent will gently say a deadline has passed.
- Claim `documents_needed` names ("pathology report") are mapped to guideline keys ("original pathology report") by token containment.
- Sessions are in memory (single process). Swap `SESSIONS` for Redis for multi-instance hosting.
- Emails are written to `OUTBOX_PATH` (default `/tmp/sop_agent_outbox.jsonl`) and shown in the UI instead of being delivered.
