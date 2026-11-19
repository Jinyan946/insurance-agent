# Insurance Claims Support Agent

This is my take on the SOP agent project: a chatbot for insurance claims support that has to follow a fixed
workflow (verify the caller → figure out what they need → handle the claim → wrap up), but still talk like
a normal person instead of a form. The agent is called **Sage** and works for a made-up company,
**MadeUp Insurance**.

**Live demo:** https://insurance-agent-fyt5.onrender.com/
(It's on Render's free tier, so the first load can take about a minute to wake up.)

**GitHub:** https://github.com/Jinyan946/insurance-agent

---

## How to run it

### Option 1: just open the live demo
Click the link above. Nothing to set up.

### Option 2: run it on your own computer

If you got this as a zip, just unzip it and `cd insurance-agent` (skip the `git clone` line).

```bash
git clone https://github.com/Jinyan946/insurance-agent.git
cd insurance-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your API key in .env
uvicorn app.main:app --port 8000 --env-file .env
```

Then open http://localhost:8000.

Or with Docker:

```bash
cp .env.example .env               # put your API key in .env
docker compose up --build
```

### Which API key?
I tested it with **Groq's free tier** (no credit card needed), so that's what `.env.example` is set up for:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_groq_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b
DEMO_TODAY=2026-03-05
```

It also works with an Anthropic key (`ANTHROPIC_API_KEY=...`) or a regular OpenAI key, and any other
OpenAI-compatible endpoint. You can also paste a key into the **Settings** button in the web page
instead of using `.env`.

If you don't set any key, it still runs in a "mock" mode (simple pattern matching + template replies),
so you can click through the whole workflow offline. The replies just sound more robotic.

**About `DEMO_TODAY`:** the sample claims have appeal deadlines in March/April 2026, which have already
passed in real life. So I pin the agent's "today" to 2026-03-05 so the demo shows "13 days left" instead
of "deadline passed 8 months ago". Delete that line to use the real date.

---

## What to try

The buttons under the chat box send sample messages for you. Click **New conversation** before each one.

1. **The test case from the assignment.** Click *Demo test case*. Margaret gives her name, DOB, SSN last 4,
   and says she's calling about her denied healthcare claim from January, all in one message. The agent
   verifies her and goes straight to claim CL-2048 without asking "how can I help you?" again. Then try
   *Follow-up*, *Can't get docs*, *Wrap up*, and answer "yes" to get the email summary.
2. **Angry caller.** Type `My name is Margaret Chen.`, then click *Angry caller*. It should calm her down,
   explain why it needs more info, and still not say anything about the claim. Click *Refuse SSN* and
   it'll offer other options (DOB, phone, email).
3. **Someone calling for their mom.** Click *Representative*, then send
   `Her DOB is 1985-03-15, phone 650-521-2836, SSN last four 4472`, then `she approved it`.
   In Settings you can switch the consent scenario to `timeout` to see what happens when mom never approves.
4. **Off-topic questions.** Click *Off-topic* three times. It says no politely, and on the third try it
   offers to connect you to a human.

The panel on the right shows what's happening behind the scenes: whether the caller is verified, what
the agent remembered, what it was told to do this turn, and a log of every step. I added this mostly
so it's easy to check the agent isn't cheating.

---

## How it works (the short version)

The main idea: **the code decides what happens, the AI only decides how to say it.**

I didn't want to just write a long prompt saying "please don't share claim info before verification",
because LLMs don't always listen. So every message goes through three steps:

1. **Pull out the useful info** (`app/extract.py`). The AI reads the message and pulls out things like
   name, DOB, which claim they mean, whether they sound angry, whether the question is off-topic.
   It does this no matter which step we're on, which is how the agent remembers stuff early
   (like "my denied claim from January" during ID verification) and uses it later.
2. **Decide what to do** (`app/sop.py`). Plain Python code checks the rules: Are there 3 matching ID
   fields? Is this person allowed to see the claim? Which claim matches what they said? Then it writes
   a short "plan" for this turn plus the only facts the AI is allowed to use.
3. **Say it nicely** (`app/agent.py`). The AI turns the plan into a normal-sounding reply.

A few things I did to keep it safe:

- During ID verification, the AI **doesn't even get the claim data**, so it can't leak it even if it wanted to.
- There's also a final check on every reply. If it mentions a claim ID, a denial reason, or a dollar
  amount before the caller is verified, the reply gets swapped out for a safe one.
- In the claim step, the AI only answers from the claim record and the document guidelines. If the answer
  isn't there, it says so and offers a human.

---

## How each requirement is covered

**The 4 phases**
- **VERIFY_ID (strict):** needs 3 of: full name, DOB, phone, email, SSN/ID last 4, all matching the
  same person. Nicknames/aliases in the data count too. The policy number helps find the record but
  doesn't count as one of the 3. If the info is wrong, it doesn't say which part was wrong (so nobody
  can guess their way in). After 3 failed tries it hands off to a human.
- **RESOLVE_INTENT (flexible):** uses whatever the caller already said to find the right claim. If
  it's unclear (Margaret has two January healthcare claims, one from 2025 and one from 2026), it asks which one.
- **PROCESS_CASE (flexible but grounded):** answers questions about the denial, missing documents,
  deadlines, how to submit, how long review takes, what to do if you can't get a document, etc. You can
  also switch to another claim mid-conversation ("what about my auto claim?").
- **POST_PROCESS:** offers to email a summary (what we talked about, claim status, next steps). The
  caller can say yes or skip. In the demo the email isn't really sent. It shows up in the "Email outbox"
  panel instead.

**Remembering things across phases.** Anything the caller mentions gets saved, even if it's for a later
step. You can see it in the "Cross-phase memory" panel.

**Staying on topic.** Off-topic questions get a polite no. After 3 of them, it suggests a human.

**Bonus: emotional support.**
- Detects frustration, anger, anxiety, confusion, and refusals.
- Acknowledges the feeling first, then continues.
- Explains *why* verification (and the policyholder's consent, for representatives) is needed.
- Never skips a step no matter how upset the caller is.
- Offers other ID options if they refuse one, and stops pushing after 3 rounds of back-and-forth.
  At that point it just asks if they'd rather talk to a human. Asking for a human directly also works anytime.

---

## Using the sample data

All 6 provided files are used:

| File | What it's for |
|---|---|
| `policyholders.json` | ID verification (including name/phone/email aliases) |
| `claims.json` | finding claims, denial reasons, missing docs, deadlines, amounts |
| `claim_schema.json` | explaining what the money fields mean if someone asks |
| `representatives.json` | checking if someone calling for another person is allowed to |
| `consent_scenarios.json` | faking the "policyholder approves on their phone" step (approve vs. timeout) |
| `required_document_guideline.json` | all the answers about documents: how to prepare them, alternatives, submission, review time |

---

## Tests

```bash
python -m pytest -q
```

23 tests that run offline (no API key needed). They cover the assignment's test case, no leaking before
verification, angry callers, refusing a field, wrong info, aliases, representative + consent (approved
and timeout), picking between similar claims, follow-up questions, off-topic handling, and the email
send/skip.

---

## Project layout

```
app/
  main.py        web server (FastAPI)
  agent.py       runs one turn: extract → decide → reply → safety check
  sop.py         the workflow rules (the important part)
  extract.py     pulls info out of each message
  tools.py       looks things up in the data (verify ID, find claims, guidelines, email)
  llm.py         talks to the AI model (Anthropic / OpenAI / Groq / mock)
  static/        the chat page
  data/          the provided sample data
tests/           the tests
```

## Stuff I'd improve with more time

- Conversations are kept in memory, so they reset if the server restarts. A real version would use a database.
- There's no login, and everyone using the demo shares my API key.
- The email is fake (it's just logged).
- It's text-only. Hooking it up to voice would be a fun next step.