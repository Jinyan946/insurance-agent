"""Turn understanding: pull every useful signal out of a caller utterance, regardless
of the current phase. Regex handles well-formed PII; the LLM handles messy language.
Output is validated/normalized so nothing downstream trusts raw model text."""
from __future__ import annotations

import re

from . import tools
from .llm import LLM

INTENTS = ["status_inquiry", "denial_question", "document_submission", "next_steps",
           "payment_question", "general_claim_question"]
EMOTIONS = ["neutral", "frustrated", "angry", "anxious", "confused"]

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": ["string", "null"], "description": "Policyholder full name if stated (NOT the representative's name)."},
        "dob": {"type": ["string", "null"], "description": "Policyholder date of birth, normalized YYYY-MM-DD."},
        "phone": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "id_last4": {"type": ["string", "null"], "description": "Last 4 of SSN or national ID."},
        "policy_number": {"type": ["string", "null"], "description": "e.g. POL-9921"},
        "caller_role": {"type": ["string", "null"], "enum": ["self", "representative", None],
                        "description": "representative if calling on behalf of someone else."},
        "rep_name": {"type": ["string", "null"], "description": "Caller's own name when they are a representative."},
        "rep_relationship": {"type": ["string", "null"], "description": "e.g. son, daughter, spouse"},
        "intent": {"type": ["string", "null"], "enum": INTENTS + [None]},
        "case_id": {"type": ["string", "null"], "description": "e.g. CL-2048"},
        "case_type": {"type": ["string", "null"], "enum": ["healthcare", "dental", "auto", None]},
        "case_status": {"type": ["string", "null"], "enum": ["denied", "open", "closed", None]},
        "case_month": {"type": ["integer", "null"], "description": "Month (1-12) the claim was filed / the service happened, if mentioned. Not from the DOB."},
        "case_year": {"type": ["integer", "null"]},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "refuses_fields": {"type": "array", "items": {"type": "string", "enum": list(tools.ID_FIELDS)},
                           "description": "Identity fields the caller explicitly refuses to give."},
        "off_topic": {"type": "boolean", "description": "True ONLY if the message is entirely unrelated to this insurance call (e.g. 'what is RL?', trivia, coding help)."},
        "wants_human": {"type": "boolean", "description": "Caller explicitly asks for a human / live agent / supervisor."},
        "accepts_offer": {"type": ["boolean", "null"], "description": "Answer to the agent's pending yes/no question, if any."},
        "conversation_done": {"type": "boolean", "description": "Caller indicates they have no more questions."},
        "cannot_get_documents": {"type": "boolean", "description": "Caller says they can't obtain requested documents or substitutes."},
        "question": {"type": ["string", "null"], "description": "Short restatement of what the caller is asking right now, if anything."},
    },
    "required": [],
}

SYSTEM = """You extract structured signals from one caller message in an insurance claims support call.
Extract EVERYTHING present, even if it belongs to a later step of the call (e.g. claim hints during identity checks).
Use the agent's previous message to interpret short answers (e.g. '4472' after being asked for SSN last four;
'yes' after a yes/no question -> accepts_offer). Never guess values that were not said.
Dates of birth must be normalized to YYYY-MM-DD. Claim month/year refer to the claim, never the DOB.
off_topic is only for requests unrelated to insurance/claims/this call; complaints, emotions and identity
questions ('why do you need my SSN?') are NOT off topic."""



_RX = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "dob_iso": re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
    "dob_us": re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b"),
    "dob_text": re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b", re.I),
    "last4": re.compile(r"(?:ssn|social|national id|id|last (?:four|4)(?: digits)?)[^0-9]{0,30}(\d{4})\b", re.I),
    "policy": re.compile(r"\bPOL-?\d{3,}\b", re.I),
    "case": re.compile(r"\bCL-?\d{3,}\b", re.I),
    "name": re.compile(r"(?i:my name is|name's|this is|i am|i'm)\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+){1,2})"),
}
_REP_MARKERS = re.compile(r"on behalf of|calling for my|for my (?:mother|mom|father|dad|wife|husband|son|daughter)|i'?m (?:her|his) (?:son|daughter|husband|wife)|my (?:mother|mom|father|dad)'s claim", re.I)
_HOLDER_NAME = re.compile(r"(?:on behalf of|my (?:mother|mom|father|dad|wife|husband)|(?:her|his) name is),?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})")
_MONTH_RX = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
_YEAR_RX = re.compile(r"\b(20[12]\d)\b")

_DOMAIN = re.compile(r"claim|insur|policy|polic|denied|deni|coverage|cover|document|report|appeal|pay|reimburs|deductib|"
                     r"verify|verif|identity|ssn|social|birth|dob|phone|email|human|agent|representative|status|"
                     r"submit|upload|portal|deadline|doctor|hospital|dental|auto|car|accident|bill|note|"
                     r"pathology|diagnos|consent|approve|help|why|refund", re.I)
_SHORT_ACK = re.compile(r"^\s*(yes|yeah|yep|sure|ok|okay|no|nope|nah|thanks?|thank you|please|go ahead|skip|send it|done|bye)\b", re.I)
_YES = re.compile(r"^\s*(yes|yeah|yep|sure|ok|okay|please|go ahead|send|do it|that works|sounds good|y)\b", re.I)
_NO = re.compile(r"^\s*(no|nope|nah|don'?t|skip|not now|no thanks|i'?m good|that'?s all|that'?s it|n)\b", re.I)


def heuristic(text: str, awaiting: str | None) -> dict:
    t = text.strip()
    low = t.lower()
    out: dict = {"refuses_fields": []}

    if m := _RX["email"].search(t):
        out["email"] = m.group(0)
    dob_span = None
    for k in ("dob_iso", "dob_us", "dob_text"):
        if m := _RX[k].search(t):
            out["dob"], dob_span = m.group(0), m.span()
            break
    scrub = t if not dob_span else t[:dob_span[0]] + " " * (dob_span[1] - dob_span[0]) + t[dob_span[1]:]
    if m := _RX["phone"].search(scrub):
        out["phone"] = m.group(0)
    if m := _RX["last4"].search(scrub):
        out["id_last4"] = m.group(1)
    elif awaiting == "id" and re.fullmatch(r"\D*(\d{4})\D*", t):
        out["id_last4"] = re.sub(r"\D", "", t)
    if m := _RX["policy"].search(t):
        out["policy_number"] = m.group(0).upper().replace("POL", "POL-").replace("--", "-")
    if m := _RX["case"].search(t):
        out["case_id"] = m.group(0).upper().replace("CL", "CL-").replace("--", "-")

    if _REP_MARKERS.search(t):
        out["caller_role"] = "representative"
        if m := _RX["name"].search(t):
            out["rep_name"] = m.group(1)
        if m := re.search(r"(?:i'?m|i am) (?:her|his|the) (son|daughter|husband|wife|spouse|caregiver)", low):
            out["rep_relationship"] = m.group(1)
        elif re.search(r"my (mother|mom|father|dad)", low):
            out["rep_relationship"] = "child"
        if m := _HOLDER_NAME.search(t):
            out["name"] = m.group(1)
    else:
        if "policyholder" in low or "my own" in low:
            out["caller_role"] = "self"
        if m := _RX["name"].search(t):
            out["name"] = m.group(1)
        elif awaiting == "id" and (m := re.match(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})(?:\.?$|,)", t)):
            out["name"] = m.group(1)

    sl = scrub.lower()
    claimy = bool(re.search(r"claim|case", sl))
    for ct, words in {"healthcare": ("healthcare", "health care", "health claim", "medical"),
                      "dental": ("dental", "dentist"), "auto": ("auto", "car ", "vehicle", "car accident")}.items():
        if any(w in sl for w in words):
            out["case_type"] = ct
            break
    if claimy:
        for w, st in tools.STATUS_SYNONYMS.items():
            if re.search(rf"\b{w}\b", sl):
                out["case_status"] = st
                break
        if m := _MONTH_RX.search(scrub):
            out["case_month"] = tools._MONTHS[m.group(1).lower()]
        if m := _YEAR_RX.search(scrub):
            out["case_year"] = int(m.group(1))

    if re.search(r"\bden(y|ied|ial)\b|rejected|why (was|did)", low):
        out["intent"] = "denial_question"
    elif re.search(r"submit|upload|send (the|my) (doc|report|note|file)|fax|mail it|format|pdf|scan", low):
        out["intent"] = "document_submission"
    elif re.search(r"status|where is|update on|how long", low):
        out["intent"] = "status_inquiry"
    elif re.search(r"next step|what (do|should) i do|what now", low):
        out["intent"] = "next_steps"
    elif re.search(r"how much|paid|payment|reimburse|net pay", low):
        out["intent"] = "payment_question"

    if re.search(r"ridiculous|absurd|unacceptable|furious|outrageous|stupid|wtf|insane|!!", low):
        out["emotion"] = "angry"
    elif re.search(r"already told you|again\?|frustrat|annoy|waste of|come on|seriously", low):
        out["emotion"] = "frustrated"
    elif re.search(r"worried|scared|anxious|afraid|stress|panic|urgent|can'?t afford", low):
        out["emotion"] = "anxious"
    elif re.search(r"confus|don'?t understand|what do you mean|why do you need", low):
        out["emotion"] = "confused"
    else:
        out["emotion"] = "neutral"

    if re.search(r"(won'?t|not going to|don'?t want to|rather not|refuse to|not comfortable|not)\s+(giving|sharing|providing|give|share|provide|telling|tell)|no way", low):
        for f, kw in {"id_last4": ("ssn", "social", "id"), "dob": ("birth", "dob"),
                      "phone": ("phone", "number"), "email": ("email",)}.items():
            if any(k in low for k in kw):
                out["refuses_fields"].append(f)
        if not out["refuses_fields"] and awaiting == "id":
            out["refuses_fields"] = ["id_last4"]

    out["wants_human"] = bool(re.search(
        r"(speak|talk|transfer|connect)\w*\s+(me\s+)?(to|with)\s+(a|an|the|some)?\s*(human|person|real|live|agent|supervisor|manager|someone|representative)|real person|live agent", low))
    if awaiting in ("yes_no", "email_consent", "handoff_offer", "anything_else"):
        if _YES.match(low):
            out["accepts_offer"] = True
        elif _NO.match(low):
            out["accepts_offer"] = False
    out["conversation_done"] = bool(re.search(
        r"that'?s all|that is all|nothing else|no more questions|that'?s it|i'?m done|all set|bye|goodbye", low)) or \
        (awaiting == "anything_else" and out.get("accepts_offer") is False)
    out["cannot_get_documents"] = bool(re.search(r"(can'?t|cannot|unable to|couldn'?t) (get|find|obtain)|don'?t have (it|them|the|any)|lost (it|the)", low))

    has_pii = any(out.get(k) for k in ("email", "phone", "dob", "id_last4", "name", "policy_number"))
    out["off_topic"] = bool(
        awaiting != "consent" and len(low.split()) >= 3 and not has_pii and not _DOMAIN.search(low) and not _SHORT_ACK.match(low) and out["emotion"] == "neutral"
        and (t.endswith("?") or re.match(r"^(what|who|how|tell|write|explain|can you|could you|give me|define)\b", low)))
    out["question"] = t if t.endswith("?") else None
    return out



def _clean(sig: dict) -> dict:
    out = dict(sig)
    for f in tools.ID_FIELDS:
        if out.get(f):
            out[f] = tools.NORMALIZERS[f](str(out[f]))
    if out.get("policy_number"):
        m = re.search(r"POL-?(\d+)", str(out["policy_number"]).upper())
        out["policy_number"] = f"POL-{m.group(1)}" if m else None
    if out.get("case_id"):
        m = re.search(r"CL-?(\d+)", str(out["case_id"]).upper())
        out["case_id"] = f"CL-{m.group(1)}" if m else None
    if out.get("case_month") and not (1 <= int(out["case_month"]) <= 12):
        out["case_month"] = None
    if out.get("intent") not in INTENTS:
        out["intent"] = None
    if out.get("emotion") not in EMOTIONS:
        out["emotion"] = "neutral"
    out["refuses_fields"] = [f for f in out.get("refuses_fields") or [] if f in tools.ID_FIELDS]
    return out


def extract(llm: LLM, text: str, phase: str, awaiting: str | None, last_agent: str | None) -> dict:
    base = heuristic(text, awaiting)
    llm_sig = llm.structured(
        SYSTEM,
        f"Current step: {phase}\nAgent's previous message: {last_agent or '(none)'}\n"
        f"Pending question type: {awaiting or 'none'}\n\nCaller message:\n{text}",
        SCHEMA, name="extract_signals")
    if llm_sig:
        merged = dict(base)
        for k, v in llm_sig.items():
            if v not in (None, "", []):
                merged[k] = v
            elif k in ("off_topic", "wants_human", "conversation_done", "cannot_get_documents"):
                merged[k] = bool(v) or False
        cleaned = _clean(merged)
        for f in tools.ID_FIELDS:
            if not cleaned.get(f) and base.get(f):
                cleaned[f] = tools.NORMALIZERS[f](base[f])
        cleaned["_source"] = "llm+regex"
        return cleaned
    out = _clean(base)
    out["_source"] = "regex"
    return out
