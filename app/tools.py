"""Deterministic, bounded tools. The LLM never touches the data directly —
the SOP controller calls these and passes only what the current phase allows."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, datetime
from pathlib import Path

from . import store

ID_FIELDS = ("name", "dob", "phone", "email", "id_last4")
MIN_ID_MATCHES = 3

# ---------------------------------------------------------------- normalizers

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def norm_name(v: str | None) -> str | None:
    if not v:
        return None
    s = re.sub(r"[^a-z\s]", " ", v.lower())
    s = " ".join(s.split())
    return s or None


def norm_phone(v: str | None) -> str | None:
    if not v:
        return None
    d = re.sub(r"\D", "", v)
    return d[-10:] if len(d) >= 10 else None


def norm_email(v: str | None) -> str | None:
    if not v:
        return None
    v = v.strip().lower().rstrip(".")
    return v if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", v) else None


def norm_last4(v: str | None) -> str | None:
    if not v:
        return None
    d = re.sub(r"\D", "", str(v))
    return d[-4:] if len(d) >= 4 else None


def norm_dob(v: str | None) -> str | None:
    """Accepts ISO, M/D/YYYY, 'March 15, 1985', '15 March 1985'. Returns YYYY-MM-DD."""
    if not v:
        return None
    s = v.strip().lower().replace(",", " ")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    toks = s.split()
    mon = next((_MONTHS[t] for t in toks if t in _MONTHS), None)
    nums = [int(t) for t in toks if t.isdigit()]
    if mon and len(nums) == 2:
        day, year = (nums[0], nums[1]) if nums[1] > 31 else (nums[1], nums[0])
        try:
            return date(year, mon, day).isoformat()
        except ValueError:
            return None
    return None


NORMALIZERS = {"name": norm_name, "dob": norm_dob, "phone": norm_phone,
               "email": norm_email, "id_last4": norm_last4}


def _record_values(p: dict, field: str) -> set[str]:
    if field == "name":
        vals = [p["name"], *p.get("name_aliases", [])]
    elif field == "phone":
        vals = [p["phone"], *p.get("phone_aliases", [])]
    elif field == "email":
        vals = [p["email"], *p.get("email_aliases", [])]
    elif field == "dob":
        vals = [p["dob"]]
    else:
        vals = [p["id_last4"]]
    return {NORMALIZERS[field](x) for x in vals if x}


def _field_matches(p: dict, field: str, value: str) -> bool:
    if field == "name":  # order-insensitive ("Chen Margaret")
        want = set(value.split())
        return any(set(v.split()) == want for v in _record_values(p, field))
    return value in _record_values(p, field)


# ---------------------------------------------------------------- identity

def verify_identity(provided: dict) -> dict:
    """provided: already-normalized values for any of ID_FIELDS (+ optional policy_number).

    Never tells the caller *which* field failed — only the controller sees details.
    """
    given = {k: v for k, v in provided.items() if k in ID_FIELDS and v}
    candidates = store.policyholders()
    pol = (provided.get("policy_number") or "").upper()
    if pol:
        located = [p for p in candidates if p["policy_number"] == pol]
        candidates = located or candidates  # policy # is a locator, not a factor

    best = None
    for p in candidates:
        matched = [f for f, v in given.items() if _field_matches(p, f, v)]
        mismatched = [f for f in given if f not in matched]
        score = (len(matched), -len(mismatched))
        if best is None or score > best[0]:
            best = (score, p, matched, mismatched)

    n_given = len(given)
    _, p, matched, mismatched = best
    if len(matched) >= MIN_ID_MATCHES and not mismatched:
        return {"status": "verified", "party_id": p["party_id"], "matched": matched,
                "n_given": n_given}
    if n_given >= MIN_ID_MATCHES:
        return {"status": "mismatch", "party_id": None, "matched": [], "n_given": n_given,
                "_debug_best_party": p["party_id"] if matched else None,
                "_debug_mismatched": mismatched}
    return {"status": "insufficient", "party_id": None, "n_given": n_given,
            "still_needed": MIN_ID_MATCHES - n_given}


def check_representative(party_id: str, rep_name: str | None, relationship: str | None) -> dict:
    rn = norm_name(rep_name)
    for r in store.representatives():
        if r["buyer_party_id"] != party_id:
            continue
        if rn and set(norm_name(r["rep_name"]).split()) == set(rn.split()):
            return {"authorized": True, "rep_name": r["rep_name"], "relationship": r["relationship"]}
    return {"authorized": False}


# ---------------------------------------------------------------- consent

class ConsentRequest:
    """Simulates an out-of-band consent prompt sent to the policyholder (SMS/app push).
    Each check() consumes the next status from the configured scenario."""

    def __init__(self, scenario: str = "default"):
        self.scenario = scenario if scenario in store.consent_scenarios() else "default"
        self.seq = store.consent_scenarios()[self.scenario]["status_sequence"]
        self.i = 0

    def check(self) -> str:
        if self.i >= len(self.seq):
            return "timeout" if self.seq[-1] == "pending" else self.seq[-1]
        s = self.seq[self.i]
        self.i += 1
        if s == "pending" and self.i >= len(self.seq) and self.seq[-1] == "pending":
            return "timeout"
        return s

    def to_dict(self):
        return {"scenario": self.scenario, "checks": self.i, "sequence": self.seq}


# ---------------------------------------------------------------- claims

STATUS_SYNONYMS = {
    "denied": "denied", "rejected": "denied", "declined": "denied",
    "open": "open", "pending": "open", "in progress": "open", "processing": "open",
    "closed": "closed", "settled": "closed", "paid": "closed", "completed": "closed",
}
CASE_TYPES = {"healthcare", "dental", "auto"}


def find_claims(party_id: str, hints: dict) -> dict:
    """Filter a party's claims by remembered hints. Returns candidates + which hints were used."""
    all_claims = store.claims_for(party_id)
    if hints.get("case_id"):
        hit = [c for c in all_claims if c["case_id"] == hints["case_id"].upper()]
        if hit:
            return {"candidates": hit, "used": ["case_id"]}

    used, cands = [], all_claims
    filters = {
        "case_type": lambda c, v: c["case_type"] == v,
        "status": lambda c, v: c["status"] == STATUS_SYNONYMS.get(v, v),
        "month": lambda c, v: int(c["created_at"][5:7]) == int(v),
        "year": lambda c, v: int(c["created_at"][:4]) == int(v),
    }
    for k, fn in filters.items():
        v = hints.get(k)
        if v in (None, ""):
            continue
        narrowed = [c for c in cands if fn(c, v)]
        used.append(k)
        cands = narrowed
    return {"candidates": cands, "used": used}


def claim_brief(c: dict) -> dict:
    """Non-sensitive one-liner for disambiguation lists (only after verification)."""
    return {"case_id": c["case_id"], "case_type": c["case_type"],
            "created_at": c["created_at"], "status": c["status"]}


def _doc_key(doc: str, table: dict) -> str | None:
    d = set(doc.lower().split())
    for k in table:
        if d <= set(k.lower().split()):
            return k
    return None


def _join(items: list[str]) -> str:
    items = [f"the {x}" for x in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def claim_context(c: dict, utterance: str = "") -> dict:
    """Everything the agent may say about one claim, pre-grounded."""
    g = store.doc_guidelines()
    ctx = {k: v for k, v in c.items() if k != "party_id"}
    ctx["field_meanings"] = {k: v["description"] for k, v in store.claim_schema()["field_descriptions"].items()}

    if c.get("appeal_deadline"):
        dl = date.fromisoformat(c["appeal_deadline"])
        days = (dl - store.today()).days
        ctx["appeal_deadline_status"] = (
            f"{days} days remaining" if days >= 0 else f"passed {-days} days ago (as of {store.today()})")

    docs = c.get("documents_needed") or []
    if docs:
        dg = []
        for d in docs:
            k1 = _doc_key(d, g["document_guidance"])
            k2 = _doc_key(d, g["document_alternative_guidance"])
            dg.append({
                "document": d,
                "how_to_prepare": g["document_guidance"][k1]["en"] if k1 else None,
                "if_unavailable": (g["document_alternative_guidance"][k2] if k2
                                   else g["document_alternative_guidance"]["default"])["en"],
            })
        ctx["document_guidance"] = dg
        ctx["general_submission_guidance"] = g["default_guidance"]["en"]
        ct = g["case_type_guidance"].get(c["case_type"])
        if ct:
            ctx["case_type_guidance"] = ct["en"]

        fill = {"case_id": c["case_id"], "documents": _join(docs),
                "average_processing_time_after_submission":
                    g["claim_followup_settings"]["average_processing_time_after_submission"]["en"]}
        u = utterance.lower()
        topics = []
        for t in g["claim_followup_guidance"]:
            if t.get("requires_documents") and not docs:
                continue
            matched = any(p in u for p in t.get("match_any", []))
            topics.append({"topic": t["topic"], "matched_current_question": matched,
                           "answer": t["en"].format(**fill)})
        ctx["followup_guidance"] = topics
        ctx["followup_fallback"] = g["claim_followup_fallback"]["en"]
        ctx["human_review_rule"] = g["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]["en"]
    return ctx


# ---------------------------------------------------------------- email / handoff

OUTBOX = Path(os.getenv("OUTBOX_PATH", "/tmp/sop_agent_outbox.jsonl"))


def send_email_summary(to: str, subject: str, body: str, session_id: str) -> dict:
    msg = {"message_id": f"MSG-{uuid.uuid4().hex[:8].upper()}", "to": to, "subject": subject,
           "body": body, "session_id": session_id, "sent_at": datetime.now().isoformat(timespec="seconds")}
    OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")
    return msg


def mask_email(e: str) -> str:
    user, dom = e.split("@")
    return f"{user[0]}{'*' * max(1, len(user) - 1)}@{dom}"
