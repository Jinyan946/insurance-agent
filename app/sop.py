"""SOP controller — the deterministic part of the harness.

Every turn:
  1. extract signals from the utterance (LLM + regex, any phase)
  2. write them to session memory (even if they belong to a later phase)
  3. apply global policies (human request, off-topic, emotion)
  4. run the current phase's step; steps may chain (VERIFY -> RESOLVE -> PROCESS) in one turn
  5. produce a TurnPlan: instructions + the ONLY facts the LLM may use + a safe fallback reply
The LLM only phrases the plan. It never decides phase transitions or sees data a gate hasn't unlocked.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from . import store, tools

VERIFY_ID, RESOLVE_INTENT, PROCESS_CASE, POST_PROCESS = "VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"
ENDED, HUMAN_HANDOFF = "ENDED", "HUMAN_HANDOFF"

MAX_VERIFY_FAILURES = 3
OFF_TOPIC_LIMIT = 3          # 3rd off-topic message -> suggest a human rep
PUSHBACK_LIMIT = 3           # turns of refusal/arguing at a gate before offering a human
FIELD_LABELS = {"name": "full name", "dob": "date of birth", "phone": "phone number on file",
                "email": "email on file", "id_last4": "last four digits of your SSN or national ID"}
NEGATIVE = {"frustrated", "angry", "anxious", "confused"}


@dataclass
class TurnPlan:
    phase: str
    instructions: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    fallback: list[str] = field(default_factory=list)
    authorized: bool = False

    def say(self, instruction: str, fallback: str | None = None):
        self.instructions.append(instruction)
        if fallback:
            self.fallback.append(fallback)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    phase: str = VERIFY_ID
    consent_scenario: str = "default"
    # identity
    id_slots: dict = field(default_factory=dict)
    policy_number: str | None = None
    caller_role: str = "self"
    rep_name: str | None = None
    rep_relationship: str | None = None
    party_id: str | None = None
    verified: bool = False
    authorized: bool = False          # verified AND (self OR rep with consent)
    verify_failures: int = 0
    refused: set = field(default_factory=set)
    consent: tools.ConsentRequest | None = None
    consent_status: str | None = None
    # cross-phase memory
    intent: str | None = None
    case_hints: dict = field(default_factory=dict)
    memory_log: list = field(default_factory=list)
    selected_case: str | None = None
    candidates: list = field(default_factory=list)
    # conversation control
    awaiting: str | None = None
    off_topic_count: int = 0
    pushback: int = 0
    negative_streak: int = 0
    topics: list = field(default_factory=list)
    email: dict = field(default_factory=dict)
    handoff: dict | None = None
    transcript: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def log(self, kind: str, **data):
        self.events.append({"t": datetime.now().strftime("%H:%M:%S"), "phase": self.phase, "kind": kind, **data})

    def goto(self, phase: str, why: str):
        if phase != self.phase:
            self.log("transition", to=phase, why=why)
            self.phase = phase

    def public_state(self) -> dict:
        return {
            "session_id": self.id, "phase": self.phase, "verified": self.verified, "authorized": self.authorized,
            "caller_role": self.caller_role, "rep_name": self.rep_name,
            "identity_fields_collected": sorted(self.id_slots), "verify_failures": self.verify_failures,
            "refused_fields": sorted(self.refused), "consent_status": self.consent_status,
            "consent": self.consent.to_dict() if self.consent else None,
            "remembered": {"intent": self.intent, "case_hints": self.case_hints, "notes": self.memory_log[-6:]},
            "selected_case": self.selected_case, "awaiting": self.awaiting,
            "off_topic_count": self.off_topic_count, "pushback": self.pushback,
            "negative_streak": self.negative_streak, "email": self.email, "handoff": self.handoff,
        }


# =============================================================== controller

class SOPController:
    def __init__(self, s: Session):
        self.s = s

    # ---------------------------------------------------------- memory
    def remember(self, sig: dict):
        s = self.s
        if not s.verified:
            for f in tools.ID_FIELDS:
                if sig.get(f):
                    if s.id_slots.get(f) != sig[f]:
                        s.log("slot", field=f)
                    s.id_slots[f] = sig[f]
                    s.refused.discard(f)
            if sig.get("policy_number"):
                s.policy_number = sig["policy_number"]
            for f in sig.get("refuses_fields", []):
                if f not in s.id_slots:
                    s.refused.add(f)
        if sig.get("caller_role") == "representative" and not s.verified:
            s.caller_role = "representative"
        if sig.get("rep_name"):
            s.rep_name = sig["rep_name"]
        if sig.get("rep_relationship"):
            s.rep_relationship = sig["rep_relationship"]

        new_hints = {k: sig.get(src) for k, src in
                     (("case_id", "case_id"), ("case_type", "case_type"), ("status", "case_status"),
                      ("month", "case_month"), ("year", "case_year")) if sig.get(src) not in (None, "")}
        changed = {k: v for k, v in new_hints.items() if s.case_hints.get(k) != v}
        intent_changed = bool(sig.get("intent")) and sig["intent"] != s.intent
        s.case_hints.update(new_hints)
        if sig.get("intent"):
            s.intent = sig["intent"]
        self.newly_remembered = bool(changed or intent_changed)
        if self.newly_remembered and s.phase == VERIFY_ID:
            note = {"phase": s.phase, "intent": sig.get("intent"), "hints": changed}
            s.memory_log.append(note)
            s.log("remembered_for_later", **note)
        return new_hints

    # ---------------------------------------------------------- main entry
    def step(self, sig: dict, utterance: str) -> TurnPlan:
        s = self.s
        plan = TurnPlan(phase=s.phase)
        if s.phase in (ENDED, HUMAN_HANDOFF):
            plan.say("The conversation is over. Briefly say so and suggest starting a new chat if they need more help.",
                     "This conversation has ended. Please start a new chat if there's anything else I can help with.")
            return plan

        prev_awaiting, s.awaiting = s.awaiting, None
        new_hints = self.remember(sig)

        # emotion bookkeeping
        emo = sig.get("emotion", "neutral")
        s.negative_streak = s.negative_streak + 1 if emo in NEGATIVE else 0
        if emo in NEGATIVE:
            s.log("emotion", emotion=emo)
            plan.say(self._empathy_instruction(emo), self._empathy_fallback(emo))
            plan.facts["caller_emotion"] = emo

        # global: explicit human request, or accepting a pending handoff offer
        if sig.get("wants_human") or (prev_awaiting == "handoff_offer" and sig.get("accepts_offer")):
            return self._handoff(plan, "caller requested a human representative")

        # global: scope guard
        if sig.get("off_topic"):
            s.off_topic_count += 1
            s.log("off_topic", count=s.off_topic_count)
            if prev_awaiting == "handoff_offer" or s.off_topic_count > OFF_TOPIC_LIMIT:
                return self._handoff(plan, "repeated out-of-scope requests")
            if s.off_topic_count >= OFF_TOPIC_LIMIT:
                plan.say("Politely decline the out-of-scope question again. Since the caller keeps asking unrelated "
                         "questions, suggest connecting them with a human representative and ask if they'd like that.",
                         "I'm sorry, I can only help with insurance claims and policy questions. Since I can't help with "
                         "that, would you like me to connect you with a human representative?")
                s.awaiting = "handoff_offer"
                return plan
            plan.say("Politely decline: you can only help with this insurance account and claims. Do not answer any "
                     "part of the unrelated question. Then steer back to the current step below.",
                     "I'm sorry, I can only help with questions about your insurance policy and claims.")
            return self._reprompt(plan, prev_awaiting)

        if prev_awaiting == "handoff_offer" and sig.get("accepts_offer") is False:
            plan.say("Caller declined the transfer. Acknowledge and continue with the current step.", "No problem, let's keep going.")

        handler = {VERIFY_ID: self._verify, RESOLVE_INTENT: self._resolve,
                   PROCESS_CASE: self._process, POST_PROCESS: self._post}[s.phase]
        return handler(plan, sig, utterance, prev_awaiting, new_hints)

    # ---------------------------------------------------------- VERIFY_ID (strict)
    def _verify(self, plan, sig, utterance, prev_awaiting, new_hints):
        s = self.s
        plan.phase = VERIFY_ID

        # consent sub-step for representatives
        if s.verified and s.caller_role == "representative" and not s.authorized:
            return self._consent_step(plan)

        res = tools.verify_identity({**s.id_slots, "policy_number": s.policy_number})
        s.log("tool", name="verify_identity", status=res["status"], n_given=res["n_given"])

        if res["status"] == "verified":
            s.verified, s.party_id = True, res["party_id"]
            s.log("verified", party_id=s.party_id, matched=res["matched"])
            s.pushback = 0
            holder = store.party(s.party_id)
            if s.caller_role == "representative":
                rep = tools.check_representative(s.party_id, s.rep_name, s.rep_relationship)
                s.log("tool", name="check_representative", authorized=rep["authorized"])
                if not rep["authorized"]:
                    plan.say("The policyholder's details matched, but the caller is NOT a representative on file. "
                             "Explain that for privacy you can only discuss the account with the policyholder or an "
                             "authorized representative. Offer: the policyholder can join/call, or transfer to a human "
                             "representative who can help set up authorization. Ask which they prefer. Disclose nothing about claims.",
                             "Thank you. I wasn't able to find you as an authorized representative on this policy, so I "
                             "can't share account details. The policyholder can call us directly, or I can connect you "
                             "with a representative who can help set up authorization. Would you like me to transfer you?")
                    s.awaiting = "handoff_offer"
                    return plan
                plan.facts["representative"] = {"name": rep["rep_name"], "relationship": rep["relationship"]}
                return self._consent_step(plan, first=True)
            s.authorized = True
            plan.facts["verified_caller_name"] = holder["name"]
            plan.say("Tell the caller (by name) that they are verified. Keep it to one short sentence.",
                     f"Thank you, {_call_name(holder)} — you're verified.")
            s.goto(RESOLVE_INTENT, "identity verified")
            return self._resolve(plan, sig, utterance, None, new_hints, chained=True)

        if res["status"] == "mismatch":
            s.verify_failures += 1
            s.id_slots.clear()
            s.log("verify_failed", failures=s.verify_failures)
            if s.verify_failures >= MAX_VERIFY_FAILURES:
                return self._handoff(plan, "identity could not be verified after multiple attempts", security=True)
            plan.say("The details provided do not match our records. Do NOT say which detail was wrong. Ask the caller "
                     f"to re-confirm three of: {self._field_list(tools.ID_FIELDS)}. Mention typos happen.",
                     "I'm sorry, those details don't match what we have on file. Could you re-confirm three of the "
                     f"following: {self._field_list(tools.ID_FIELDS)}?")
            s.awaiting = "id"
            return plan

        # insufficient
        gave_new = any(sig.get(f) for f in tools.ID_FIELDS)
        stalled = not gave_new and (sig.get("refuses_fields") or sig.get("emotion") in NEGATIVE
                                    or sig.get("intent") or sig.get("question"))
        s.pushback = s.pushback + 1 if stalled else 0
        options = [f for f in tools.ID_FIELDS if f not in s.id_slots and f not in s.refused]
        need = res["still_needed"]
        plan.facts["identity_fields_received"] = [FIELD_LABELS[f] for f in s.id_slots]
        plan.facts["identity_fields_still_needed"] = need
        plan.facts["acceptable_identity_options"] = [FIELD_LABELS[f] for f in options]

        if self.newly_remembered:
            plan.say("Briefly acknowledge the reason they're calling (in their own words) and say you'll pull it up right "
                     "after verification. Do not confirm whether such a claim exists or any detail about it.",
                     "I've noted what you're calling about and will pull it up as soon as you're verified.")

        if sig.get("refuses_fields") or len(options) < need or s.pushback:
            plan.say("Explain briefly WHY verification is required: claim records contain protected health and personal "
                     "information, and verifying protects the caller from someone else accessing their account. "
                     "No claim details can be shared until then.",
                     "I understand. Your claim records contain protected personal and health information, so I need "
                     "to confirm I'm speaking with the right person before I can discuss them — it's what keeps anyone "
                     "else from accessing your account.")
        if len(options) < need:
            plan.say("There are not enough remaining identity options. Offer to transfer to a human representative "
                     "who can verify another way, and ask if they'd like that.",
                     "Since we can't complete verification here, would you like me to connect you with a "
                     "representative who can verify you another way?")
            s.awaiting = "handoff_offer"
            return plan
        if s.pushback >= PUSHBACK_LIMIT:
            plan.say("You've explained this several times. Stop persuading: offer, without pressure, either to continue "
                     "verifying OR to be transferred to a human representative. Ask which they prefer.",
                     "I don't want to keep you going in circles. We can finish verification together, or I can "
                     "transfer you to a representative — which would you prefer?")
            s.awaiting = "handoff_offer"
            return plan
        if s.refused:
            plan.say(f"Reassure them they don't have to share {self._field_list(s.refused)}; offer the other options.",
                     f"That's completely fine — you don't have to share your {self._field_list(s.refused)}.")
        got = len(s.id_slots)
        lead = (f"Thanks, I have {got} of the 3 details I need. " if got else
                "To get started, I need to verify your identity. ")
        plan.say(f"Ask for {need} more identity detail(s), letting the caller choose from the acceptable options. "
                 "Ask naturally, not as a form. Do not reveal or confirm whether any detail already given is correct.",
                 lead + f"Could you share {need} more of the following: {self._field_list(options)}?")
        s.awaiting = "id"
        return plan

    def _consent_step(self, plan: TurnPlan, first: bool = False) -> TurnPlan:
        s = self.s
        holder = store.party(s.party_id)
        if first or not s.consent:
            s.consent = tools.ConsentRequest(s.consent_scenario)
            s.log("tool", name="request_consent", scenario=s.consent.scenario)
        status = s.consent.check()
        s.consent_status = status
        s.log("tool", name="check_consent", status=status)
        masked = "***-***-" + holder["phone"][-4:]
        plan.facts["policyholder_first_name"] = holder["name"].split()[0]
        if status == "approved":
            s.authorized = True
            s.goto(RESOLVE_INTENT, "representative consent approved")
            plan.say("The policyholder approved the authorization request. Tell the caller they're all set.",
                     f"{holder['name'].split()[0]} has approved the request, so you're all set.")
            return self._resolve(plan, {}, "", None, {}, chained=True)
        if status == "pending":
            plan.facts["consent_sent_to"] = masked
            plan.say("A consent request was sent to the policyholder's phone on file (give the masked number). Explain "
                     "that because they're calling on someone else's behalf, the policyholder must approve before claim "
                     "details can be shared. Ask the caller to let you know once it's approved." if first else
                     "Consent is still pending. Say you're still waiting for the policyholder to approve, and ask them "
                     "to let you know once they have.",
                     (f"Thanks — the details check out. Because you're calling on {holder['name'].split()[0]}'s behalf, "
                      f"I've sent an authorization request to the phone on file ending in {holder['phone'][-4:]}. "
                      "Once she approves it, let me know and we'll continue.") if first else
                     "I'm still waiting on the approval. Let me know once it's been accepted on the phone.")
            s.awaiting = "consent"
            return plan
        # timeout / denied
        plan.say("The authorization request expired without approval. Explain that claim details can't be shared "
                 "without the policyholder's consent. Offer: have the policyholder call directly, or transfer to a "
                 "human representative. Ask which they prefer.",
                 "Unfortunately the authorization request expired without approval, so I can't share the claim "
                 "details. The policyholder can call us directly, or I can transfer you to a representative. "
                 "Would you like the transfer?")
        s.awaiting = "handoff_offer"
        return plan

    # ---------------------------------------------------------- RESOLVE_INTENT (flexible)
    def _resolve(self, plan, sig, utterance, prev_awaiting, new_hints, chained=False):
        s = self.s
        plan.phase = RESOLVE_INTENT
        plan.authorized = True
        claims = store.claims_for(s.party_id)
        if not claims:
            plan.say("There are no claims on file for this policy. Say so, and ask whether they'd like to be connected "
                     "with a representative (e.g. to file a new claim or for general policy questions).",
                     "I don't see any claims on file for your policy. Would you like me to connect you with a "
                     "representative who can help you file a new claim or answer policy questions?")
            s.awaiting = "handoff_offer"
            return plan

        cands = []
        if prev_awaiting == "case_choice" and new_hints and s.candidates:
            pool = [c for c in claims if c["case_id"] in s.candidates]
            only_new = tools.find_claims(s.party_id, new_hints)["candidates"]
            cands = [c for c in only_new if c in pool] or only_new
        if not cands:
            r = tools.find_claims(s.party_id, s.case_hints)
            cands = r["candidates"]
            if not cands and new_hints:
                cands = tools.find_claims(s.party_id, new_hints)["candidates"]
        s.log("tool", name="find_claims", hints=s.case_hints, n=len(cands))

        if len(claims) == 1:
            cands = claims
        has_hint = bool(s.case_hints)
        if len(cands) == 1 and (has_hint or len(claims) == 1):
            c = cands[0]
            s.selected_case = c["case_id"]
            if not s.intent:
                s.intent = "denial_question" if c["status"] == "denied" else "status_inquiry"
            used_memory = bool(s.memory_log)
            s.log("case_resolved", case_id=c["case_id"], intent=s.intent, from_memory=used_memory)
            plan.facts["resolved_from_earlier_mention"] = used_memory
            whose = "her" if s.caller_role == "representative" else "your"
            plan.say("Confirm which claim you pulled up (type, date filed, case ID)"
                     + (", connecting it to what the caller mentioned earlier so they don't need to repeat it" if used_memory else "")
                     + ".", f"I've pulled up {whose} {c['case_type']} claim {c['case_id']}, filed {c['created_at']}.")
            s.goto(PROCESS_CASE, "case resolved")
            return self._process(plan, sig, utterance, None, {}, opening=True)

        s.candidates = [c["case_id"] for c in (cands or claims)]
        plan.facts["claims_on_file"] = [tools.claim_brief(c) for c in (cands or claims)]
        if has_hint and not cands:
            plan.say("No claim matches what the caller described. Say so plainly, list the claims on file briefly "
                     "(type, date, status), and ask which one they mean.",
                     "I don't see a claim matching that description. Here's what I have on file: "
                     + self._brief_list(claims) + ". Which one are you calling about?")
        elif has_hint:
            plan.say("More than one claim matches. Briefly list the matching claims (type, date, status) and ask which one.",
                     "I found more than one claim that could match: " + self._brief_list(cands) + ". Which one do you mean?")
        else:
            plan.say("Ask what they're calling about today. Mention briefly the claims on file (type, date, status) "
                     "to make it easy to choose.",
                     "How can I help you today? I see these claims on your policy: " + self._brief_list(claims) + ".")
        s.awaiting = "case_choice"
        return plan

    # ---------------------------------------------------------- PROCESS_CASE (flexible, grounded)
    def _process(self, plan, sig, utterance, prev_awaiting, new_hints, opening=False):
        s = self.s
        plan.phase = PROCESS_CASE
        plan.authorized = True

        # switching to a different claim mid-case?
        current = next(c for c in store.claims() if c["case_id"] == s.selected_case)
        if not opening and new_hints and current not in tools.find_claims(s.party_id, new_hints)["candidates"]:
            r = tools.find_claims(s.party_id, new_hints)["candidates"]
            if len(r) == 1:
                s.log("case_switched", frm=s.selected_case, to=r[0]["case_id"])
                s.selected_case = r[0]["case_id"]
                plan.say("The caller switched to another claim. Say you've pulled it up.",
                         f"Sure — switching to your {r[0]['case_type']} claim {r[0]['case_id']}.")
                opening = True
            elif len(r) > 1:
                s.case_hints = dict(new_hints)
                s.goto(RESOLVE_INTENT, "caller referenced a different, ambiguous claim")
                return self._resolve(plan, sig, utterance, None, new_hints)

        if not opening and (sig.get("conversation_done") or (prev_awaiting == "anything_else" and sig.get("accepts_offer") is False)):
            s.goto(POST_PROCESS, "caller has no more questions")
            return self._post(plan, sig, utterance, None, {}, entering=True)

        claim = next(c for c in store.claims() if c["case_id"] == s.selected_case)
        ctx = tools.claim_context(claim, utterance)
        plan.facts["claim"] = ctx
        plan.facts["caller_intent"] = s.intent
        topic = sig.get("question") or s.intent or "claim overview"
        s.topics.append({"case_id": claim["case_id"], "topic": topic})

        plan.say("Answer the caller's question (or, if they just got here, their remembered intent: "
                 f"'{s.intent}') using ONLY facts in FACTS.claim. Interpret messy wording sensibly. "
                 "If a followup_guidance item has matched_current_question=true, base the answer on it. "
                 "If the answer isn't in the facts, say you don't have that information and offer a human claims rep. "
                 "Never predict or promise an outcome (e.g. that an appeal will succeed). Explain money fields in "
                 "plain language using field_meanings if amounts come up. If the appeal deadline has passed, say so "
                 "gently and suggest a human rep to review options.")
        plan.fallback.append(self._claim_fallback(claim, ctx, sig, opening))
        if sig.get("cannot_get_documents") and claim.get("documents_needed"):
            plan.say("The caller can't get the requested documents. Walk through the if_unavailable alternatives; if "
                     "they've exhausted those, apply human_review_rule and offer a human claims representative.")
        plan.say("End by asking if there's anything else you can help with on this claim (one short question).",
                 "Is there anything else I can help you with on this claim?")
        s.awaiting = "anything_else"
        return plan

    # ---------------------------------------------------------- POST_PROCESS (semi-strict)
    def _post(self, plan, sig, utterance, prev_awaiting, new_hints, entering=False):
        s = self.s
        plan.phase = POST_PROCESS
        plan.authorized = True
        holder = store.party(s.party_id)
        masked = tools.mask_email(holder["email"])
        plan.facts["email_on_file_masked"] = masked

        if not entering and prev_awaiting == "email_consent":
            if sig.get("accepts_offer") is True:
                s.email = {"decision": "send", "to_masked": masked}
                s.log("email_decision", decision="send")
                return plan  # server generates + sends summary, then finishes the plan (needs the LLM)
            if sig.get("accepts_offer") is False:
                s.email = {"decision": "skip"}
                s.log("email_decision", decision="skip")
                plan.say("Caller skipped the email. Acknowledge, briefly recap the single most important next step, "
                         "thank them and close warmly.",
                         ("No problem, I won't send an email. " + self._next_step_line()).strip() + " Thanks for calling, take care!")
                s.goto(ENDED, "email skipped")
                return plan
            if sig.get("intent") or new_hints or sig.get("question"):
                s.goto(PROCESS_CASE, "caller had another question")
                return self._process(plan, sig, utterance, None, new_hints)

        plan.say("Offer to email a summary of today's call (what was discussed, claim status/outcome, and next steps) "
                 "to the masked email on file. Make clear it's optional: they can say yes or skip.",
                 f"Before you go, would you like me to email a summary of today's call — what we discussed, your claim "
                 f"status, and next steps — to {masked}? You can also skip it.")
        s.awaiting = "email_consent"
        return plan

    def finish_email(self, plan: TurnPlan, summary: dict) -> TurnPlan:
        s = self.s
        holder = store.party(s.party_id)
        msg = tools.send_email_summary(holder["email"], summary["subject"], summary["body"], s.id)
        s.email.update({"message_id": msg["message_id"], "subject": msg["subject"], "body": msg["body"]})
        s.log("tool", name="send_email_summary", message_id=msg["message_id"])
        plan.facts["email_message_id"] = msg["message_id"]
        plan.say("Confirm the summary email was sent to the masked address, thank them and close warmly.",
                 f"Done — I've sent the summary to {tools.mask_email(holder['email'])}. Thanks for calling, take care!")
        s.goto(ENDED, "email sent")
        return plan

    # ---------------------------------------------------------- handoff & helpers
    def _handoff(self, plan: TurnPlan, reason: str, security: bool = False) -> TurnPlan:
        s = self.s
        s.handoff = {"reason": reason, "verified": s.verified, "party_id": s.party_id if s.authorized else None,
                     "case": s.selected_case, "intent": s.intent, "case_hints": s.case_hints,
                     "emotion_streak": s.negative_streak}
        s.log("handoff", reason=reason)
        s.goto(HUMAN_HANDOFF, reason)
        plan.phase = HUMAN_HANDOFF
        if security:
            plan.say("Verification failed too many times. Calmly say you're unable to verify identity over chat and "
                     "are transferring them to a representative who can help verify another way. No account details.",
                     "I'm sorry, I'm not able to verify your identity here. I'm transferring you to a representative "
                     "who can help verify you another way.")
        else:
            plan.say("Tell the caller you're transferring them to a human representative now, and that you're passing "
                     "along what's been covered so they won't have to start over. Thank them for their patience.",
                     "I'm connecting you with a human representative now and passing along what we've covered so you "
                     "won't have to start over. Thank you for your patience.")
        return plan

    def _reprompt(self, plan, prev_awaiting):
        s = self.s
        s.awaiting = prev_awaiting
        prompts = {
            VERIFY_ID: ("Remind them you still need to verify identity and ask for the remaining details.",
                        "To help with your claim, I still need to verify your identity."),
            RESOLVE_INTENT: ("Ask which claim they're calling about.", "Which claim can I help you with today?"),
            PROCESS_CASE: ("Ask if they have any other questions about their claim.", "Do you have any other questions about your claim?"),
            POST_PROCESS: ("Re-ask whether they'd like the email summary or to skip it.",
                           "Would you like the email summary of today's call, or should I skip it?"),
        }
        i, f = prompts[s.phase]
        plan.say(i, f)
        return plan

    def _next_step_line(self) -> str:
        if not self.s.selected_case:
            return ""
        c = next(c for c in store.claims() if c["case_id"] == self.s.selected_case)
        if c.get("documents_needed"):
            return f"The main next step is submitting the {' and '.join(c['documents_needed'])} for claim {c['case_id']}."
        return ""

    def _claim_fallback(self, c, ctx, sig, opening) -> str:
        q = (sig.get("question") or "").lower()
        matched = [t for t in ctx.get("followup_guidance", []) if t["matched_current_question"]]
        if matched:
            return matched[0]["answer"]
        if sig.get("cannot_get_documents") and ctx.get("document_guidance"):
            return " ".join(d["if_unavailable"] for d in ctx["document_guidance"])
        if c["status"] == "denied" and (opening or s_in(q, "why", "denied", "denial")):
            dl = f" The appeal deadline is {c['appeal_deadline']} — {ctx.get('appeal_deadline_status')}." if c.get("appeal_deadline") else ""
            return (f"It was denied because {c['denial_reason']}. To have it reconsidered, we need "
                    f"{tools._join(c['documents_needed'])}.{dl}")
        if s_in(q, "pay", "paid", "amount", "how much") or sig.get("intent") == "payment_question":
            return (f"The allowed maximum is ${c['allowed_max_amount']}, the expected reimbursement is "
                    f"${c['expected_reimbursement_amount']}, and the net amount paid so far is ${c['net_pay']}.")
        return f"This claim is currently {c['status']}: {c['summary']}."

    @staticmethod
    def _field_list(fields) -> str:
        labels = [FIELD_LABELS[f] for f in fields]
        return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + ", or " + labels[-1]

    @staticmethod
    def _brief_list(cs) -> str:
        return "; ".join(f"{c['case_type']} claim {c['case_id']} filed {c['created_at']} ({c['status']})" for c in cs)

    @staticmethod
    def _empathy_instruction(emo: str) -> str:
        return {"angry": "The caller is angry. Open with a calm, sincere acknowledgment (not defensive, no blame), then continue.",
                "frustrated": "The caller is frustrated. Open by acknowledging the frustration briefly and sincerely, then continue.",
                "anxious": "The caller is anxious. Open with reassurance and a calm tone; be clear about what happens next.",
                "confused": "The caller is confused. Acknowledge it and explain simply, one thing at a time."}[emo]

    @staticmethod
    def _empathy_fallback(emo: str) -> str:
        return {"angry": "I hear you, and I'm sorry this has been so frustrating.",
                "frustrated": "I completely understand the frustration, and I appreciate your patience.",
                "anxious": "I understand this is stressful — I'm here to help you through it.",
                "confused": "No problem, let me explain."}[emo]


def _call_name(p: dict) -> str:
    parts = p["name"].split()
    return parts[0] if len(parts) == 2 else p["name"]


def s_in(text: str, *words) -> bool:
    return any(w in text for w in words)
