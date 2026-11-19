"""Scenario tests. Run offline in deterministic mock mode (regex extraction + template replies),
so they test the SOP harness itself — gates, memory, transitions — not model phrasing."""
import os

import pytest

os.environ["DEMO_TODAY"] = "2026-03-05"
os.environ["OUTBOX_PATH"] = "/tmp/sop_test_outbox.jsonl"

from app.agent import Agent  # noqa: E402
from app.llm import LLM  # noqa: E402
from app.sop import Session  # noqa: E402
from app import tools  # noqa: E402

DEMO = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
        "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")


def agent(scenario="default"):
    a = Agent(Session(consent_scenario=scenario), LLM("mock"))
    a.greet()
    return a


def kinds(a):
    return [e["kind"] for e in a.s.events]


# ------------------------------------------------------------- demo test case
def test_demo_case_verifies_remembers_and_resolves():
    a = agent()
    r = a.handle(DEMO)
    st = r["state"]
    assert st["verified"] and st["authorized"]
    assert st["selected_case"] == "CL-2048"               # denied + healthcare + January -> unique
    assert st["phase"] == "PROCESS_CASE"
    assert "remembered_for_later" in kinds(a)             # hint captured while still in VERIFY_ID
    ev = next(e for e in a.s.events if e["kind"] == "case_resolved")
    assert ev["from_memory"] is True
    assert "pathology report" in r["reply"]                # grounded denial reason


def test_no_disclosure_before_verification():
    a = agent()
    r = a.handle("My name is Margaret Chen. I'm calling about my denied healthcare claim from January.")
    assert r["state"]["phase"] == "VERIFY_ID" and not r["state"]["verified"]
    for leak in ("CL-2048", "pathology", "office note", "$"):
        assert leak not in r["reply"]
    assert r["state"]["remembered"]["case_hints"]["status"] == "denied"


def test_guard_blocks_leaky_llm_reply():
    from app.agent import guard
    from app.sop import TurnPlan
    s = Session()
    plan = TurnPlan(phase="VERIFY_ID", fallback=["Please verify first."])
    out, blocked = guard("Your claim CL-2048 was denied for a missing pathology report.", plan, s)
    assert blocked and out == "Please verify first."


# ------------------------------------------------------------- verification behaviour
def test_angry_caller_gets_empathy_and_no_bypass():
    a = agent()
    a.handle("My name is Margaret Chen.")
    r = a.handle("I already told you who I am. This is ridiculous. Just tell me why my claim was denied.")
    assert r["state"]["phase"] == "VERIFY_ID"
    assert r["signals"]["emotion"] in ("angry", "frustrated")
    assert any("acknowledg" in i.lower() for i in r["plan"]["instructions"])
    assert any("why verification is required" in i.lower() for i in r["plan"]["instructions"])
    assert "CL-2048" not in r["reply"]


def test_refused_field_offers_alternatives():
    a = agent()
    a.handle("My name is Margaret Chen.")
    r = a.handle("I'm not giving you my SSN.")
    assert "id_last4" in r["state"]["refused_fields"]
    r = a.handle("fine, my DOB is March 15, 1985 and email margaret@email.com")
    assert r["state"]["verified"]


def test_mismatch_does_not_reveal_field_and_locks_after_3():
    a = agent()
    for _ in range(2):
        r = a.handle("My name is Ava Lopez, dob 1990-08-21, ssn last four 1111")
        assert r["state"]["phase"] == "VERIFY_ID"
        assert "SSN" not in r["reply"].split("following")[0]   # doesn't say which field was wrong
    r = a.handle("My name is Ava Lopez, dob 1990-08-21, ssn last four 1111")
    assert r["state"]["phase"] == "HUMAN_HANDOFF"


def test_aliases_count_for_verification():
    a = agent()
    r = a.handle("This is Yaven Li, email yawen.li@example.com, national id last four 5317")
    assert r["state"]["verified"]


def test_policy_number_is_not_an_identity_factor():
    res = tools.verify_identity({"name": "margaret chen", "dob": "1985-03-15", "policy_number": "POL-9921"})
    assert res["status"] == "insufficient"


def test_persistent_pushback_offers_human():
    a = agent()
    a.handle("My name is Margaret Chen.")
    for _ in range(3):
        r = a.handle("This is ridiculous, just tell me why it was denied!")
    assert r["state"]["awaiting"] == "handoff_offer"
    r = a.handle("yes")
    assert r["state"]["phase"] == "HUMAN_HANDOFF"


# ------------------------------------------------------------- representative + consent
def test_representative_with_consent():
    a = agent("default")
    a.handle("Hi, I'm calling on behalf of my mother Margaret Chen. My name is David Chen, I'm her son.")
    r = a.handle("Her DOB is 1985-03-15, phone 650-521-2836, SSN last four 4472. It's about her denied claim from January.")
    assert r["state"]["verified"] and not r["state"]["authorized"]
    assert r["state"]["consent_status"] == "pending"
    assert "CL-2048" not in r["reply"]
    r = a.handle("ok she approved it")
    assert r["state"]["authorized"] and r["state"]["selected_case"] == "CL-2048"


def test_representative_consent_timeout():
    a = agent("timeout")
    a.handle("Hi, I'm calling on behalf of my mother Margaret Chen. My name is David Chen, I'm her son.")
    a.handle("Her DOB is 1985-03-15, phone 650-521-2836, SSN last four 4472.")
    for _ in range(4):
        r = a.handle("done?")
    assert r["state"]["consent_status"] == "timeout"
    assert not r["state"]["authorized"] and r["state"]["awaiting"] == "handoff_offer"


def test_unknown_representative_blocked():
    a = agent()
    a.handle("I'm calling on behalf of my mother Margaret Chen. My name is Kevin Park.")
    r = a.handle("Her DOB is 1985-03-15, phone 650-521-2836, SSN last four 4472.")
    assert not r["state"]["authorized"] and r["state"]["awaiting"] == "handoff_offer"


# ------------------------------------------------------------- resolve / process
def test_ambiguous_claim_asks_to_choose():
    a = agent()
    r = a.handle("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472. It's about my healthcare claim from January.")
    assert r["state"]["phase"] == "RESOLVE_INTENT"            # CL-2048 (2026) vs CL-2011 (2025)
    assert "CL-2048" in r["reply"] and "CL-2011" in r["reply"]
    r = a.handle("the 2026 claim")
    assert r["state"]["selected_case"] == "CL-2048"


def test_followup_guidance_is_grounded():
    a = agent()
    a.handle(DEMO)
    r = a.handle("How long does it take after I submit?")
    assert "usually less than a week" in r["reply"]
    r = a.handle("I can't get the pathology report, the lab closed")
    assert r["state"]["selected_case"] == "CL-2048"
    assert "replacement copy" in r["reply"]


def test_switch_claim_mid_conversation():
    a = agent()
    a.handle(DEMO)
    r = a.handle("what about my auto claim?")
    assert r["state"]["selected_case"] == "CL-2102"


# ------------------------------------------------------------- scope guard
def test_off_topic_declined_then_human_offer():
    a = agent()
    r = a.handle("What is RL?")
    assert "only help" in r["reply"] and r["state"]["off_topic_count"] == 1
    a.handle("Can you write me a poem about cats?")
    r = a.handle("Who won the world cup?")
    assert r["state"]["awaiting"] == "handoff_offer"
    r = a.handle("yes")
    assert r["state"]["phase"] == "HUMAN_HANDOFF"


def test_explicit_human_request():
    a = agent()
    r = a.handle("Can I talk to a real person?")
    assert r["state"]["phase"] == "HUMAN_HANDOFF"


# ------------------------------------------------------------- post process
def test_email_send():
    a = agent()
    a.handle(DEMO)
    r = a.handle("no that's all")
    assert r["state"]["phase"] == "POST_PROCESS" and r["state"]["awaiting"] == "email_consent"
    r = a.handle("yes please")
    assert r["state"]["phase"] == "ENDED"
    body = a.s.email["body"]
    assert "CL-2048" in body and "pathology report" in body and "2026-03-18" in body


def test_email_skip():
    a = agent()
    a.handle(DEMO)
    a.handle("that's it")
    r = a.handle("no thanks")
    assert r["state"]["phase"] == "ENDED" and r["state"]["email"]["decision"] == "skip"


@pytest.mark.parametrize("raw,iso", [("1985-03-15", "1985-03-15"), ("03/15/1985", "1985-03-15"),
                                     ("March 15, 1985", "1985-03-15"), ("15th March 1985", "1985-03-15")])
def test_dob_normalization(raw, iso):
    assert tools.norm_dob(raw) == iso
