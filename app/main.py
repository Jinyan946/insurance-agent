"""FastAPI server: chat API + static test UI."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Agent
from .llm import LLM
from .sop import Session

app = FastAPI(title="Insurance SOP Agent")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

SESSIONS: dict[str, Agent] = {}


class NewSession(BaseModel):
    provider: str | None = None      # anthropic | openai | mock (default: auto from env)
    api_key: str | None = None       # optional; overrides env for this session only, never stored on disk
    model: str | None = None
    consent_scenario: str = "default"


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    llm = LLM()
    return {"default_provider": llm.provider, "default_model": llm.model,
            "env_key_present": not llm.is_mock}


@app.post("/api/session")
def new_session(body: NewSession):
    llm = LLM(body.provider, body.api_key, body.model)
    s = Session(consent_scenario=body.consent_scenario)
    agent = Agent(s, llm)
    SESSIONS[s.id] = agent
    greeting = agent.greet()
    return {"session_id": s.id, "reply": greeting, "provider": llm.provider, "model": llm.model,
            "state": s.public_state()}


@app.post("/api/chat")
def chat(body: ChatIn):
    agent = SESSIONS.get(body.session_id)
    if not agent:
        raise HTTPException(404, "unknown session")
    if not body.message.strip():
        raise HTTPException(400, "empty message")
    return agent.handle(body.message.strip()[:2000])


@app.get("/api/session/{sid}/events")
def events(sid: str):
    agent = SESSIONS.get(sid)
    if not agent:
        raise HTTPException(404, "unknown session")
    return {"events": agent.s.events, "email": agent.s.email, "handoff": agent.s.handoff}


@app.get("/healthz")
def healthz():
    return {"ok": True, "provider_default": LLM().provider, "port": os.getenv("PORT", "8000")}
