"""Read-only access to the demo fixtures (policyholders, claims, guidance...)."""
from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))


def _load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def policyholders() -> list[dict]:
    return _load("policyholders.json")


@lru_cache
def claims() -> list[dict]:
    return _load("claims.json")


@lru_cache
def representatives() -> list[dict]:
    return _load("representatives.json")


@lru_cache
def consent_scenarios() -> dict:
    return _load("consent_scenarios.json")


@lru_cache
def doc_guidelines() -> dict:
    return _load("required_document_guideline.json")


@lru_cache
def claim_schema() -> dict:
    return _load("claim_schema.json")


def party(party_id: str) -> dict | None:
    return next((p for p in policyholders() if p["party_id"] == party_id), None)


def claims_for(party_id: str) -> list[dict]:
    return [c for c in claims() if c["party_id"] == party_id]


def today() -> date:
    """Demo clock. Override with DEMO_TODAY=YYYY-MM-DD for deterministic demos."""
    v = os.getenv("DEMO_TODAY")
    return date.fromisoformat(v) if v else date.today()
