"""Thin provider layer: Anthropic, any OpenAI-compatible endpoint (OpenAI, Groq, ...), or offline 'mock'.

Uses plain HTTPS (stdlib urllib) instead of vendor SDKs, so there are no SDK/httpx version conflicts.
Any failure is recorded in `last_error` (shown in the UI) and the SOP falls back to safe template text.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request

log = logging.getLogger("llm")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")

DEFAULT_MODELS = {
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
}

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _SSL = ssl.create_default_context()


def resolve_provider(provider: str | None, api_key: str | None) -> tuple[str, str | None]:
    """Pick provider from explicit choice, else whichever env key exists, else mock."""
    p = (provider or os.getenv("LLM_PROVIDER") or "").lower()
    if p == "mock":
        return "mock", None
    if p in ("anthropic", "openai"):
        key = api_key or os.getenv(f"{p.upper()}_API_KEY")
        return (p, key) if key else ("mock", None)
    if api_key:  # key given without provider: guess from prefix
        return ("anthropic" if api_key.startswith("sk-ant") else "openai"), api_key
    for p in ("anthropic", "openai"):
        if os.getenv(f"{p.upper()}_API_KEY"):
            return p, os.getenv(f"{p.upper()}_API_KEY")
    return "mock", None


def _post(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json",
                                          "user-agent": "insurance-sop-agent/1.0", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from None


class LLM:
    def __init__(self, provider: str | None = None, api_key: str | None = None, model: str | None = None):
        self.provider, self._key = resolve_provider(provider, api_key)
        self.model = model or DEFAULT_MODELS.get(self.provider)
        self.base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.last_error: str | None = None

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def _fail(self, what: str, e: Exception):
        self.last_error = f"{what} failed: {e}"
        log.warning(self.last_error)

    def _anthropic(self, body: dict) -> dict:
        return _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": self._key, "anthropic-version": "2023-06-01"},
                     {"model": self.model, **body})

    def _openai(self, body: dict) -> dict:
        if "gpt-oss" in (self.model or ""):  
            body = {**body, "reasoning_effort": "low", "max_tokens": max(body.get("max_tokens", 0), 2000)}
        return _post(f"{self.base_url}/chat/completions",
                     {"authorization": f"Bearer {self._key}"}, {"model": self.model, **body})

    def structured(self, system: str, user: str, schema: dict, name: str = "record") -> dict | None:
        if self.is_mock:
            return None
        try:
            if self.provider == "anthropic":
                r = self._anthropic({
                    "max_tokens": 1024, "temperature": 0, "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "tools": [{"name": name, "description": "Record the extracted fields.", "input_schema": schema}],
                    "tool_choice": {"type": "tool", "name": name}})
                return next(b["input"] for b in r["content"] if b["type"] == "tool_use")
            r = self._openai({
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "tools": [{"type": "function", "function": {"name": name, "description": "Record the extracted fields.",
                                                            "parameters": schema}}],
                "tool_choice": {"type": "function", "function": {"name": name}}})
            return json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        except Exception as e:
            self._fail("structured call", e)
            return None

    def chat(self, system: str, messages: list[dict], max_tokens: int = 500) -> str | None:
        if self.is_mock:
            return None
        try:
            if self.provider == "anthropic":
                r = self._anthropic({"max_tokens": max_tokens, "temperature": 0.4, "system": system,
                                     "messages": messages})
                return "".join(b["text"] for b in r["content"] if b["type"] == "text").strip()
            r = self._openai({"temperature": 0.4, "max_tokens": max_tokens,
                              "messages": [{"role": "system", "content": system}, *messages]})
            text = (r["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise RuntimeError(f"model returned an empty reply (finish_reason={r['choices'][0].get('finish_reason')})")
            return text
        except Exception as e:
            self._fail("chat call", e)
            return None