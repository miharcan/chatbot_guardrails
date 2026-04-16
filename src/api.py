"""Lightweight HTTP API for guardrail decisions."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.guardrail import classify_text


app = FastAPI(title="chatbot-guardrails-api", version="0.1.0")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
PRIVILEGED_API_KEYS = {k.strip() for k in os.getenv("PRIVILEGED_API_KEYS", "").split(",") if k.strip()}
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_RATE_LIMIT_STATE: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = threading.Lock()


class DecideRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Candidate assistant response text to classify.")
    mode: Literal["rule", "auto", "llm"] = Field(
        default="rule",
        description="rule=deterministic baseline, auto=rule with optional LLM escalation, llm=force LLM judge",
    )


class DecideResponse(BaseModel):
    label: Literal["safe", "unsafe"]
    confidence: float
    rationale: str
    mode: str
    blocked_criteria: list[str]
    support_signals: list[str]
    blocked_reasons: list[str]
    support_reasons: list[str]


def _client_ip(x_forwarded_for: str | None, x_real_ip: str | None) -> str:
    if isinstance(x_forwarded_for, str) and x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    if isinstance(x_real_ip, str) and x_real_ip:
        return x_real_ip.strip()
    return "unknown"


def _require_auth_and_tier(
    authorization: str | None,
    x_api_key: str | None,
) -> tuple[str, bool]:
    has_auth_config = bool(API_AUTH_TOKEN or API_KEYS)
    auth_identity = "anonymous"
    is_privileged = False

    if x_api_key:
        key = x_api_key.strip()
        if key in API_KEYS:
            auth_identity = f"key:{key}"
            is_privileged = key in PRIVILEGED_API_KEYS
        else:
            raise HTTPException(status_code=401, detail="Unauthorized")
    elif authorization:
        expected = f"Bearer {API_AUTH_TOKEN}"
        if API_AUTH_TOKEN and authorization.strip() == expected:
            auth_identity = "bearer:token"
            is_privileged = True
        else:
            raise HTTPException(status_code=401, detail="Unauthorized")
    elif has_auth_config:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return auth_identity, is_privileged


def _enforce_rate_limit(bucket: str) -> None:
    now = time.time()
    lower_bound = now - RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LIMIT_LOCK:
        q = _RATE_LIMIT_STATE[bucket]
        while q and q[0] < lower_bound:
            q.popleft()
        if len(q) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/decide", response_model=DecideResponse)
def decide(
    req: DecideRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_forwarded_for: str | None = Header(default=None),
    x_real_ip: str | None = Header(default=None),
) -> DecideResponse:
    auth_identity, is_privileged = _require_auth_and_tier(authorization, x_api_key)
    client_ip = _client_ip(x_forwarded_for, x_real_ip)
    _enforce_rate_limit(f"{auth_identity}|{client_ip}")

    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Input too long (max {MAX_TEXT_CHARS} chars)")

    if req.mode in {"auto", "llm"} and not is_privileged:
        raise HTTPException(status_code=403, detail="Mode restricted to privileged API keys")

    result = classify_text(req.text, mode=req.mode)
    return DecideResponse(**result)
