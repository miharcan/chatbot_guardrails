"""Mental health guardrail core logic."""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # Optional at runtime; environment variables can still be provided externally.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


# Runtime config defaults.
OPENAI_MODEL = "gpt-4.1-mini"
LLM_TIMEOUT_S = 10.0
AUTO_ESCALATE_CONF_THRESHOLD = 0.70
CONF_SIGMOID_K = 1.4
CONF_MIN = 0.51
CONF_MAX = 0.99
CONF_NO_SIGNAL = 0.58

BLOCKED_WEIGHTS = {
    "BLOCKED_1": 1.0,
    "BLOCKED_2": 1.2,
    "BLOCKED_3": 1.1,
    "BLOCKED_4": 1.2,
    "BLOCKED_5": 1.3,
}
ALLOWED_WEIGHTS = {"ALLOWED_1": 0.4, "ALLOWED_2": 0.3, "ALLOWED_3": 0.3}

BLOCKED_PATTERNS = {
    "BLOCKED_1": [
        r"\byou should\b",
        r"\byou need to\b",
        r"\bstart\b",
        r"\bdo this\b",
        r"\btry this\b",
        r"\bhere is what to do\b",
        r"\bfollow this plan\b",
        r"\bwhat should i do\b",
        r"\bwhat should i do tonight\b",
        r"\bshould i follow\b",
    ],
    "BLOCKED_2": [
        r"\bgiven your\b",
        r"\bfrom what you described\b",
        r"\bbased on your\b",
        r"\bfor your anxiety\b",
        r"\bfor your depression\b",
        r"\byour panic\b",
        r"\bgiven my\b",
        r"\bmy panic attacks\b",
    ],
    "BLOCKED_3": [
        r"\bcbt\b",
        r"\btherapy type\b",
        r"\bmedication\b",
        r"\bssri\b",
        r"\btreatment plan\b",
        r"\bdose\b",
        r"\bbest therapy\b",
    ],
    "BLOCKED_4": [
        r"\byou have\b",
        r"\byou likely have\b",
        r"\bsounds like\b",
        r"\bthis is depression\b",
        r"\bclinical depression\b",
        r"\bdiagnos\w*\b",
    ],
    "BLOCKED_5": [
        r"\bif you are suicidal, do\b",
        r"\bmake a safety plan\b",
        r"\bremove means\b",
        r"\bself-harm plan\b",
        r"\bhow to avoid hurting yourself\b",
    ],
}

ALLOWED_PATTERNS = {
    "ALLOWED_1": [
        r"\bi am sorry\b",
        r"\bthat sounds really hard\b",
        r"\byour feelings are valid\b",
        r"\byou are not alone\b",
        r"\bi hear you\b",
        r"\byou are not broken\b",
        r"\bi am here\b",
    ],
    "ALLOWED_2": [
        r"\byou deserve support\b",
        r"\bit can help to reach out\b",
        r"\bwhen you are ready\b",
        r"\bif you want, i can listen\b",
    ],
    "ALLOWED_3": [
        r"\blicensed professional\b",
        r"\bemergency services\b",
        r"\btrusted person\b",
        r"\bcrisis hotline\b",
    ],
}

BLOCKED_REASON_MAP = {
    "BLOCKED_1": "Prescriptive mental-health action language",
    "BLOCKED_2": "Personalization to user symptoms or history",
    "BLOCKED_3": "Treatment recommendation or ranking",
    "BLOCKED_4": "Diagnosis-like interpretation",
    "BLOCKED_5": "Crisis procedural instructions beyond immediate escalation",
}

SUPPORT_REASON_MAP = {
    "ALLOWED_1": "Emotional validation and empathetic support",
    "ALLOWED_2": "Non-directive encouragement",
    "ALLOWED_3": "Generic referral or escalation to trusted/professional help",
}


def load_env_candidates() -> None:
    for candidate in (".env", "../.env"):
        load_dotenv(candidate, override=False)


load_env_candidates()
OPENAI_MODEL = os.getenv("OPENAI_GUARD_MODEL", OPENAI_MODEL)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def find_triggers(text: str, pattern_map: dict[str, list[str]]) -> dict[str, list[str]]:
    lowered_text = text.lower()
    hits: dict[str, list[str]] = {}
    for key, patterns in pattern_map.items():
        matched = [pattern for pattern in patterns if re.search(pattern, lowered_text)]
        if matched:
            hits[key] = matched
    return hits


def confidence_from_scores(block_score: float, allow_score: float) -> float:
    if block_score == 0 and allow_score == 0:
        return CONF_NO_SIGNAL
    margin = block_score - allow_score
    prob = 1.0 / (1.0 + math.exp(-CONF_SIGMOID_K * margin))
    return round(max(CONF_MIN, min(CONF_MAX, prob)), 3)


def guard_decide(text: str) -> dict[str, Any]:
    """Deterministic rule-based decision."""
    if not isinstance(text, str) or not text.strip():
        return {
            "is_advice": True,
            "confidence_score": CONF_NO_SIGNAL,
            "rationale": "Input was empty/invalid; conservative default applied.",
            "blocked_criteria": [],
            "support_signals": [],
        }

    blocked_hits = find_triggers(text, BLOCKED_PATTERNS)
    allowed_hits = find_triggers(text, ALLOWED_PATTERNS)
    blocked_score = sum(BLOCKED_WEIGHTS[c] for c in blocked_hits)
    allowed_score = sum(ALLOWED_WEIGHTS[a] for a in allowed_hits)

    if blocked_score > 0:
        is_advice = True
        decision_reason = f"Blocking criteria found: {', '.join(sorted(blocked_hits))}."
    elif allowed_score > 0:
        is_advice = False
        decision_reason = f"Allowed signals found: {', '.join(sorted(allowed_hits))}."
    else:
        is_advice = True
        decision_reason = "No clear allow/block signal; conservative ambiguity rule applies (block)."

    confidence = confidence_from_scores(blocked_score, allowed_score)
    rationale = (
        f"Decision={'BLOCKED_ADVICE' if is_advice else 'ALLOWED_SUPPORT'}; "
        f"block_score={blocked_score:.2f}, allow_score={allowed_score:.2f}. {decision_reason}"
    )

    return {
        "is_advice": is_advice,
        "confidence_score": confidence,
        "rationale": rationale,
        "blocked_criteria": sorted(blocked_hits),
        "support_signals": sorted(allowed_hits),
    }


def llm_available() -> bool:
    return bool(OPENAI_API_KEY)


def build_judge_prompt(text: str) -> str:
    return f"""You are a policy judge for chatbot safety.
Task: detect personalized mental health advice.

Policy:
- BLOCK if any of BLOCKED_1..BLOCKED_5 apply:
  BLOCKED_1 prescriptive mental-health actions
  BLOCKED_2 personalization to user symptoms/history
  BLOCKED_3 treatment recommendation/ranking
  BLOCKED_4 diagnosis-like interpretation
  BLOCKED_5 crisis handling instructions beyond immediate emergency/professional escalation
- ALLOW only empathetic, non-directive support.
- If uncertain or mixed, choose BLOCK (conservative default).

Return strict JSON only with keys:
{{
  "is_advice": bool,
  "confidence_score": float,
  "rationale": str,
  "blocked_criteria": list[str]
}}

Text to judge:
{text}
"""


def estimate_cost_usd(
    input_tokens: int, output_tokens: int, price_in_per_m: float = 0.40, price_out_per_m: float = 1.60
) -> float:
    return round(
        (input_tokens / 1_000_000) * price_in_per_m + (output_tokens / 1_000_000) * price_out_per_m,
        6,
    )


def llm_judge_decide(text: str, timeout_s: float = LLM_TIMEOUT_S) -> dict[str, Any] | None:
    if not llm_available():
        return None
    try:
        from openai import OpenAI

        started = time.time()
        response = OpenAI(api_key=OPENAI_API_KEY).responses.create(
            model=OPENAI_MODEL,
            input=build_judge_prompt(text),
            max_output_tokens=250,
            timeout=timeout_s,
        )
        elapsed_ms = int((time.time() - started) * 1000)

        raw = (response.output_text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{") : raw.rfind("}") + 1]

        data = json.loads(raw)
        usage = getattr(response, "usage", None)
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        data["latency_ms"] = elapsed_ms
        data["input_tokens"] = in_tokens
        data["output_tokens"] = out_tokens
        data["cost_usd"] = estimate_cost_usd(in_tokens, out_tokens)
        data["judge_mode"] = "llm"
        data["confidence_score"] = float(max(0.0, min(1.0, data.get("confidence_score", 0.5))))
        data.setdefault("support_signals", [])
        return data
    except Exception:
        return None


def guard_decide_llm(text: str, mode: str = "auto") -> dict[str, Any]:
    """Hybrid guard with rule-first routing and optional LLM escalation."""
    started = time.time()
    if mode == "llm":
        llm_out = llm_judge_decide(text)
        if llm_out is not None:
            return llm_out
        pred = guard_decide(text)
        pred["judge_mode"] = "rule"
        pred["latency_ms"] = int((time.time() - started) * 1000)
        pred["escalated_from_rule"] = False
        return pred

    pred = guard_decide(text)
    pred["judge_mode"] = "rule"
    pred["latency_ms"] = int((time.time() - started) * 1000)

    if mode == "rule":
        pred["escalated_from_rule"] = False
        return pred

    weak_evidence = not pred.get("blocked_criteria") and not pred.get("support_signals")
    low_confidence = pred.get("confidence_score", 0.0) < AUTO_ESCALATE_CONF_THRESHOLD
    if weak_evidence or low_confidence:
        llm_out = llm_judge_decide(text)
        if llm_out is not None:
            llm_out["escalated_from_rule"] = True
            return llm_out

    pred["escalated_from_rule"] = False
    return pred


def classify_text(text: str, mode: str = "rule") -> dict[str, Any]:
    """
    Public API contract for services: text in, safe/unsafe out.

    safe  -> supportive output allowed
    unsafe -> personalized mental-health advice should be blocked
    """
    decision = guard_decide_llm(text, mode=mode) if mode in {"auto", "llm", "rule"} else guard_decide(text)
    label = "unsafe" if decision.get("is_advice", True) else "safe"
    blocked_criteria = list(decision.get("blocked_criteria", []))
    support_signals = list(decision.get("support_signals", []))
    blocked_reasons = [BLOCKED_REASON_MAP.get(code, f"Policy criterion triggered: {code}") for code in blocked_criteria]
    support_reasons = [SUPPORT_REASON_MAP.get(code, f"Support signal triggered: {code}") for code in support_signals]

    return {
        "label": label,
        "confidence": float(decision.get("confidence_score", CONF_NO_SIGNAL)),
        "rationale": str(decision.get("rationale", "")),
        "mode": str(decision.get("judge_mode", "rule")),
        "blocked_criteria": blocked_criteria,
        "support_signals": support_signals,
        "blocked_reasons": blocked_reasons,
        "support_reasons": support_reasons,
    }
