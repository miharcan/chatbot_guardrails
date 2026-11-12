> **GCP retirement — 11 November 2026:** Cloud deployment is archived and shut down. Historical cloud URLs and deployment examples below are retained as reference. Do not run them. See [retirement records](docs/gcp-retirement/README.md).

# Mental Health Guardrails API

Production-minded guardrail service that detects **personalized mental health advice** and returns a structured safety decision (`safe` / `unsafe`) with confidence and explainability.

This code demonstrates:
- sociotechnical boundary design (support vs advice),
- LLM-as-a-judge + deterministic guardrail modes,
- explainable outputs with policy criteria and human-readable reasons,
- deployable API with authentication, rate limiting, and spend controls.

It is built around a challenge framing that emphasizes the ability to:
- translate a complex social safety problem into a technical solution,
- design rigorous evaluation systems (LLM-as-a-judge) for real time guardrails,
- link model probability and human understanding.

## Challenge Context

A general-purpose assistant is deployed globally and users often seek emotional support.
The system should remain empathetic, while blocking personalized mental health advice due to safety and liability risk.

This implementation focuses on a runnable, self-standing code deliverable (Part A):
- policy boundary implementation,
- judge pipeline,
- explainability interface,
- standardized API contract.

This is an MVP implementation and not a complete solution.

## Beyond Mental Health

Mental health is the demonstrated use case in this repository, but the same guardrail architecture is domain-agnostic:
- define a domain boundary,
- encode policy criteria,
- score and explain decisions,
- route low-confidence/ambiguous cases for stronger judging.

The same pattern can be applied to finance, for example:
- allow general education and neutral explanation,
- block personalized financial advice or high-risk prescriptive guidance,
- provide clear rationale for non-technical stakeholders.

## API Contract

Guardrail scope:
- this service evaluates **assistant outputs before display** (not user inputs).

Endpoint:
- `POST /decide`

Input:
- `{"text": "...", "mode": "rule|auto|llm"}`

Output:
- `label`: `safe | unsafe`
- `confidence`
- `rationale`
- `blocked_criteria`
- `support_signals`
- `blocked_reasons`
- `support_reasons`

## Repository Structure

- `src/guardrail.py`: policy criteria, scoring, confidence, hybrid routing
- `src/api.py`: FastAPI service (`/health`, `/decide`)
- `Dockerfile`: containerized runtime
- `requirements.txt`: runtime dependencies

## Quick Start (Recommended)

### 1. Run with Docker
Build:
```bash
docker build -t chatbot-guardrails:local .
```

Run:
```bash
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY=your_key_here \
  -e OPENAI_GUARD_MODEL=gpt-4.1-mini \
  chatbot-guardrails:local
```

Test (safe example):
```bash
curl -s http://127.0.0.1:8080/decide \
  -H 'Content-Type: application/json' \
  -d '{"text":"I hear you and I am here with you.","mode":"rule"}'
```

Test (unsafe example):
```bash
curl -s http://127.0.0.1:8080/decide \
  -H 'Content-Type: application/json' \
  -d '{"text":"You should take panadol for that.","mode":"rule"}'
```

### 2. Run locally without Docker
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

## Hosted GCP API (Test Access)

The project is also deployed on GCP Cloud Run and can be tested via hosted API.

To request test access:
- Email: `mihael.arcan@gmail.com`
- Subject: `Guardrails API access request`
- Include: use case, expected request volume, and whether you need `rule` only or `auto/llm`

Important:
- API keys are issued manually (there is no automatic key generator endpoint in this repo)
- Standard keys can use `mode=rule`
- Privileged keys can use `mode=auto` and `mode=llm`
- Keys may be rotated or revoked for abuse

Example hosted call (after receiving a key and URL):
```bash
curl -s https://URL/decide \
  -H 'X-API-Key: ASSIGNED_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"text":"I hear you and I am here with you.","mode":"rule"}'
```

## Example API Outputs

Safe (`rule`):
```json
{
  "label": "safe",
  "confidence": 0.51,
  "rationale": "Decision=ALLOWED_SUPPORT; block_score=0.00, allow_score=0.40. Allowed signals found: ALLOWED_1.",
  "mode": "rule",
  "blocked_criteria": [],
  "support_signals": ["ALLOWED_1"],
  "blocked_reasons": [],
  "support_reasons": ["Emotional validation and empathetic support"]
}
```

Unsafe (`rule`):
```json
{
  "label": "unsafe",
  "confidence": 0.802,
  "rationale": "Decision=BLOCKED_ADVICE; block_score=1.00, allow_score=0.00. Blocking criteria found: BLOCKED_1.",
  "mode": "rule",
  "blocked_criteria": ["BLOCKED_1"],
  "support_signals": [],
  "blocked_reasons": ["Prescriptive mental-health action language"],
  "support_reasons": []
}
```

Restricted mode with standard key (`403`):
```json
{
  "detail": "Mode restricted to privileged API keys"
}
```

## Controls

Built-in controls:
- API key authentication (`X-API-Key`)
- privileged key gating for expensive modes (`auto`/`llm`)
- rate limiting per key and IP
- max input length guard
- Cloud Run max instance and timeout controls

## Notes

This is a safety MVP for research and engineering evaluation.
It is not a clinical diagnosis or treatment system.
