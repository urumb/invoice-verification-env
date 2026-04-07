from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

from env.models import Action


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.getenv("INVOICE_ENV_URL", "http://127.0.0.1:8000")
EPISODES_PER_DIFFICULTY = int(os.getenv("INVOICE_EPISODES", "5"))

POLICY_PROMPT = """\
You are evaluating employee invoices against company policy.

Approve when the expense has a clear business purpose, acceptable documentation, and does not violate spending policy.
Reject when the expense is personal, undocumented, excessive, fraudulent, or violates policy.

You MUST respond with ONLY valid JSON — no markdown, no code fences, no extra text.
Return strict JSON with exactly these keys:
- "decision": "approve" or "reject"
- "reason": concise policy-based explanation (string)
- "confidence": float between 0 and 1

Example:
{"decision": "reject", "reason": "Personal expense not reimbursable under policy.", "confidence": 0.92}
"""


def model_to_dict(model: Action) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def cleanup_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


# ---------------------------------------------------------------------------
# Heuristic fallback agent (no API key needed)
# ---------------------------------------------------------------------------

class HeuristicInvoiceAgent:
    def predict(self, invoice: Dict[str, Any]) -> Action:
        description = str(invoice.get("description", "")).lower()
        category = str(invoice.get("category", "")).lower()
        amount = float(invoice.get("amount", 0))
        receipt = bool(invoice.get("receipt", False))

        reject_terms = [
            "personal", "birthday", "gaming", "grocery", "household",
            "luxury", "alcohol", "first-class", "first class", "spa",
            "video game",
        ]
        approve_terms = [
            "client", "conference", "training", "software", "hosting",
            "office", "business", "professional", "remote work", "stipend",
            "wellness program", "overtime", "candidate", "laptop",
        ]

        if "team-building" in description or "team building" in description:
            return Action(
                decision="approve",
                reason="Approved because the expense supports an approved team-building event with a stated business purpose.",
                confidence=0.74,
            )

        if not receipt and amount > 20:
            return Action(
                decision="reject",
                reason="Rejected because the invoice is missing a receipt and exceeds the undocumented expense threshold.",
                confidence=0.88,
            )

        if category in {"personal", "groceries", "entertainment"} or any(
            term in description for term in reject_terms
        ):
            return Action(
                decision="reject",
                reason="Rejected because the expense appears personal or policy violating rather than a valid business expense.",
                confidence=0.86,
            )

        if "local seminar" in description and amount >= 500:
            return Action(
                decision="reject",
                reason="Rejected because local travel accommodation appears excessive for an event that could be commuted.",
                confidence=0.76,
            )

        if "candidate" in description and not receipt:
            return Action(
                decision="reject",
                reason="Rejected because meals for recruiting still require a receipt under policy.",
                confidence=0.81,
            )

        if any(term in description for term in approve_terms) or category in {
            "office_supplies", "office_equipment", "electronics", "software",
            "training", "travel", "transportation", "infrastructure", "legal",
            "accommodation", "health", "rent",
        }:
            return Action(
                decision="approve",
                reason="Approved because the invoice has a valid business purpose, aligns with an allowed category, and has acceptable support.",
                confidence=0.72,
            )

        return Action(
            decision="reject",
            reason="Rejected because the business justification is unclear and the expense should be reviewed manually.",
            confidence=0.6,
        )


# ---------------------------------------------------------------------------
# OpenAI-powered agent
# ---------------------------------------------------------------------------

class OpenAIInvoiceAgent:
    def __init__(self) -> None:
        self._fallback = HeuristicInvoiceAgent()
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=self._api_key) if self._api_key else None
        )

    def predict(self, invoice: Dict[str, Any]) -> Optional[Action]:
        if self._client is None:
            return self._fallback.predict(invoice)

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": POLICY_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps({"invoice": invoice}, ensure_ascii=True),
                    },
                ],
            )
            content = completion.choices[0].message.content or "{}"
            payload = _safe_parse_json(content)
            if payload is None:
                print(f"  ⚠ LLM returned unparseable JSON, falling back to heuristic.")
                return self._fallback.predict(invoice)
            return Action(**payload)
        except Exception as exc:
            print(f"  ⚠ OpenAI call failed ({exc}), falling back to heuristic.")
            return self._fallback.predict(invoice)


# ---------------------------------------------------------------------------
# Safe JSON parsing
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON; return None on failure instead of raising."""
    # Strip markdown code fences if the model wraps the response
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        stripped = "\n".join(lines)

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
        print(f"  ⚠ JSON parsed but got {type(data).__name__} instead of dict.")
        return None
    except json.JSONDecodeError as exc:
        print(f"  ⚠ JSON parse error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def is_server_ready(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/state", timeout=1)
        return response.ok
    except requests.RequestException:
        return False


def ensure_server(base_url: str) -> Optional[subprocess.Popen]:
    if is_server_ready(base_url):
        return None

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app", "--host", host, "--port", str(port),
        ],
        cwd=ROOT_DIR,
    )

    for _ in range(20):
        if is_server_ready(base_url):
            return process
        time.sleep(0.5)

    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("FastAPI server did not start in time.")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_difficulty(
    agent: OpenAIInvoiceAgent, difficulty: str, episodes: int
) -> float:
    rewards: list[float] = []

    for episode in range(1, episodes + 1):
        # ── Reset ──
        reset_resp = requests.post(
            f"{BASE_URL}/reset",
            json={"difficulty": difficulty},
            timeout=10,
        )
        reset_resp.raise_for_status()
        observation = reset_resp.json()

        # ── Predict ──
        action = agent.predict(observation["invoice"])
        if action is None:
            print(f"  [{difficulty}] episode {episode}: SKIPPED (no valid action)")
            continue

        # ── Step ──
        step_resp = requests.post(
            f"{BASE_URL}/step",
            json=model_to_dict(action),
            timeout=10,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = float(result["reward"])
        info = result["info"]
        rewards.append(reward)

        decision_status = "✓" if info.get("decision_correct") else "✗"
        matched = info.get("matched_keywords", [])

        print(
            f"  [{difficulty}] episode {episode}: "
            f"decision={action.decision} [{decision_status}]  "
            f"reward={reward:.2f}  "
            f"keywords_matched={len(matched)}"
        )

    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def main() -> None:
    server_process = ensure_server(BASE_URL)
    if server_process is not None:
        atexit.register(cleanup_process, server_process)

    agent = OpenAIInvoiceAgent()

    print("=" * 60)
    print("  Invoice Verification — Inference Run")
    print("=" * 60)

    overall_scores: Dict[str, float] = {}

    for difficulty in ("easy", "medium", "hard"):
        print(f"\n── {difficulty.upper()} ({EPISODES_PER_DIFFICULTY} episodes) ──")
        avg = evaluate_difficulty(agent, difficulty, EPISODES_PER_DIFFICULTY)
        overall_scores[difficulty] = avg
        print(f"  → {difficulty} average reward: {avg:.2f}")

    grand_avg = sum(overall_scores.values()) / len(overall_scores)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for diff, score in overall_scores.items():
        print(f"  {diff:>8s}: {score:.2f}")
    print(f"  {'overall':>8s}: {grand_avg:.2f}")
    print("=" * 60)

    cleanup_process(server_process)


if __name__ == "__main__":
    main()
