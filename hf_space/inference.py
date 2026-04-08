"""Inference runner for the Invoice Verification environment.

Supports three agent modes:
    1. **Rule-based** (default) — deterministic policy engine, no API key needed
    2. **OpenAI** — uses GPT model via ``OPENAI_API_KEY`` env var
    3. **Heuristic** — legacy keyword matcher (fallback)

Usage::

    python inference.py                    # defaults: seed=42, 5 episodes
    python inference.py --seed 123
    python inference.py --episodes 10
    python inference.py --use-llm
"""
from __future__ import annotations

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

import argparse
import atexit
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

from env.models import Action
from env.policy import evaluate_invoice

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = API_BASE_URL
DOCKER_BASE_URL = "http://127.0.0.1:7860"

POLICY_PROMPT = """\
You are evaluating employee invoices against company policy.

Approve when the expense has a clear business purpose, acceptable documentation, and does not violate spending policy.
Reject when the expense is personal, undocumented, excessive, fraudulent, or violates policy.

You MUST respond with ONLY valid JSON — no markdown, no code fences, no extra text.
Return strict JSON with exactly these keys:
- "decision": "approve" or "reject"
- "reason": concise policy-based explanation referencing specific invoice fields (amount, category, receipt, date)
- "confidence": float between 0 and 1

Example:
{"decision": "reject", "reason": "Rejected: receipt is missing and amount $420.00 exceeds the waiver threshold. Category 'personal' is not allowed.", "confidence": 0.92}
"""


# ---------------------------------------------------------------------------
# Episode result tracking
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Captures the outcome of a single evaluation episode."""
    difficulty: str
    decision: str
    expected_decision: str
    decision_correct: bool
    confidence: float
    reward: float
    matched_keywords: List[str] = field(default_factory=list)
    referenced_fields: List[str] = field(default_factory=list)


@dataclass
class RunMetrics:
    """Aggregated metrics across all episodes."""
    total: int = 0
    correct: int = 0
    approvals: int = 0
    rejections: int = 0
    total_confidence: float = 0.0
    total_reward: float = 0.0
    episodes: List[EpisodeResult] = field(default_factory=list)

    def record(self, ep: EpisodeResult) -> None:
        """Record a completed episode."""
        self.total += 1
        if ep.decision_correct:
            self.correct += 1
        if ep.decision == "approve":
            self.approvals += 1
        else:
            self.rejections += 1
        self.total_confidence += ep.confidence
        self.total_reward += ep.reward
        self.episodes.append(ep)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def approval_rate(self) -> float:
        return self.approvals / self.total if self.total else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejections / self.total if self.total else 0.0

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.total if self.total else 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def model_to_dict(model: Action) -> Dict[str, Any]:
    """Convert a Pydantic Action model to a plain dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def cleanup_process(process: Optional[subprocess.Popen]) -> None:
    """Gracefully terminate a subprocess."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON; return None on failure instead of raising."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        stripped = "\n".join(lines)

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
        print(f"  [!] JSON parsed but got {type(data).__name__} instead of dict.")
        return None
    except json.JSONDecodeError as exc:
        print(f"  [!] JSON parse error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RuleBasedAgent:
    """Deterministic agent powered by the centralized policy engine."""

    def predict(self, invoice: Dict[str, Any]) -> Action:
        """Evaluate an invoice using ``env.policy.evaluate_invoice``."""
        result = evaluate_invoice(invoice)
        return Action(
            decision=result["decision"],
            reason=" ".join(result["reasons"]),
            confidence=result["confidence"],
        )


class HeuristicInvoiceAgent:
    """Legacy keyword-based heuristic agent (fallback)."""

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
                reason="Approved: expense supports an approved team-building event with a stated business purpose.",
                confidence=0.74,
            )
        if not receipt and amount > 20:
            return Action(
                decision="reject",
                reason=f"Rejected: receipt is missing and amount ${amount:.2f} exceeds the undocumented expense threshold.",
                confidence=0.88,
            )
        if category in {"personal", "groceries", "entertainment"} or any(
            term in description for term in reject_terms
        ):
            return Action(
                decision="reject",
                reason=f"Rejected: category '{category}' or description indicates a personal or policy-violating expense.",
                confidence=0.86,
            )
        if "local seminar" in description and amount >= 500:
            return Action(
                decision="reject",
                reason=f"Rejected: amount ${amount:.2f} for local travel accommodation appears excessive.",
                confidence=0.76,
            )
        if "candidate" in description and not receipt:
            return Action(
                decision="reject",
                reason="Rejected: receipt is required for recruiting meals under policy.",
                confidence=0.81,
            )
        if any(term in description for term in approve_terms) or category in {
            "office_supplies", "office_equipment", "electronics", "software",
            "training", "travel", "transportation", "infrastructure", "legal",
            "accommodation", "health", "rent",
        }:
            return Action(
                decision="approve",
                reason=f"Approved: category '{category}' is allowed, amount ${amount:.2f} is within policy, and receipt is present.",
                confidence=0.72,
            )
        return Action(
            decision="reject",
            reason="Rejected: business justification is unclear - the expense should be reviewed manually.",
            confidence=0.6,
        )


class OpenAIInvoiceAgent:
    """LLM-powered agent using the OpenAI API."""

    def __init__(self) -> None:
        self._fallback = RuleBasedAgent()
        self._api_key = os.getenv("OPENAI_API_KEY") or HF_TOKEN
        self._model = MODEL_NAME
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=self._api_key) if self._api_key else None
        )

    def predict(self, invoice: Dict[str, Any]) -> Action:
        """Query the LLM; fall back to the rule-based agent on failure."""
        if self._client is None:
            return self._fallback.predict(invoice)

        try:
            prompt = (
                f"{POLICY_PROMPT}\n\n"
                f"Invoice JSON:\n{json.dumps({'invoice': invoice}, ensure_ascii=True)}"
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are an invoice verification agent."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            payload = _safe_parse_json(content)
            if payload is None:
                print("  [!] LLM returned unparseable JSON, falling back to rule-based.")
                return self._fallback.predict(invoice)
            return Action(**payload)
        except Exception as exc:
            print(f"  [!] OpenAI call failed ({exc}), falling back to rule-based.")
            return self._fallback.predict(invoice)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def is_server_ready(base_url: str) -> bool:
    """Check if the FastAPI server is responding."""
    try:
        response = requests.get(f"{base_url}/state", timeout=1)
        return response.ok
    except requests.RequestException:
        return False


def resolve_base_url(base_url: str) -> str:
    """Auto-detect whether to use the local or Docker server."""
    if os.getenv("API_BASE_URL"):
        return base_url
    if is_server_ready(base_url):
        return base_url
    if is_server_ready(DOCKER_BASE_URL):
        return DOCKER_BASE_URL
    return base_url


def ensure_server(base_url: str, seed: int) -> Optional[subprocess.Popen]:
    """Start the FastAPI server if it isn't already running."""
    if is_server_ready(base_url):
        return None

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    env = os.environ.copy()
    env["INVOICE_ENV_SEED"] = str(seed)

    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app", "--host", host, "--port", str(port),
        ],
        cwd=ROOT_DIR,
        env=env,
    )

    for _ in range(20):
        if is_server_ready(base_url):
            return process
        time.sleep(0.5)

    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("FastAPI server did not start in time.")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agent(agent_type: str) -> RuleBasedAgent | HeuristicInvoiceAgent | OpenAIInvoiceAgent:
    """Instantiate the requested agent type."""
    agents = {
        "rule": RuleBasedAgent,
        "heuristic": HeuristicInvoiceAgent,
        "openai": OpenAIInvoiceAgent,
    }
    cls = agents.get(agent_type)
    if cls is None:
        raise ValueError(f"Unknown agent type '{agent_type}'. Choose from: {list(agents)}")
    return cls()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_difficulty(
    agent: RuleBasedAgent | HeuristicInvoiceAgent | OpenAIInvoiceAgent,
    difficulty: str,
    episodes: int,
    metrics: RunMetrics,
) -> float:
    """Run *episodes* evaluation rounds at the given difficulty.

    Returns the average reward for this difficulty level.
    """
    rewards: List[float] = []

    for episode in range(1, episodes + 1):
        # ── Reset ──
        payload = {"difficulty": difficulty}
        if episode == 1 and hasattr(metrics, "_seed"):
            payload["seed"] = metrics._seed
            
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset",
            json=payload,
            timeout=10,
        )
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
        session_id = reset_data.get("session_id")
        observation = reset_data.get("observation", reset_data)

        # ── Predict ──
        action = agent.predict(observation["invoice"])

        # ── Step ──
        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            params={"session_id": session_id} if session_id is not None else None,
            json=model_to_dict(action),
            timeout=10,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = float(result["reward"])
        info = result["info"]
        rewards.append(reward)

        # ── Record metrics ──
        ep = EpisodeResult(
            difficulty=difficulty,
            decision=action.decision,
            expected_decision="approve" if info.get("decision_correct") and action.decision == "approve"
                             else ("reject" if info.get("decision_correct") and action.decision == "reject"
                                   else ("reject" if action.decision == "approve" else "approve")),
            decision_correct=bool(info.get("decision_correct", False)),
            confidence=action.confidence,
            reward=reward,
            matched_keywords=info.get("matched_keywords", []),
            referenced_fields=info.get("referenced_fields", []),
        )
        metrics.record(ep)

        decision_icon = "Y" if ep.decision_correct else "N"
        kw_count = len(ep.matched_keywords)
        ref_count = len(ep.referenced_fields)

        print(
            f"  [{difficulty}] ep {episode:>2d}: "
            f"decision={action.decision:<7s} [{decision_icon}]  "
            f"reward={reward:.2f}  "
            f"conf={action.confidence:.2f}  "
            f"kw={kw_count}  refs={ref_count}"
        )

    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(metrics: RunMetrics, difficulty_scores: Dict[str, float]) -> None:
    """Print a clean structured summary of the evaluation run."""
    width = 60
    print("\n" + "=" * width)
    print("  EVALUATION SUMMARY")
    print("=" * width)

    # Per-difficulty
    print("\n  Per-Difficulty Rewards:")
    for diff, score in difficulty_scores.items():
        bar = "#" * int(score * 20)
        print(f"    {diff:>8s}: {score:.4f}  {bar}")

    grand_avg = sum(difficulty_scores.values()) / len(difficulty_scores) if difficulty_scores else 0.0
    print(f"    {'overall':>8s}: {grand_avg:.4f}")

    # Aggregate metrics
    print(f"\n  Aggregate Metrics ({metrics.total} episodes):")
    print(f"    Accuracy       : {metrics.accuracy:.2%}  ({metrics.correct}/{metrics.total})")
    print(f"    Approval Rate  : {metrics.approval_rate:.2%}")
    print(f"    Rejection Rate : {metrics.rejection_rate:.2%}")
    print(f"    Avg Confidence : {metrics.avg_confidence:.4f}")
    print(f"    Avg Reward     : {metrics.avg_reward:.4f}")

    print("=" * width)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference against the Invoice Verification environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("INVOICE_RANDOM_SEED", "42")),
        help="Random seed for deterministic execution (default: 42)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.getenv("INVOICE_EPISODES", "5")),
        help="Number of episodes per difficulty level (default: 5)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use OpenAI agent for inference (requires OPENAI_API_KEY). Falls back to rule-based if absent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"LLM model string to use if --use-llm is provided (default: {MODEL_NAME})",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, seed RNG, run evaluation, print metrics."""
    args = parse_args()
    global API_BASE_URL

    # Deterministic seeding
    random.seed(args.seed)

    API_BASE_URL = resolve_base_url(API_BASE_URL)
    server_process = ensure_server(API_BASE_URL, seed=args.seed)
    if server_process is not None:
        atexit.register(cleanup_process, server_process)

    if args.use_llm:
        if not os.getenv("OPENAI_API_KEY"):
            print("LLM requested but no API key found. Falling back to rule-based agent.")
            agent = RuleBasedAgent()
            agent_label = "Rule-based (fallback)"
        else:
            agent = OpenAIInvoiceAgent()
            agent._model = args.model  # Override model via CLI
            agent_label = f"OpenAI ({args.model})"
    else:
        agent = RuleBasedAgent()
        agent_label = "Rule-based"

    metrics = RunMetrics()
    metrics._seed = args.seed

    print("=" * 60)
    print("  Invoice Verification - Inference Run")
    print(f"  Agent: {agent_label}  |  Seed: {args.seed}  |  Episodes/diff: {args.episodes}")
    print("=" * 60)

    difficulty_scores: Dict[str, float] = {}

    for difficulty in ("easy", "medium", "hard"):
        print(f"\n-- {difficulty.upper()} ({args.episodes} episodes) --")
        avg = evaluate_difficulty(agent, difficulty, args.episodes, metrics)
        difficulty_scores[difficulty] = avg
        print(f"  -> {difficulty} average reward: {avg:.4f}")

    print_summary(metrics, difficulty_scores)

    cleanup_process(server_process)


if __name__ == "__main__":
    main()
