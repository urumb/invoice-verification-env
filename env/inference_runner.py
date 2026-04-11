from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .models import Action, dump_model
from .openenv_adapter import OpenEnvAdapter
from .policy import (
    MAX_AMOUNT,
    evaluate_invoice,
    is_valid_category,
    is_valid_date,
    parse_amount,
    parse_receipt,
)

load_dotenv()

try:
    import numpy as np
except ImportError:
    np = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
TASKS = ("easy", "medium", "hard")
STAGES = ("analyze", "flag_issues", "final_decision")
ANALYZE_ACTIONS = {
    "verify_policy_fields",
    "check_documentation",
    "reconcile_totals",
    "review_line_items",
    "inspect_vendor",
}
FLAG_ACTIONS = {
    "flag_document_gap",
    "flag_total_mismatch",
    "flag_vendor_conflict",
    "flag_policy_risk",
    "confirm_clean_invoice",
}

SYSTEM_PROMPT = """
You are the LLM invoice reviewer inside an OpenEnv benchmark.
Inspect the invoice yourself instead of imitating a rule-based policy engine.
Use the full invoice, the current stage, the previous findings, and the derived checks.
Write concise step-by-step reasoning in one line like:
"1) ... 2) ... 3) ..."

Return exactly one JSON object:
{
  "stage": "analyze|flag_issues|final_decision",
  "action": "specific stage action or final approve/reject",
  "reasoning": "single-line step-by-step reasoning",
  "confidence": 0.0
}

Rules:
- stage must exactly match current_stage
- analyze action should be specific, such as verify_policy_fields, check_documentation,
  reconcile_totals, review_line_items, or inspect_vendor
- flag_issues action should be specific, such as flag_document_gap, flag_total_mismatch,
  flag_vendor_conflict, flag_policy_risk, or confirm_clean_invoice
- final_decision action must be approve or reject
- reasoning must reference concrete invoice evidence
- output JSON only
""".strip()


@dataclass
class EpisodeResult:
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
    total: int = 0
    correct: int = 0
    approvals: int = 0
    rejections: int = 0
    total_confidence: float = 0.0
    total_reward: float = 0.0
    episodes: List[EpisodeResult] = field(default_factory=list)

    def record(self, episode: EpisodeResult) -> None:
        self.total += 1
        self.correct += int(episode.decision_correct)
        self.approvals += int(episode.decision == "approve")
        self.rejections += int(episode.decision != "approve")
        self.total_confidence += episode.confidence
        self.total_reward += episode.reward
        self.episodes.append(episode)

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


class CompatibleOpenEnv:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._env = OpenEnvAdapter(seed=seed)

    def reset(self, seed: Optional[int] = None, task: str = "easy") -> Dict[str, Any]:
        return self._env.reset(seed=seed, difficulty=task)

    def step(self, action: Action | Dict[str, Any]):
        payload = action if isinstance(action, dict) else dump_model(action)
        return self._env.step(payload)

    def close(self) -> None:
        self._env.close()


class BaseStageAgent:
    name: str

    def act(self, observation: Dict[str, Any], seed: int) -> Action:
        raise NotImplementedError


class RuleBasedStageAgent(BaseStageAgent):
    name = "rule"

    def act(self, observation: Dict[str, Any], seed: int) -> Action:
        return rule_based_action(observation)


class LLMStageAgent(BaseStageAgent):
    name = "llm"

    def __init__(self, client: Optional[OpenAI]) -> None:
        self._client = client

    def act(self, observation: Dict[str, Any], seed: int) -> Action:
        if self._client is None:
            return rule_based_action(observation)
        return request_llm_action(self._client, observation, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference against the invoice verification environment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Print both the rule-based run and the LLM run using the same output format.",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def single_line(value: Any) -> str:
    return " ".join(str(value).split())


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end >= start:
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_amount(value: Any) -> str:
    parsed = _safe_float(value)
    return "unknown" if parsed is None else f"${parsed:.2f}"


def _line_items_total(invoice: Dict[str, Any]) -> Optional[float]:
    line_items = invoice.get("line_items")
    if not isinstance(line_items, list) or not line_items:
        return None
    total = 0.0
    for item in line_items:
        if not isinstance(item, dict):
            return None
        line_total = _safe_float(item.get("line_total"))
        if line_total is None:
            quantity = _safe_float(item.get("quantity"))
            unit_price = _safe_float(item.get("unit_price"))
            if quantity is None or unit_price is None:
                return None
            line_total = quantity * unit_price
        total += line_total
    return round(total, 2)


def build_invoice_context(invoice: Dict[str, Any]) -> Dict[str, Any]:
    amount = parse_amount(invoice.get("amount"))
    subtotal = _safe_float(invoice.get("subtotal"))
    tax_amount = _safe_float(invoice.get("tax_amount"))
    expected_tax = _safe_float(invoice.get("expected_tax_amount"))
    computed_total = _safe_float(invoice.get("computed_total"))
    reported_total = _safe_float(invoice.get("reported_total"))
    line_total = _line_items_total(invoice)
    vendor_name = str(invoice.get("vendor_name", "")).strip()
    vendor_registered_name = str(invoice.get("vendor_registered_name", "")).strip()
    vendor_status = str(invoice.get("vendor_status", "")).strip().lower()
    department = str(invoice.get("department", "")).strip().lower()
    approver_department = str(invoice.get("approver_department", "")).strip().lower()
    anomaly_flags = [single_line(item) for item in invoice.get("anomaly_flags", [])]
    missing_fields = [single_line(item) for item in invoice.get("missing_fields", [])]
    policy = evaluate_invoice(invoice)

    if computed_total is None and subtotal is not None and tax_amount is not None:
        computed_total = round(subtotal + tax_amount, 2)

    reported_total_delta = None if computed_total is None or reported_total is None else round(reported_total - computed_total, 2)
    tax_delta = None if expected_tax is None or tax_amount is None else round(tax_amount - expected_tax, 2)
    line_total_delta = None if line_total is None or subtotal is None else round(subtotal - line_total, 2)

    vendor_match = None
    if vendor_name and vendor_registered_name:
        vendor_match = vendor_name.lower() == vendor_registered_name.lower()

    return {
        "amount": _format_amount(amount),
        "policy_limit": _format_amount(MAX_AMOUNT),
        "receipt_present": parse_receipt(invoice.get("receipt", False)),
        "category_allowed": is_valid_category(invoice.get("category")),
        "date_valid": is_valid_date(str(invoice.get("date", "")).strip()),
        "line_item_count": len(invoice.get("line_items", [])) if isinstance(invoice.get("line_items"), list) else 0,
        "line_items_total": _format_amount(line_total),
        "subtotal": _format_amount(subtotal),
        "tax_amount": _format_amount(tax_amount),
        "expected_tax_amount": _format_amount(expected_tax),
        "computed_total": _format_amount(computed_total),
        "reported_total": _format_amount(reported_total),
        "reported_total_delta": reported_total_delta,
        "tax_delta": tax_delta,
        "line_total_delta": line_total_delta,
        "vendor_name": vendor_name or None,
        "vendor_registered_name": vendor_registered_name or None,
        "vendor_status": vendor_status or None,
        "vendor_name_matches_registration": vendor_match,
        "department": department or None,
        "approver_department": approver_department or None,
        "department_alignment": (department == approver_department) if department and approver_department else None,
        "purchase_order": invoice.get("purchase_order"),
        "missing_fields": missing_fields,
        "anomaly_flags": anomaly_flags,
        "policy_decision_hint": policy.get("decision"),
        "policy_reasons_hint": [single_line(item) for item in policy.get("expected_reasoning", [])],
    }


def choose_specific_stage_action(stage: str, invoice: Dict[str, Any], proposed_action: str, reasoning: str) -> str:
    normalized = single_line(proposed_action).lower().replace(" ", "_")
    context = build_invoice_context(invoice)

    if stage == "final_decision":
        return "reject" if normalized == "reject" else "approve"

    if stage == "analyze" and normalized in ANALYZE_ACTIONS:
        return normalized
    if stage == "flag_issues" and normalized in FLAG_ACTIONS:
        return normalized

    evidence = " ".join(
        [
            single_line(reasoning).lower(),
            " ".join(context.get("anomaly_flags") or []).lower(),
            " ".join(context.get("missing_fields") or []).lower(),
        ]
    )

    if stage == "analyze":
        if not context["receipt_present"] or context["missing_fields"]:
            return "check_documentation"
        if (
            context.get("reported_total_delta") not in (None, 0.0)
            or context.get("tax_delta") not in (None, 0.0)
            or context.get("line_total_delta") not in (None, 0.0)
            or any(token in evidence for token in ("mismatch", "total", "tax"))
        ):
            return "reconcile_totals"
        if (
            context.get("vendor_status") not in (None, "", "verified")
            or context.get("vendor_name_matches_registration") is False
            or "vendor" in evidence
        ):
            return "inspect_vendor"
        if context.get("line_item_count", 0) > 1 or "line item" in evidence:
            return "review_line_items"
        return "verify_policy_fields"

    if not context["receipt_present"] or context["missing_fields"] or "receipt" in evidence:
        return "flag_document_gap"
    if (
        context.get("reported_total_delta") not in (None, 0.0)
        or context.get("tax_delta") not in (None, 0.0)
        or context.get("line_total_delta") not in (None, 0.0)
        or any(token in evidence for token in ("mismatch", "rounding", "total"))
    ):
        return "flag_total_mismatch"
    if (
        context.get("vendor_status") not in (None, "", "verified")
        or context.get("vendor_name_matches_registration") is False
        or context.get("department_alignment") is False
        or any(token in evidence for token in ("vendor", "department", "conflict"))
    ):
        return "flag_vendor_conflict"
    if context.get("anomaly_flags"):
        return "flag_policy_risk"
    return "confirm_clean_invoice"


def rule_based_action(observation: Dict[str, Any]) -> Action:
    stage = str(observation.get("stage", "analyze"))
    invoice = observation.get("invoice", {})
    previous_findings = [single_line(item) for item in observation.get("previous_findings", [])]
    policy_result = evaluate_invoice(invoice)
    context = build_invoice_context(invoice)

    if stage == "analyze":
        reasoning = single_line(
            "; ".join(
                [
                    f"1) amount {context['amount']} against policy limit {context['policy_limit']}",
                    f"2) receipt is {'present' if context['receipt_present'] else 'missing'} and category is {'allowed' if context['category_allowed'] else 'not allowed'}",
                    f"3) date validity is {'ok' if context['date_valid'] else 'invalid'} with {context['line_item_count']} line items",
                ]
            )
        )
        action = "inspect_invoice"
        confidence = 0.80
    elif stage == "flag_issues":
        issues = context["anomaly_flags"] or context["missing_fields"] or [single_line(item) for item in policy_result.get("expected_reasoning", [])]
        if issues and policy_result.get("decision") == "reject":
            reasoning = single_line(
                "; ".join(
                    [
                        "1) invoice shows concrete discrepancies",
                        f"2) key issues: {'; '.join(issues)}",
                        "3) these findings should feed the final decision",
                    ]
                )
            )
            action = "identify_issues"
            confidence = 0.86
        else:
            reasoning = single_line(
                "; ".join(
                    [
                        "1) no material anomalies were found",
                        "2) totals and documentation appear internally consistent",
                        "3) the invoice can move toward approval unless later evidence conflicts",
                    ]
                )
            )
            action = "no_issues"
            confidence = 0.83
    else:
        decision = "reject" if policy_result.get("decision") == "reject" or context["anomaly_flags"] else "approve"
        supporting = previous_findings or context["anomaly_flags"] or policy_result.get("expected_reasoning", [])
        reasoning = single_line(
            "; ".join(
                [
                    f"1) prior findings: {'; '.join(single_line(item) for item in supporting) or 'no blocking findings'}",
                    f"2) policy outcome points to {decision}",
                    f"3) final action is {decision}",
                ]
            )
        )
        action = decision
        confidence = 0.90 if decision == "reject" else 0.88

    return Action(stage=stage if stage in STAGES else "analyze", action=action, reasoning=reasoning, confidence=confidence)


def normalize_model_action(payload: Optional[Dict[str, Any]], observation: Dict[str, Any], fallback: Action) -> Action:
    if not isinstance(payload, dict):
        return fallback

    stage = str(payload.get("stage", observation.get("stage", fallback.stage))).strip().lower()
    if stage not in STAGES:
        stage = fallback.stage

    reasoning = single_line(payload.get("reasoning", payload.get("reason", ""))) or fallback.reasoning
    try:
        confidence = float(payload.get("confidence", fallback.confidence))
    except (TypeError, ValueError):
        confidence = fallback.confidence
    confidence = max(0.0, min(1.0, confidence))

    proposed_action = str(payload.get("action", "")).strip()
    invoice = observation.get("invoice", {})

    if stage == "final_decision":
        action = proposed_action.lower()
        if action not in {"approve", "reject"}:
            context = build_invoice_context(invoice)
            evidence = f"{reasoning.lower()} {' '.join(context.get('anomaly_flags') or []).lower()}"
            if any(token in evidence for token in ("reject", "mismatch", "missing", "unverified", "conflict", "invalid")):
                action = "reject"
            elif any(token in evidence for token in ("approve", "consistent", "valid", "clean")):
                action = "approve"
            else:
                return fallback
    else:
        action = choose_specific_stage_action(stage, invoice, proposed_action, reasoning)

    return Action(stage=stage, action=action, reasoning=reasoning, confidence=confidence)


def request_llm_action(client: OpenAI, observation: Dict[str, Any], seed: int) -> Action:
    fallback = rule_based_action(observation)
    allowed_actions = sorted(ANALYZE_ACTIONS if observation.get("stage") == "analyze" else FLAG_ACTIONS)
    if observation.get("stage") == "final_decision":
        allowed_actions = ["approve", "reject"]

    user_payload = {
        "seed": seed,
        "current_stage": observation.get("stage"),
        "previous_findings": observation.get("previous_findings", []),
        "allowed_actions": allowed_actions,
        "invoice": observation.get("invoice", {}),
        "derived_checks": build_invoice_context(observation.get("invoice", {})),
    }

    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, sort_keys=True)},
        ],
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }

    try:
        response = client.chat.completions.create(**request_kwargs)
    except Exception:
        try:
            request_kwargs.pop("response_format", None)
            response = client.chat.completions.create(**request_kwargs)
        except Exception:
            return fallback

    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        if message is not None:
            content = _text_from_content(message.content)

    return normalize_model_action(extract_json_object(content), observation, fallback)


def episode_seed(base_seed: int, difficulty_index: int, episode_index: int) -> int:
    return base_seed + (difficulty_index * 1000) + episode_index


def evaluate_difficulty(
    agent: BaseStageAgent,
    difficulty: str,
    episodes: int,
    metrics: RunMetrics,
    base_seed: int,
) -> float:
    rewards: List[float] = []

    for episode in range(1, episodes + 1):
        env = CompatibleOpenEnv(seed=base_seed)
        step_rewards: List[float] = []
        matched_keywords: List[str] = []
        referenced_fields: List[str] = []
        final_action = Action(stage="final_decision", action="reject", reasoning="fallback", confidence=0.0)
        final_info: Dict[str, Any] = {}

        try:
            current_seed = episode_seed(base_seed, TASKS.index(difficulty), episode)
            observation = env.reset(seed=current_seed, task=difficulty)
            source_invoice = dict(observation.get("invoice", {}))
            done = False

            while not done:
                action = agent.act(observation, current_seed)
                observation, reward, done, info = env.step(action)
                step_rewards.append(float(reward))
                matched_keywords.extend(str(item) for item in info.get("matched_keywords", []))
                referenced_fields.extend(str(item) for item in info.get("referenced_fields", []))
                if done:
                    final_action = action
                    final_info = info

            episode_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
            rewards.append(episode_reward)
            expected_decision = str(final_info.get("expected_decision") or evaluate_invoice(source_invoice)["decision"])
            final_decision = final_action.action.strip().lower()

            result = EpisodeResult(
                difficulty=difficulty,
                decision=final_decision,
                expected_decision=expected_decision,
                decision_correct=bool(final_info.get("decision_correct", final_decision == expected_decision)),
                confidence=float(final_action.confidence),
                reward=episode_reward,
                matched_keywords=sorted(set(matched_keywords)),
                referenced_fields=sorted(set(referenced_fields)),
            )
            metrics.record(result)

            marker = "Y" if result.decision_correct else "N"
            print(
                f"  [{difficulty}] ep {episode:>2d}: "
                f"decision={result.decision:<7s} [{marker}]  "
                f"reward={result.reward:.2f}  "
                f"conf={result.confidence:.2f}  "
                f"kw={len(result.matched_keywords)}  refs={len(result.referenced_fields)}"
            )
        except Exception:
            fallback_result = EpisodeResult(
                difficulty=difficulty,
                decision="reject",
                expected_decision="reject",
                decision_correct=False,
                confidence=0.0,
                reward=0.0,
            )
            metrics.record(fallback_result)
            rewards.append(0.0)
            print(f"  [{difficulty}] ep {episode:>2d}: decision=reject  [N]  reward=0.00  conf=0.00  kw=0  refs=0")
        finally:
            env.close()

    return sum(rewards) / len(rewards) if rewards else 0.0


def print_summary(metrics: RunMetrics, difficulty_scores: Dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    print("\n  Per-Difficulty Rewards:")
    for difficulty, score in difficulty_scores.items():
        bar = "#" * int(score * 20)
        print(f"    {difficulty:>8s}: {score:.4f}  {bar}")

    overall = sum(difficulty_scores.values()) / len(difficulty_scores) if difficulty_scores else 0.0
    print(f"    {'overall':>8s}: {overall:.4f}")

    print(f"\n  Aggregate Metrics ({metrics.total} episodes):")
    print(f"    Accuracy       : {metrics.accuracy:.2%}  ({metrics.correct}/{metrics.total})")
    print(f"    Approval Rate  : {metrics.approval_rate:.2%}")
    print(f"    Rejection Rate : {metrics.rejection_rate:.2%}")
    print(f"    Avg Confidence : {metrics.avg_confidence:.4f}")
    print(f"    Avg Reward     : {metrics.avg_reward:.4f}")
    print("=" * 60)


def run_agent(agent: BaseStageAgent, seed: int, episodes: int) -> None:
    metrics = RunMetrics()
    print("=" * 60)
    print("  Invoice Verification - Inference Run")
    print(f"  Agent: {agent.name}  |  Seed: {seed}  |  Episodes/diff: {episodes}")
    print("=" * 60)

    difficulty_scores: Dict[str, float] = {}
    for difficulty in TASKS:
        print(f"\n-- {difficulty.upper()} ({episodes} episodes) --")
        average_reward = evaluate_difficulty(agent, difficulty, episodes, metrics, seed)
        difficulty_scores[difficulty] = average_reward
        print(f"  -> {difficulty} average reward: {average_reward:.4f}")

    print_summary(metrics, difficulty_scores)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    global MODEL_NAME
    MODEL_NAME = args.model

    client = OpenAI(base_url=API_BASE_URL, api_key=LLM_API_KEY) if args.use_llm and LLM_API_KEY else None
    run_agent(RuleBasedStageAgent(), args.seed, args.episodes)

    if args.use_llm:
        print()
        run_agent(LLMStageAgent(client), args.seed, args.episodes)
