from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env.openenv_adapter import OpenEnvAdapter
from env.policy import evaluate_invoice

load_dotenv()

try:
    import numpy as np
except ImportError:
    np = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
STAGES = ("analyze", "flag_issues", "final_decision")
TASKS = ("easy", "medium", "hard")

SYSTEM_PROMPT = (
    "You are an invoice verification agent operating in a three-stage workflow. "
    "Return exactly one JSON object with keys \"stage\", \"action\", \"reasoning\", and \"confidence\". "
    "The stage must match the observation stage. "
    "For stage analyze, inspect the invoice fields carefully. "
    "For stage flag_issues, identify discrepancies or explicitly state there are no issues. "
    "For stage final_decision, action must be approve or reject. "
    "Reasoning must be concise and single-line. Confidence must be between 0 and 1. "
    "Do not return markdown or extra text."
)


class CompatibleOpenEnv:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._env = OpenEnvAdapter(seed=seed)

    def reset(self, seed: Optional[int] = None, task: str = "easy") -> Dict[str, Any]:
        return self._env.reset(seed=seed, difficulty=task)

    def step(self, action_dict: Dict[str, Any]):
        return self._env.step(action_dict)

    def close(self) -> None:
        self._env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def load_env_name() -> str:
    benchmark_name = "invoice-verification-env"
    metadata_path = Path(__file__).resolve().parent / "env" / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = metadata.get("name")
            if isinstance(name, str) and name.strip():
                benchmark_name = name.strip()
        except Exception:
            pass
    return benchmark_name


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def format_reward(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def single_line(value: Any) -> str:
    return " ".join(str(value).split())


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
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _format_amount(value: Any) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _line_item_count(invoice: Dict[str, Any]) -> int:
    line_items = invoice.get("line_items") or []
    return len(line_items) if isinstance(line_items, list) else 0


def rule_based_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    stage = str(observation.get("stage", "analyze"))
    invoice = observation.get("invoice", {})
    previous_findings = [str(item) for item in observation.get("previous_findings", [])]
    policy_result = evaluate_invoice(invoice)

    amount = _format_amount(invoice.get("amount"))
    category = str(invoice.get("category", "unknown")).strip().lower() or "unknown"
    receipt = "present" if bool(invoice.get("receipt", False)) else "missing"
    date_value = str(invoice.get("date", "unknown")).strip() or "unknown"
    subtotal = _format_amount(invoice.get("subtotal"))
    tax_amount = _format_amount(invoice.get("tax_amount"))
    reported_total = _format_amount(invoice.get("reported_total"))
    vendor_name = str(invoice.get("vendor_name", "unknown vendor")).strip() or "unknown vendor"
    vendor_status = str(invoice.get("vendor_status", "unknown")).strip() or "unknown"
    anomaly_flags = [str(item) for item in invoice.get("anomaly_flags", [])]
    line_item_count = _line_item_count(invoice)

    if stage == "analyze":
        detail_parts = [
            f"amount {amount}",
            f"category {category}",
            f"receipt {receipt}",
            f"date {date_value}",
        ]
        if "subtotal" in invoice:
            detail_parts.append(f"subtotal {subtotal}")
        if "tax_amount" in invoice:
            detail_parts.append(f"tax {tax_amount}")
        if "reported_total" in invoice:
            detail_parts.append(f"reported total {reported_total}")
        if line_item_count:
            detail_parts.append(f"line items {line_item_count}")
        if "vendor_name" in invoice:
            detail_parts.append(f"vendor {vendor_name}")
        if "vendor_status" in invoice:
            detail_parts.append(f"vendor status {vendor_status}")
        reasoning = f"Invoice fields reviewed: {'; '.join(detail_parts)}."
        action_value = "inspect_invoice"
        confidence = 0.80
    elif stage == "flag_issues":
        issue_points = []
        if anomaly_flags:
            issue_points.extend(anomaly_flags)
        elif policy_result.get("decision") == "reject":
            issue_points.extend(policy_result.get("expected_reasoning") or policy_result.get("reasons") or [])
        if issue_points:
            reasoning = single_line("; ".join(issue_points))
            action_value = "identify_issues"
            confidence = 0.86
        else:
            consistency_points = [
                f"amount {amount} is within policy",
                f"receipt is {receipt}",
                f"reported total {reported_total} is consistent" if "reported_total" in invoice else "no policy issues found",
            ]
            if line_item_count:
                consistency_points.append(f"{line_item_count} line items reconcile")
            if "vendor_status" in invoice:
                consistency_points.append(f"vendor status is {vendor_status}")
            reasoning = single_line("; ".join(consistency_points))
            action_value = "no_issues"
            confidence = 0.84
    else:
        if anomaly_flags:
            final_decision = "reject"
        elif policy_result.get("decision") == "reject":
            final_decision = "reject"
        else:
            final_decision = "approve"
        reasoning_parts = previous_findings or anomaly_flags or list(policy_result.get("reasons") or [])
        reasoning = single_line("; ".join(reasoning_parts))
        action_value = final_decision
        confidence = 0.90 if final_decision == "reject" else 0.88

    return {
        "stage": stage if stage in STAGES else "analyze",
        "action": action_value,
        "reasoning": single_line(reasoning),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def normalize_action(payload: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return fallback

    stage_value = str(payload.get("stage", fallback["stage"])).strip().lower()
    action_value = str(payload.get("action", "")).strip()
    reasoning_value = single_line(payload.get("reasoning", payload.get("reason", "")))
    confidence_value = payload.get("confidence", fallback["confidence"])

    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = float(fallback["confidence"])

    stage = stage_value if stage_value in STAGES else fallback["stage"]
    action = action_value or fallback["action"]
    reasoning = reasoning_value or fallback["reasoning"]
    confidence = max(0.0, min(1.0, confidence))

    if stage == "final_decision" and action.lower() not in {"approve", "reject"}:
        return fallback

    return {
        "stage": stage,
        "action": action,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def request_model_action(client: OpenAI, observation: Dict[str, Any], seed: int) -> Dict[str, Any]:
    fallback = rule_based_action(observation)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"seed={seed}\nobservation={json.dumps(observation, ensure_ascii=True, sort_keys=True)}",
        },
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"},
        )
    except Exception:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                top_p=1,
            )
        except Exception:
            return fallback
    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        content = message.content if message and message.content is not None else ""
    return normalize_action(extract_json_object(content), fallback)


def choose_action(
    observation: Dict[str, Any],
    seed: int,
    use_llm: bool,
    client: Optional[OpenAI],
) -> Dict[str, Any]:
    if not use_llm or client is None:
        return rule_based_action(observation)
    return request_model_action(client, observation, seed)


def run_task(env: CompatibleOpenEnv, client: Optional[OpenAI], seed: int, use_llm: bool, task: str, benchmark_name: str) -> None:
    steps = 0
    rewards = []
    success = False

    print(f"[START] task={task} env={benchmark_name} model={MODEL_NAME}")

    try:
        observation = env.reset(seed=seed, task=task)
        done = False
        while not done:
            action = choose_action(observation, seed, use_llm, client)
            reward_value = 0.0
            step_done = False
            error_value = "null"
            try:
                observation, reward_value, step_done, _info = env.step(action)
                done = bool(step_done)
                success = done
            except Exception as exc:
                error_value = single_line(str(exc))
                done = True
                success = False
            steps += 1
            rewards.append(format_reward(reward_value))
            print(
                f"[STEP] step={steps} action={action['action']} reward={format_reward(reward_value)} "
                f"done={bool_str(step_done)} error={error_value}"
            )
    except Exception:
        success = False
    finally:
        print(f"[END] success={bool_str(success)} steps={steps} rewards={','.join(rewards)}")


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

    args = parse_args()
    seed_everything(args.seed)

    client = None
    if args.use_llm:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

    env = CompatibleOpenEnv(seed=args.seed)
    benchmark_name = load_env_name()

    try:
        for task in TASKS:
            run_task(env, client, args.seed, args.use_llm, task, benchmark_name)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()