from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from env.openenv_adapter import OpenEnvAdapter
from env.policy import evaluate_invoice

try:
    import numpy as np
except ImportError:
    np = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = (
    "You are an invoice verification agent. "
    "Return exactly one JSON object with keys "
    '"action", "reasoning", and "confidence". '
    'The "action" must be either "approve" or "reject". '
    'The "reasoning" must be a concise single-line explanation. '
    'The "confidence" must be a float between 0 and 1. '
    "Do not return markdown or extra text."
)


class CompatibleOpenEnv:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._env = OpenEnvAdapter(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self._env.reset(seed=seed)

    def step(self, action_dict: Dict[str, Any]):
        translated_action = {
            "decision": action_dict["action"],
            "reason": action_dict["reasoning"],
            "confidence": action_dict["confidence"],
        }
        return self._env.step(translated_action)

    def close(self) -> None:
        self._env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def load_names() -> tuple[str, str]:
    benchmark_name = "invoice-verification-env"
    task_name = "invoice"
    metadata_path = Path(__file__).resolve().parent / "env" / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = metadata.get("name")
            if isinstance(name, str) and name.strip():
                benchmark_name = name.strip()
        except Exception:
            pass
    normalized = benchmark_name.replace("_", "-").split("-")
    if normalized and normalized[0]:
        task_name = normalized[0]
    return task_name, benchmark_name


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


def fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    invoice = observation.get("invoice", {})
    result = evaluate_invoice(invoice)
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "action": result.get("decision", "reject"),
        "reasoning": single_line(" ".join(result.get("reasons", ["Fallback decision."]))),
        "confidence": confidence,
    }


def normalize_action(payload: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return fallback
    action_value = payload.get("action", payload.get("decision", ""))
    reasoning_value = payload.get("reasoning", payload.get("reason", ""))
    confidence_value = payload.get("confidence", fallback["confidence"])
    action = str(action_value).strip().lower()
    reasoning = single_line(reasoning_value)
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = fallback["confidence"]
    confidence = max(0.0, min(1.0, confidence))
    if action not in {"approve", "reject"}:
        action = fallback["action"]
    if not reasoning:
        reasoning = fallback["reasoning"]
    return {
        "action": action,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def request_model_action(client: OpenAI, observation: Dict[str, Any], seed: int) -> Dict[str, Any]:
    fallback = fallback_action(observation)
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


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

    args = parse_args()
    seed_everything(args.seed)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )
    env = CompatibleOpenEnv(seed=args.seed)
    task_name, benchmark_name = load_names()

    steps = 0
    rewards = []
    success = False

    print(f"[START] task={task_name} env={benchmark_name} model={MODEL_NAME}")

    try:
        observation = env.reset(seed=args.seed)
        done = False
        while not done:
            action = request_model_action(client, observation, args.seed)
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
        try:
            env.close()
        except Exception:
            pass
        print(f"[END] success={bool_str(success)} steps={steps} rewards={','.join(rewards)}")


if __name__ == "__main__":
    main()