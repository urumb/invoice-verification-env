from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openenv_   core import Environment

from .environment import InvoiceEnvironment
from .models import Action, Difficulty, dump_model

try:
    import numpy as np
except ImportError:
    np = None


_REGISTRY: Dict[str, type] = {}

_DEFAULT_METADATA: Dict[str, Any] = {
    "name": "invoice-verification",
    "description": "OpenEnv-compatible invoice verification environment.",
    "reward_description": (
        "Rewards are normalized to [0, 1]. Analysis and issue-flagging stages reward "
        "field coverage and anomaly detection. The final decision stage rewards the "
        "correct approve/reject action plus evidence-grounded reasoning."
    ),
    "action_space": {
        "type": "object",
        "properties": {
            "stage": {
                "type": "string",
                "enum": ["analyze", "flag_issues", "final_decision"],
            },
            "action": {
                "type": "string",
                "minLength": 1,
            },
            "reasoning": {
                "type": "string",
                "minLength": 1,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["stage", "action", "reasoning", "confidence"],
    },
    "observation_space": {
        "type": "object",
        "properties": {
            "stage": {
                "type": "string",
                "enum": ["analyze", "flag_issues", "final_decision"],
            },
            "invoice": {
                "type": "object",
            },
            "previous_findings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["stage", "invoice", "previous_findings"],
    },
    "reward_range": [0, 1],
}


def register_environment(name: str, env_cls: type) -> None:
    _REGISTRY[name] = env_cls


class OpenEnvAdapter(Environment):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._seed: Optional[int] = None
        self._env = InvoiceEnvironment(seed=seed)
        if seed is not None:
            self.seed(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        effective_seed = self._seed if seed is None else seed
        if effective_seed is not None:
            self.seed(effective_seed)

        result = self._env.reset(difficulty=difficulty, seed=effective_seed)
        return {
            "stage": result.stage,
            "invoice": copy.deepcopy(result.invoice),
            "previous_findings": copy.deepcopy(result.previous_findings),
        }

    def step(
        self,
        action: Action | Dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not isinstance(action, Action):
            action = Action(**action)

        result = self._env.step(action)

        observation = {
            "stage": result.observation.stage,
            "invoice": copy.deepcopy(result.observation.invoice),
            "previous_findings": copy.deepcopy(result.observation.previous_findings),
        }
        reward = float(result.reward.value)
        done = bool(result.done)
        info = copy.deepcopy(result.info) if result.info else {}
        info["stage"] = action.stage
        info["reasoning"] = action.reasoning
        info["confidence"] = action.confidence
        info["keywords_matched"] = info.get("matched_keywords", [])
        info["reward"] = dump_model(result.reward)

        return observation, reward, done, info

    def get_metadata(self) -> Dict[str, Any]:
        metadata_file = Path(__file__).parent / "metadata.json"
        if not metadata_file.exists():
            return copy.deepcopy(_DEFAULT_METADATA)

        try:
            with metadata_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return copy.deepcopy(_DEFAULT_METADATA)

        metadata = copy.deepcopy(_DEFAULT_METADATA)
        metadata.update({k: v for k, v in data.items() if v is not None})
        return metadata

    @property
    def state(self) -> Dict[str, Any]:
        return self.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        state = self._env.state()
        return {
            "step_count": int(state.step_count),
            "current_invoice": copy.deepcopy(state.current_invoice),
            "stage": state.stage,
            "previous_findings": copy.deepcopy(state.previous_findings),
        }

    def seed(self, seed: int) -> None:
        self._seed = int(seed)
        random.seed(self._seed)
        if np is not None:
            np.random.seed(self._seed)
        if hasattr(self._env, "_rng"):
            self._env._rng = random.Random(self._seed)

    def close(self) -> None:
        return None


register_environment("invoice-verification", OpenEnvAdapter)
