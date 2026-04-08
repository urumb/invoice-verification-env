from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openenv.core import Environment

from .environment import InvoiceEnvironment
from .models import Action, Difficulty

try:
    import numpy as np
except ImportError:
    np = None


_REGISTRY: Dict[str, type] = {}

_DEFAULT_METADATA: Dict[str, Any] = {
    "name": "invoice-verification",
    "description": "OpenEnv-compatible invoice verification environment.",
    "action_space": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approve", "reject"],
            },
            "reason": {
                "type": "string",
                "minLength": 1,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["decision", "reason", "confidence"],
    },
    "observation_space": {
        "type": "object",
        "properties": {
            "invoice": {
                "type": "object",
            }
        },
        "required": ["invoice"],
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
        return {"invoice": copy.deepcopy(result.invoice)}

    def step(
        self,
        action: Action | Dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not isinstance(action, Action):
            action = Action(**action)

        result = self._env.step(action)

        observation = {"invoice": copy.deepcopy(result.observation.invoice)}
        reward = float(result.reward)
        done = bool(result.done)
        info = copy.deepcopy(result.info) if result.info else {}
        info["reason"] = action.reason
        info["confidence"] = action.confidence
        info["keywords_matched"] = info.get("matched_keywords", [])

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
        state = self._env.state()
        return {
            "step_count": int(state.step_count),
            "current_invoice": copy.deepcopy(state.current_invoice),
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
