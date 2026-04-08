from __future__ import annotations

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

# Global registry
_REGISTRY: Dict[str, type] = {}

def register_environment(name: str, env_cls: type) -> None:
    _REGISTRY[name] = env_cls


class OpenEnvAdapter(Environment):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._seed = seed
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
        if seed is not None:
            self.seed(seed)
        elif self._seed is not None:
            seed = self._seed

        result = self._env.reset(difficulty=difficulty, seed=seed)
        return {"invoice": result.invoice}

    def step(
        self,
        action: Action | Dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:

        if not isinstance(action, Action):
            action = Action(**action)

        result = self._env.step(action)

        observation = {"invoice": result.observation.invoice}
        reward = result.reward
        done = result.done
        
        info = dict(result.info) if result.info else {}
        info["reason"] = getattr(action, "reason", "No reason provided")
        info["confidence"] = getattr(action, "confidence", 0.0)
        info["keywords_matched"] = info.get("matched_keywords", [])

        return observation, reward, done, info

    def get_metadata(self) -> Dict[str, Any]:
        metadata_file = Path(__file__).parent / "metadata.json"
        default_metadata: Dict[str, Any] = {
            "name": "invoice-verification",
            "description": "RL environment for verifying invoices based on policy rules",
            "action_space": {
                "decision": ["approve", "reject"],
                "confidence": [0, 1]
            },
            "observation_space": {
                "fields": ["amount", "category", "date", "description", "receipt"]
            },
            "reward_range": [0, 1]
        }
        
        if not metadata_file.exists():
            return default_metadata
            
        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "name": data.get("name", default_metadata["name"]),
                "description": data.get("description", default_metadata["description"]),
                "action_space": data.get("action_space", default_metadata["action_space"]),
                "observation_space": data.get("observation_space", default_metadata["observation_space"]),
                "reward_range": data.get("reward_range", default_metadata["reward_range"])
            }
        except (json.JSONDecodeError, OSError):
            return default_metadata

    @property
    def state(self) -> Dict[str, Any]:
        st = self._env.state()
        import json
        return json.loads(json.dumps({
            "step_count": st.step_count,
            "current_invoice": st.current_invoice,
        }, default=str))

    def seed(self, seed: int) -> None:
        self._seed = seed
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)

    def close(self) -> None:
        pass


register_environment("invoice-verification", OpenEnvAdapter)