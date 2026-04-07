from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openenv.core import Environment

from .environment import InvoiceEnvironment
from .models import Action, Difficulty

# ── Global Environment Registry ──
_REGISTRY: Dict[str, type] = {}

def register_environment(name: str, env_cls: type) -> None:
    """Register an environment class globally."""
    _REGISTRY[name] = env_cls


class OpenEnvAdapter(Environment):
    """Adapter wrapping the core InvoiceEnvironment into openenv-core format."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._env = InvoiceEnvironment(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reset the environment state and return observation."""
        result = self._env.reset(difficulty=difficulty, seed=seed)
        return result.invoice

    def step(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step and return (observation, reward, done, info)."""
        result = self._env.step(action)
        return (
            result.observation.invoice,
            result.reward,
            result.done,
            result.info,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Fetch metadata dictionary from env/metadata.json."""
        metadata_file = Path(__file__).parent / "metadata.json"
        if not metadata_file.exists():
            return {}
        with metadata_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def state(self) -> Dict[str, Any]:
        """Return the internal environment state."""
        st = self._env.state()
        return {"step_count": st.step_count, "current_invoice": st.current_invoice}

    def close(self) -> None:
        """Cleanup logic if any."""
        pass

# Automatically register our environment on import
register_environment("invoice-verification", OpenEnvAdapter)
