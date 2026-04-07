from __future__ import annotations

import os
import random
from typing import Optional, cast

from .grader import build_feedback, grade
from .models import Action, Difficulty, Observation, State, StepResult, TaskRecord
from .tasks import get_random_task


class InvoiceEnvironment:
    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed_value = os.getenv("INVOICE_ENV_SEED")
            seed = int(seed_value) if seed_value is not None else None

        self._step_count = 0
        self._current_task: Optional[TaskRecord] = None
        self._episode_done = False
        self._rng = random.Random(seed)

    def reset(self, difficulty: Optional[Difficulty] = None) -> Observation:
        chosen_difficulty = difficulty or cast(
            Difficulty, self._rng.choice(["easy", "medium", "hard"])
        )
        self._current_task = get_random_task(chosen_difficulty, rng=self._rng)
        self._step_count = 0
        self._episode_done = False
        return Observation(invoice=dict(self._current_task.invoice))

    def step(self, action: Action) -> StepResult:
        if self._current_task is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset().")

        reward = grade(action, self._current_task)
        self._step_count += 1
        self._episode_done = True

        observation = Observation(invoice=dict(self._current_task.invoice))
        info = build_feedback(action, self._current_task, reward)

        return StepResult(
            observation=observation,
            reward=reward,
            done=True,
            info=info,
        )

    def state(self) -> State:
        current_invoice = dict(self._current_task.invoice) if self._current_task else {}
        return State(step_count=self._step_count, current_invoice=current_invoice)
