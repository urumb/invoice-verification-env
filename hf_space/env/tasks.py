from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .models import Difficulty, TaskRecord


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DATA_CACHE: Dict[str, List[TaskRecord]] = {}


def load_tasks(difficulty: Difficulty) -> List[TaskRecord]:
    if difficulty not in _DATA_CACHE:
        dataset_path = DATA_DIR / f"{difficulty}.json"
        if not dataset_path.exists():
            raise ValueError(f"Dataset for difficulty '{difficulty}' was not found.")

        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            raw_records = json.load(dataset_file)

        _DATA_CACHE[difficulty] = [TaskRecord(**record) for record in raw_records]

    return _DATA_CACHE[difficulty]


def get_random_task(
    difficulty: Difficulty, rng: Optional[random.Random] = None
) -> TaskRecord:
    tasks = load_tasks(difficulty)
    if not tasks:
        raise ValueError(f"Dataset for difficulty '{difficulty}' is empty.")
    chooser = rng or random
    return chooser.choice(tasks)
