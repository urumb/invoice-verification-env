from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Decision = Literal["approve", "reject"]
Difficulty = Literal["easy", "medium", "hard"]


class Action(BaseModel):
    decision: Decision
    reason: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Observation(BaseModel):
    invoice: Dict[str, Any]


class State(BaseModel):
    step_count: int = Field(default=0, ge=0)
    current_invoice: Dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    invoice: Dict[str, Any]
    decision: Decision
    keywords: List[str]


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    difficulty: Optional[Difficulty] = None
