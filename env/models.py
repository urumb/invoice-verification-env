from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Decision = Literal["approve", "reject"]
Difficulty = Literal["easy", "medium", "hard"]
Stage = Literal["analyze", "flag_issues", "final_decision"]


class Action(BaseModel):
    stage: Stage
    action: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Observation(BaseModel):
    stage: Stage
    invoice: Dict[str, Any]
    previous_findings: List[str] = Field(default_factory=list)


class State(BaseModel):
    step_count: int = Field(default=0, ge=0)
    current_invoice: Dict[str, Any] = Field(default_factory=dict)
    stage: Optional[Stage] = None
    previous_findings: List[str] = Field(default_factory=list)


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
    seed: Optional[int] = None
