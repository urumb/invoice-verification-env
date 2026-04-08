from __future__ import annotations

import uuid
from typing import Any
from fastapi import Body, FastAPI, HTTPException

from env.environment import InvoiceEnvironment
from env.models import Action, Observation, ResetRequest, State, StepResult
# Delay OpenEnvAdapter import or import safely if it brings extra deps
from env.openenv_adapter import OpenEnvAdapter


app = FastAPI(
    title="Invoice Verification Environment",
    description="OpenEnv-compatible invoice verification environment.",
    version="1.0.0",
)

_environments: dict[str, InvoiceEnvironment] = {}

def _get_env(session_id: str | None) -> tuple[InvoiceEnvironment, str]:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in _environments:
        _environments[session_id] = InvoiceEnvironment()
    return _environments[session_id], session_id


class SessionObservation(Observation):
    session_id: str


class SessionStepResult(StepResult):
    session_id: str


class SessionState(State):
    session_id: str


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Invoice Verification Environment is running."}


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    try:
        adapter = OpenEnvAdapter()
        return adapter.get_metadata()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset", response_model=SessionObservation)
def reset_environment(
    session_id: str | None = None,
    request: ResetRequest | None = Body(default=None)
) -> SessionObservation:
    env, sid = _get_env(session_id)
    payload = request or ResetRequest()
    
    if payload.seed is not None:
        import random
        random.seed(payload.seed)
        try:
            import numpy as np
            np.random.seed(payload.seed)
        except ImportError:
            pass
        if hasattr(env, "_rng"):
            env._rng = random.Random(payload.seed)

    try:
        obs = env.reset(payload.difficulty, seed=payload.seed)
        return SessionObservation(invoice=obs.invoice, session_id=sid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=SessionStepResult)
def step_environment(
    action: Action,
    session_id: str | None = None
) -> SessionStepResult:
    env, sid = _get_env(session_id)
    try:
        result = env.step(action)
        return SessionStepResult(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info,
            session_id=sid
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=SessionState)
def get_state(session_id: str | None = None) -> SessionState:
    env, sid = _get_env(session_id)
    st = env.state()
    return SessionState(
        step_count=st.step_count,
        current_invoice=st.current_invoice,
        session_id=sid
    )
