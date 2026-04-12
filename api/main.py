from __future__ import annotations

import os
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from env.models import Action, Difficulty, Observation, State, schema_for
from env.openenv_adapter import OpenEnvAdapter


APP_NAME = "invoice-verification"
APP_VERSION = "1.0.1"

app = FastAPI(
    title="Invoice Verification Environment",
    description="OpenEnv-compatible invoice verification environment.",
    version=APP_VERSION,
)

_sessions: Dict[str, OpenEnvAdapter] = {}
_sessions_lock = threading.Lock()


class ResetPayload(BaseModel):
    session_id: Optional[str] = None
    difficulty: Optional[Difficulty] = None
    seed: Optional[int] = None


class StepPayload(BaseModel):
    session_id: Optional[str] = None
    stage: str
    action: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class EpisodeResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float = Field(..., gt=0.0, lt=1.0)
    done: bool
    info: Dict[str, Any]
    session_id: str


class SessionStateResponse(State):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    name: str
    version: str


class SchemaResponse(BaseModel):
    action: Dict[str, Any]
    observation: Dict[str, Any]
    state: Dict[str, Any]


class JsonRpcError(BaseModel):
    code: int
    message: str


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Any = None
    error: JsonRpcError


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _get_or_create_adapter(session_id: Optional[str]) -> tuple[OpenEnvAdapter, str]:
    sid = session_id or _new_session_id()
    with _sessions_lock:
        adapter = _sessions.get(sid)
        if adapter is None:
            adapter = OpenEnvAdapter()
            _sessions[sid] = adapter
    return adapter, sid


def _reset_response(
    session_id: str,
    observation: Dict[str, Any],
    adapter: OpenEnvAdapter,
    difficulty: Optional[Difficulty],
    seed: Optional[int],
) -> EpisodeResponse:
    return EpisodeResponse(
        observation=observation,
        reward=0.01,
        done=False,
        info={
            "session_id": session_id,
            "difficulty": difficulty,
            "seed": seed,
            "state": adapter.state,
        },
        session_id=session_id,
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": APP_NAME,
        "status": "ok",
        "version": APP_VERSION,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", name=APP_NAME, version=APP_VERSION)


@app.get("/metadata")
def get_metadata() -> Dict[str, Any]:
    return OpenEnvAdapter().get_metadata()


@app.get("/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    return SchemaResponse(
        action=schema_for(Action),
        observation=schema_for(Observation),
        state=schema_for(State),
    )


@app.post("/mcp", response_model=JsonRpcResponse)
def mcp_endpoint() -> JsonRpcResponse:
    return JsonRpcResponse(
        error=JsonRpcError(code=-32600, message="MCP is not enabled for this environment."),
    )


@app.post("/reset", response_model=EpisodeResponse)
def reset_environment(
    request: Optional[ResetPayload] = Body(default=None),
    session_id: Optional[str] = Query(default=None),
    difficulty: Optional[Difficulty] = Query(default=None),
    seed: Optional[int] = Query(default=None),
) -> EpisodeResponse:
    payload = request or ResetPayload()
    effective_session_id = payload.session_id or session_id
    effective_difficulty = payload.difficulty if payload.difficulty is not None else difficulty
    effective_seed = payload.seed if payload.seed is not None else seed

    adapter, sid = _get_or_create_adapter(effective_session_id)

    try:
        observation = adapter.reset(seed=effective_seed, difficulty=effective_difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc

    return _reset_response(sid, observation, adapter, effective_difficulty, effective_seed)


@app.post("/step", response_model=EpisodeResponse)
def step_environment(
    request: StepPayload = Body(...),
    session_id: Optional[str] = Query(default=None),
) -> EpisodeResponse:
    effective_session_id = request.session_id or session_id
    adapter, sid = _get_or_create_adapter(effective_session_id)
    action = Action(
        stage=request.stage,
        action=request.action,
        reasoning=request.reasoning,
        confidence=request.confidence,
    )

    try:
        observation, reward, done, info = adapter.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc

    response_info = dict(info)
    response_info["session_id"] = sid
    response_info["state"] = adapter.state

    return EpisodeResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=response_info,
        session_id=sid,
    )


@app.get("/state", response_model=SessionStateResponse)
def get_state(session_id: Optional[str] = Query(default=None)) -> SessionStateResponse:
    adapter, sid = _get_or_create_adapter(session_id)
    state = adapter.state
    return SessionStateResponse(
        step_count=state.get("step_count", 0),
        current_invoice=state.get("current_invoice", {}),
        stage=state.get("stage"),
        previous_findings=state.get("previous_findings", []),
        session_id=sid,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
