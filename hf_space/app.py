"""FastAPI and Gradio application wrapper for Hugging Face Spaces."""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from typing import Any, Dict, Optional

import gradio as gr
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for candidate in (PARENT_DIR, CURRENT_DIR):
    if os.path.isdir(os.path.join(candidate, "env")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from env.models import Action, Difficulty, State, TaskRecord
from env.openenv_adapter import OpenEnvAdapter
from env.policy import evaluate_invoice


APP_NAME = "invoice-verification"
APP_VERSION = "1.0.1"

_base_app = FastAPI(
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
    decision: str
    reason: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class EpisodeResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float = Field(..., ge=0.0, le=1.0)
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


def _model_schema(model_cls: type[BaseModel]) -> Dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def _reset_response(
    session_id: str,
    observation: Dict[str, Any],
    adapter: OpenEnvAdapter,
    difficulty: Optional[Difficulty],
    seed: Optional[int],
) -> EpisodeResponse:
    return EpisodeResponse(
        observation=observation,
        reward=0.0,
        done=False,
        info={
            "session_id": session_id,
            "difficulty": difficulty,
            "seed": seed,
            "state": adapter.state,
        },
        session_id=session_id,
    )


def _prepare_custom_episode(adapter: OpenEnvAdapter, invoice: Dict[str, Any]) -> None:
    policy_result = evaluate_invoice(invoice)
    adapter._env._current_task = TaskRecord(
        invoice=invoice,
        decision=policy_result["decision"],
        keywords=list(policy_result.get("expected_reasoning") or []),
    )
    adapter._env._step_count = 0
    adapter._env._episode_done = False


@_base_app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": APP_NAME,
        "status": "ok",
        "version": APP_VERSION,
    }


@_base_app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", name=APP_NAME, version=APP_VERSION)


@_base_app.get("/metadata")
def get_metadata() -> Dict[str, Any]:
    return OpenEnvAdapter().get_metadata()


@_base_app.get("/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    return SchemaResponse(
        action=_model_schema(Action),
        observation={
            "type": "object",
            "properties": {
                "invoice": {"type": "object"},
            },
            "required": ["invoice"],
        },
        state=_model_schema(State),
    )


@_base_app.post("/mcp", response_model=JsonRpcResponse)
def mcp_endpoint() -> JsonRpcResponse:
    return JsonRpcResponse(
        error=JsonRpcError(code=-32600, message="MCP is not enabled for this environment."),
    )


@_base_app.post("/reset", response_model=EpisodeResponse)
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


@_base_app.post("/step", response_model=EpisodeResponse)
def step_environment(
    request: StepPayload = Body(...),
    session_id: Optional[str] = Query(default=None),
) -> EpisodeResponse:
    effective_session_id = request.session_id or session_id
    adapter, sid = _get_or_create_adapter(effective_session_id)
    action = Action(
        decision=request.decision,
        reason=request.reason,
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


@_base_app.get("/state", response_model=SessionStateResponse)
def get_state(session_id: Optional[str] = Query(default=None)) -> SessionStateResponse:
    adapter, sid = _get_or_create_adapter(session_id)
    state = adapter.state
    return SessionStateResponse(
        step_count=state.get("step_count", 0),
        current_invoice=state.get("current_invoice", {}),
        session_id=sid,
    )


def gradio_interface(invoice_json_str: str) -> tuple[str, str, str]:
    try:
        invoice = json.loads(invoice_json_str)
        if not isinstance(invoice, dict):
            return ("Error", "0.00", "Invoice JSON must be an object.")

        policy_result = evaluate_invoice(invoice)
        action = Action(
            decision=policy_result["decision"],
            reason=" ".join(policy_result["reasons"]),
            confidence=policy_result["confidence"],
        )

        session_id = _new_session_id()
        adapter, _ = _get_or_create_adapter(session_id)
        _prepare_custom_episode(adapter, invoice)
        _, reward, done, info = adapter.step(action)

        message = info.get("message", "No explanation available.")
        return (
            action.decision,
            f"{reward:.2f}",
            f"{message}\nDone: {done}\nReason: {action.reason}",
        )
    except json.JSONDecodeError:
        return ("Error", "0.00", "Invalid JSON format.")
    except Exception as exc:
        return ("Error", "0.00", f"Error: {exc}")


with gr.Blocks(title="Invoice Verification Demo") as demo:
    gr.Markdown("# Invoice Verification Environment")
    gr.Markdown("Submit an invoice JSON manually and evaluate it.")
    gr.Markdown("API endpoints remain available at `/`, `/metadata`, `/reset`, `/step`, `/state`, `/health`, and `/schema`.")

    with gr.Row():
        with gr.Column():
            invoice_in = gr.Textbox(
                label="Invoice JSON",
                lines=10,
                value='{"amount": 100.0, "category": "travel", "date": "2023-01-01", "description": "Client taxi ride", "receipt": true}',
            )
            btn = gr.Button("Evaluate Invoice")

        with gr.Column():
            decision_out = gr.Textbox(label="Decision", interactive=False)
            reward_out = gr.Textbox(label="Reward", interactive=False)
            msg_out = gr.Textbox(label="Explanation", interactive=False)

    btn.click(
        fn=gradio_interface,
        inputs=[invoice_in],
        outputs=[decision_out, reward_out, msg_out],
    )


app = gr.mount_gradio_app(_base_app, demo, path="/ui")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "hf_space.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
