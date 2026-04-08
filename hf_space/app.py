"""FastAPI and Gradio application wrapper for Hugging Face Spaces."""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict

import gradio as gr
from fastapi import FastAPI, HTTPException

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for candidate in (CURRENT_DIR, PARENT_DIR):
    if os.path.isdir(os.path.join(candidate, "env")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

from env.models import Action, TaskRecord
from env.openenv_adapter import OpenEnvAdapter
from env.policy import evaluate_invoice
from inference import RuleBasedAgent


_adapters: Dict[str, OpenEnvAdapter] = {}


def _get_adapter(session_id: str | None) -> tuple[OpenEnvAdapter, str]:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in _adapters:
        _adapters[session_id] = OpenEnvAdapter()
    return _adapters[session_id], session_id


app = FastAPI(title="Invoice Verification HF Space", version="1.0.0")


def _prepare_custom_episode(adapter: OpenEnvAdapter, invoice: Dict[str, Any]) -> None:
    policy_result = evaluate_invoice(invoice)
    adapter._env._current_task = TaskRecord(
        invoice=invoice,
        decision=policy_result["decision"],
        keywords=policy_result.get("violations") or [],
    )
    adapter._env._step_count = 0
    adapter._env._episode_done = False


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "HF Space running"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return OpenEnvAdapter().get_metadata()


@app.post("/reset")
def api_reset(session_id: str | None = None, seed: int | None = None) -> Dict[str, Any]:
    adapter, sid = _get_adapter(session_id)
    if seed is not None:
        adapter.seed(seed)

    try:
        result = adapter.reset(seed=seed)
        result["session_id"] = sid
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Reset failed: {exc}") from exc


@app.post("/step")
def api_step(action: Action, session_id: str | None = None) -> Dict[str, Any]:
    adapter, sid = _get_adapter(session_id)
    try:
        observation, reward, done, info = adapter.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "session_id": sid,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc


@app.get("/state")
def api_state(session_id: str | None = None) -> Dict[str, Any]:
    adapter, sid = _get_adapter(session_id)
    state = dict(adapter.state)
    state["session_id"] = sid
    return state


def gradio_interface(invoice_json_str: str) -> tuple[str, str, str]:
    try:
        invoice = json.loads(invoice_json_str)
        if not isinstance(invoice, dict):
            return ("Error", "0.0", "Invoice JSON must be an object.")

        agent = RuleBasedAgent()
        action = agent.predict(invoice)

        session_id = str(uuid.uuid4())
        adapter, _ = _get_adapter(session_id)
        _prepare_custom_episode(adapter, invoice)
        _, reward, done, info = adapter.step(action)

        message = info.get("message", "No explanation available.")
        return (
            action.decision,
            f"{reward:.2f}",
            f"{message}\nDone: {done}\nReason: {action.reason}",
        )
    except json.JSONDecodeError:
        return ("Error", "0.0", "Invalid JSON format.")
    except Exception as exc:
        return ("Error", "0.0", f"Error: {exc}")


with gr.Blocks(title="Invoice Verification Demo") as demo:
    gr.Markdown("# Invoice Verification Environment")
    gr.Markdown("Submit an invoice JSON manually and evaluate it.")

    with gr.Row():
        with gr.Column():
            invoice_in = gr.Textbox(
                label="Invoice JSON",
                lines=10,
                value='{\n  "amount": 100.0,\n  "category": "travel",\n  "date": "2023-01-01",\n  "description": "Flight",\n  "receipt": true\n}',
            )
            btn = gr.Button("Submit Action")

        with gr.Column():
            decision_out = gr.Textbox(label="Decision", interactive=False)
            reward_out = gr.Textbox(label="Reward", interactive=False)
            msg_out = gr.Textbox(label="Explanation", interactive=False)

    btn.click(
        fn=gradio_interface,
        inputs=[invoice_in],
        outputs=[decision_out, reward_out, msg_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
