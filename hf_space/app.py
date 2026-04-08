"""FastAPI and Gradio application wrapper for Hugging Face Spaces."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import gradio as gr
from fastapi import FastAPI, HTTPException

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.models import Action, TaskRecord
from env.openenv_adapter import OpenEnvAdapter
from env.policy import evaluate_invoice
from inference import RuleBasedAgent


_adapter = OpenEnvAdapter()
app = FastAPI(title="Invoice Verification HF Space", version="1.0.0")


def _prepare_custom_episode(invoice: Dict[str, Any]) -> None:
    policy_result = evaluate_invoice(invoice)
    keywords = policy_result.get("violations") or ["amount", "category", "receipt", "date"]

    _adapter._env._current_task = TaskRecord(
        invoice=invoice,
        decision=policy_result["decision"],
        keywords=keywords,
    )
    _adapter._env._step_count = 0
    _adapter._env._episode_done = False


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "HF Space running"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return _adapter.get_metadata()


@app.post("/reset")
def api_reset() -> Dict[str, Any]:
    try:
        return _adapter.reset()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Reset failed: {exc}") from exc


@app.post("/step")
def api_step(action: Action) -> Dict[str, Any]:
    try:
        observation, reward, done, info = _adapter.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc


def gradio_interface(invoice_json_str: str) -> tuple[str, str, str]:
    try:
        invoice = json.loads(invoice_json_str)
        if not isinstance(invoice, dict):
            return ("Error", "0.0", "Invoice JSON must be an object.")

        agent = RuleBasedAgent()
        action = agent.predict(invoice)

        _prepare_custom_episode(invoice)
        _, reward, done, info = _adapter.step(action)

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


app = gr.mount_gradio_app(app, demo, path="/gradio")
