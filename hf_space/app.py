"""FastAPI and Gradio Application wrapper for Hugging Face Spaces."""
from __future__ import annotations

import gradio as gr
from fastapi import FastAPI, HTTPException

import sys
import os

# Ensure the root codebase is discoverable so we can import 'env'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.models import Action
from env.openenv_adapter import OpenEnvAdapter


# Initialize internal wrapped state
_adapter = OpenEnvAdapter()
app = FastAPI(title="Invoice Verification HF Space", version="1.0.0")


@app.get("/")
def read_root():
    """Root heartbeat."""
    return {"status": "HF Space running"}


@app.post("/reset")
def api_reset():
    """Reset the environment."""
    obs = _adapter.reset()
    return {"observation": obs}


@app.post("/step")
def api_step(action: Action):
    """Step the environment."""
    try:
        obs, reward, done, info = _adapter.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def gradio_interface(decision: str, reason: str, confidence: float):
    """Minimal function mapping the UI components to our adapter."""
    try:
        action = Action(decision=decision, reason=reason, confidence=confidence)
        obs, reward, done, info = _adapter.step(action)
        _adapter.reset()  # Reset right after the demonstration payload
        
        return (
            str(obs),
            f"{reward:.2f}",
            info.get("message", "No strict details provided.")
        )
    except Exception as exc:
        return ("{}", "0.0", f"Error: {str(exc)}")


# Setup minimal Gradio interface
with gr.Blocks(title="Invoice Verification Demo") as demo:
    gr.Markdown("# Invoice Verification Environment")
    gr.Markdown("Submit an evaluation action manually against the agent environment logic.")
    
    with gr.Row():
        with gr.Column():
            dec_in = gr.Dropdown(choices=["approve", "reject"], label="Decision", value="approve")
            reason_in = gr.Textbox(label="Reason", placeholder="Must reference invoice fields like 'amount', 'category'...")
            conf_in = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.9, label="Confidence")
            btn = gr.Button("Submit Action")
            
        with gr.Column():
            obs_out = gr.Textbox(label="Current Environment State / Observation", interactive=False)
            rew_out = gr.Textbox(label="Received Reward", interactive=False)
            msg_out = gr.Textbox(label="Grader Explanation", interactive=False)
            
    # Whenever the app loads, let UI start fresh
    demo.load(lambda: str(_adapter.reset()), inputs=None, outputs=obs_out)
    btn.click(fn=gradio_interface, inputs=[dec_in, reason_in, conf_in], outputs=[obs_out, rew_out, msg_out])

# Mount Gradio safely within the fastAPI application runtime
app = gr.mount_gradio_app(app, demo, path="/gradio")
