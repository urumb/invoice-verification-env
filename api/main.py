from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from env.environment import InvoiceEnvironment
from env.models import Action, Observation, ResetRequest, State, StepResult


app = FastAPI(
    title="Invoice Verification Environment",
    description="OpenEnv-compatible invoice verification environment.",
    version="1.0.0",
)
environment = InvoiceEnvironment()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Invoice Verification Environment is running."}


@app.post("/reset", response_model=Observation)
def reset_environment(request: ResetRequest | None = Body(default=None)) -> Observation:
    payload = request or ResetRequest()
    try:
        return environment.reset(payload.difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step_environment(action: Action) -> StepResult:
    try:
        return environment.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=State)
def get_state() -> State:
    return environment.state()
