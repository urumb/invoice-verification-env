# Invoice Verification Environment
A deterministic, OpenEnv-compatible Mini RL Environment designed to benchmark agent reasoning over expense reports.

## 🚀 Overview
This isn't a standard classification pipeline—it's a full-fledged Mini RL Environment. Rather than just making binary predictions, agents play within a sandbox where they must evaluate invoices against company policies, provide quality reasoning, and determine confidence. Built with deterministic evaluation and OpenEnv compatibility from day one, it features built-in reward shaping that evaluates the quality of agent reasoning alongside base correctness.

## 🧠 Why This Project Stands Out
- **RL-Style Agent Evaluation**: Moves beyond simple text classification into a fully interactive state-action-reward loop.
- **Deterministic & Reproducible**: Full seed control ensures every run yields the exact same state transitions and results.
- **Reasoning-Aware Rewards**: Agents are explicitly rewarded for matching key reasoning traits, preventing "right answer, wrong reason" scenarios.
- **Deploy Anywhere**: Built totally API-first, making it instantly deployable locally, on OpenEnv, or Hugging Face.

## ⚙️ Architecture
- **InvoiceEnvironment (Core)**: The underlying state machine managing episodes, invoice selection, grading, and dynamic feedback.
- **OpenEnvAdapter**: A standard wrapper that adheres to the OpenEnv specification, ensuring easy plug-and-play with external evaluation frameworks.
- **FastAPI Backend**: Exposes the environment natively over HTTP to support remote agent inference.
- **Inference Pipeline**: A robust `inference.py` test suite with rule-based fallbacks, deterministic tracking, and CLI ease of use.
- **HF Space**: A custom Gradio interface wrapping the environment, exposing both web UI and API.

## 📦 Setup Instructions

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## 🧪 Run Evaluation

```bash
python inference.py --seed 42
python inference.py --use-llm
```

## 🔌 API Endpoints
The FastAPI server natively exposes the following routes for environment interaction:
- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /metadata`

## 🤖 OpenEnv Compatibility
The environment seamlessly plugs into major RL benchmarking suites via `OpenEnvAdapter`. The adapter wraps the underlying environment to support standard `reset` and `step` loop interfaces, returning structured observation, reward, done flag, and info payloads. Complete environment schema configuration is defined statically via `metadata.json`.

## 🌐 Hugging Face Deployment
Fully configured for Hugging Face Spaces. The `hf_space/app.py` script automatically mounts the core FastAPI application while also serving a clean Gradio web interface. All core environment endpoints remain interactively exposed out of the box.

## 📊 Results
The environment natively logs run metrics, yielding high-visibility evaluations. By setting deterministic seeds, you achieve identical run states allowing stable agent A/B testing. Inference explicitly tracks raw accuracy alongside confidence and reward optimization metrics, delivering highly reproducible benchmarks.

## 🏆 Why This Wins
This environment succeeds because it is **reproducible**, deeply **scalable**, and built **API-first**. It shifts the focus from brittle predictive models to genuine agent evaluation ready for modern foundational model benchmarking.
