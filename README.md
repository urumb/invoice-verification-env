# Invoice Verification RL Environment

## Overview
The Invoice Verification Environment provides an RL-style setup for validating employee expense reports and invoices based on established company policy rules. It functions deterministically but allows building an AI or LLM-based agent policy that must decide whether to approve or reject invoices with confidence.

## Features
- **Deterministic evaluation**: Fully supports setting random seeds to guarantee repeatable tests across difficulties.
- **OpenEnv compatible**: Adapter allows drop-in support for any standard `OpenEnv` usage.
- **Reward shaping**: Agents are rewarded based not only on correct decisions but also the quality and confidence of their explanations/reasoning.
- **FastAPI endpoints**: Ready to be queried over HTTP.
- **Hugging Face ready**: A Gradio wrapper is available to deploy cleanly to a Spaces project.

## Run locally
Launch the backend server locally:
```bash
uvicorn api.main:app --reload
```

## Run evaluation
After starting the environment (or letting it automatically spawn the server), run the inference script to evaluate agents:
```bash
python inference.py --seed 42
```
To run the OpenAI agent (ensure you have `OPENAI_API_KEY` set):
```bash
python inference.py --use-llm
```

## Design Notes
- A baseline **Rule-based agent** is used to ensure out-of-the-box evaluation without API keys.
- **Reward depends on correctness + explanation quality**: Agents must mention matched reasoning traits.
- Environment supports **multiple priority/difficulty levels** (`easy`, `medium`, `hard`) which varies the trickiness of the invoices.

## Result
Achieves consistent deterministic performance across runs. Ready for evaluation or local modification.
