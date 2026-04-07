# Invoice Verification Environment

OpenEnv-compatible Python environment where an agent reviews an invoice and decides whether to approve or reject it based on company policy.

## Features

- Typed `Action`, `Observation`, and `State` models with Pydantic
- Strict `reset()`, `step()`, and `state()` environment flow
- FastAPI service with `/reset`, `/step`, and `/state`
- Three difficulty datasets with 10 tasks each
- `inference.py` runner that calls the API and uses OpenAI when configured
- Docker support on port `7860`

## Project Structure

```text
invoice-verification-env/
├── api/
│   └── main.py
├── data/
│   ├── easy.json
│   ├── hard.json
│   └── medium.json
├── env/
│   ├── environment.py
│   ├── grader.py
│   ├── models.py
│   └── tasks.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Local Setup

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

## API Endpoints

- `POST /reset` with optional body `{"difficulty": "easy" | "medium" | "hard"}`
- `POST /step` with an `Action` payload
- `GET /state`

## Run Inference

`inference.py` will start the API automatically if it is not already running.

```bash
python inference.py
```

To use the OpenAI API, set:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini
```

Without an API key, the script falls back to a deterministic heuristic agent so the project remains runnable.

## Docker

```bash
docker build -t invoice-verification-env .
docker run -p 7860:7860 invoice-verification-env
```
