---
title: Invoice Verification Env
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📄 Invoice Verification RL Environment (OpenEnv)

A deterministic, multi-step, multi-task reinforcement learning environment for evaluating agent reasoning in invoice verification.
  
---

## 🚀 Overview

This project models invoice verification as a **multi-step decision-making process**, not just classification.

Agents must:

1. Analyze invoice data
2. Identify inconsistencies
3. Make a final decision (approve/reject)

Each action is evaluated using a **structured reward function** that considers correctness, reasoning, and confidence.

---

## 🧠 Key Highlights

* 🔁 **Multi-step reasoning environment**
* 📊 **Multi-task benchmark** (easy / medium / hard)
* 🎯 **Deterministic evaluation (seed-based)**
* 🤖 Supports **Rule-based + LLM agents**
* 🧪 **Reproducible benchmarking setup**
* ⚙️ Fully **OpenEnv compliant**
* 🐳 Dockerized for deployment

---

## 🧩 Task Design

### 🟢 Easy

* Clear valid/invalid invoices
* Obvious errors (missing fields, incorrect totals)

### 🟡 Medium

* Subtle inconsistencies
* Tax mismatches, rounding issues

### 🔴 Hard

* Complex reasoning required
* Multiple conflicting fields
* Vendor anomalies

---

## 🔁 Multi-Step Workflow

Each episode follows:

1. `inspect_invoice` → analyze fields
2. `identify_issues` → detect problems
3. `final_decision` → approve/reject

---

## 🎯 Reward Function

Reward ∈ [0, 1], based on:

* ✔ Decision correctness
* 🧠 Reasoning quality
* 📌 Issue identification
* ⚠ Penalties for incorrect stage order

---

## 🤖 Agents

### Rule-Based Agent

* Deterministic logic
* Serves as baseline

### LLM Agent

* Uses OpenAI-compatible API
* Generates reasoning-based decisions
* Supports structured JSON outputs

---

## 📊 Example Output

```
[START] task=easy env=invoice-verification-env model=gpt-4.1-mini
[STEP] step=1 action=inspect_invoice reward=0.35 done=false error=null
[STEP] step=2 action=identify_issues reward=0.50 done=false error=null
[STEP] step=3 action=reject reward=0.80 done=true error=null
[END] success=true steps=3 rewards=0.35,0.50,0.80
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload
```

Run evaluation:

```bash
python inference.py --seed 42
python inference.py --seed 42 --use-llm
```

---

## 🐳 Docker

Build:

```bash
docker build -t invoice-env .
```

Run:

```bash
docker run -p 8000:8000 --env-file .env invoice-env
```

---

## 🔌 API Endpoints

* `POST /reset`
* `POST /step`
* `GET /state`
* `GET /metadata`

---

## 📦 OpenEnv Compliance

* ✔ reset(), step(), state()
* ✔ Typed schemas
* ✔ Deterministic rewards
* ✔ openenv.yaml included

Validated using:

```bash
openenv validate
```

---

## 🏆 Why This Stands Out

Unlike standard classification tasks, this project:

* Evaluates **reasoning, not just accuracy**
* Provides **multi-step interaction**
* Enables **agent benchmarking**
* Ensures **reproducibility**

---

## 🏁 Summary

A complete OpenEnv-compatible RL environment for benchmarking intelligent agents on structured decision-making tasks like invoice verification.
