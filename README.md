---
title: failureiq-env
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# FailureIQ OpenEnv

FailureIQ is an AI evaluation environment that tests whether agents can identify the true root cause hidden inside noisy, misleading production logs — a task engineers face daily in real systems.

FailureIQ simulates a real-world debugging task: diagnosing the true root cause from noisy, wrapped stack traces. In production systems, logs often include multiple wrapper exceptions that distract engineers from the deepest "Caused by" error. This environment trains and evaluates agents on that exact skill.

> The environment simulates real-world debugging scenarios where logs contain multiple layers of exception wrapping. Agents must identify the true root cause by filtering noisy stack traces and prioritizing the deepest causal signal.

## Why this matters
- Logs are noisy and full of wrapper exceptions that obscure the real issue.
- The true root cause is often buried deep in stack traces.
- Misdiagnosis wastes engineering time and leads to repeated incidents.

FailureIQ evaluates whether an agent can cut through this noise and reason toward the correct root cause — not just surface-level errors.

## Why this is real-world
- Engineers routinely face nested exceptions that hide the real issue.
- Junior engineers often report the top-level error instead of the root cause.
- This environment forces reasoning: scan, filter noise, prioritize the deepest cause.

## OpenEnv API
- `reset()` returns the initial observation for a task.
- `step(action)` accepts a diagnosis and returns observation, reward, done, info.
- `state()` returns the internal state (task id, step, best score).

## Action Space
`FailureIQAction`
- `action_type: str` One of: `fetch_logs`, `rank_logs`, `classify_issue`, `request_code`, `request_config`, `request_data`, `propose_fix`, `submit_solution`
- `response: str` Free-text fix proposal or final diagnosis
- `category: str` Optional classification (`code`, `config`, `data`)
- `top_k: int` Optional number of ranked log lines

## Observation Space
`FailureIQObservation`
- `task_id: str`
- `difficulty: str` (easy, medium, hard)
- `phase: str` current phase (`need_logs`, `analysis`, `investigation`, `final`)
- `message: str` guidance or feedback
- `available_actions: List[str]` allowed actions in the current phase
- `log: str | None` raw log text (only after `fetch_logs`)
- `ranked_logs: List[str] | None` top-ranked log lines (after `rank_logs`)
- `classification: str | None` remembered category
- `code_snippet: str | None`, `config_snippet: str | None`, `data_context: str | None`
- `step: int`
- `max_steps: int`

## Tasks (Easy -> Medium -> Hard)
1. **code_null_pointer**: Nested exceptions with a `NullPointerException` as the deepest cause.
2. **config_oom**: Misleading timeout and executor loss with `OutOfMemoryError` as the true cause.
3. **data_constraint**: Wrapped database error where the real issue is a `varchar` length constraint.

## Reward Shaping
- Cumulative reward across the trajectory for: fetching logs, ranking logs, correct classification, and relevant context requests.
- Fix proposals scored by expected fix keywords and validation steps.
- Final score for the true root cause (deepest “Caused by”).
- Penalty for invalid actions, wrong classification, or running out of steps.
- Episode ends when submitted or max steps reached.

## Multi-step Flow
1. `reset()` returns a brief failure notice (no logs yet).
2. `fetch_logs` retrieves raw logs.
3. `rank_logs` returns the highest-signal lines to avoid flooding the model.
4. `classify_issue` sets `code`/`config`/`data`.
5. Request relevant context (`request_code` / `request_config` / `request_data`).
6. `propose_fix` to earn partial reward.
7. `submit_solution` with the root cause for the final score.

## Project Layout
- `failureiq/env.py` Environment implementation
- `failureiq/tasks.py` Task specs and logs
- `failureiq/graders.py` Deterministic grading logic
- `app.py` FastAPI server for HF Spaces
- `inference.py` Baseline inference script
- `openenv.yaml` OpenEnv metadata
- `Dockerfile` Container build

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Demo UI
Open the demo interface at:
- http://localhost:8000/
- http://localhost:8000/ui

## Example Usage
```bash
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' \
  -d '{"action_type": "fetch_logs"}'
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' \
  -d '{"action_type": "rank_logs", "top_k": 8}'
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' \
  -d '{"action_type": "classify_issue", "category": "config"}'
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' \
  -d '{"action_type": "request_config"}'
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' \
  -d '{"action_type": "propose_fix", "response": "Increase spark.executor.memory to 6g"}'
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' \
  -d '{"action_type": "submit_solution", "response": "OutOfMemoryError: Java heap space"}'
```

## Baseline Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="YOUR_MODEL"
export HF_TOKEN="YOUR_HF_TOKEN"
python inference.py
```

The script prints per-task scores and the average score.

## Baseline Scores
Run `python inference.py` once and paste the numbers here:
- code_null_pointer: 1.00
- config_oom: 1.00
- data_constraint: 1.00
- Average: 1.00

## Deployment (HF Space + Docker)
Build and run:
```bash
docker build -t failureiq .
docker run -p 8000:7860 failureiq
```

## Validation
Run the OpenEnv validator (if available in your setup):
```bash
openenv validate
```

# failureiq-openenv
AI-powered incident debugging environment that identifies root causes from noisy logs and proposes fixes using structured reasoning step by step.
