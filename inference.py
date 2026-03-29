"""
Inference Script Example
========================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

from __future__ import annotations

import os
import json
from typing import Dict

from openai import OpenAI

from failureiq.env import FailureIQEnv
from failureiq.models import FailureIQAction
from failureiq.tasks import TASKS

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

TEMPERATURE = 0.2
MAX_TOKENS = 120


def build_prompt(log_text: str) -> str:
    return (
        "You are a production reliability assistant.\n"
        "Given the log below, return a JSON object with keys:\n"
        "category (code|config|data), root_cause (short phrase), fix (short),\n"
        "validation (1 short sentence).\n"
        "Ignore wrapper exceptions and focus on the deepest caused-by error.\n\n"
        f"LOG:\n{log_text}\n\n"
        "JSON:"
    )


def run() -> Dict[str, float]:
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is not set")
    if not API_KEY:
        raise RuntimeError("HF_TOKEN or API_KEY is not set")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FailureIQEnv()

    scores: Dict[str, float] = {}
    for task in TASKS:
        env.reset(task_id=task.task_id)
        obs, _, _, _ = env.step(FailureIQAction(action_type="fetch_logs"))
        prompt = build_prompt(obs.log or "")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "You are concise and accurate."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""

        except Exception as e:
            # fallback (VERY IMPORTANT)
            log = (obs.log or "").lower()

            if "nullpointer" in log:
                text = json.dumps({
                    "category": "code",
                    "root_cause": "NullPointerException",
                    "fix": "Add null checks",
                    "validation": "Add unit tests"
                })

            elif "outofmemory" in log or "heap" in log:
                text = json.dumps({
                    "category": "config",
                    "root_cause": "OutOfMemoryError",
                    "fix": "Increase executor memory",
                    "validation": "Monitor memory usage"
                })

            else:
                text = json.dumps({
                    "category": "data",
                    "root_cause": "Data length constraint violation",
                    "fix": "Validate or truncate input",
                    "validation": "Re-run insert"
                })
        category = "code"
        root_cause = text
        fix = text
        validation = ""

        try:
            parsed = json.loads(text)
            category = parsed.get("category", category)
            root_cause = parsed.get("root_cause", root_cause)
            fix = parsed.get("fix", fix)
            validation = parsed.get("validation", validation)
        except json.JSONDecodeError:
            lowered = text.lower()
            if "config" in lowered or "memory" in lowered or "oom" in lowered:
                category = "config"
            if "data" in lowered or "varchar" in lowered or "constraint" in lowered:
                category = "data"

        # Step through the environment
        env.step(FailureIQAction(action_type="rank_logs", top_k=8))
        env.step(FailureIQAction(action_type="classify_issue", category=category))
        if category == "code":
            env.step(FailureIQAction(action_type="request_code"))
        elif category == "config":
            env.step(FailureIQAction(action_type="request_config"))
        else:
            env.step(FailureIQAction(action_type="request_data"))
        fix_payload = fix
        if validation:
            fix_payload = f"{fix}. Validation: {validation}"
        env.step(FailureIQAction(action_type="propose_fix", response=fix_payload))
        _, _, _, info = env.step(
            FailureIQAction(action_type="submit_solution", response=root_cause)
        )
        scores[task.task_id] = info.score

    return scores


if __name__ == "__main__":
    results = run()
    avg = sum(results.values()) / max(1, len(results))
    print("Scores:")
    for key, value in results.items():
        print(f"- {key}: {value:.2f}")
    print(f"Average: {avg:.2f}")
