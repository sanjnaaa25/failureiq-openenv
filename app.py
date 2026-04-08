from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel

from failureiq.env import FailureIQEnv
from failureiq.models import FailureIQAction

app = FastAPI(title="FailureIQ", version="0.1.0")

ENV = FailureIQEnv()
UI_PATH = Path(__file__).parent / "ui" / "index.html"


class ResetRequest(BaseModel):
    task_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui_root() -> str:
    return UI_PATH.read_text(encoding="utf-8")


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return UI_PATH.read_text(encoding="utf-8")


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict:
    task_id = req.task_id if req else None
    obs = ENV.reset(task_id=task_id)
    return {"observation": obs.model_dump()}


@app.get("/reset")
def reset_get(task_id: str | None = None) -> dict:
    obs = ENV.reset(task_id=task_id)
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: FailureIQAction) -> dict:
    obs, reward, done, info = ENV.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }


@app.get("/step")
def step_get(
    action_type: str = Query(...),
    response: str | None = Query(None),
    category: str | None = Query(None),
    top_k: int | None = Query(None),
) -> dict:
    action = FailureIQAction(
        action_type=action_type,
        response=response,
        category=category,
        top_k=top_k,
    )
    obs, reward, done, info = ENV.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }


@app.get("/state")
def state() -> dict:
    return {"state": ENV.state()}
