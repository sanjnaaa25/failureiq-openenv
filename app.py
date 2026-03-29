from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from failureiq.env import FailureIQEnv
from failureiq.models import FailureIQAction

app = FastAPI(title="FailureIQ", version="0.1.0")

ENV = FailureIQEnv()


class ResetRequest(BaseModel):
    task_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict:
    task_id = req.task_id if req else None
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


@app.get("/state")
def state() -> dict:
    return {"state": ENV.state()}
