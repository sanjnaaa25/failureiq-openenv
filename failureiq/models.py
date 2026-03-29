from __future__ import annotations

from typing import List, Optional
from typing import Literal

from pydantic import BaseModel, Field


class FailureIQObservation(BaseModel):
    """Observation returned to the agent."""

    task_id: str
    difficulty: str
    phase: str
    message: str
    available_actions: List[str]
    log: Optional[str] = None
    ranked_logs: Optional[List[str]] = None
    classification: Optional[str] = None
    code_snippet: Optional[str] = None
    config_snippet: Optional[str] = None
    data_context: Optional[str] = None
    step: int
    max_steps: int


class FailureIQAction(BaseModel):
    """Action provided by the agent."""

    action_type: Literal[
        "fetch_logs",
        "rank_logs",
        "classify_issue",
        "request_code",
        "request_config",
        "request_data",
        "propose_fix",
        "submit_solution",
    ] = Field(..., description="Type of action to perform")
    response: Optional[str] = Field(
        None, description="Free-text response, fix proposal, or final diagnosis"
    )
    category: Optional[Literal["code", "config", "data"]] = None
    top_k: Optional[int] = Field(10, description="How many ranked log lines to return")


class FailureIQReward(BaseModel):
    """Reward signal returned after each step."""

    value: float


class FailureIQInfo(BaseModel):
    """Extra info for debugging and grading."""

    score: float
    done_reason: Optional[str] = None
    expected_keywords: Optional[List[str]] = None
    rationale: Optional[str] = None
