from __future__ import annotations

from typing import Tuple

from .tasks import TaskSpec


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def grade_root_cause(response: str, task: TaskSpec) -> Tuple[float, str]:
    """Returns (score, rationale) for root cause only."""
    if not response:
        return 0.0, "Empty response."

    normalized = _normalize(response)

    has_root = any(k in normalized for k in task.root_cause_keywords)
    has_mislead = any(k in normalized for k in task.misleading_keywords)

    if has_root:
        score = 1.0
        rationale = "Mentions the root cause."
    else:
        score = 0.0
        rationale = "No signal for the root cause."

    # Penalize if the answer focuses on misleading top-level errors
    if has_mislead and not has_root:
        score = max(0.0, score - 0.3)
        rationale += " Focuses on a misleading top-level error."

    return score, rationale


def grade_fix(response: str, task: TaskSpec) -> Tuple[float, str]:
    """Returns (score, rationale) for fix quality."""
    if not response:
        return 0.0, "Empty fix response."

    normalized = _normalize(response)
    has_fix = any(k in normalized for k in task.fix_keywords)
    has_validation = any(k.lower() in normalized for k in task.validation_steps) or any(
        k in normalized for k in ["validate", "re-run", "unit test", "monitor"]
    )

    score = 0.0
    if has_fix:
        score += 0.6
    if has_validation:
        score += 0.4

    if score >= 0.9:
        return score, "Mentions a fix and validation steps."
    if score > 0.0:
        return score, "Partial fix/validation provided."
    return 0.0, "Fix does not reference expected correction."
