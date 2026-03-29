from __future__ import annotations

from typing import Dict, Optional, Tuple, List

from .graders import grade_root_cause, grade_fix
from .models import FailureIQAction, FailureIQInfo, FailureIQObservation, FailureIQReward
from .tasks import TASKS, TaskSpec


class FailureIQEnv:
    """OpenEnv-style environment for root cause detection in logs."""

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self._task_index = 0
        self._current_task: Optional[TaskSpec] = None
        self._step = 0
        self._best_score = 0.0
        self._done = False
        self._phase = "need_logs"
        self._classification: Optional[str] = None
        self._last_ranked: Optional[List[str]] = None

    def _select_task(self, task_id: Optional[str]) -> TaskSpec:
        if task_id:
            for task in TASKS:
                if task.task_id == task_id:
                    return task
            raise ValueError(f"Unknown task_id: {task_id}")
        task = TASKS[self._task_index % len(TASKS)]
        self._task_index += 1
        return task

    def reset(self, task_id: Optional[str] = None) -> FailureIQObservation:
        self._current_task = self._select_task(task_id)
        self._step = 0
        self._best_score = 0.0
        self._done = False
        self._phase = "need_logs"
        self._classification = None
        self._last_ranked = None
        return self._build_observation(
            message="Pipeline failed. Fetch logs to begin analysis.",
        )

    def step(
        self, action: FailureIQAction
    ) -> Tuple[FailureIQObservation, FailureIQReward, bool, FailureIQInfo]:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            info = FailureIQInfo(score=self._best_score, done_reason="episode_done")
            obs = self._build_observation(message="Episode already finished.")
            return obs, FailureIQReward(value=0.0), True, info

        self._step += 1
        reward_value = 0.0
        rationale = None

        if action.action_type == "fetch_logs" and self._phase == "need_logs":
            self._phase = "analysis"
            reward_value = 0.05
            message = "Logs fetched. You may rank logs or classify the issue."
            obs = self._build_observation(message=message, log=self._current_task.log)
        elif action.action_type == "rank_logs" and self._phase in ("analysis", "investigation"):
            self._last_ranked = self._rank_logs(self._current_task.log, action.top_k or 10)
            reward_value = 0.05
            message = "Top-ranked log lines returned. Classify the issue next."
            obs = self._build_observation(message=message, ranked_logs=self._last_ranked)
        elif action.action_type == "classify_issue" and self._phase in ("analysis", "investigation"):
            if action.category not in ("code", "config", "data"):
                reward_value = -0.1
                message = "Invalid category. Choose: code, config, or data."
                obs = self._build_observation(message=message)
            else:
                self._classification = action.category
                self._phase = "investigation"
                if action.category == self._current_task.category:
                    reward_value = 0.3
                    message = f"Correct category: {action.category}. Request code/config/data as needed."
                else:
                    reward_value = -0.1
                    message = f"Category recorded: {action.category}. You may continue investigating."
                obs = self._build_observation(message=message, classification=self._classification)
        elif action.action_type == "request_code" and self._phase == "investigation":
            reward_value = 0.1 if self._current_task.category == "code" else 0.0
            message = "Code context returned."
            obs = self._build_observation(message=message, code_snippet=self._current_task.code_snippet)
        elif action.action_type == "request_config" and self._phase == "investigation":
            reward_value = 0.1 if self._current_task.category == "config" else 0.0
            message = "Config context returned."
            obs = self._build_observation(message=message, config_snippet=self._current_task.config_snippet)
        elif action.action_type == "request_data" and self._phase == "investigation":
            reward_value = 0.1 if self._current_task.category == "data" else 0.0
            message = "Data context returned."
            obs = self._build_observation(message=message, data_context=self._current_task.data_context)
        elif action.action_type == "propose_fix" and self._phase == "investigation":
            fix_score, fix_rationale = grade_fix(action.response or "", self._current_task)
            reward_value = 0.3 * fix_score
            rationale = fix_rationale
            self._phase = "final"
            message = "Fix proposal recorded. Submit final root cause."
            obs = self._build_observation(message=message, classification=self._classification)
        elif action.action_type == "submit_solution" and self._phase in ("investigation", "final"):
            score, rationale = grade_root_cause(action.response or "", self._current_task)
            reward_value = score
            self._done = True
            done = True
            self._best_score = max(0.0, min(1.0, self._best_score + reward_value))
            info = FailureIQInfo(
                score=self._best_score,
                done_reason="solved" if score >= 0.95 else "submitted",
                expected_keywords=self._current_task.root_cause_keywords,
                rationale=rationale,
            )
            obs = self._build_observation(message="Submission received.")
            return obs, FailureIQReward(value=reward_value), done, info
        else:
            reward_value = -0.1
            message = "Action not allowed in current phase."
            obs = self._build_observation(message=message)

        done = self._step >= self.max_steps
        self._done = done
        if done and action.action_type != "submit_solution":
            reward_value = max(-0.2, reward_value - 0.2)

        self._best_score = max(0.0, min(1.0, self._best_score + reward_value))
        info = FailureIQInfo(
            score=self._best_score,
            done_reason="max_steps" if done else None,
            expected_keywords=self._current_task.root_cause_keywords,
            rationale=rationale,
        )
        return obs, FailureIQReward(value=reward_value), done, info

    def state(self) -> Dict[str, object]:
        return {
            "task_id": self._current_task.task_id if self._current_task else None,
            "step": self._step,
            "max_steps": self.max_steps,
            "best_score": self._best_score,
            "done": self._done,
            "phase": self._phase,
            "classification": self._classification,
        }

    def _build_observation(
        self,
        message: str,
        log: Optional[str] = None,
        ranked_logs: Optional[List[str]] = None,
        classification: Optional[str] = None,
        code_snippet: Optional[str] = None,
        config_snippet: Optional[str] = None,
        data_context: Optional[str] = None,
    ) -> FailureIQObservation:
        return FailureIQObservation(
            task_id=self._current_task.task_id if self._current_task else "",
            difficulty=self._current_task.difficulty if self._current_task else "",
            phase=self._phase,
            message=message,
            available_actions=self._available_actions(),
            log=log,
            ranked_logs=ranked_logs,
            classification=classification or self._classification,
            code_snippet=code_snippet,
            config_snippet=config_snippet,
            data_context=data_context,
            step=self._step,
            max_steps=self.max_steps,
        )

    def _available_actions(self) -> List[str]:
        if self._phase == "need_logs":
            return ["fetch_logs"]
        if self._phase == "analysis":
            return ["rank_logs", "classify_issue"]
        if self._phase == "investigation":
            return [
                "rank_logs",
                "classify_issue",
                "request_code",
                "request_config",
                "request_data",
                "propose_fix",
                "submit_solution",
            ]
        if self._phase == "final":
            return ["submit_solution"]
        return []

    def _rank_logs(self, log_text: str, top_k: int) -> List[str]:
        lines = [line for line in log_text.splitlines() if line.strip()]
        scored = []
        for line in lines:
            score = 0
            lowered = line.lower()
            if "caused by" in lowered:
                score += 3
            if "error" in lowered:
                score += 2
            if "exception" in lowered:
                score += 2
            if "outofmemory" in lowered or "nullpointer" in lowered or "psql" in lowered:
                score += 2
            scored.append((score, line))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in scored[: max(1, min(top_k, len(scored)))]]
