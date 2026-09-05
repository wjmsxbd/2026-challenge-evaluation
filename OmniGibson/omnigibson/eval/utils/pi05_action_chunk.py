"""PI0.5 action-chunk post-processing for all 100 2026 challenge tasks."""

from collections import deque
import dataclasses
import importlib.util
import logging
import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Must match behavior-1k-solution's champion_2026 checkpoint metadata.
TASK_NUM_STAGES = (
    5, 6, 15, 15, 14, 12, 9, 15, 10, 15,
    7, 13, 10, 15, 15, 15, 15, 11, 13, 12,
    14, 15, 9, 15, 15, 15, 15, 15, 15, 15,
    11, 10, 10, 13, 5, 5, 14, 6, 8, 10,
    5, 15, 8, 15, 12, 11, 9, 14, 15, 15,
    15, 9, 12, 14, 13, 11, 6, 5, 15, 7,
    5, 13, 8, 5, 15, 13, 12, 8, 14, 5,
    10, 15, 10, 15, 13, 13, 11, 5, 5, 12,
    10, 10, 9, 8, 15, 14, 11, 12, 15, 5,
    5, 11, 5, 5, 15, 15, 6, 15, 15, 9,
)


@lru_cache(maxsize=1)
def _load_correction_functions():
    path = os.environ.get("PI05_CORRECTION_RULES_PATH")
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "PI05_CORRECTION_RULES_PATH must point to behavior-1k-solution's correction_rules.py"
        )
    spec = importlib.util.spec_from_file_location("pi05_runtime_correction_rules", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PI0.5 correction rules from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_correction_rules, module.check_gripper_variation


@dataclasses.dataclass
class B1KActionChunkConfig:
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    apply_eval_tricks: bool = True
    # Carry the unexecuted tail of one prediction into the next request. This
    # is the action-chunk "maintenance" / inpainting prefix used by the
    # champion evaluator.
    enable_action_chunk_maintenance: bool = True
    # Compress a longer predicted chunk into fewer simulator actions by
    # interpolation (when execute_in_n_steps < actions_to_execute).
    enable_compression: bool = True
    action_horizon: int = 30

    def validate(self) -> None:
        for name in (
            "actions_to_execute",
            "execute_in_n_steps",
            "history_len",
            "votes_to_promote",
            "action_horizon",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.actions_to_keep < 0:
            raise ValueError("actions_to_keep must be non-negative")
        if self.actions_to_execute + self.actions_to_keep > self.action_horizon:
            raise ValueError("actions_to_execute + actions_to_keep exceeds action_horizon")
        if self.execute_in_n_steps > self.actions_to_execute:
            raise ValueError("execute_in_n_steps cannot exceed actions_to_execute")
        if self.votes_to_promote > self.history_len:
            raise ValueError("votes_to_promote cannot exceed history_len")


class B1KActionChunkPostprocessor:
    """Own stage voting, correction rules, compression, and inpainting carry."""

    def __init__(self, config: B1KActionChunkConfig, task_id: int | None = None) -> None:
        config.validate()
        self.config = config
        self.task_id = None
        self.current_stage = 0
        self.prediction_history = deque([], maxlen=config.history_len)
        self.next_initial_actions: np.ndarray | None = None
        self.step_count = 0
        self.prediction_count = 0
        if task_id is not None:
            self.set_task(task_id)

    def reset(self, task_id: int | None = None) -> None:
        if task_id is not None:
            self._validate_task_id(task_id)
            self.task_id = int(task_id)
        self.current_stage = 0
        self.prediction_history.clear()
        self.next_initial_actions = None
        self.step_count = 0
        self.prediction_count = 0

    def set_task(self, task_id: int) -> bool:
        self._validate_task_id(task_id)
        task_id = int(task_id)
        if self.task_id == task_id:
            return False
        old_task_id = self.task_id
        self.task_id = task_id
        self.current_stage = 0
        self.prediction_history.clear()
        self.next_initial_actions = None
        logger.info(
            "Task change detected: %s -> %s (max stages: %s)",
            old_task_id,
            task_id,
            TASK_NUM_STAGES[task_id],
        )
        return True

    def process(
        self,
        raw_actions: np.ndarray,
        predicted_subtask_logits: np.ndarray | None,
        current_state: np.ndarray,
    ) -> np.ndarray:
        if self.task_id is None:
            raise ValueError("task_id must be set before processing an action chunk")

        actions = np.asarray(raw_actions)
        while actions.ndim > 2 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] < 23:
            raise ValueError(f"Expected action chunk shaped [H, >=23], got {actions.shape}")
        if actions.shape[0] < self.config.execute_in_n_steps:
            raise ValueError(
                f"Action chunk horizon {actions.shape[0]} is shorter than "
                f"execute_in_n_steps={self.config.execute_in_n_steps}"
            )
        actions = np.asarray(actions[:, :23]).copy()
        current_state = np.asarray(current_state).reshape(-1)
        if current_state.shape != (23,):
            raise ValueError(f"Expected current state shape (23,), got {current_state.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(current_state).all():
            raise ValueError("Action chunk and current state must contain only finite values")

        # SPEEDUP_EVAL: keep the champion's task/stage-specific action correction
        # in the evaluator, rather than coupling it to the persistent policy server.
        should_compress = (
            self.config.enable_compression
            and self.config.execute_in_n_steps < self.config.actions_to_execute
        )
        if self.config.apply_eval_tricks:
            apply_correction_rules, check_gripper_variation = _load_correction_functions()
            actions_before = actions.copy()
            actions, corrected_stage = apply_correction_rules(
                self.task_id, self.current_stage, current_state, actions
            )
            if corrected_stage != self.current_stage:
                logger.info(
                    "Correction rule changed stage %s -> %s (task %s, step %s)",
                    self.current_stage,
                    corrected_stage,
                    self.task_id,
                    self.step_count,
                )
                self.current_stage = corrected_stage
                self.prediction_history.clear()
            if not np.allclose(actions_before, actions, rtol=1e-3):
                logger.info(
                    "Correction rule modified actions (max diff %.4f, task %s, stage %s)",
                    np.max(np.abs(actions_before - actions)),
                    self.task_id,
                    self.current_stage,
                )

            if should_compress:
                has_high_variation, left_variation, right_variation = check_gripper_variation(
                    actions, self.config.actions_to_execute
                )
                if has_high_variation:
                    should_compress = False
                    logger.info(
                        "Gripper variation disabled compression (left %.4f, right %.4f)",
                        left_variation,
                        right_variation,
                    )

        # SPEEDUP_EVAL: execute a 26-step prediction in 20 simulator actions when
        # safe, while retaining the tail as the next inpainting prefix.
        actions_to_execute = (
            self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
        )
        inpainting_end = actions_to_execute + self.config.actions_to_keep
        # SPEEDUP_EVAL: carry the chunk tail into the next request so consecutive
        # batched inferences remain temporally continuous.
        if (
            self.config.enable_action_chunk_maintenance
            and self.config.actions_to_keep
            and len(actions) >= inpainting_end
        ):
            self.next_initial_actions = actions[actions_to_execute:inpainting_end].copy()
        else:
            self.next_initial_actions = None

        # SPEEDUP_EVAL: interpolate the compressed chunk and rescale base motion so
        # reducing policy calls does not reduce the accumulated [vx, vy, wz] travel.
        execution_chunk = actions[:actions_to_execute].copy()
        if should_compress:
            execution_chunk = self._interpolate_actions(execution_chunk, self.config.execute_in_n_steps)
            execution_chunk[:, :3] *= actions_to_execute / self.config.execute_in_n_steps

        self.prediction_count += 1
        if predicted_subtask_logits is not None:
            self.update_current_stage(predicted_subtask_logits)
        return np.asarray(execution_chunk, dtype=np.float32)

    def update_current_stage(self, predicted_subtask_logits: np.ndarray) -> None:
        if self.task_id is None:
            return
        # SPEEDUP_EVAL: use a short vote history instead of switching stage on one
        # noisy logit prediction; this is part of the champion action protocol.
        logits = np.asarray(predicted_subtask_logits).squeeze()
        if logits.ndim != 1:
            raise ValueError(f"Expected 1-D subtask logits, got {logits.shape}")

        max_stage = TASK_NUM_STAGES[self.task_id] - 1
        predicted_stage = min(int(np.argmax(logits)), max_stage)
        self.prediction_history.append(predicted_stage)
        if len(self.prediction_history) != self.config.history_len:
            return

        next_stage = self.current_stage + 1
        if next_stage > max_stage:
            return
        votes_for_next = sum(pred == next_stage for pred in self.prediction_history)
        votes_to_skip = sum(pred == next_stage + 1 for pred in self.prediction_history)
        votes_to_go_back = sum(pred == self.current_stage - 1 for pred in self.prediction_history)
        old_stage = self.current_stage
        reason = None
        if votes_for_next >= self.config.votes_to_promote:
            self.current_stage = next_stage
            reason = "advanced"
        elif votes_to_skip == self.config.history_len:
            self.current_stage = next_stage
            reason = "skipped"
        elif votes_to_go_back == self.config.history_len and self.current_stage > 0:
            self.current_stage -= 1
            reason = "went back"
        if reason is not None:
            self.prediction_history.clear()
            logger.info(
                "Stage %s: %s -> %s (task %s, step %s)",
                reason,
                old_stage,
                self.current_stage,
                self.task_id,
                self.step_count,
            )

    def record_executed_actions(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("Executed action count must be non-negative")
        self.step_count += count

    @staticmethod
    def _interpolate_actions(actions: np.ndarray, target_steps: int) -> np.ndarray:
        original_indices = np.linspace(0, len(actions) - 1, len(actions))
        target_indices = np.linspace(0, len(actions) - 1, target_steps)
        interpolated = np.zeros((target_steps, actions.shape[1]), dtype=np.float64)
        for dim in range(actions.shape[1]):
            interpolator = interp1d(original_indices, actions[:, dim], kind="cubic")
            interpolated[:, dim] = interpolator(target_indices)
        return interpolated

    @staticmethod
    def _validate_task_id(task_id: int) -> None:
        if not 0 <= int(task_id) < len(TASK_NUM_STAGES):
            raise ValueError(f"task_id must be in [0, {len(TASK_NUM_STAGES) - 1}], got {task_id}")
