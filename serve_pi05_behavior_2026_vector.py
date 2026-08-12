"""Persistent raw-action-chunk PI0.5 server for the 2026 100-task checkpoint."""

import dataclasses
import enum
import logging
import os
import time
import traceback
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import msgpack_numpy
from openpi_client.image_tools import resize_with_pad
import tyro
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from b1k.models.observation import Observation
from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.policies import policy_config as policy_config
from b1k.policies.checkpoint_switcher import CheckpointSwitcher
from b1k.shared import normalize
from b1k.training import config as training_config


logger = logging.getLogger(__name__)
RESIZE_SIZE = 224


class EnvMode(enum.Enum):
    ALOHA_SIM = "aloha_sim"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    host: str = "localhost"
    port: int = 8000
    norm_stats_path: str | None = None
    inference_only: bool = True
    dynamic_batching: bool = True
    dynamic_batch_max_size: int = 2
    dynamic_batch_wait_ms: float = 0.0
    dynamic_batch_granularity: int = 1
    num_steps: int = 20
    disable_fast_auxiliary: bool = True
    task_checkpoint_mapping: str | None = None
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def _scalar_int(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one value, got {array.shape}")
    return int(array.reshape(-1)[0])


def _normalize_raw_actions(actions: Any, action_horizon: int = 30) -> np.ndarray:
    actions = np.asarray(actions)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[0] < action_horizon or actions.shape[1] < 23:
        raise ValueError(f"Expected actions shaped [>={action_horizon}, >=23], got {actions.shape}")
    actions = np.asarray(actions[:action_horizon, :23], dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Model returned NaN or Inf actions")
    return actions


def _load_norm_stats(path: str | None):
    if path is None:
        return None
    stats_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    return normalize.deserialize_json(open(stats_path, encoding="utf-8").read())


def _prepare_model_input(obs: dict, task_id: int, current_stage: int) -> dict:
    head = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
    left = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
    right = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
    return {
        "observation/egocentric_camera": resize_with_pad(head, RESIZE_SIZE, RESIZE_SIZE),
        "observation/wrist_image_left": resize_with_pad(left, RESIZE_SIZE, RESIZE_SIZE),
        "observation/wrist_image_right": resize_with_pad(right, RESIZE_SIZE, RESIZE_SIZE),
        "observation/state": np.asarray(obs["robot_r1::proprio"]),
        "tokenized_prompt": np.asarray([task_id, current_stage], dtype=np.int32),
        "tokenized_prompt_mask": np.asarray([True, True], dtype=bool),
        "subtask_state": np.asarray(current_stage, dtype=np.int32),
    }


def _normalize_initial_actions(policy, obs: dict, initial_actions: np.ndarray) -> jnp.ndarray:
    # msgpack_numpy reconstructs arrays as read-only views over the receive
    # buffer, while the delta-action transform updates this array in place.
    transformed = policy._input_transform({**obs, "actions": np.array(initial_actions, copy=True)})
    return jnp.asarray(transformed["actions"])


def _infer_batch(policy, observations: list[dict], initial_actions: list[np.ndarray | None]) -> list[dict]:
    transformed_inputs = [
        policy._input_transform(jax.tree.map(lambda value: value, observation))
        for observation in observations
    ]
    inputs = jax.tree.map(
        lambda *values: jnp.stack([jnp.asarray(value) for value in values], axis=0),
        *transformed_inputs,
    )
    policy._rng, sample_rng = jax.random.split(policy._rng)
    sample_kwargs = dict(policy._sample_kwargs)

    present = [item is not None and np.asarray(item).size > 0 for item in initial_actions]
    if any(present) and not all(present):
        raise ValueError("Every sample in a batch must consistently provide initial_actions")
    if all(present):
        normalized = [
            _normalize_initial_actions(policy, obs, np.asarray(actions))
            for obs, actions in zip(observations, initial_actions)
        ]
        if len({tuple(value.shape) for value in normalized}) != 1:
            raise ValueError("All initial_actions in a batch must have the same shape")
        sample_kwargs["initial_actions"] = jnp.stack(normalized, axis=0)

    start = time.monotonic()
    actions, subtask_logits = policy._sample_actions(
        sample_rng,
        Observation.from_dict(inputs),
        **sample_kwargs,
    )
    infer_ms = (time.monotonic() - start) * 1000.0
    actions = np.asarray(actions)
    subtask_logits = np.asarray(subtask_logits)
    states = np.asarray(inputs["state"])
    if actions.shape[0] != len(observations) or subtask_logits.shape[0] != len(observations):
        raise RuntimeError(f"Unexpected model batch shapes: {actions.shape}, {subtask_logits.shape}")

    results = []
    for index in range(len(observations)):
        result = policy._output_transform(
            {
                "state": states[index],
                "actions": actions[index],
                "subtask_logits": subtask_logits[index],
            }
        )
        result["policy_timing"] = {"infer_ms": infer_ms}
        results.append(result)
    return results


class RawChunkPolicy:
    def __init__(self, policy, checkpoint_switcher=None, max_batch_size: int = 2) -> None:
        self.policy = policy
        self.checkpoint_switcher = checkpoint_switcher
        self.max_batch_size = int(max_batch_size)
        self.request_count = 0

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, request: dict) -> dict:
        observations = request.get("observation_batch")
        if not isinstance(observations, list) or not observations:
            raise ValueError("Raw chunk server requires a non-empty observation_batch")
        if len(observations) > self.max_batch_size:
            raise ValueError(
                f"Raw chunk server accepts at most {self.max_batch_size} observations, got {len(observations)}"
            )
        task_ids = []
        model_inputs = []
        initial_actions = []
        for index, obs in enumerate(observations):
            task_id = _scalar_int(obs.get("task_id"), f"observation_batch[{index}].task_id")
            current_stage = _scalar_int(
                obs.get("current_stage"), f"observation_batch[{index}].current_stage"
            )
            if not 0 <= task_id < len(TASK_NUM_STAGES):
                raise ValueError(f"task_id must be in [0, {len(TASK_NUM_STAGES) - 1}], got {task_id}")
            if not 0 <= current_stage < TASK_NUM_STAGES[task_id]:
                raise ValueError(f"Invalid stage {current_stage} for task {task_id}")
            task_ids.append(task_id)
            model_inputs.append(_prepare_model_input(obs, task_id, current_stage))
            prefix = obs.get("initial_actions")
            initial_actions.append(None if prefix is None else np.asarray(prefix))

        if len(set(task_ids)) != 1:
            raise ValueError(f"One vector request must contain one task ID, got {sorted(set(task_ids))}")
        if self.checkpoint_switcher is not None:
            self.policy = self.checkpoint_switcher.get_policy_for_task(task_ids[0])

        outputs = _infer_batch(self.policy, model_inputs, initial_actions)
        action_chunks = []
        logits = []
        timings = []
        for output in outputs:
            action_chunks.append(_normalize_raw_actions(output["actions"]))
            output_logits = np.asarray(output["subtask_logits"], dtype=np.float32).squeeze()
            if output_logits.ndim != 1 or np.isnan(output_logits).any() or not np.isfinite(output_logits).any():
                raise ValueError(f"Invalid subtask logits: {output_logits.shape}")
            logits.append(output_logits)
            timings.append(output.get("policy_timing", {}))

        self.request_count += 1
        chunks = np.stack(action_chunks)
        return {
            "action": chunks[:, 0].copy(),
            "action_chunk": chunks,
            "subtask_logits": np.stack(logits),
            "predicted_stage": np.asarray([int(np.argmax(value)) for value in logits], dtype=np.int32),
            "policy_timing": timings,
            "batch_size": len(observations),
        }


class RawChunkWebsocketServer(WebsocketPolicyServer):
    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        previous_total = None
        while True:
            try:
                started = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in request:
                    self._policy.reset()
                    continue
                inference_started = time.monotonic()
                response = self._policy.act(request)
                infer_ms = (time.monotonic() - inference_started) * 1000.0
                timing = response.setdefault("server_timing", {})
                timing["infer_ms"] = infer_ms
                timing["request_batch_size"] = int(response.get("batch_size", 0))
                if previous_total is not None:
                    timing["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                error = traceback.format_exc()
                logger.error("Raw chunk server error:\n%s", error)
                await websocket.send(error)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                raise


def main(args: Args) -> None:
    if not isinstance(args.policy, Checkpoint):
        raise ValueError("policy:checkpoint is required")
    if not args.inference_only:
        raise ValueError("This entrypoint only supports --inference-only")
    if args.dynamic_batch_max_size != 2 or args.dynamic_batch_granularity != 1:
        raise ValueError("The two-env evaluator requires max policy batch size 2 and granularity 1")
    if len(TASK_NUM_STAGES) != 100:
        raise RuntimeError(
            f"PI05_REPO exposes {len(TASK_NUM_STAGES)} tasks; use its champion_2026 checkout"
        )

    config = training_config.get_config(args.policy.config)
    if getattr(config.model, "num_tasks", None) != 100:
        raise RuntimeError(f"Policy config {args.policy.config!r} is not a 100-task config")
    if args.disable_fast_auxiliary:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, use_fast_auxiliary=False),
        )
    norm_stats = _load_norm_stats(args.norm_stats_path)
    policy = policy_config.create_trained_policy(
        config,
        args.policy.dir,
        sample_kwargs={"num_steps": args.num_steps},
        norm_stats=norm_stats,
    )

    checkpoint_switcher = None
    if args.task_checkpoint_mapping:
        checkpoint_switcher = CheckpointSwitcher(
            config_path=args.task_checkpoint_mapping,
            training_config=config,
            sample_kwargs={"num_steps": args.num_steps},
            norm_stats=norm_stats,
        )
    metadata = dict(policy.metadata or {})
    metadata["b1k_protocol"] = {
        "version": 2,
        "mode": "raw_action_chunk",
        "action_horizon": 30,
        "action_dim": 23,
        "task_count": len(TASK_NUM_STAGES),
        "task_num_stages": list(TASK_NUM_STAGES),
        "proprioception_schema": "r1pro_v3_61",
        "policy_batch_size": args.dynamic_batch_max_size,
        "capabilities": [
            "action_chunk",
            "subtask_logits",
            "request_stage",
            "observation_batch",
            "persistent_server",
        ],
    }
    server = RawChunkWebsocketServer(
        policy=RawChunkPolicy(policy, checkpoint_switcher, max_batch_size=args.dynamic_batch_max_size),
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logger.info(
        "Persistent 2026 raw-chunk server: tasks=100 checkpoint=%s policy_batch=%s",
        args.policy.dir,
        args.dynamic_batch_max_size,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
