"""Implementation of the 2026 vectorized action-chunk evaluator."""

import copy
import json
import logging
import os
import tempfile
import traceback
from pathlib import Path
from signal import SIGINT, signal
from typing import Any

import cv2
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import augment_rooms, get_task_relevant_room_types, load_available_tasks
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from omnigibson.eval.evaluator import (
    DEFAULT_ROBOT_CONFIG_PATH,
    EVAL_BASE_LINK_MASS,
    EVAL_HEAD_HORIZONTAL_APERTURE,
    LIGHT_EVAL_TASKS,
)
from omnigibson.eval.policies import WebsocketPolicy
from omnigibson.eval.utils.eval_utils import (
    EVAL_TIMEOUT_MULTIPLIER,
    PROPRIOCEPTION_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
    get_robot_camera_names,
)
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.eval.utils.score_utils import load_human_stats
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, TaskMetric
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


# The checkpoint was trained with the 2025 256-D proprio tensor, while the
# official 2026 R1Pro config exposes a 61-D challenge-safe tensor. The PI0.5
# input transform only reads these six fields from the legacy tensor. Repack
# exactly those fields and leave global pose / base position entries at zero.
_PI05_PROPRIO_TARGET_SLICES = {
    "base_qvel": np.s_[253:256],
    "trunk_qpos": np.s_[236:240],
    "arm_left_qpos": np.s_[158:165],
    "gripper_left_qpos": np.s_[193:195],
    "arm_right_qpos": np.s_[197:204],
    "gripper_right_qpos": np.s_[232:234],
}
_PI05_PROPRIO_DIM = 256
_R1PRO_2026_PROPRIO_DIM = max(index.stop for index in PROPRIOCEPTION_INDICES["R1Pro"].values())
_MAX_GRIPPER_WIDTH = 0.1
_POLICY_IMAGE_SIZE = 224


def _select_policy_base_qvel(
    proprio: th.Tensor,
    absolute_base_qvel: th.Tensor,
    frame: str,
) -> th.Tensor:
    """Select the base velocity frame exposed to the policy.

    OmniGibson's current holonomic-base proprioception reports robot-local
    [vx, vy, wz]. Legacy training data used the raw virtual-joint velocities,
    which are expressed in the articulation's fixed canonical frame.
    """
    if frame == "relative":
        return proprio
    if frame != "absolute":
        raise ValueError(f"Unknown PI0.5 base velocity frame: {frame}")

    base_qvel = th.as_tensor(absolute_base_qvel, dtype=proprio.dtype, device=proprio.device)
    if base_qvel.shape != (3,):
        raise ValueError(f"Expected 3-D absolute base qvel, got shape {tuple(base_qvel.shape)}")
    converted = proprio.clone()
    converted[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]] = base_qvel
    return converted


def _configure_policy_camera_resolution(robot_cfg: dict) -> None:
    """Set policy camera resolution before VisionSensor render products exist."""
    sensor_config = robot_cfg.setdefault("sensor_config", {})
    vision_config = sensor_config.setdefault("VisionSensor", {})
    sensor_kwargs = vision_config.setdefault("sensor_kwargs", {})
    sensor_kwargs["image_height"] = _POLICY_IMAGE_SIZE
    sensor_kwargs["image_width"] = _POLICY_IMAGE_SIZE


def _adapt_proprio_for_pi05(proprio: th.Tensor) -> th.Tensor:
    """Repack official 2026 R1Pro proprio into the checkpoint's legacy indices."""
    if proprio.shape[-1] != _R1PRO_2026_PROPRIO_DIM:
        raise ValueError(
            "This PI0.5 adapter requires the official 2026 R1Pro "
            f"{_R1PRO_2026_PROPRIO_DIM}-D proprio layout, got shape {tuple(proprio.shape)}"
        )
    adapted = proprio.new_zeros((*proprio.shape[:-1], _PI05_PROPRIO_DIM))
    for key, target_slice in _PI05_PROPRIO_TARGET_SLICES.items():
        adapted[..., target_slice] = proprio[..., PROPRIOCEPTION_INDICES["R1Pro"][key]]
    return adapted


def _extract_pi05_state(proprio: np.ndarray) -> np.ndarray:
    """Extract the 23-D action state from either supported R1Pro layout."""
    proprio = np.asarray(proprio)
    if proprio.shape[-1] == _R1PRO_2026_PROPRIO_DIM:
        indices = PROPRIOCEPTION_INDICES["R1Pro"]
    elif proprio.shape[-1] == _PI05_PROPRIO_DIM:
        indices = _PI05_PROPRIO_TARGET_SLICES
    else:
        raise ValueError(f"Unsupported PI0.5 proprio shape: {proprio.shape}")
    left_gripper = proprio[..., indices["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper = proprio[..., indices["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    return np.concatenate(
        [
            proprio[..., indices["base_qvel"]],
            proprio[..., indices["trunk_qpos"]],
            proprio[..., indices["arm_left_qpos"]],
            2.0 * (left_gripper / _MAX_GRIPPER_WIDTH) - 1.0,
            proprio[..., indices["arm_right_qpos"]],
            2.0 * (right_gripper / _MAX_GRIPPER_WIDTH) - 1.0,
        ],
        axis=-1,
    )


def _plain_dict(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _atomic_json_dump(data: Any, output_path: Path, **dump_kwargs: Any) -> None:
    """Write JSON through a same-directory temporary file and atomically replace the target."""
    # SPEEDUP_EVAL: never expose a half-written metric marker to the scheduler or
    # output validator if the evaluator is interrupted during a large run.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, **dump_kwargs)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _close_video_writer(writer) -> None:
    if writer is None:
        return
    container, stream = writer
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _snapshot_policy_value(value: Any) -> Any:
    """Copy an observation tree out of live simulator/sensor buffers."""
    if th.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot_policy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_policy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_policy_value(item) for item in value)
    return copy.deepcopy(value)


def _sync_robot_after_pose_override(robot) -> None:
    """Reset articulation and controller state after loading a cached pose."""
    robot.keep_still()
    if getattr(robot, "n_joints", 0) > 0:
        positions = robot.get_joint_positions()
        robot.set_joint_positions(positions=positions, drive=False)
        robot.set_joint_velocities(velocities=th.zeros_like(positions), drive=False)
    robot.keep_still()


def _convert_pose_to_world(scene, pose: dict) -> dict:
    converted = dict(pose)
    position = th.as_tensor(pose["position"], dtype=th.float32)
    orientation = th.as_tensor(pose["orientation"], dtype=th.float32)
    position, orientation = scene.convert_scene_relative_pose_to_world(position, orientation)
    converted["position"] = position.detach().cpu().tolist()
    converted["orientation"] = orientation.detach().cpu().tolist()
    return converted


def _robot_poses_in_world(scene, robot_poses: dict) -> dict:
    return {name: [_convert_pose_to_world(scene, pose) for pose in poses] for name, poses in robot_poses.items()}


class VectorChunkEvaluator:
    """Run one 2026 task in synchronized environments using serialized policy requests."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.task_id = int(cfg.task.id)
        self.task_name = str(cfg.task.name)
        self.skip_action_chunk_rendering = bool(cfg.get("skip_action_chunk_rendering", False))
        # SPEEDUP_EVAL: keep both synchronized vector environments on the exact
        # same seed so slot assignment does not change the rollout RNG stream.
        # This intentionally mirrors the server's fixed JAX seed (0) and makes
        # each group reset reproducible independent of its slot.
        self.env_seeds = [0 for _ in range(self.num_envs)]
        self.should_sync_lights = self.task_name in LIGHT_EVAL_TASKS
        self.human_stats = load_human_stats(self.task_name)

        from omnigibson.eval.utils.pi05_action_chunk import (
            B1KActionChunkConfig,
            B1KActionChunkPostprocessor,
        )

        action_cfg = cfg.action_chunk
        self.postprocessor_config = B1KActionChunkConfig(
            actions_to_execute=int(action_cfg.actions_to_execute),
            actions_to_keep=int(action_cfg.actions_to_keep),
            execute_in_n_steps=int(action_cfg.execute_in_n_steps),
            history_len=int(action_cfg.history_len),
            votes_to_promote=int(action_cfg.votes_to_promote),
            apply_eval_tricks=bool(action_cfg.apply_eval_tricks),
            enable_action_chunk_maintenance=bool(action_cfg.get("enable_action_chunk_maintenance", True)),
            enable_compression=bool(action_cfg.get("enable_compression", True)),
            action_horizon=int(action_cfg.action_horizon),
        )
        self.postprocessors = [
            B1KActionChunkPostprocessor(self.postprocessor_config, task_id=self.task_id)
            for _ in range(self.num_envs)
        ]

        env_config = self._build_environment_config()
        # SPEEDUP_EVAL: create all slots inside one Isaac Sim process; policy
        # requests and simulator ticks are shared by the active slots below.
        self.vector_env = og.VectorEnvironment(self.num_envs, env_config, seeds=self.env_seeds)
        self.envs = self.vector_env.envs
        for env in self.envs:
            env._eval_robot_config = self.robot_eval_config
        self.envs = [instantiate(cfg.env_wrapper, env=env) for env in self.envs]
        self.robots = [env.robots[0] for env in self.envs]
        self.robot_camera_names = get_robot_camera_names(self.robots[0].name, self.robot_eval_config)
        self._validate_robot_and_cameras()
        self._apply_robot_eval_settings()

        self.policy = WebsocketPolicy(host=str(cfg.host), port=int(cfg.port))
        self._validate_server_metadata()
        self.video_writers = [None] * self.num_envs
        self.video_paths = [None] * self.num_envs
        self.light_synchronizers = [None] * self.num_envs
        self._closed = False
        signal(SIGINT, self._sigint_handler)

        logger.info(
            "Vector chunk evaluation: envs=%s request_batch_max=%s env_seeds=%s actions=%s->%s "
            "base_velocity_frame=%s action_chunk_maintenance=%s compression=%s "
            "get_obs/metrics every action=true rendering=per-action-or-chunk-skip "
            "skip_action_chunk_rendering=%s "
            "video_every_action=%s viewer_camera=%s",
            self.num_envs,
            self.num_envs,
            self.env_seeds,
            self.postprocessor_config.actions_to_execute,
            self.postprocessor_config.execute_in_n_steps,
            str(cfg.base_velocity_frame),
            self.postprocessor_config.enable_action_chunk_maintenance,
            self.postprocessor_config.enable_compression,
            self.skip_action_chunk_rendering,
            bool(cfg.write_video),
            bool(gm.RENDER_VIEWER_CAMERA),
        )

    def _build_environment_config(self) -> dict:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        if self.task_name not in available_tasks:
            raise ValueError(f"Unknown BEHAVIOR task: {self.task_name}")
        task_cfg = available_tasks[self.task_name][0]
        config = generate_basic_environment_config(task_name=self.task_name, task_cfg=task_cfg)
        # SPEEDUP_EVAL: load only rooms relevant to this activity to reduce scene
        # construction, physics, and rendering cost without changing task objects.
        if bool(self.cfg.partial_scene_load):
            rooms = get_task_relevant_room_types(activity_name=self.task_name)
            config["scene"]["load_room_types"] = augment_rooms(rooms, task_cfg["scene_model"], self.task_name)

        robot_path = self.cfg.get("robot_config") or DEFAULT_ROBOT_CONFIG_PATH
        robot_cfg = _plain_dict(OmegaConf.load(str(robot_path)))
        if "model" not in robot_cfg or "name" not in robot_cfg:
            raise ValueError("Robot config must include canonical 'model' and 'name' fields")
        if "type" in robot_cfg:
            raise ValueError("Robot config must use canonical 'model', not 'type'")
        robot_cfg["model"] = robot_cfg["model"].lower()
        self.robot_eval_config = _plain_dict(robot_cfg.pop("eval", None)) or {}
        # SPEEDUP_EVAL: configure 224x224 before render products are created;
        # changing an initialized camera would force a costly/destructive rebuild.
        _configure_policy_camera_resolution(robot_cfg)
        logger.info(
            "Policy camera resolution configured before sensor creation: %sx%s",
            _POLICY_IMAGE_SIZE,
            _POLICY_IMAGE_SIZE,
        )
        robot_cfg["position"] = task_cfg["robot_start_position"]
        robot_cfg["orientation"] = task_cfg["robot_start_orientation"]
        config["robots"] = [robot_cfg]

        if self.cfg.max_steps is None:
            max_steps_multiplier = float(self.cfg.get("max_steps_multiplier", EVAL_TIMEOUT_MULTIPLIER))
            max_steps = int(self.human_stats["length"] * max_steps_multiplier)
            timeout_description = f" ({max_steps_multiplier}x mean human length)"
        else:
            max_steps = int(self.cfg.max_steps)
            timeout_description = " (absolute override)"
        config["task"]["termination_config"]["max_steps"] = max_steps
        config["task"]["include_obs"] = False
        logger.info(
            "Official 2026 dynamics: physics/render/action=%s/%s/%s Hz; timeout=%s steps%s",
            config["env"]["physics_frequency"],
            config["env"]["rendering_frequency"],
            config["env"]["action_frequency"],
            max_steps,
            timeout_description,
        )
        return config

    def _validate_robot_and_cameras(self) -> None:
        if any(len(env.robots) != 1 for env in self.envs):
            raise ValueError("Vector PI0.5 evaluation requires exactly one robot per environment")
        if any(robot.model != "r1pro" or robot.name != "robot_r1" for robot in self.robots):
            raise ValueError("This PI0.5 checkpoint requires an R1Pro named 'robot_r1' in every environment")
        if any(robot.action_dim != 23 for robot in self.robots):
            raise ValueError(f"PI0.5 action chunks require action_dim=23, got {[r.action_dim for r in self.robots]}")

        required_roles = {"head", "left_wrist", "right_wrist"}
        missing_roles = sorted(required_roles - set(self.robot_camera_names))
        if missing_roles:
            raise ValueError(f"Robot eval.camera_sensor_names is missing roles: {missing_roles}")
        expected_camera_names = {
            "head": "robot_r1::robot_r1:zed_link:Camera:0",
            "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
            "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        }
        if self.robot_camera_names != expected_camera_names:
            raise ValueError(
                "This PI0.5 checkpoint requires the standard R1Pro head and wrist camera names; "
                f"got {self.robot_camera_names}"
            )
        for slot, robot in enumerate(self.robots):
            missing_sensors = [
                name.split("::", 1)[1]
                for name in self.robot_camera_names.values()
                if name.split("::", 1)[1] not in robot.sensors
            ]
            if missing_sensors:
                raise ValueError(f"Vector slot {slot} is missing camera sensors: {missing_sensors}")

    def _apply_robot_eval_settings(self) -> None:
        og.sim.stop()
        for robot in self.robots:
            if robot.model in ("r1", "r1pro"):
                robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
            head_sensor_name = self.robot_camera_names["head"].split("::", 1)[1]
            robot.sensors[head_sensor_name].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE
        og.sim.play()

    def _validate_server_metadata(self) -> None:
        from omnigibson.eval.utils.pi05_action_chunk import TASK_NUM_STAGES

        metadata = self.policy.get_server_metadata()
        protocol = metadata.get("b1k_protocol", {}) if isinstance(metadata, dict) else {}
        capabilities = set(protocol.get("capabilities", [])) if isinstance(protocol, dict) else set()
        required = {"action_chunk", "subtask_logits", "request_stage", "observation_batch"}
        if protocol.get("version") != 2 or protocol.get("mode") != "raw_action_chunk":
            raise RuntimeError(f"Vector evaluation requires raw-action protocol v2, got {protocol!r}")
        if not required.issubset(capabilities):
            raise RuntimeError(f"Policy server is missing capabilities: {sorted(required - capabilities)}")
        if protocol.get("task_count") != 100:
            raise RuntimeError(f"2026 evaluation requires a 100-task policy server, got {protocol!r}")
        if tuple(protocol.get("task_num_stages", ())) != TASK_NUM_STAGES:
            raise RuntimeError("Evaluator/server 2026 task-stage metadata does not match")
        if protocol.get("proprioception_schema") != str(self.cfg.proprioception_schema):
            raise RuntimeError(
                "Evaluator/server proprioception schema mismatch: "
                f"{self.cfg.proprioception_schema} != {protocol.get('proprioception_schema')}"
            )
        server_batch_size = protocol.get("policy_batch_size")
        if server_batch_size != self.num_envs:
            raise RuntimeError(
                f"Evaluator requires policy max batch size {self.num_envs}, got {server_batch_size}"
            )

    def _instance_path(self, env, instance_id: int) -> str:
        scene_model = env.task.scene_name
        filename = env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=env.task.activity_name,
            activity_definition_id=env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        path = get_task_instance_path(
            scene_model,
            f"{scene_model}_task_{env.task.activity_name}_instances/{filename}-tro_state",
            mode=str(self.cfg.mode),
        )
        if path is None:
            raise FileNotFoundError(
                f"Could not find 2026 {self.cfg.mode} task instance {instance_id} for {self.task_name}"
            )
        return path

    def _load_instance_state(self, slot: int, instance_id: int) -> None:
        env = self.envs[slot]
        robot = self.robots[slot]
        path = self._instance_path(env, instance_id)
        env.task.activity_instance_id = instance_id
        with open(path, "r") as file:
            state = recursively_convert_to_torch(json.load(file))

        robot_poses = state.pop("robot_poses", None)
        for key, entity_state in state.items():
            entity = env.task.object_scope.get(key)
            if entity is None:
                raise KeyError(f"Task object {key!r} from {path} is missing in vector slot {slot}")
            if slot != 0 and isinstance(entity_state, dict) and isinstance(entity_state.get("root_link"), dict):
                root_link = dict(entity_state["root_link"])
                if "pos" in root_link and "ori" in root_link:
                    position, orientation = env.scene.convert_scene_relative_pose_to_world(
                        root_link["pos"], root_link["ori"]
                    )
                    entity_state = dict(entity_state)
                    root_link["pos"], root_link["ori"] = position, orientation
                    entity_state["root_link"] = root_link
            entity.load_state(entity_state, serialized=False)

        if robot_poses is None:
            raise KeyError(f"robot_poses is missing in {path}")
        world_robot_poses = _robot_poses_in_world(env.scene, robot_poses)
        normalized_poses = {name.lower(): poses for name, poses in world_robot_poses.items()}
        if "robot" in normalized_poses:
            available_poses = normalized_poses["robot"]
        elif robot.model in normalized_poses:
            available_poses = normalized_poses[robot.model]
        else:
            raise KeyError(f"No generic or model-specific robot pose for {robot.model} in {path}")
        robot_pose = available_poses[0]
        robot.set_position_orientation(robot_pose["position"], robot_pose["orientation"])
        _sync_robot_after_pose_override(robot)
        env.scene.write_task_metadata(key="robot_poses", data=world_robot_poses)

        if self.should_sync_lights:
            set_light_control_toggles(env.task.object_scope.values(), True)

    def _reset_light_synchronizer(self, slot: int) -> None:
        if not self.should_sync_lights:
            self.light_synchronizers[slot] = None
            return
        synchronizer = LightToggleSynchronizer(self.envs[slot].scene)
        synchronizer.reset_from_current_state()
        self.light_synchronizers[slot] = synchronizer

    def _sync_lights(self, slots: list[int]) -> list[dict] | None:
        if not self.should_sync_lights:
            return None
        for slot in slots:
            self.light_synchronizers[slot].sync_from_current_state()
        for _ in range(3):
            og.sim.render()
        return [self.envs[slot].get_obs()[0] for slot in slots]

    def _activate_slots(self, slots: list[int]) -> None:
        for slot in slots:
            robot = self.robots[slot]
            robot.control_enabled = True
            robot.wake()

    def _park_slot(self, slot: int) -> None:
        """Stop an inactive robot from reusing its last controller goal in later global simulator steps."""
        # SPEEDUP_EVAL: a finished slot remains in the global scene list for stable
        # PhysX indices, so explicitly zero and sleep it while other slots continue.
        robot = self.robots[slot]
        positions = robot.get_joint_positions().clone()
        if not bool(th.isfinite(positions).all()):
            raise RuntimeError(f"Cannot park vector slot {slot}: robot joint positions are non-finite")
        zeros = th.zeros_like(positions)

        # Disable batched controller and assisted-grasp updates, clear its stored goals at the current configuration,
        # and explicitly zero both state and drive targets before sleeping the articulation.
        robot.control_enabled = False
        robot.set_joint_positions(positions, drive=False)
        robot.set_joint_velocities(zeros, drive=False)
        robot.set_joint_positions(positions, drive=True)
        robot.set_joint_velocities(zeros, drive=True)
        robot.set_joint_efforts(zeros)
        robot.keep_still()
        robot.sleep()

    def _load_group(self, instance_ids: list[int], rollout_id: int) -> dict[int, dict]:
        # SPEEDUP_EVAL: reset/load a pair as one synchronized group. This avoids
        # advancing one slot while another slot is being restored.
        slots = list(range(len(instance_ids)))
        logger.info(
            "Resetting vector group RNGs: instances=%s slot_seeds=%s",
            instance_ids,
            [self.env_seeds[slot] for slot in slots],
        )
        self._activate_slots(slots)
        for inactive_slot in sorted(set(range(self.num_envs)) - set(slots)):
            self._park_slot(inactive_slot)
        # Match the baseline evaluator's reset before each TRO load. Restore both scenes before a shared physics / render
        # step so the first slot is not advanced again while the second slot resets.
        self.vector_env.reset_synchronized(env_indices=slots, get_obs=True)
        for slot in slots:
            self._reset_light_synchronizer(slot)
        self._sync_lights(slots)
        for slot, instance_id in enumerate(instance_ids):
            self._load_instance_state(slot, instance_id)

        og.sim.update_handles()
        for _ in range(25):
            og.sim.step_physics()
            for slot in slots:
                for instance, entity in self.envs[slot].task.object_scope.items():
                    if not is_system_bddl_inst(instance) and entity is not None:
                        entity.keep_still()

        for slot in slots:
            self.envs[slot].scene.update_initial_file()
        # Match Evaluator.load_task_instance(): restore the freshly snapshotted states and take one shared physics step.
        self.vector_env.reset_scenes_synchronized(env_indices=slots)
        for slot in slots:
            self._reset_light_synchronizer(slot)

        # Match the rollout Evaluator.reset(): a second scene restore / physics step, task bookkeeping reset, one normal
        # rendered simulator step, three render flushes, and one observation read per slot.
        raw_observations, _ = self.vector_env.reset_synchronized(env_indices=slots, get_obs=True)
        for slot in slots:
            self._reset_light_synchronizer(slot)
        synced_obs = self._sync_lights(slots)
        if synced_obs is not None:
            raw_observations = synced_obs

        self.policy.reset()
        records = {}
        for slot, (instance_id, raw_obs) in enumerate(zip(instance_ids, raw_observations)):
            self.postprocessors[slot].reset(task_id=self.task_id)
            metrics = [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]
            for metric in metrics:
                metric.reset(self.envs[slot])
            records[slot] = {
                "instance_id": int(instance_id),
                "rollout_id": int(rollout_id),
                "obs": self._preprocess_obs(slot, raw_obs),
                "metrics": metrics,
                "final_info": None,
            }
        return records

    def _preprocess_obs(self, slot: int, obs: dict) -> dict:
        obs = flatten_obs_dict(obs)
        robot = self.robots[slot]
        base_pose = robot.get_position_orientation()
        camera_poses = []
        for camera_name in self.robot_camera_names.values():
            sensor_name = camera_name.split("::", 1)[1]
            camera = robot.sensors[sensor_name]
            direct_camera_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_camera_pose, np.zeros(16)):
                camera_pose = camera.get_position_orientation()
            else:
                camera_pose = T.mat2pose(
                    th.tensor(
                        np.linalg.inv(np.reshape(direct_camera_pose, (4, 4)).T),
                        dtype=th.float32,
                    )
                )
            camera_poses.append(th.cat(T.relative_pose_transform(*camera_pose, *base_pose)))
        obs[f"{robot.name}::cam_rel_poses"] = th.cat(camera_poses, dim=-1)
        proprio = obs[f"{robot.name}::proprio"]
        # SPEEDUP_EVAL: adapt the official 2026 observation layout to the legacy
        # champion checkpoint while keeping the controller action frame separate.
        proprio = _select_policy_base_qvel(
            proprio,
            robot.get_joint_velocities()[robot.base_control_idx],
            str(self.cfg.base_velocity_frame),
        )
        if str(self.cfg.proprioception_schema) == "r1pro_v3_61":
            if proprio.shape[-1] != _R1PRO_2026_PROPRIO_DIM:
                raise ValueError(
                    f"Expected official 2026 {_R1PRO_2026_PROPRIO_DIM}-D proprio, got {tuple(proprio.shape)}"
                )
        elif str(self.cfg.proprioception_schema) == "r1pro_v2_256":
            proprio = _adapt_proprio_for_pi05(proprio)
        else:
            raise ValueError(f"Unknown PI0.5 proprioception schema: {self.cfg.proprioception_schema}")
        obs[f"{robot.name}::proprio"] = proprio
        obs["task_id"] = th.tensor([self.task_id], dtype=th.int64)
        return obs

    def _policy_obs(self, slot: int, obs: dict) -> dict:
        # SPEEDUP_EVAL: carry stage and the retained action prefix in the request;
        # these fields let the persistent server reproduce the champion protocol.
        request = dict(obs)
        env_step = int(self.envs[slot]._current_step)
        request["env_step"] = env_step
        request["episode_step"] = env_step
        processor = self.postprocessors[slot]
        request["current_stage"] = th.tensor([processor.current_stage], dtype=th.int64)
        if (
            self.postprocessor_config.enable_action_chunk_maintenance
            and processor.next_initial_actions is not None
        ):
            request["initial_actions"] = th.from_numpy(processor.next_initial_actions.copy()).to(th.float32)
        return request

    def _infer_chunks(self, active_slots: list[int], records: dict[int, dict]) -> dict[int, np.ndarray]:
        if not active_slots:
            return {}
        # SPEEDUP_EVAL: one observation_batch produces one model batch (normally
        # batch-2, or batch-1 after a slot terminates).
        requests = [self._policy_obs(slot, records[slot]["obs"]) for slot in active_slots]
        states = [
            _extract_pi05_state(
                records[slot]["obs"][f"{self.robots[slot].name}::proprio"].detach().cpu().numpy().copy()
            )
            for slot in active_slots
        ]
        response = self.policy.infer(_snapshot_policy_value({"observation_batch": requests}))
        action_chunks = np.asarray(response.get("action_chunk"))
        logits = np.asarray(response.get("subtask_logits"))
        batch_size = len(active_slots)
        expected_prefix = (batch_size, self.postprocessor_config.action_horizon)
        if action_chunks.shape[:2] != expected_prefix:
            raise RuntimeError(f"Unexpected action_chunk shape {action_chunks.shape}; expected {expected_prefix} + D")
        if logits.ndim != 2 or logits.shape[0] != batch_size:
            raise RuntimeError(f"Unexpected subtask_logits shape: {logits.shape}; expected batch {batch_size}")
        return {
            slot: self.postprocessors[slot].process(action_chunks[index], logits[index], states[index])
            for index, slot in enumerate(active_slots)
        }

    def _write_video(self, slot: int, obs: dict) -> None:
        if self.video_paths[slot] is None:
            return
        left = cv2.resize(
            obs[self.robot_camera_names["left_wrist"] + "::rgb"].detach().cpu().numpy(), (224, 224)
        )
        right = cv2.resize(
            obs[self.robot_camera_names["right_wrist"] + "::rgb"].detach().cpu().numpy(), (224, 224)
        )
        head = cv2.resize(obs[self.robot_camera_names["head"] + "::rgb"].detach().cpu().numpy(), (448, 448))
        frame = np.expand_dims(np.hstack([np.vstack([left, right]), head]), 0)
        if self.video_writers[slot] is None:
            self.video_writers[slot] = create_video_writer(
                fpath=str(self.video_paths[slot]), resolution=frame.shape[1:3], rate=int(self.cfg.video_fps)
            )
        write_video(frame, video_writer=self.video_writers[slot], batch_size=1, mode="rgb")

    def _finish_slot(self, slot: int, record: dict, json_dir: Path) -> dict:
        env = self.envs[slot]
        success = bool(env.task.success)
        result = {
            "task": self.task_name,
            "instance_id": record["instance_id"],
            "rollout_id": record["rollout_id"],
            "steps": int(env._current_step),
            "success": success,
        }
        for metric in record["metrics"]:
            result.update(metric.aggregate(env))
        output_path = json_dir / f"{self.task_name}_{record['instance_id']}_{record['rollout_id']}.json"
        _atomic_json_dump(result, output_path, indent=2, default=float, allow_nan=False)
        _close_video_writer(self.video_writers[slot])
        self.video_writers[slot] = None
        self.video_paths[slot] = None
        logger.info(
            "Result: instance=%s slot=%s steps=%s success=%s q_score=%s -> %s",
            record["instance_id"],
            slot,
            result["steps"],
            success,
            result.get("q_score", {}).get("final"),
            output_path,
        )
        return result

    def _execute_chunks(
        self,
        execution_chunks: dict[int, np.ndarray],
        records: dict[int, dict],
        active: set[int],
        json_dir: Path,
    ) -> list[dict]:
        completed = []
        latest_observations = {}
        max_actions = max(len(chunk) for chunk in execution_chunks.values())
        request_slots = sorted(execution_chunks)

        # SPEEDUP_EVAL: advance all still-active slots together, preserving the
        # official render/observation/reward/termination path after every action.
        for action_index in range(max_actions):
            step_slots = [
                slot for slot in request_slots if slot in active and action_index < len(execution_chunks[slot])
            ]
            if not step_slots:
                break
            actions = []
            for slot in step_slots:
                action = th.as_tensor(execution_chunks[slot][action_index], dtype=th.float32)
                # PI0.5 base actions follow the R1Pro controller convention and
                # are always robot-local [vx, vy, wz]. base_velocity_frame only
                # selects the base qvel representation exposed in observations.
                actions.append(action)
            # The fast four-instance profile skips rendering only for action
            # chunk steps 2..10 (1-based). Chunk boundaries and all other steps
            # retain the normal rendered simulator path.
            skip_chunk_rendering = getattr(self, "skip_action_chunk_rendering", False)
            render_step = not (skip_chunk_rendering and 1 <= action_index <= 9)
            if skip_chunk_rendering:
                observations, rewards, terminated, truncated, infos = self.vector_env.step(
                    actions, env_indices=step_slots, render=render_step
                )
            else:
                # Keep the baseline call shape for custom/test vector environments that
                # predate the optional render keyword.
                observations, rewards, terminated, truncated, infos = self.vector_env.step(
                    actions, env_indices=step_slots
                )
            synced_obs = self._sync_lights(step_slots) if render_step else None
            if synced_obs is not None:
                observations = synced_obs

            for slot, action, obs, reward, is_terminated, is_truncated, info in zip(
                step_slots, actions, observations, rewards, terminated, truncated, infos
            ):
                latest_observations[slot] = obs
                self.postprocessors[slot].record_executed_actions()
                for metric in records[slot]["metrics"]:
                    metric.step(env=self.envs[slot], action=action, obs=obs, reward=reward, terminated=is_terminated,
                                truncated=is_truncated, info=info)
                if bool(self.cfg.write_video):
                    self._write_video(slot, flatten_obs_dict(obs))
                if is_terminated or is_truncated:
                    records[slot]["final_info"] = info
                    active.remove(slot)
                    # Aggregate immediately. Other active slots continue to advance the shared
                    # simulator, which must not change this slot's terminal predicate state.
                    completed.append(self._finish_slot(slot, records[slot], json_dir))
                    self._park_slot(slot)

        for slot in request_slots:
            if slot not in active or slot not in latest_observations:
                continue
            records[slot]["obs"] = self._preprocess_obs(slot, latest_observations[slot])
        return completed

    def _run_group(
        self,
        instance_ids: list[int],
        rollout_id: int,
        json_dir: Path,
        video_dir: Path,
    ) -> list[dict]:
        records = self._load_group(instance_ids, rollout_id=rollout_id)
        active = set(records)
        for slot, record in records.items():
            if bool(self.cfg.write_video):
                self.video_paths[slot] = video_dir / (
                    f"{self.task_name}_{record['instance_id']}_{record['rollout_id']}.mp4"
                )
            logger.info("Starting instance=%s rollout=%s in vector slot=%s", record["instance_id"], rollout_id, slot)

        results = []
        while active:
            execution_chunks = self._infer_chunks(sorted(active), records)
            results.extend(self._execute_chunks(execution_chunks, records, active, json_dir))
        return results

    def run(self, instance_ids: list[int], rollout_id: int = 0) -> list[dict]:
        root = Path(str(self.cfg.output_dir))
        json_dir = root / "json"
        video_dir = root / "videos"
        json_dir.mkdir(parents=True, exist_ok=True)
        if bool(self.cfg.write_video):
            video_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for start in range(0, len(instance_ids), self.num_envs):
            group = instance_ids[start : start + self.num_envs]
            logger.info("Loading vector instance group: %s", group)
            results.extend(self._run_group(group, rollout_id, json_dir, video_dir))

        expected_keys = {(int(instance_id), int(rollout_id)) for instance_id in instance_ids}
        actual_keys = {(int(result["instance_id"]), int(result["rollout_id"])) for result in results}
        if len(results) != len(expected_keys) or actual_keys != expected_keys:
            raise RuntimeError(
                "Vector evaluation did not produce the exact requested rollout set: "
                f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}, result_count={len(results)}"
            )

        completion_path = root / "evaluation_complete.json"
        _atomic_json_dump(
            {
                "task": self.task_name,
                "task_id": self.task_id,
                "mode": str(self.cfg.mode),
                "instance_ids": [int(instance_id) for instance_id in instance_ids],
                "num_rollouts": 1,
                "num_vector_envs": self.num_envs,
                "result_count": len(results),
                "write_video": bool(self.cfg.write_video),
            },
            completion_path,
            indent=2,
            allow_nan=False,
        )
        logger.info("Vector evaluation completed: %s", completion_path)
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in self.video_writers:
            _close_video_writer(writer)
        self.video_writers = [None] * self.num_envs
        self.vector_env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame) -> None:
        logger.warning("SIGINT or CTRL-C detected")
        self.close()
        raise SystemExit(130)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.close()
