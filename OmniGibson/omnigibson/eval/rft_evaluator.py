"""Successful-trajectory collection using the existing vector action-chunk evaluator."""

import copy
import logging
from pathlib import Path

from omnigibson.eval.utils.eval_utils import flatten_obs_dict
from omnigibson.eval.utils.rft_pose_perturbator import perturb_pose, sample_seed
from omnigibson.eval.utils.rft_recorder import (
    CAMERA_ROLES,
    RFTRecorder,
    completed_episode,
    episode_path,
    write_json,
)
from omnigibson.eval.vector_evaluator import VectorChunkEvaluator, _sync_robot_after_pose_override

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RFTVectorEvaluator(VectorChunkEvaluator):
    """Keep native per-step observations separate from policy chunk-boundary observations."""

    def __init__(self, cfg):
        if cfg.get("skip_action_chunk_rendering", False):
            raise ValueError("RFT collection requires rendering every action step")
        self.recorders = {}
        self.sample_contexts = {}
        super().__init__(cfg)
        self.server_metadata = self.policy.get_server_metadata()
        frequencies = self.envs[0].env_config
        self.capture_fps = int(frequencies["action_frequency"])
        if (
            self.capture_fps != frequencies["rendering_frequency"]
            or self.capture_fps != frequencies["action_frequency"]
        ):
            raise ValueError("RFT capture requires equal, integer action and rendering frequencies")

    def _load_group(self, instance_ids: list[int], rollout_id: int) -> dict[int, dict]:
        if self.recorders:
            raise RuntimeError("Previous RFT group still has open recordings")
        self.sample_contexts = {}
        for slot, instance_id in enumerate(instance_ids):
            seed = sample_seed(int(self.cfg.seed), self.task_id, instance_id, rollout_id)
            self.env_seeds[slot] = seed
            self.sample_contexts[slot] = {
                "task": self.task_name,
                "task_id": self.task_id,
                "instance_id": instance_id,
                "sample_id": rollout_id,
                "mode": str(self.cfg.mode),
                "environment_seed": seed,
            }
        self.vector_env.seeds = list(self.env_seeds)
        return super()._load_group(instance_ids, rollout_id)

    def _load_instance_state(self, slot: int, instance_id: int) -> None:
        super()._load_instance_state(slot, instance_id)
        env, robot = self.envs[slot], self.robots[slot]
        # Start from the freshly loaded TRO every time. Update the stored pose before
        # _load_group snapshots / resets the scene so the perturbation survives reset.
        poses = copy.deepcopy(env.scene.get_task_metadata("robot_poses"))
        keys = {key.lower(): key for key in poses}
        key = keys["robot"] if "robot" in keys else keys[robot.model]
        pose = poses[key][0]
        position, orientation, perturbation = perturb_pose(
            pose["position"],
            pose["orientation"],
            seed=self.sample_contexts[slot]["environment_seed"],
            translation=float(self.cfg.rft.translation) if self.cfg.rft.perturb_pose else 0.0,
            yaw_degrees=float(self.cfg.rft.yaw_degrees) if self.cfg.rft.perturb_pose else 0.0,
        )
        pose["position"], pose["orientation"] = position, orientation
        robot.set_position_orientation(position, orientation)
        _sync_robot_after_pose_override(robot)
        env.scene.write_task_metadata("robot_poses", poses)
        self.sample_contexts[slot]["pose_perturbation"] = perturbation
        self.sample_contexts[slot]["scene"] = env.task.scene_name

    def _snapshot_observation(self, slot: int, obs: dict) -> None:
        flat_obs = flatten_obs_dict(obs)
        robot = self.robots[slot]
        self.recorders[slot].set_observation(
            flat_obs[f"{robot.name}::proprio"],
            {role: flat_obs[self.robot_camera_names[role] + "::rgb"] for role in CAMERA_ROLES},
            base_qpos=robot.get_joint_positions()[robot.base_control_idx],
            base_qvel=robot.get_joint_velocities()[robot.base_control_idx],
        )

    def _on_rollout_start(self, slot: int, record: dict, obs: dict) -> None:
        metadata = {
            **self.sample_contexts[slot],
            "policy_checkpoint": str(self.cfg.rft.policy_checkpoint),
            "policy_checkpoint_source": "collector_argument",
            "policy_proprioception_schema": str(self.cfg.proprioception_schema),
            "policy_base_velocity_frame": str(self.cfg.base_velocity_frame),
            "server_metadata": self.server_metadata,
        }
        path = episode_path(Path(str(self.cfg.output_dir)), record["instance_id"], record["rollout_id"])
        self.recorders[slot] = RFTRecorder(path, metadata, fps=self.capture_fps)
        self._snapshot_observation(slot, obs)

    def _before_action_step(self, slots: list[int], actions: list, records: dict[int, dict]) -> None:
        for slot, action in zip(slots, actions):
            self.recorders[slot].before_step(action)

    def _after_action_step(self, slot: int, obs: dict, reward, terminated, truncated, info: dict) -> None:
        self.recorders[slot].after_step(reward, terminated, truncated)
        if not (terminated or truncated):
            self._snapshot_observation(slot, obs)

    def _on_rollout_end(self, slot: int, record: dict, result: dict) -> None:
        recorder = self.recorders[slot]
        recorder.finish(result)
        del self.recorders[slot]
        result["trajectory_dir"] = str(recorder.path)
        if result["success"]:
            logger.info(
                "RFT success: instance=%s sample=%s slot=%s steps=%s -> %s",
                record["instance_id"],
                record["rollout_id"],
                slot,
                result["steps"],
                recorder.path,
            )

    def collect(
        self, instance_ids: list[int], *, num_rollouts: int, sample_start: int, successes_per_instance: int
    ) -> dict:
        """Run a bounded sample range, skipping committed samples and satisfied instance quotas."""
        root = Path(str(self.cfg.output_dir))
        metrics_dir, video_dir = root / "metrics", root / "videos"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.write_video:
            video_dir.mkdir(parents=True, exist_ok=True)
        successes = {instance_id: 0 for instance_id in instance_ids}
        completed = {}
        sample_ids = range(sample_start, sample_start + num_rollouts)
        # Count all committed samples in this invocation's range before testing quotas.
        for instance_id in instance_ids:
            for sample_id in sample_ids:
                result = completed_episode(
                    episode_path(root, instance_id, sample_id),
                    expected_identity={"task_id": self.task_id, "instance_id": instance_id, "sample_id": sample_id},
                )
                if result is not None:
                    completed[instance_id, sample_id] = result
                    successes[instance_id] += int(result["success"])

        for sample_id in sample_ids:
            pending = [
                instance_id
                for instance_id in instance_ids
                if (instance_id, sample_id) not in completed
                and (not successes_per_instance or successes[instance_id] < successes_per_instance)
            ]
            for start in range(0, len(pending), self.num_envs):
                group = pending[start : start + self.num_envs]
                results = self._run_group(group, sample_id, metrics_dir, video_dir)
                for result in results:
                    instance_id = result["instance_id"]
                    completed[instance_id, sample_id] = result
                    successes[instance_id] += int(result["success"])
                self._write_summary(
                    root, instance_ids, sample_start, num_rollouts, successes_per_instance, completed, successes
                )
        return self._write_summary(
            root, instance_ids, sample_start, num_rollouts, successes_per_instance, completed, successes
        )

    def _write_summary(self, root, instance_ids, sample_start, num_rollouts, target, completed, successes):
        summary = {
            "task": self.task_name,
            "task_id": self.task_id,
            "mode": str(self.cfg.mode),
            "instance_ids": instance_ids,
            "sample_start": sample_start,
            "max_rollouts_per_instance": num_rollouts,
            "successes_per_instance": target,
            "completed_rollouts": len(completed),
            "successful_rollouts": sum(successes.values()),
            "successes_by_instance": successes,
            "target_reached": all(count >= target for count in successes.values()) if target else None,
        }
        write_json(root / "collection_summary.json", summary)
        return summary

    def close(self) -> None:
        for recorder in list(self.recorders.values()):
            try:
                recorder.abort()
            except Exception:
                logger.exception("Failed to clean up unpublished RFT episode %s", recorder.staging)
        self.recorders.clear()
        super().close()
