"""Collect successful 2026 BEHAVIOR rollouts as NPZ + three RGB videos.

Requires the same raw-action-chunk websocket server as eval_vector. No dataset
conversion or training runs here. Each process collects one task in one or more
synchronized environments; --num-rollouts bounds attempts per instance.
"""

import argparse
import logging
import math
from pathlib import Path

from omnigibson.eval.eval_vector import build_parser
from omnigibson.eval.utils.cpu_utils import apply_cpu_config, configure_runtime_thread_pools, resolve_cpu_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(mode="train", output_dir="outputs/rft", num_rollouts=10, apply_eval_tricks=False)
    parser.add_argument(
        "--policy-checkpoint", required=True, help="Checkpoint identifier for provenance and resume checks."
    )
    parser.add_argument(
        "--sample-start", type=int, default=0, help="First sample ID; useful for separate collection batches."
    )
    parser.add_argument(
        "--successes-per-instance",
        type=int,
        default=0,
        help="Stop sampling each instance after this many successes (0: exhaust --num-rollouts).",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip complete samples in an existing compatible collection."
    )
    parser.add_argument("--perturb-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--perturb-translation", type=float, default=0.15, help="Maximum absolute local X/Y offset in metres."
    )
    parser.add_argument("--perturb-yaw-degrees", type=float, default=15.0, help="Maximum absolute initial yaw offset.")
    args = parser.parse_args(argv)
    if args.mode == "hidden_test":
        parser.error("RFT collection supports train and public_test instances only")
    if args.num_envs < 1 or args.num_rollouts < 1:
        parser.error("--num-envs and --num-rollouts must be positive")
    if args.sample_start < 0 or args.successes_per_instance < 0:
        parser.error("--sample-start and --successes-per-instance must be non-negative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if len(set(args.instance_indices)) != len(args.instance_indices) or any(i < 0 for i in args.instance_indices):
        parser.error("--instance-indices must be unique and non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.max_steps_multiplier) or args.max_steps_multiplier <= 0:
        parser.error("--max-steps-multiplier must be finite and positive")
    for name in ("perturb_translation", "perturb_yaw_degrees"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def main() -> None:
    args = parse_args()
    cpu_config = resolve_cpu_config(
        cpu_affinity=args.cpu_affinity,
        cpu_cores_per_env=args.cpu_cores_per_env,
        cpu_worker_index=args.cpu_worker_index,
        cpu_num_threads=args.cpu_num_threads,
    )
    apply_cpu_config(cpu_config)
    configure_runtime_thread_pools(cpu_config.num_threads)

    from omegaconf import OmegaConf

    import omnigibson as og
    from omnigibson.eval.evaluator import DEFAULT_ROBOT_CONFIG_PATH, resolve_instance_ids
    from omnigibson.eval.rft_evaluator import RFTVectorEvaluator
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES, seed_everything
    from omnigibson.eval.utils.pi05_action_chunk import B1KActionChunkConfig
    from omnigibson.eval.utils.rft_recorder import collection_directory
    from omnigibson.macros import gm

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if args.task_name not in TASK_NAMES_TO_INDICES:
        raise SystemExit(f"Unknown 2026 task: {args.task_name}")
    task_id = TASK_NAMES_TO_INDICES[args.task_name]
    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, args.mode)
    robot_config_path = Path(args.robot_config or DEFAULT_ROBOT_CONFIG_PATH).expanduser().resolve()
    robot_config = OmegaConf.to_container(OmegaConf.load(robot_config_path), resolve=True)
    action_chunk = {
        "actions_to_execute": args.actions_to_execute,
        "actions_to_keep": args.actions_to_keep,
        "execute_in_n_steps": args.execute_in_n_steps,
        "history_len": args.stage_history_len,
        "votes_to_promote": args.stage_votes_to_promote,
        "apply_eval_tricks": args.apply_eval_tricks,
        "enable_action_chunk_maintenance": args.action_chunk_maintenance,
        "enable_compression": args.compression,
        "action_horizon": args.action_horizon,
    }
    B1KActionChunkConfig(**action_chunk).validate()
    task_dir = Path(args.output_dir).expanduser().resolve() / f"task-{task_id:04d}_{args.task_name}"
    settings = {
        "task": {"name": args.task_name, "id": task_id},
        "instance_ids": instance_ids,
        "mode": args.mode,
        "seed": args.seed,
        "robot": robot_config,
        "policy_checkpoint": args.policy_checkpoint,
        "proprioception_schema": args.pi05_proprioception_schema,
        "base_velocity_frame": args.pi05_base_velocity_frame,
        "action_chunk": action_chunk,
        "max_steps": args.max_steps,
        "max_steps_multiplier": args.max_steps_multiplier,
        "partial_scene_load": args.partial_scene_load,
        "env_wrapper": args.env_wrapper,
        "perturb_pose": args.perturb_pose,
        "perturb_translation": args.perturb_translation,
        "perturb_yaw_degrees": args.perturb_yaw_degrees,
    }
    cfg = OmegaConf.create(
        {
            "task": settings["task"],
            "host": args.host,
            "port": args.port,
            "robot_config": str(robot_config_path),
            "mode": args.mode,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "max_steps_multiplier": args.max_steps_multiplier,
            "env_wrapper": {"_target_": args.env_wrapper},
            "output_dir": str(task_dir),
            "write_video": args.write_video,
            "video_fps": args.video_fps,
            "skip_action_chunk_rendering": False,
            "partial_scene_load": args.partial_scene_load,
            "proprioception_schema": args.pi05_proprioception_schema,
            "base_velocity_frame": args.pi05_base_velocity_frame,
            "seed": args.seed,
            "action_chunk": action_chunk,
            "rft": {
                "policy_checkpoint": args.policy_checkpoint,
                "perturb_pose": args.perturb_pose,
                "translation": args.perturb_translation,
                "yaw_degrees": args.perturb_yaw_degrees,
            },
        }
    )
    gm.HEADLESS = args.headless
    gm.RENDER_VIEWER_CAMERA = args.render_viewer_camera
    seed_everything(args.seed)
    with collection_directory(task_dir, settings, resume=args.resume):
        evaluator = None
        try:
            evaluator = RFTVectorEvaluator(cfg)
            with evaluator:
                summary = evaluator.collect(
                    instance_ids,
                    num_rollouts=args.num_rollouts,
                    sample_start=args.sample_start,
                    successes_per_instance=args.successes_per_instance,
                )
                logger.info("RFT collection finished: %s", summary)
        finally:
            # The context manager closes a constructed evaluator. Constructor failures
            # can still leave an Isaac Sim process without that context manager.
            if evaluator is None and og.sim is not None:
                og.shutdown()


if __name__ == "__main__":
    main()
