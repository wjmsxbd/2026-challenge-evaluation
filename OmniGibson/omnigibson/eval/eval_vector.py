"""Vectorized PI0.5 action-chunk evaluator for the 2026 BEHAVIOR Challenge.

This entry point keeps the official single-environment evaluator unchanged.
It runs two environments in one Isaac Sim process and sends their observations
together in one raw-action-chunk websocket request with policy batch size 2.
"""

import argparse
import logging
import os
from pathlib import Path

from omnigibson.eval.utils.cpu_utils import (
    apply_cpu_config,
    configure_runtime_thread_pools,
    format_cpu_affinity,
    resolve_cpu_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--robot-config", type=str, default=None)
    parser.add_argument("--instance-indices", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default="public_test")
    parser.add_argument("--num-rollouts", type=int, default=1)
    # SPEEDUP_EVAL: default to two simulator slots so one policy request can
    # serve both active rollouts on a GPU.
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Fixed environment RNG seed (default: 0).")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Absolute episode timeout in steps. When set, overrides --max-steps-multiplier.",
    )
    parser.add_argument(
        "--max-steps-multiplier",
        type=float,
        default=1.5,
        help="Episode timeout as a multiple of the mean human-demo length (default: 1.5).",
    )
    parser.add_argument("--env-wrapper", default="omnigibson.eval.wrappers.DefaultWrapper")
    parser.add_argument("--output-dir", default="/tmp/b1k_eval_vector")
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    # SPEEDUP_EVAL: task-relevant room loading is enabled by default to reduce
    # scene initialization and runtime object/render overhead.
    parser.add_argument("--partial-scene-load", action=argparse.BooleanOptionalAction, default=True)
    # SPEEDUP_EVAL: the viewer camera is not a policy input or submission video;
    # disabling it avoids an extra render product in headless throughput runs.
    parser.add_argument("--render-viewer-camera", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--actions-to-execute", type=int, default=26)
    parser.add_argument("--actions-to-keep", type=int, default=4)
    parser.add_argument("--execute-in-n-steps", type=int, default=20)
    parser.add_argument("--stage-history-len", type=int, default=3)
    parser.add_argument("--stage-votes-to-promote", type=int, default=2)
    parser.add_argument("--apply-eval-tricks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument(
        "--pi05-proprioception-schema",
        choices=("r1pro_v3_61", "r1pro_v2_256"),
        default="r1pro_v3_61",
    )
    parser.add_argument(
        "--pi05-base-velocity-frame",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Base qvel frame exposed to PI0.5: absolute uses legacy raw virtual-joint velocities, while "
            "relative uses robot-local velocities. Policy base actions are always robot-local."
        ),
    )

    parser.add_argument("--cpu-affinity", default=None)
    parser.add_argument("--cpu-cores-per-env", type=int, default=None)
    parser.add_argument("--cpu-worker-index", type=int, default=None)
    parser.add_argument("--cpu-num-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        cpu_config = resolve_cpu_config(
            cpu_affinity=args.cpu_affinity,
            cpu_cores_per_env=args.cpu_cores_per_env,
            cpu_worker_index=args.cpu_worker_index,
            cpu_num_threads=args.cpu_num_threads,
        )
        apply_cpu_config(cpu_config)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"CPU configuration error: {exc}") from exc

    configure_runtime_thread_pools(cpu_config.num_threads)

    from omegaconf import OmegaConf

    from omnigibson.eval.evaluator import resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES, seed_everything
    from omnigibson.eval.vector_evaluator import VectorChunkEvaluator
    from omnigibson.macros import gm
    from omnigibson.utils.ui_utils import create_module_logger

    logger = create_module_logger(module_name=__name__)
    logger.setLevel(logging.INFO)

    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")
    if args.num_rollouts != 1:
        raise SystemExit("The 2026 challenge protocol requires --num-rollouts 1")
    if not 0 <= args.seed < 2**32:
        raise SystemExit("--seed must be in [0, 2**32)")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.max_steps_multiplier <= 0:
        raise SystemExit("--max-steps-multiplier must be positive")
    if args.mode == "hidden_test":
        raise SystemExit("Hidden 2026 task instances are reserved for organizer-run final evaluation")

    task_id = TASK_NAMES_TO_INDICES.get(args.task_name)
    if task_id is None:
        raise SystemExit(f"Unknown 2026 challenge task: {args.task_name}")
    gm.HEADLESS = args.headless
    gm.RENDER_VIEWER_CAMERA = args.render_viewer_camera
    # SPEEDUP_EVAL: seed Python/NumPy/Torch/CUDA/Warp before constructing or
    # resetting environments so paired slots are reproducible.
    seed = seed_everything(args.seed)
    logger.info("Seeded environment Python, NumPy, Torch, CUDA, and Warp RNGs with seed=%s", seed)
    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)

    logger.info(
        "Evaluator resources: pid=%s worker_index=%s cpus=%s num_threads=%s CUDA_VISIBLE_DEVICES=%s",
        os.getpid(),
        cpu_config.worker_index,
        format_cpu_affinity(os.sched_getaffinity(0)),
        cpu_config.num_threads,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    logger.info(
        "2026 protocol: task_id=%s task=%s split=%s indices=%s resolved_instance_ids=%s rollouts=1",
        task_id,
        args.task_name,
        args.mode,
        args.instance_indices,
        instance_ids,
    )

    cfg = OmegaConf.create(
        {
            "task": {"name": args.task_name, "id": task_id},
            "host": args.host,
            "port": args.port,
            "robot_config": str(Path(args.robot_config).expanduser()) if args.robot_config else None,
            "mode": args.mode,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "max_steps_multiplier": args.max_steps_multiplier,
            "env_wrapper": {"_target_": args.env_wrapper},
            "output_dir": str(Path(args.output_dir).expanduser()),
            "write_video": args.write_video,
            "video_fps": args.video_fps,
            "partial_scene_load": args.partial_scene_load,
            "proprioception_schema": args.pi05_proprioception_schema,
            "base_velocity_frame": args.pi05_base_velocity_frame,
            "seed": seed,
            "action_chunk": {
                "actions_to_execute": args.actions_to_execute,
                "actions_to_keep": args.actions_to_keep,
                "execute_in_n_steps": args.execute_in_n_steps,
                "history_len": args.stage_history_len,
                "votes_to_promote": args.stage_votes_to_promote,
                "apply_eval_tricks": args.apply_eval_tricks,
                "action_horizon": args.action_horizon,
            },
        }
    )

    with VectorChunkEvaluator(cfg) as evaluator:
        evaluator.run([int(instance_id) for instance_id in instance_ids], rollout_id=0)


if __name__ == "__main__":
    main()
