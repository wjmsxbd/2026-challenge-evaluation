#!/usr/bin/env bash
set -euo pipefail

# Task ID 0 (turning_on_radio), public instances 0-3, one GPU with two synchronized environments.
# Keep the 43-D joint+EEF model, but execute base velocity + torso joints + arm EEF IK.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
export PI_BEHAVIOR_TORCH_CHECKPOINT="${PI_BEHAVIOR_TORCH_CHECKPOINT:-/mnt/data/oss/moqi-sh-oss/wangjm/ckpt/b1k/pi_behavior_b1k_2026/pi_behavior_eef_both/checkpoints/steps_16000_model.safetensors}"
export PI_BEHAVIOR_CONFIG_YAML="${PI_BEHAVIOR_CONFIG_YAML:-${PI_BEHAVIOR_TORCH_CHECKPOINT%/checkpoints/*}/config.full.yaml}"
# EEF-both uses 43-D statistics matching the checkpoint's saved training config.
export PI05_ASSETS_ROOT="${PI05_ASSETS_ROOT:-${SCRIPT_DIR}/../pretrain/outputs/assets/pi_behavior_eef/both}"
export PI05_NORM_STATS_PATH="${PI05_NORM_STATS_PATH:-${PI05_ASSETS_ROOT}/behavior-1k/2026-challenge-demos/norm_stats.json}"
export PI05_SERVER_SCRIPT="${PI05_SERVER_SCRIPT:-${SCRIPT_DIR}/../pretrain/launch/pi_behavior/serve_pi_behavior_2026_vector_torch_hybrid_eef.py}"
export BEHAVIOR_ENV_DIR="${BEHAVIOR_ENV_DIR:-/mnt/data/nas/F_zone_nas/wangjm/miniconda3/envs/behavior_2026}"
# The relocated NAS Conda installation still embeds its old mount prefix.
# Use the local Conda installation to activate the existing NAS environment.
export BEHAVIOR_CONDA_SH="${BEHAVIOR_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

# Robot-local [vx, vy, wz], including yaw velocity; there is no absolute yaw
# angle in the 61-D policy observation. Base actions are already robot-local.
export PI05_BASE_VELOCITY_FRAME=relative
export PI05_ROBOT_CONFIG="${PI05_ROBOT_CONFIG:-${SCRIPT_DIR}/OmniGibson/omnigibson/eval/r1pro_hybrid_eef.yaml}"
export TASK_IDS='0'
export TASK_LIMIT=1
export EVAL_INSTANCE_INDICES='0 1 2 3'
export VECTOR_ENVS_PER_PROCESS=2
export INSTANCE_CHUNK_SIZE=2
export PI05_DYNAMIC_BATCH_MAX_SIZE=2
export PI05_DYNAMIC_BATCH_GRANULARITY=1
export NUM_GPUS=1
export GPU_IDS='0'
export EVAL_MAX_STEPS_MULTIPLIER=1.2
# An inherited absolute step limit would override the multiplier.
unset EVAL_MAX_STEPS
# The first hybrid-control validation deliberately avoids 23-D joint-specific
# correction rules and temporal operations that do not preserve SO(3).
export PI05_APPLY_EVAL_TRICKS=false
export PI05_ACTION_CHUNK_MAINTENANCE=false
export PI05_COMPRESSION=false
export PI05_ACTIONS_TO_EXECUTE=20
export PI05_ACTIONS_TO_KEEP=0
export PI05_EXECUTE_IN_N_STEPS=20
# Render every action step and save every resulting frame to the evaluation video.
export PI05_SKIP_ACTION_CHUNK_RENDERING=false
export EVAL_WRITE_VIDEO=true
export PRETRAIN_EVAL_LAUNCHER="${PRETRAIN_EVAL_LAUNCHER:-${SCRIPT_DIR}/../pretrain/launch/pi_behavior/run_eval_2026_torch.sh}"
exec bash "${SCRIPT_DIR}/run_eval_2026_persistent.sh" "$@"
