#!/usr/bin/env bash
set -euo pipefail

# Tasks 50-99, public instances 0-3, eight GPUs with two synchronized environments per GPU.
# Execute the 43-D joint+EEF checkpoint through its original 23-D joint-control path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
export PI_BEHAVIOR_TORCH_CHECKPOINT="${PI_BEHAVIOR_TORCH_CHECKPOINT:-/mnt/data/oss/moqi-sh-oss/wangjm/ckpt/b1k/pi_behavior_b1k_2026/pi_behavior_eef_both/checkpoints/steps_54000_model.safetensors}"
export PI_BEHAVIOR_CONFIG_YAML="${PI_BEHAVIOR_CONFIG_YAML:-${PI_BEHAVIOR_TORCH_CHECKPOINT%/checkpoints/*}/config.full.yaml}"
# The checkpoint was trained with joint+EEF supervision, so retain its 43-D statistics even though only 23-D
# physical joint actions are executed by the original server.
export PI05_ASSETS_ROOT="${PI05_ASSETS_ROOT:-${SCRIPT_DIR}/../pretrain/outputs/assets/pi_behavior_eef/both}"
export PI05_NORM_STATS_PATH="${PI05_NORM_STATS_PATH:-${PI05_ASSETS_ROOT}/behavior-1k/2026-challenge-demos/norm_stats.json}"
export PI05_SERVER_SCRIPT="${PI05_SERVER_SCRIPT:-${SCRIPT_DIR}/../pretrain/launch/pi_behavior/serve_pi_behavior_2026_vector_torch.py}"
export PI05_ROBOT_CONFIG="${PI05_ROBOT_CONFIG:-${SCRIPT_DIR}/OmniGibson/omnigibson/eval/r1pro.yaml}"
export BEHAVIOR_ENV_DIR="${BEHAVIOR_ENV_DIR:-/mnt/data/nas/F_zone_nas/wangjm/miniconda3/envs/behavior_2026}"
# The relocated NAS Conda installation still embeds its old mount prefix.
# Use the local Conda installation to activate the existing NAS environment.
export BEHAVIOR_CONDA_SH="${BEHAVIOR_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

# Robot-local [vx, vy, wz], including yaw velocity.
export PI05_BASE_VELOCITY_FRAME=relative
printf -v TASK_IDS '%s ' {50..99}
export TASK_IDS
export TASK_LIMIT=50
export EVAL_INSTANCE_INDICES='0 1 2 3'
export VECTOR_ENVS_PER_PROCESS=2
export INSTANCE_CHUNK_SIZE=2
export PI05_DYNAMIC_BATCH_MAX_SIZE=2
export PI05_DYNAMIC_BATCH_GRANULARITY=1
export NUM_GPUS=8
export GPU_IDS='0 1 2 3 4 5 6 7'
export EVAL_MAX_STEPS_MULTIPLIER=1.2
# An inherited absolute step limit would override the multiplier.
unset EVAL_MAX_STEPS

# Restore the champion joint-action postprocessing protocol: reference corrections,
# action-tail inpainting, and 26-to-20 temporal compression.
export PI05_APPLY_EVAL_TRICKS=true
export PI05_ACTION_CHUNK_MAINTENANCE=true
export PI05_COMPRESSION=true
export PI05_ACTIONS_TO_EXECUTE=26
export PI05_ACTIONS_TO_KEEP=4
export PI05_EXECUTE_IN_N_STEPS=20

# Render and save every executed simulator action frame.
export PI05_SKIP_ACTION_CHUNK_RENDERING=false
export EVAL_WRITE_VIDEO=true
export PRETRAIN_EVAL_LAUNCHER="${PRETRAIN_EVAL_LAUNCHER:-${SCRIPT_DIR}/../pretrain/launch/pi_behavior/run_eval_2026_torch.sh}"
exec bash "${SCRIPT_DIR}/run_eval_2026_persistent.sh" "$@"
