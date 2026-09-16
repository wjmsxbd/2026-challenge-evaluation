#!/usr/bin/env bash
set -euo pipefail

# Load the Torch checkpoint with the persistent scheduler.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
export OLD_SCHEDULER="${BEHAVIOR_ROOT}/run_pi05_behavior_2026_eval_persistent.sh"
export PI_BEHAVIOR_TORCH_CHECKPOINT="${PI_BEHAVIOR_TORCH_CHECKPOINT:-/mnt/data/ckpt/b1k/pi_behavior_b1k_2026/pretrain_champion_2026_torch_pretrain_bs64_global2048_w_oema_bf16_0907_v2/checkpoints/steps_80000_model.safetensors}"
export PI_BEHAVIOR_CONFIG_YAML="${PI_BEHAVIOR_CONFIG_YAML:-${PI_BEHAVIOR_TORCH_CHECKPOINT%/checkpoints/*}/config.full.yaml}"
export BEHAVIOR_ENV_DIR="${BEHAVIOR_ENV_DIR:-/mnt/data_nas/wangjm/miniconda3/envs/behavior_2026}"
PRETRAIN_EVAL_LAUNCHER="${PRETRAIN_EVAL_LAUNCHER:-${SCRIPT_DIR}/../pretrain/launch/pi_behavior/run_eval_2026_torch.sh}"
[[ -f "${PRETRAIN_EVAL_LAUNCHER}" ]] || {
  echo "Pretrain launcher not found: ${PRETRAIN_EVAL_LAUNCHER}" >&2
  exit 1
}
exec bash "${PRETRAIN_EVAL_LAUNCHER}" "$@"
