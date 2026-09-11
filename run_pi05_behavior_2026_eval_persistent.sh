#!/usr/bin/env bash
set -euo pipefail

# Persistent Isaac evaluation with a rank-0 dynamic scheduler for PAI DLC.
# Run this launcher once per node; each GPU keeps its policy and Isaac processes.
#
# Default protocol:
#   - 100 official tasks (2026 task IDs 0-99)
#   - public instance indices 0-9
#   - one rollout per instance
#   - official 120/30/30 Hz dynamics and task-specific 1.5x human timeout
#   - one persistent policy server + one two-slot VectorEnvironment per GPU
#   - instance-pair chunks are the scheduling units; chunks from one task may run
#     concurrently on different GPUs and are merged into the official task output
#   - one policy request containing both environments per synchronized inference step
#
# The default submission profile writes MP4 videos and validates them with the metrics.
# For a metrics-only run without videos, use:
#   EVAL_PROFILE=throughput bash run_eval_2026_persistent.sh
#
# Single-GPU smoke test:
#   GPU_IDS=0 NUM_GPUS=1 TASK_IDS=0 TASK_LIMIT=1 \
#     EVAL_INSTANCE_INDICES='0 1' EVAL_MAX_STEPS=100 \
#     bash run_eval_2026_persistent.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
# Match the policy source and runtime used by pretrain/launch/pi_behavior/run_eval_2026.sh.
CURRENT_PRETRAIN_ROOT="${CURRENT_PRETRAIN_ROOT:-${BEHAVIOR_ROOT}/../pretrain}"
PI05_REPO="${PI05_REPO:-${CURRENT_PRETRAIN_ROOT}/jax_behavior}"
PI05_SERVER_SCRIPT="${PI05_SERVER_SCRIPT:-${PI05_REPO}/scripts/serve_pi_behavior_2026_vector.py}"
PI05_POLICY_CONFIG="${PI05_POLICY_CONFIG:-pi_behavior_b1k_2026}"
PI05_POLICY_DIR="${PI05_POLICY_DIR:-/mnt/data/ckpt/[b1k]/pi_behavior_b1k_2026/20260819_b1k_2026_full_dlc_4node_bs2048_downsample6_val005_fast30hz_no_ki_from_pi05/59000}"
PI05_INFERENCE_CKPT_SUFFIX="${PI05_INFERENCE_CKPT_SUFFIX:-_inference}"
PI05_AUTO_CONVERT_CKPT="${PI05_AUTO_CONVERT_CKPT:-true}"
PI05_CONVERT_SCRIPT="${PI05_CONVERT_SCRIPT:-${CURRENT_PRETRAIN_ROOT}/../behavior-1k-solution/scripts/merge_sharded_params_for_inference.py}"
PI05_ASSETS_ROOT="${PI05_ASSETS_ROOT:-${CURRENT_PRETRAIN_ROOT}/../behavior-1k-solution/outputs/assets/pi_behavior_b1k_2026}"
PI05_NORM_STATS_PATH="${PI05_NORM_STATS_PATH:-${PI05_ASSETS_ROOT}/behavior-1k/2026-challenge-demos/norm_stats.json}"
PI05_TASK_CHECKPOINT_MAPPING="${PI05_TASK_CHECKPOINT_MAPPING:-${CURRENT_PRETRAIN_ROOT}/task_checkpoint_mapping.json}"
USE_PI05_TASK_CHECKPOINT_MAPPING="${USE_PI05_TASK_CHECKPOINT_MAPPING:-false}"
# Reuse the reference launcher's managed venv; policy source comes from PI05_REPO.
PI05_ENV_DIR="${PI05_ENV_DIR:-${CURRENT_PRETRAIN_ROOT}/../behavior-1k-solution/.venv}"
PI05_PYTHON="${PI05_PYTHON:-${PI05_ENV_DIR}/bin/python}"
# Use the NAS-backed evaluator environment by default. Override
# BEHAVIOR_ENV_DIR explicitly when running on another host.
BEHAVIOR_ENV_DIR="${BEHAVIOR_ENV_DIR:-/mnt/data_nas/wangjm/miniconda3/envs/behavior_2026}"
BEHAVIOR_PYTHON="${BEHAVIOR_PYTHON:-${BEHAVIOR_ENV_DIR}/bin/python}"
DRIVER_FIX_SCRIPT="${DRIVER_FIX_SCRIPT:-${HOME}/driver_fix/activate.sh}"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-${BEHAVIOR_ROOT}/datasets}"
TASK_STATS_FILE="${TASK_STATS_FILE:-${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/metadata/task.jsonl}"
PI05_RESOLVED_POLICY_DIR=""

TASK_IDS="${TASK_IDS:-}"
TASK_LIMIT="${TASK_LIMIT:-100}"
EVAL_INSTANCE_INDICES="${EVAL_INSTANCE_INDICES:-0 1 2 3 4 5 6 7 8 9}"
EVAL_INSTANCE_INDICES="${EVAL_INSTANCE_INDICES//,/ }"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-}"
EVAL_MAX_STEPS_MULTIPLIER="${EVAL_MAX_STEPS_MULTIPLIER:-1.5}"
EVAL_SEED="${EVAL_SEED:-0}"

NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-8}}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_IDS="${GPU_IDS//,/ }"
VECTOR_ENVS_PER_PROCESS="${VECTOR_ENVS_PER_PROCESS:-2}"
# Number of public instances in one request to the evaluator. Keeping this
# equal to VECTOR_ENVS_PER_PROCESS preserves synchronized batch-2 inference.
INSTANCE_CHUNK_SIZE="${INSTANCE_CHUNK_SIZE:-${VECTOR_ENVS_PER_PROCESS}}"
PORT_BASE="${PORT_BASE:-7100}"
PORT_STRIDE="${PORT_STRIDE:-100}"
PI05_SERVER_HOST="${PI05_SERVER_HOST:-localhost}"
PI05_CLIENT_HOST="${PI05_CLIENT_HOST:-localhost}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
DLC_SCHEDULER_SCRIPT="${DLC_SCHEDULER_SCRIPT:-${BEHAVIOR_ROOT}/OmniGibson/omnigibson/eval/utils/pi05_dynamic_scheduler.py}"
DLC_SCHEDULER_BIND_HOST="${DLC_SCHEDULER_BIND_HOST:-0.0.0.0}"
DLC_SCHEDULER_ADVERTISE_HOST="${DLC_SCHEDULER_ADVERTISE_HOST:-${MASTER_ADDR:-}}"
DLC_SCHEDULER_PORT="${DLC_SCHEDULER_PORT:-6900}"
DLC_SCHEDULER_REQUEST_TIMEOUT="${DLC_SCHEDULER_REQUEST_TIMEOUT:-60}"

PI05_DYNAMIC_BATCH_MAX_SIZE="${PI05_DYNAMIC_BATCH_MAX_SIZE:-2}"
PI05_DYNAMIC_BATCH_WAIT_MS="${PI05_DYNAMIC_BATCH_WAIT_MS:-0}"
PI05_DYNAMIC_BATCH_GRANULARITY="${PI05_DYNAMIC_BATCH_GRANULARITY:-1}"
PI05_ACTIONS_TO_EXECUTE="${PI05_ACTIONS_TO_EXECUTE:-26}"
PI05_ACTIONS_TO_KEEP="${PI05_ACTIONS_TO_KEEP:-4}"
PI05_EXECUTE_IN_N_STEPS="${PI05_EXECUTE_IN_N_STEPS:-20}"
PI05_HISTORY_LEN="${PI05_HISTORY_LEN:-3}"
PI05_VOTES_TO_PROMOTE="${PI05_VOTES_TO_PROMOTE:-2}"
PI05_NUM_STEPS="${PI05_NUM_STEPS:-20}"
PI05_APPLY_EVAL_TRICKS="${PI05_APPLY_EVAL_TRICKS:-true}"
PI05_DISABLE_FAST_AUXILIARY="${PI05_DISABLE_FAST_AUXILIARY:-true}"
PI05_PROPRIOCEPTION_SCHEMA="${PI05_PROPRIOCEPTION_SCHEMA:-r1pro_v3_61}"
PI05_BASE_VELOCITY_FRAME="${PI05_BASE_VELOCITY_FRAME:-absolute}"

EVAL_PROFILE="${EVAL_PROFILE:-submission}"
case "${EVAL_PROFILE}" in
  throughput) PROFILE_WRITE_VIDEO=false ;;
  submission) PROFILE_WRITE_VIDEO=true ;;
  *) echo "EVAL_PROFILE must be throughput or submission, got: ${EVAL_PROFILE}" >&2; exit 2 ;;
esac
EVAL_WRITE_VIDEO="${EVAL_WRITE_VIDEO:-${PROFILE_WRITE_VIDEO}}"
EVAL_PARTIAL_SCENE_LOAD="${EVAL_PARTIAL_SCENE_LOAD:-true}"
EVAL_FAIL_FAST="${EVAL_FAIL_FAST:-true}"
EVAL_MAX_TASK_ATTEMPTS="${EVAL_MAX_TASK_ATTEMPTS:-5}"
# Optional wall-clock limit per request, including first startup (0 disables it).
EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-0}"
EVAL_WORKER_SHUTDOWN_TIMEOUT="${EVAL_WORKER_SHUTDOWN_TIMEOUT:-30}"

if [[ "${EVAL_PROFILE}" == submission && "${EVAL_WRITE_VIDEO}" != true ]]; then
  echo "EVAL_PROFILE=submission requires EVAL_WRITE_VIDEO=true; use throughput for a no-video run." >&2
  exit 2
fi

CPU_RESERVE_CORES="${CPU_RESERVE_CORES:-auto}"
EVAL_CPU_CORES="${EVAL_CPU_CORES:-auto}"
EVAL_CPU_NUM_THREADS="${EVAL_CPU_NUM_THREADS:-auto}"
SERVER_CPU_NUM_THREADS="${SERVER_CPU_NUM_THREADS:-2}"

# PAI DLC starts this script once on every Worker node. WORLD_SIZE is the
# number of nodes, RANK is this node's index, and NPROC_PER_NODE is the
# number of local GPUs exposed to each Worker.
DLC_WORLD_SIZE="${WORLD_SIZE:-1}"
DLC_RANK="${RANK:-0}"
DLC_NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS:-8}}"
DLC_BARRIER_TIMEOUT="${DLC_BARRIER_TIMEOUT:-1800}"
DLC_HEARTBEAT_INTERVAL="${DLC_HEARTBEAT_INTERVAL:-60}"
DLC_HEARTBEAT_STALE_TIMEOUT="${DLC_HEARTBEAT_STALE_TIMEOUT:-1800}"
DLC_RUN_KEY="${DLC_RUN_KEY:-${PAI_JOB_ID:-}}"

[[ "${DLC_WORLD_SIZE}" =~ ^[1-9][0-9]*$ && "${DLC_RANK}" =~ ^(0|[1-9][0-9]*)$ \
   && "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "WORLD_SIZE and NUM_GPUS must be positive integers; RANK must be non-negative." >&2
  exit 2
}
DLC_MULTINODE=false
(( DLC_WORLD_SIZE > 1 )) && DLC_MULTINODE=true

if [[ -z "${DLC_RUN_KEY}" ]]; then
  if [[ "${DLC_MULTINODE}" == true ]]; then
    echo "DLC_RUN_KEY or PAI_JOB_ID is required for a multi-node run." >&2
    exit 2
  fi
  DLC_RUN_KEY="$(date +%Y%m%d_%H%M%S)"
fi

EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${BEHAVIOR_ROOT}/logs/pi05_behavior_2026_persistent_outputs}"
EVAL_LOG_PARENT="${EVAL_LOG_PARENT:-${BEHAVIOR_ROOT}/logs}"
RESUME_LOG_DIR="${GLOBAL_LOG_DIR:-${LOG_DIR:-}}"
SCRIPT_START_EPOCH="$(date +%s)"

TOTAL_SCHEDULER_SLOTS=$((DLC_WORLD_SIZE * NUM_GPUS))
OUTPUT_VALIDATOR="${BEHAVIOR_ROOT}/OmniGibson/omnigibson/eval/utils/pi05_output_validator.py"
RESUME_HELPER="${BEHAVIOR_ROOT}/OmniGibson/omnigibson/eval/utils/pi05_persistent_resume.py"

DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: bash run_pi05_behavior_2026_eval_persistent.sh [--log-dir EXISTING_LOG_DIR] [--base-velocity-frame absolute|relative] [--dry-run] [--help]

Core overrides:
  EVAL_PROFILE              submission (videos) or throughput (no videos), default submission.
  TASK_IDS                  Space/comma-separated 2026 task IDs in [0,99], default all 100.
  TASK_LIMIT                Limit queued tasks after selection, default 100.
  EVAL_INSTANCE_INDICES     Public split indices, default '0 1 2 3 4 5 6 7 8 9'.
  EVAL_SEED                 Fixed environment RNG seed, default 0.
  EVAL_MAX_STEPS            Optional absolute timeout override; empty uses the human-length multiplier.
  EVAL_MAX_STEPS_MULTIPLIER Multiplier applied to mean human-demo length, default 1.5.
  EVAL_MAX_TASK_ATTEMPTS    Attempts per instance chunk before terminal failure, default 5.
  EVAL_REQUEST_TIMEOUT      Per-chunk wall-clock limit including startup, default 0 (disabled).
  EVAL_WORKER_SHUTDOWN_TIMEOUT  Graceful Isaac shutdown limit in seconds, default 30.
  GPU_IDS / NUM_GPUS        GPU IDs and number of colocated env/server pairs.
  INSTANCE_CHUNK_SIZE       Instances per scheduled chunk, defaulting to the vector-env count (2).
  TASK_STATS_FILE           Per-task human statistics used for load balancing.
  CURRENT_PRETRAIN_ROOT     Pretrain checkout, default the sibling pretrain directory.
  PI05_REPO                 100-task PI0.5 source, default ${CURRENT_PRETRAIN_ROOT}/jax_behavior.
  PI05_SERVER_SCRIPT        Policy server, default ${PI05_REPO}/scripts/serve_pi_behavior_2026_vector.py.
  PI05_POLICY_DIR           Training or merged inference checkpoint directory.
  PI05_AUTO_CONVERT_CKPT    Automatically merge sharded training checkpoints, default true.
  PI05_CONVERT_SCRIPT       Sharded-checkpoint merge script.
  PI05_NORM_STATS_PATH      2026 checkpoint normalization statistics.
  PI05_BASE_VELOCITY_FRAME  Policy observation base qvel frame: absolute (legacy raw) or relative (robot-local), default absolute. Actions are always robot-local.
  PI05_ENV_DIR              Policy runtime, default the sibling behavior-1k-solution/.venv.
  BEHAVIOR_ENV_DIR          2026 evaluator conda environment directory.
  DRIVER_FIX_SCRIPT         Script sourced when starting an evaluator process, default ~/driver_fix/activate.sh.
  WORLD_SIZE / RANK         DLC node count / node rank (not torchrun process ranks), default 1 / 0.
  NPROC_PER_NODE            Available GPUs per node; NUM_GPUS defaults to this value, or 8.
  DLC_RUN_KEY               Optional coordination key, default PAI_JOB_ID; directory names are always dates.
  DLC_SCHEDULER_ADVERTISE_HOST  Rank-0 address reachable from every node, default MASTER_ADDR or rank-0 IP.
  DLC_SCHEDULER_PORT        Preferred rank-0 HTTP scheduler port, default 6900.
  DLC_SCHEDULER_REQUEST_TIMEOUT  Retry budget per HTTP operation in seconds, default 60.
  DLC_BARRIER_TIMEOUT      Startup barrier timeout in seconds, default 1800.
  DLC_HEARTBEAT_INTERVAL / DLC_HEARTBEAT_STALE_TIMEOUT  Rank heartbeat interval / stale limit, 60 / 1800 seconds.
  GLOBAL_LOG_DIR / LOG_DIR  Existing evaluation log directory to resume (also --log-dir); omitted starts from scratch.
  EVAL_LOG_PARENT           Parent for fresh date-named log directories, default ${BEHAVIOR_ROOT}/logs.
  EVAL_LOG_ROOT             Shared NAS output root; all nodes must use the same paths.

Rank 0 builds a longest-task-first queue and serves it over HTTP. Every GPU
worker claims one instance chunk, completes it, then requests the next chunk.
NAS holds outputs and coordination markers; the queue has a single writer on
rank 0. Only rank 0 merges task outputs and issues the global run certificate.
Use the same command on every DLC node; do not wrap this script in torchrun.
Without a log directory, rank 0 creates a new YYYYMMDD_HHMMSS run and starts
from scratch. Specify --log-dir (or LOG_DIR) to resume only that evaluation.
Finished instances are validated and skipped; incomplete instances restart from
their initial state. Outputs stay in the original run, while each resume's logs
are kept under the specified log directory's .sessions/<date>/ subdirectory.
No simulator state or action chunk is restored. Dry runs only plan on rank 0.

The default checkpoint is the 100-task 2026 PI_BEHAVIOR checkpoint. If the
requested training checkpoint is sharded, an existing sibling ending in
_inference is selected or created automatically. One server is loaded once per
GPU worker and reused until that worker's task queue is empty.
Each worker also retains one Isaac process. Same-task chunks reuse environments;
task changes clear and rebuild both scenes while preserving the Isaac app.
Failed requests restart the evaluator before retrying; output checks are retained.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-dir)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--log-dir requires an existing log directory" >&2; exit 2; }
      RESUME_LOG_DIR="$2"
      shift 2
      ;;
    --log-dir=*)
      RESUME_LOG_DIR="${1#*=}"
      [[ -n "${RESUME_LOG_DIR}" ]] || { echo "--log-dir requires an existing log directory" >&2; exit 2; }
      shift
      ;;
    --base-velocity-frame)
      [[ $# -ge 2 ]] || { echo "--base-velocity-frame requires absolute or relative" >&2; exit 2; }
      PI05_BASE_VELOCITY_FRAME="$2"
      shift 2
      ;;
    --base-velocity-frame=*)
      PI05_BASE_VELOCITY_FRAME="${1#*=}"
      shift
      ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command not found: $1" >&2; exit 1; }
}

validate_bool() {
  case "$2" in
    true|false) ;;
    *) echo "$1 must be true or false, got: $2" >&2; exit 1 ;;
  esac
}

validate_positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$1 must be a positive integer, got: $2" >&2; exit 1; }
}

validate_positive_float() {
  awk -v value="$2" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }' || {
    echo "$1 must be a positive number, got: $2" >&2
    exit 1
  }
}

validate_seed() {
  [[ "$2" =~ ^[0-9]+$ ]] && (( $2 < 4294967296 )) \
    || { echo "$1 must be an integer in [0, 4294967296), got: $2" >&2; exit 1; }
}

is_inference_checkpoint() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/params/_METADATA" ]] || return 1
  [[ -f "${checkpoint_dir}/params/manifest.ocdbt" && -d "${checkpoint_dir}/params/d" ]]
}

checkpoint_process_count() {
  local checkpoint_dir="$1" shard_dir count=0
  for shard_dir in "${checkpoint_dir}"/params/ocdbt.process_*; do
    [[ -d "${shard_dir}" ]] || continue
    count=$((count + 1))
  done
  (( count > 0 )) || return 1
  printf '%s\n' "${count}"
}

resolve_policy_checkpoint() {
  local requested="${PI05_POLICY_DIR%/}"
  local converted="${requested}${PI05_INFERENCE_CKPT_SUFFIX}"
  local process_count conversion_log conversion_lock
  if is_inference_checkpoint "${requested}"; then
    PI05_RESOLVED_POLICY_DIR="${requested}"
  elif is_inference_checkpoint "${converted}"; then
    echo "[pi05 convert] using existing inference checkpoint: ${converted}"
    PI05_RESOLVED_POLICY_DIR="${converted}"
  elif [[ "${PI05_AUTO_CONVERT_CKPT}" != true ]]; then
    echo "No server-readable PI0.5 checkpoint found and PI05_AUTO_CONVERT_CKPT=false." >&2
    echo "Requested: ${requested}" >&2
    echo "Inference fallback: ${converted}" >&2
    return 1
  elif [[ ! -d "${requested}" ]]; then
    echo "PI0.5 checkpoint directory not found: ${requested}" >&2
    return 1
  elif [[ ! -f "${PI05_CONVERT_SCRIPT}" ]]; then
    echo "PI0.5 checkpoint conversion script not found: ${PI05_CONVERT_SCRIPT}" >&2
    return 1
  elif ! process_count="$(checkpoint_process_count "${requested}")"; then
    echo "Checkpoint is neither server-readable nor a sharded OCDBT checkpoint: ${requested}" >&2
    return 1
  else
    mkdir -p "${LOG_DIR}" "$(dirname "${converted}")"
    conversion_log="${LOG_DIR}/checkpoint_conversion.log"
    conversion_lock="${converted}.conversion.lock"
    if ! (
      flock -x 9
      if is_inference_checkpoint "${converted}"; then
        echo "[pi05 convert] another launcher completed the conversion: ${converted}"
        exit 0
      fi
      echo "[pi05 convert] merging ${process_count} checkpoint shards"
      echo "[pi05 convert] source: ${requested}"
      echo "[pi05 convert] destination: ${converted}"
      echo "[pi05 convert] progress log: ${conversion_log}"
      cd "${PI05_REPO}"
      "${PI05_PYTHON}" "${PI05_CONVERT_SCRIPT}" \
        --src "${requested}" \
        --dst "${converted}" \
        --num-processes "${process_count}" \
        --overwrite \
        >>"${conversion_log}" 2>&1
    ) 9>"${conversion_lock}"; then
      echo "PI0.5 checkpoint conversion failed; see ${conversion_log}" >&2
      return 1
    fi
    if ! is_inference_checkpoint "${converted}"; then
      echo "Converted checkpoint is still not server-readable: ${converted}" >&2
      echo "See conversion log: ${conversion_log}" >&2
      return 1
    fi
    echo "[pi05 convert] inference checkpoint ready: ${converted}"
    PI05_RESOLVED_POLICY_DIR="${converted}"
  fi
}

validate_pi05_2026_source() {
  python3 - \
    "${PI05_REPO}/src/b1k/models/pi_behavior_config.py" \
    "${PI05_REPO}/src/b1k/training/config.py" \
    "${PI05_POLICY_CONFIG}" <<'PY'
import ast
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
policy_config = sys.argv[3]
for path in (model_path, config_path):
    if not path.is_file():
        raise SystemExit(f"Missing PI0.5 source file: {path}")

tree = ast.parse(model_path.read_text(encoding="utf-8"))
task_stages = None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "TASK_NUM_STAGES" for target in node.targets
    ):
        task_stages = ast.literal_eval(node.value)
        break
if task_stages is None or len(task_stages) != 100:
    raise SystemExit(
        f"{model_path} exposes {len(task_stages) if task_stages is not None else 0} tasks; "
        "PI05_REPO must expose all 100 tasks for the 2026 checkpoint"
    )

config_text = config_path.read_text(encoding="utf-8")
if f'name="{policy_config}"' not in config_text:
    raise SystemExit(f"Policy config {policy_config!r} is missing from {config_path}")
print(f"Validated PI0.5 2026 source: tasks={len(task_stages)}, config={policy_config}")
PY
}

AVAILABLE_CPU_IDS=()
EVAL_CPU_LISTS=()
SERVER_CPU_LISTS=()
RESOLVED_CPU_RESERVE=0

resolve_cpu_allocation() {
  mapfile -t AVAILABLE_CPU_IDS < <(python3 - <<'PY'
import os
for cpu_id in sorted(os.sched_getaffinity(0)):
    print(cpu_id)
PY
  )
  local available_count="${#AVAILABLE_CPU_IDS[@]}"
  (( available_count > 0 )) || { echo "No CPUs available in launcher cpuset." >&2; exit 1; }

  if [[ "${CPU_RESERVE_CORES}" == auto ]]; then
    RESOLVED_CPU_RESERVE=$((available_count / 32))
    (( RESOLVED_CPU_RESERVE < 2 )) && RESOLVED_CPU_RESERVE=2
    (( RESOLVED_CPU_RESERVE > 8 )) && RESOLVED_CPU_RESERVE=8
  else
    validate_positive_int CPU_RESERVE_CORES "${CPU_RESERVE_CORES}"
    RESOLVED_CPU_RESERVE="${CPU_RESERVE_CORES}"
  fi
  validate_positive_int SERVER_CPU_NUM_THREADS "${SERVER_CPU_NUM_THREADS}"

  local usable=$((available_count - RESOLVED_CPU_RESERVE))
  local server_total=$((NUM_GPUS * SERVER_CPU_NUM_THREADS))
  (( usable > server_total )) || { echo "Too few CPUs after server allocation." >&2; exit 1; }
  if [[ "${EVAL_CPU_CORES}" == auto ]]; then
    EVAL_CPU_CORES=$(((usable - server_total) / NUM_GPUS))
  else
    validate_positive_int EVAL_CPU_CORES "${EVAL_CPU_CORES}"
  fi
  (( EVAL_CPU_CORES > 0 )) || { echo "No CPU cores remain for evaluators." >&2; exit 1; }
  if [[ "${EVAL_CPU_NUM_THREADS}" == auto ]]; then
    EVAL_CPU_NUM_THREADS="${EVAL_CPU_CORES}"
  else
    validate_positive_int EVAL_CPU_NUM_THREADS "${EVAL_CPU_NUM_THREADS}"
  fi
  (( EVAL_CPU_NUM_THREADS <= EVAL_CPU_CORES )) || {
    echo "EVAL_CPU_NUM_THREADS cannot exceed EVAL_CPU_CORES." >&2
    exit 1
  }

  local required=$((NUM_GPUS * EVAL_CPU_CORES + server_total + RESOLVED_CPU_RESERVE))
  (( required <= available_count )) || {
    echo "CPU allocation exceeds cpuset: required=${required}, available=${available_count}." >&2
    exit 1
  }

  local worker offset cpu_list
  for ((worker = 0; worker < NUM_GPUS; worker++)); do
    offset=$((worker * EVAL_CPU_CORES))
    cpu_list="$(IFS=,; echo "${AVAILABLE_CPU_IDS[*]:${offset}:${EVAL_CPU_CORES}}")"
    EVAL_CPU_LISTS+=("${cpu_list}")
  done
  for ((worker = 0; worker < NUM_GPUS; worker++)); do
    offset=$((NUM_GPUS * EVAL_CPU_CORES + worker * SERVER_CPU_NUM_THREADS))
    cpu_list="$(IFS=,; echo "${AVAILABLE_CPU_IDS[*]:${offset}:${SERVER_CPU_NUM_THREADS}}")"
    SERVER_CPU_LISTS+=("${cpu_list}")
  done
}

validate_environment() {
  require_command flock
  require_command curl
  require_command python3
  require_command taskset
  require_command setsid
  validate_positive_int NUM_GPUS "${NUM_GPUS}"
  validate_positive_int DLC_WORLD_SIZE "${DLC_WORLD_SIZE}"
  validate_positive_int DLC_NPROC_PER_NODE "${DLC_NPROC_PER_NODE}"
  validate_positive_int DLC_BARRIER_TIMEOUT "${DLC_BARRIER_TIMEOUT}"
  validate_positive_int DLC_HEARTBEAT_INTERVAL "${DLC_HEARTBEAT_INTERVAL}"
  validate_positive_int DLC_HEARTBEAT_STALE_TIMEOUT "${DLC_HEARTBEAT_STALE_TIMEOUT}"
  validate_positive_int DLC_SCHEDULER_PORT "${DLC_SCHEDULER_PORT}"
  validate_positive_int DLC_SCHEDULER_REQUEST_TIMEOUT "${DLC_SCHEDULER_REQUEST_TIMEOUT}"
  (( DLC_SCHEDULER_PORT <= 65535 )) || {
    echo "DLC_SCHEDULER_PORT must be at most 65535, got: ${DLC_SCHEDULER_PORT}" >&2
    exit 1
  }
  [[ "${DLC_RANK}" =~ ^[0-9]+$ ]] || {
    echo "DLC_RANK must be a non-negative integer, got: ${DLC_RANK}" >&2
    exit 1
  }
  (( DLC_RANK < DLC_WORLD_SIZE )) || {
    echo "DLC_RANK=${DLC_RANK} must be smaller than DLC_WORLD_SIZE=${DLC_WORLD_SIZE}." >&2
    exit 1
  }
  (( NUM_GPUS <= DLC_NPROC_PER_NODE )) || {
    echo "NUM_GPUS=${NUM_GPUS} cannot exceed DLC_NPROC_PER_NODE=${DLC_NPROC_PER_NODE}." >&2
    exit 1
  }
  [[ "${DLC_RUN_KEY}" =~ ^[A-Za-z0-9._-]+$ && "${DLC_RUN_KEY}" != . && "${DLC_RUN_KEY}" != .. ]] || {
    echo "DLC_RUN_KEY contains unsafe path characters: ${DLC_RUN_KEY}" >&2
    exit 1
  }
  validate_positive_int VECTOR_ENVS_PER_PROCESS "${VECTOR_ENVS_PER_PROCESS}"
  validate_positive_int INSTANCE_CHUNK_SIZE "${INSTANCE_CHUNK_SIZE}"
  validate_positive_int PI05_DYNAMIC_BATCH_MAX_SIZE "${PI05_DYNAMIC_BATCH_MAX_SIZE}"
  validate_positive_int PI05_DYNAMIC_BATCH_GRANULARITY "${PI05_DYNAMIC_BATCH_GRANULARITY}"
  validate_positive_int TASK_LIMIT "${TASK_LIMIT}"
  validate_positive_int EVAL_MAX_TASK_ATTEMPTS "${EVAL_MAX_TASK_ATTEMPTS}"
  [[ "${EVAL_REQUEST_TIMEOUT}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "EVAL_REQUEST_TIMEOUT must be a non-negative integer." >&2; exit 2;
  }
  validate_positive_int EVAL_WORKER_SHUTDOWN_TIMEOUT "${EVAL_WORKER_SHUTDOWN_TIMEOUT}"
  validate_positive_float EVAL_MAX_STEPS_MULTIPLIER "${EVAL_MAX_STEPS_MULTIPLIER}"
  [[ -z "${EVAL_MAX_STEPS}" ]] || validate_positive_int EVAL_MAX_STEPS "${EVAL_MAX_STEPS}"
  validate_seed EVAL_SEED "${EVAL_SEED}"
  validate_bool EVAL_WRITE_VIDEO "${EVAL_WRITE_VIDEO}"
  validate_bool EVAL_PARTIAL_SCENE_LOAD "${EVAL_PARTIAL_SCENE_LOAD}"
  validate_bool EVAL_FAIL_FAST "${EVAL_FAIL_FAST}"
  validate_bool PI05_APPLY_EVAL_TRICKS "${PI05_APPLY_EVAL_TRICKS}"
  validate_bool PI05_DISABLE_FAST_AUXILIARY "${PI05_DISABLE_FAST_AUXILIARY}"
  validate_bool PI05_AUTO_CONVERT_CKPT "${PI05_AUTO_CONVERT_CKPT}"
  validate_bool USE_PI05_TASK_CHECKPOINT_MAPPING "${USE_PI05_TASK_CHECKPOINT_MAPPING}"
  [[ "${EVAL_WRITE_VIDEO}" != true ]] || require_command ffprobe

  (( VECTOR_ENVS_PER_PROCESS == 2 )) || {
    echo "This launcher requires VECTOR_ENVS_PER_PROCESS=2 for synchronized two-env simulation." >&2
    exit 1
  }
  (( INSTANCE_CHUNK_SIZE <= VECTOR_ENVS_PER_PROCESS )) || {
    echo "INSTANCE_CHUNK_SIZE cannot exceed VECTOR_ENVS_PER_PROCESS." >&2
    exit 1
  }
  (( PI05_DYNAMIC_BATCH_MAX_SIZE == 2 && PI05_DYNAMIC_BATCH_GRANULARITY == 1 )) || {
    echo "This launcher requires policy max batch size 2 and granularity 1." >&2
    exit 1
  }
  [[ -x "${PI05_PYTHON}" ]] || { echo "PI05_PYTHON is not executable: ${PI05_PYTHON}" >&2; exit 1; }
  [[ -x "${BEHAVIOR_PYTHON}" ]] || {
    echo "BEHAVIOR_PYTHON is not executable: ${BEHAVIOR_PYTHON}" >&2
    exit 1
  }
  [[ -r "${DRIVER_FIX_SCRIPT}" ]] || {
    echo "GPU driver activation script is not readable: ${DRIVER_FIX_SCRIPT}" >&2
    exit 1
  }
  [[ -f "${PI05_SERVER_SCRIPT}" ]] || { echo "PI0.5 vector server is missing: ${PI05_SERVER_SCRIPT}" >&2; exit 1; }
  [[ -f "${BEHAVIOR_ROOT}/OmniGibson/omnigibson/eval/eval_persistent.py" ]] || {
    echo "2026 vector evaluator is missing under ${BEHAVIOR_ROOT}." >&2
    exit 1
  }
  [[ -f "${OUTPUT_VALIDATOR}" ]] || {
    echo "PI0.5 output validator is missing: ${OUTPUT_VALIDATOR}" >&2
    exit 1
  }
  [[ -f "${RESUME_HELPER}" ]] || { echo "Persistent resume helper is missing: ${RESUME_HELPER}" >&2; exit 1; }
  [[ -f "${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/metadata/B100_task_misc.csv" ]] || {
    echo "2026 challenge metadata is missing under ${OMNIGIBSON_DATA_PATH}." >&2
    exit 1
  }
  [[ -f "${TASK_STATS_FILE}" ]] || {
    echo "2026 task statistics are missing: ${TASK_STATS_FILE}" >&2
    exit 1
  }
  [[ -f "${PI05_NORM_STATS_PATH}" ]] || {
    echo "2026 checkpoint normalization stats are missing: ${PI05_NORM_STATS_PATH}" >&2
    exit 1
  }
  if [[ "${USE_PI05_TASK_CHECKPOINT_MAPPING}" == true && ! -f "${PI05_TASK_CHECKPOINT_MAPPING}" ]]; then
    echo "PI05_TASK_CHECKPOINT_MAPPING not found: ${PI05_TASK_CHECKPOINT_MAPPING}" >&2
    exit 1
  fi
  [[ "${PI05_PROPRIOCEPTION_SCHEMA}" == r1pro_v3_61 ]] || {
    echo "The 2026 checkpoint requires PI05_PROPRIOCEPTION_SCHEMA=r1pro_v3_61." >&2
    exit 1
  }
  case "${PI05_BASE_VELOCITY_FRAME}" in
    absolute|relative) ;;
    *)
      echo "PI05_BASE_VELOCITY_FRAME must be absolute or relative, got: ${PI05_BASE_VELOCITY_FRAME}" >&2
      exit 1
      ;;
  esac

  [[ -f "${DLC_SCHEDULER_SCRIPT}" ]] || {
    echo "Dynamic scheduler service is missing: ${DLC_SCHEDULER_SCRIPT}" >&2; exit 1;
  }
  validate_pi05_2026_source

  read -r -a GPU_ID_LIST <<<"${GPU_IDS}"
  (( ${#GPU_ID_LIST[@]} >= NUM_GPUS )) || {
    echo "Need ${NUM_GPUS} GPU IDs, got: ${GPU_ID_LIST[*]}" >&2
    exit 1
  }
  GPU_ID_LIST=("${GPU_ID_LIST[@]:0:${NUM_GPUS}}")
  declare -A seen_gpus=()
  local gpu_id
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: ${gpu_id}" >&2; exit 1; }
    [[ -z "${seen_gpus[${gpu_id}]:-}" ]] || { echo "Duplicate GPU ID: ${gpu_id}" >&2; exit 1; }
    seen_gpus["${gpu_id}"]=1
  done

  read -r -a INSTANCE_INDEX_LIST <<<"${EVAL_INSTANCE_INDICES}"
  (( ${#INSTANCE_INDEX_LIST[@]} > 0 )) || { echo "EVAL_INSTANCE_INDICES cannot be empty." >&2; exit 1; }
  declare -A seen_instances=()
  local index
  for index in "${INSTANCE_INDEX_LIST[@]}"; do
    [[ "${index}" =~ ^[0-9]+$ ]] || { echo "Invalid public instance index: ${index}" >&2; exit 1; }
    (( index >= 0 && index < 20 )) || { echo "Public instance index must be in [0,19]: ${index}" >&2; exit 1; }
    [[ -z "${seen_instances[${index}]:-}" ]] || { echo "Duplicate instance index: ${index}" >&2; exit 1; }
    seen_instances["${index}"]=1
  done

  resolve_cpu_allocation
}

validate_gpu_runtime() {
  require_command nvidia-smi
  local gpu_count gpu_id
  gpu_count="$(nvidia-smi -L 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  (( gpu_count > 0 )) || { echo "nvidia-smi did not report a usable GPU." >&2; exit 1; }
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    (( gpu_id < gpu_count )) || { echo "GPU ID ${gpu_id} is out of range; found ${gpu_count} GPUs." >&2; exit 1; }
  done
}

build_task_queue() {
  TASK_IDS="${TASK_IDS//,/ }"
  mkdir -p "${QUEUE_DIR}"
  SELECTED_TASK_IDS="${TASK_IDS}" TASK_LIMIT_VALUE="${TASK_LIMIT}" \
    NUM_SCHEDULER_SLOTS="${TOTAL_SCHEDULER_SLOTS}" INSTANCE_COUNT_VALUE="${#INSTANCE_INDEX_LIST[@]}" \
    INSTANCE_INDICES_VALUE="${INSTANCE_INDEX_LIST[*]}" \
    VECTOR_ENVS_VALUE="${VECTOR_ENVS_PER_PROCESS}" INSTANCE_CHUNK_SIZE_VALUE="${INSTANCE_CHUNK_SIZE}" \
    MAX_STEPS_OVERRIDE="${EVAL_MAX_STEPS}" \
    MAX_STEPS_MULTIPLIER="${EVAL_MAX_STEPS_MULTIPLIER}" \
    python3 - \
      "${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/metadata/B100_task_misc.csv" \
      "${TASK_STATS_FILE}" "${QUEUE_FILE}" "${ONLINE_QUEUE_FILE}" "${SCHEDULE_FILE}" <<'PY'
import csv
import json
import math
import os
import sys

metadata_path, stats_path, output_path, online_queue_path, schedule_path = sys.argv[1:]
with open(metadata_path, newline="", encoding="utf-8") as file:
    tasks = {int(row["Task ID"]): row["Task"] for row in csv.DictReader(file)}
with open(stats_path, encoding="utf-8") as file:
    stats = {
        int(row["task_index"]): row
        for line in file
        if line.strip()
        for row in [json.loads(line)]
    }

raw_ids = os.environ.get("SELECTED_TASK_IDS", "").split()
task_ids = [int(value) for value in raw_ids] if raw_ids else list(range(100))
if len(task_ids) != len(set(task_ids)):
    raise SystemExit("TASK_IDS contains duplicates")
unsupported = [task_id for task_id in task_ids if task_id not in range(100)]
if unsupported:
    raise SystemExit(f"2026 task IDs must be in 0-99; got {unsupported}")
limit = int(os.environ["TASK_LIMIT_VALUE"])
selected_ids = task_ids[:limit]
missing_stats = [task_id for task_id in selected_ids if task_id not in stats]
if missing_stats:
    raise SystemExit(f"Missing human statistics for task IDs: {missing_stats}")

num_slots = int(os.environ["NUM_SCHEDULER_SLOTS"])
instance_count = int(os.environ["INSTANCE_COUNT_VALUE"])
instance_indices = [int(value) for value in os.environ["INSTANCE_INDICES_VALUE"].split()]
vector_envs = int(os.environ["VECTOR_ENVS_VALUE"])
chunk_size = int(os.environ["INSTANCE_CHUNK_SIZE_VALUE"])
max_steps_override = os.environ.get("MAX_STEPS_OVERRIDE", "")
max_steps_multiplier = float(os.environ["MAX_STEPS_MULTIPLIER"])
if num_slots <= 0 or instance_count <= 0 or vector_envs <= 0 or chunk_size <= 0:
    raise SystemExit("Scheduler slots, instance count, and vector env count must be positive")
if len(instance_indices) != instance_count:
    raise SystemExit("INSTANCE_INDICES_VALUE does not match INSTANCE_COUNT_VALUE")
if chunk_size > vector_envs:
    raise SystemExit("INSTANCE_CHUNK_SIZE cannot exceed VECTOR_ENVS_PER_PROCESS")

scheduled_tasks = []
for selection_order, task_id in enumerate(selected_ids):
    if max_steps_override:
        timeout_steps = int(max_steps_override)
    else:
        timeout_steps = int(float(stats[task_id]["length"]) * max_steps_multiplier)
    if timeout_steps <= 0:
        raise SystemExit(f"Estimated timeout must be positive for task ID {task_id}, got {timeout_steps}")
    for chunk_number, start in enumerate(range(0, instance_count, chunk_size)):
        chunk_indices = instance_indices[start : start + chunk_size]
        scheduled_tasks.append(
            {
                "task_id": task_id,
                "task_name": tasks[task_id],
                "instance_indices": chunk_indices,
                "timeout_steps": timeout_steps,
                "estimated_slot_steps": timeout_steps * len(chunk_indices),
                "chunk_steps": timeout_steps * len(chunk_indices),
                "selection_order": (selection_order, chunk_number),
            }
        )

# Online scheduling queue: longest estimated task-step chunks first.  Workers
# claim from rank 0 over HTTP at runtime, so faster GPUs naturally receive
# more work instead of being constrained by an offline per-GPU assignment.
ordered_tasks = sorted(scheduled_tasks, key=lambda item: (-item["timeout_steps"], item["selection_order"]))

with open(output_path, "w", encoding="utf-8") as file:
    for task_id in sorted(selected_ids, key=lambda task_id: (-int(float(stats[task_id]["length"]) * max_steps_multiplier) if not max_steps_override else -int(max_steps_override), task_id)):
        file.write(f"{task_id}\t{tasks[task_id]}\n")

os.makedirs(os.path.dirname(os.path.abspath(online_queue_path)), exist_ok=True)
with open(online_queue_path, "w", encoding="utf-8") as queue_file:
    for order, task in enumerate(ordered_tasks, start=1):
        indices_text = ",".join(str(index) for index in task["instance_indices"])
        queue_file.write(
            f'{task["task_id"]}\t{task["task_name"]}\t{indices_text}\t'
            f'{task["timeout_steps"]}\t{task["chunk_steps"]}\t{order}\n'
        )

with open(schedule_path, "w", encoding="utf-8") as schedule_file:
    schedule_file.write("order\tworker\ttask_id\ttask_name\tinstance_indices\ttimeout_steps\tchunk_steps\testimated_steps\tspeed_source\n")
    for order, task in enumerate(ordered_tasks, start=1):
        indices_text = ",".join(str(index) for index in task["instance_indices"])
        schedule_file.write(
            f'{order}\tpending\t{task["task_id"]}\t{task["task_name"]}\t{indices_text}\t'
            f'{task["timeout_steps"]}\t{task["chunk_steps"]}\t{task["chunk_steps"]}\tstep_lpt_online\n'
        )
PY
}

is_port_free() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ! ss -ltn "( sport = :${port} )" | grep -q ":${port}"
  elif command -v lsof >/dev/null 2>&1; then
    ! lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null 2>&1
  else
    return 0
  fi
}

find_free_port() {
  local port="$1"
  while ! is_port_free "${port}"; do port=$((port + 1)); done
  (( port <= 65535 )) || { echo "No free port at or above $1." >&2; return 1; }
  echo "${port}"
}

check_run_configuration() {
  # GPU IDs and CPU affinity may differ between hosts; the evaluation protocol
  # and number of workers per host must agree before any work is dispatched.
  local -a config=(
    "world_size=${DLC_WORLD_SIZE}" "num_gpus=${NUM_GPUS}"
    "task_ids=${TASK_IDS//,/ }" "task_limit=${TASK_LIMIT}"
    "instance_indices=${INSTANCE_INDEX_LIST[*]}" "chunk_size=${INSTANCE_CHUNK_SIZE}"
    "num_envs=${VECTOR_ENVS_PER_PROCESS}" "max_steps=${EVAL_MAX_STEPS}"
    "max_steps_multiplier=${EVAL_MAX_STEPS_MULTIPLIER}" "seed=${EVAL_SEED}"
    "write_video=${EVAL_WRITE_VIDEO}" "partial_scene_load=${EVAL_PARTIAL_SCENE_LOAD}"
    "policy_config=${PI05_POLICY_CONFIG}" "policy_dir=${PI05_POLICY_DIR}"
    "policy_repo=${PI05_REPO}" "policy_server=${PI05_SERVER_SCRIPT}"
    "norm_stats=${PI05_NORM_STATS_PATH}" "base_velocity_frame=${PI05_BASE_VELOCITY_FRAME}"
    "apply_eval_tricks=${PI05_APPLY_EVAL_TRICKS}" "fail_fast=${EVAL_FAIL_FAST}"
    "actions_to_execute=${PI05_ACTIONS_TO_EXECUTE}" "actions_to_keep=${PI05_ACTIONS_TO_KEEP}"
    "execute_in_n_steps=${PI05_EXECUTE_IN_N_STEPS}" "history_len=${PI05_HISTORY_LEN}"
    "votes_to_promote=${PI05_VOTES_TO_PROMOTE}" "num_steps=${PI05_NUM_STEPS}"
    "disable_fast_auxiliary=${PI05_DISABLE_FAST_AUXILIARY}" "proprioception_schema=${PI05_PROPRIOCEPTION_SCHEMA}"
    "use_task_checkpoint_mapping=${USE_PI05_TASK_CHECKPOINT_MAPPING}"
    "task_checkpoint_mapping=$([[ "${USE_PI05_TASK_CHECKPOINT_MAPPING}" == true ]] && printf '%s' "${PI05_TASK_CHECKPOINT_MAPPING}" || true)"
    "skip_action_chunk_rendering=${PI05_SKIP_ACTION_CHUNK_RENDERING:-false}"
    "run_output_root=${RUN_OUTPUT_ROOT}" "dry_run=${DRY_RUN}"
  )
  python3 - "${RUN_CONFIG_FILE}" "${DLC_RANK}" "${config[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = dict(value.split("=", 1) for value in sys.argv[3:])
if sys.argv[2] == "0":
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
else:
    expected = json.loads(path.read_text(encoding="utf-8"))
    differences = {key: (expected.get(key), value) for key, value in config.items() if expected.get(key) != value}
    if differences:
        raise SystemExit(f"Rank {sys.argv[2]} configuration differs from rank 0 (rank0, local): {differences}")
PY
}

resolve_scheduler_advertise_host() {
  local host="${DLC_SCHEDULER_ADVERTISE_HOST}"
  if [[ -z "${host}" || ( "${DLC_MULTINODE}" == true && "${host}" =~ ^(localhost|127\.) ) ]]; then
    host="$(hostname -I 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i !~ /^127\./) {print $i; exit}}')"
  fi
  [[ -n "${host}" ]] || host="$(hostname -f)"
  if [[ "${DLC_MULTINODE}" == true && "${host}" =~ ^(localhost|127\.) ]]; then
    echo "Rank-0 scheduler address is not reachable from other nodes: ${host}" >&2
    return 1
  fi
  printf '%s\n' "${host}"
}

start_dynamic_scheduler() {
  local port advertise_host scheduler_log deadline
  port="$(find_free_port "${DLC_SCHEDULER_PORT}")"
  advertise_host="$(resolve_scheduler_advertise_host)" || return 1
  scheduler_log="${LOG_DIR}/dynamic_scheduler_rank0_port${port}.log"
  setsid python3 "${DLC_SCHEDULER_SCRIPT}" \
    --host "${DLC_SCHEDULER_BIND_HOST}" \
    --port "${port}" \
    --queue-file "${ONLINE_QUEUE_FILE}" \
    --stop-file "${STOP_FILE}" \
    --journal "${SCHEDULER_JOURNAL_FILE}" \
    --token "${DLC_RUN_KEY}" \
    >"${scheduler_log}" 2>&1 &
  SCHEDULER_PID="$!"

  deadline=$((SECONDS + DLC_BARRIER_TIMEOUT))
  while (( SECONDS < deadline )); do
    kill -0 "${SCHEDULER_PID}" >/dev/null 2>&1 || {
      echo "Rank-0 dynamic scheduler exited; see ${scheduler_log}" >&2
      return 1
    }
    if curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
      SCHEDULER_URL="http://${advertise_host}:${port}"
      write_shared_marker "${SCHEDULER_ENDPOINT_FILE}" "${SCHEDULER_URL}"
      echo "Rank-0 dynamic scheduler: ${SCHEDULER_URL} (PID=${SCHEDULER_PID})"
      return 0
    fi
    sleep 1
  done
  echo "Timed out starting rank-0 dynamic scheduler; see ${scheduler_log}" >&2
  return 1
}

connect_dynamic_scheduler() {
  local deadline=$((SECONDS + DLC_BARRIER_TIMEOUT))
  wait_for_shared_file "${SCHEDULER_ENDPOINT_FILE}" "rank-0 dynamic scheduler endpoint"
  SCHEDULER_URL="$(<"${SCHEDULER_ENDPOINT_FILE}")"
  [[ "${SCHEDULER_URL}" =~ ^http://[^/]+:[0-9]+$ ]] || {
    echo "Invalid dynamic scheduler endpoint: ${SCHEDULER_URL}" >&2
    return 1
  }
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' --max-time 2 -fsS "${SCHEDULER_URL}/healthz" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Cannot reach rank-0 dynamic scheduler: ${SCHEDULER_URL}" >&2
  return 1
}

write_shared_marker() {
  local target="$1"
  shift
  local temporary="${target}.tmp.rank${DLC_RANK}.$$"
  mkdir -p "$(dirname "${target}")"
  printf '%s\n' "$*" >"${temporary}"
  mv -T -- "${temporary}" "${target}"
}

initialize_run_paths() {
  local -a timestamp_args=()
  local session_parent="${EVAL_LOG_ROOT}"
  if [[ -n "${RESUME_LOG_DIR}" ]]; then
    RESUME_LOG_DIR="$(cd "${RESUME_LOG_DIR}" && pwd)"
    RUN_OUTPUT_ROOT="$(python3 "${RESUME_HELPER}" source-root --log-dir "${RESUME_LOG_DIR}")"
    session_parent="${RUN_OUTPUT_ROOT}/.sessions"
  fi
  [[ "${DRY_RUN}" != true ]] || timestamp_args+=(--dry-run)
  RUN_TS="$(python3 "${RESUME_HELPER}" timestamp --output-root "${session_parent}" \
    --key "${DLC_RUN_KEY}" --rank "${DLC_RANK}" --world-size "${DLC_WORLD_SIZE}" \
    --timeout "${DLC_BARRIER_TIMEOUT}" "${timestamp_args[@]}")"

  SESSION_OUTPUT_ROOT="${session_parent}/${RUN_TS}"
  if [[ -n "${RESUME_LOG_DIR}" ]]; then
    GLOBAL_LOG_DIR="${RESUME_LOG_DIR}/.sessions/${RUN_TS}"
  else
    GLOBAL_LOG_DIR="${EVAL_LOG_PARENT}/pi05_behavior_2026_persistent_${RUN_TS}"
    RUN_OUTPUT_ROOT="${SESSION_OUTPUT_ROOT}"
  fi
  LOG_DIR="${GLOBAL_LOG_DIR}/rank-${DLC_RANK}"

  # Queue, summaries, stop signal and coordination markers are shared through
  # NAS. Process logs and PID files remain rank-local.
  QUEUE_FILE="${GLOBAL_LOG_DIR}/task_queue.tsv"
  QUEUE_DIR="${GLOBAL_LOG_DIR}/task_queues"
  ONLINE_QUEUE_FILE="${QUEUE_DIR}/online.tsv"
  WORKER_RECORD_DIR="${GLOBAL_LOG_DIR}/worker_records"
  CHUNK_RESULTS_FILE="${GLOBAL_LOG_DIR}/chunk_results.tsv"
  SCHEDULE_FILE="${GLOBAL_LOG_DIR}/task_schedule.tsv"
  ASSIGNMENTS_FILE="${GLOBAL_LOG_DIR}/task_assignments.tsv"
  RESULTS_FILE="${GLOBAL_LOG_DIR}/results.tsv"
  ATTEMPTS_FILE="${GLOBAL_LOG_DIR}/attempts.tsv"
  PID_DIR="${LOG_DIR}/pids"
  STOP_FILE="${GLOBAL_LOG_DIR}/stop_requested"
  ATTEMPTS_ROOT="${SESSION_OUTPUT_ROOT}/.attempts"
  CHUNKS_ROOT="${SESSION_OUTPUT_ROOT}/.chunks"
  RUN_MANIFEST="${RUN_OUTPUT_ROOT}/run_manifest.json"
  RUN_COMPLETE="${RUN_OUTPUT_ROOT}/run_complete.json"

  COORD_DIR="${GLOBAL_LOG_DIR}/coord"
  RUN_CONFIG_FILE="${COORD_DIR}/run_config.json"
  CHECKPOINT_READY_FILE="${COORD_DIR}/checkpoint.path"
  QUEUE_READY_FILE="${COORD_DIR}/queue.ready"
  START_FILE="${COORD_DIR}/start.ready"
  RANK_READY_FILE="${COORD_DIR}/rank-${DLC_RANK}.ready"
  RANK_DONE_FILE="${COORD_DIR}/rank-${DLC_RANK}.done"
  RANK_STATUS_FILE="${COORD_DIR}/rank-${DLC_RANK}.status"
  RANK_HEARTBEAT_FILE="${COORD_DIR}/rank-${DLC_RANK}.heartbeat"
  FINAL_STATUS_FILE="${COORD_DIR}/final.status"
  SCHEDULER_ENDPOINT_FILE="${COORD_DIR}/scheduler.endpoint"
  SCHEDULER_JOURNAL_FILE="${COORD_DIR}/scheduler.journal.jsonl"
  TIMING_FILE="${GLOBAL_LOG_DIR}/timing.tsv"
}

wait_for_shared_file() {
  local target="$1"
  local description="$2"
  local deadline=$((SECONDS + DLC_BARRIER_TIMEOUT))
  local heartbeat_rank=""
  local heartbeat_file=""
  local heartbeat_epoch now age
  local heartbeat_deadline=$((SECONDS + DLC_HEARTBEAT_STALE_TIMEOUT))

  # Startup barriers retain a fixed timeout. Rank completion and final
  # validation are governed by peer heartbeats instead.
  if [[ "${target}" =~ /rank-([0-9]+)\.(done|status)$ ]]; then
    heartbeat_rank="${BASH_REMATCH[1]}"
  elif [[ "${target}" == "${FINAL_STATUS_FILE}" ]]; then
    heartbeat_rank=0
  fi

  if [[ -n "${heartbeat_rank}" ]]; then
    heartbeat_file="${COORD_DIR}/rank-${heartbeat_rank}.heartbeat"
  fi

  while [[ ! -e "${target}" ]]; do
    if [[ -z "${heartbeat_rank}" && -e "${STOP_FILE}" ]]; then
      echo "Run stopped while waiting for ${description}: ${target}" >&2
      return 1
    fi
    if [[ -n "${heartbeat_rank}" ]]; then
      heartbeat_epoch=""

      if [[ -e "${heartbeat_file}" ]]; then
        heartbeat_epoch="$(
          sed -n 's/.*epoch=\([0-9][0-9]*\).*/\1/p' \
            "${heartbeat_file}" 2>/dev/null || true
        )"
      fi

      if [[ "${heartbeat_epoch}" =~ ^[0-9]+$ ]]; then
        now="$(date +%s)"
        age=$((now - heartbeat_epoch))
        (( age >= 0 )) || age=0

        if (( age > DLC_HEARTBEAT_STALE_TIMEOUT )); then
          echo "Rank ${heartbeat_rank} heartbeat is stale (${age}s); " \
               "while waiting for ${description}: ${target}" >&2
          return 1
        fi
      elif (( SECONDS >= heartbeat_deadline )); then
        echo "No valid heartbeat from rank ${heartbeat_rank} within " \
             "${DLC_HEARTBEAT_STALE_TIMEOUT}s; " \
             "while waiting for ${description}: ${target}" >&2
        return 1
      fi
    elif (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${description}: ${target}" >&2
      return 1
    fi

    sleep 2
  done
}

HEARTBEAT_PID=""

start_rank_heartbeat() {
  [[ -z "${HEARTBEAT_PID}" ]] || {
    echo "Rank heartbeat is already running: PID=${HEARTBEAT_PID}" >&2
    return 1
  }

  local launcher_pid="${BASHPID}"
  (
    sleep_pid=""
    trap 'kill "${sleep_pid}" 2>/dev/null || true; exit 0' INT TERM

    while kill -0 "${launcher_pid}" 2>/dev/null; do
      if ! write_shared_marker \
          "${RANK_HEARTBEAT_FILE}" \
          "rank=${DLC_RANK} epoch=$(date +%s) phase=alive"; then
        echo "Failed to refresh rank ${DLC_RANK} heartbeat; retrying." >&2
      fi

      sleep "${DLC_HEARTBEAT_INTERVAL}" &
      sleep_pid="$!"
      wait "${sleep_pid}" || true
    done
  ) &

  HEARTBEAT_PID="$!"
}

stop_rank_heartbeat() {
  local pid="${HEARTBEAT_PID:-}"
  [[ -n "${pid}" ]] || return 0

  kill "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
  HEARTBEAT_PID=""
}

SCHEDULER_PID=""
SCHEDULER_URL=""

stop_dynamic_scheduler() {
  local pid="${SCHEDULER_PID:-}"
  [[ -n "${pid}" ]] || return 0
  kill "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
  SCHEDULER_PID=""
}

scheduler_post() {
  local endpoint="$1"
  shift
  local deadline=$((SECONDS + DLC_SCHEDULER_REQUEST_TIMEOUT))
  local response
  local -a curl_args=(
    --noproxy '*'
    --connect-timeout 5
    --max-time 15
    -fsS
    -H "X-Scheduler-Token: ${DLC_RUN_KEY}"
    -X POST
  )
  while (( $# > 0 )); do
    curl_args+=(--data-urlencode "$1")
    shift
  done
  while (( SECONDS < deadline )); do
    if response="$(curl "${curl_args[@]}" "${SCHEDULER_URL}/${endpoint}")"; then
      printf '%s\n' "${response}"
      return 0
    fi
    sleep 2
  done
  echo "Scheduler request timed out: ${SCHEDULER_URL}/${endpoint}" >&2
  return 1
}

claim_next_task() {
  local worker="$1" response
  response="$(scheduler_post claim "worker=${worker}")" || return 2
  case "${response}" in
    assigned$'\t'*) printf '%s\n' "${response#*$'\t'}" ;;
    empty|stopped) return 1 ;;
    *)
      echo "Unexpected scheduler claim response for worker ${worker}: ${response}" >&2
      return 2
      ;;
  esac
}

complete_task_claim() {
  local worker="$1" queue_order="$2" completion_status="$3" response
  response="$(scheduler_post complete \
    "worker=${worker}" \
    "queue_order=${queue_order}" \
    "status=${completion_status}")" || return 1
  case "${response}" in
    complete|already_complete) return 0 ;;
    *)
      echo "Unexpected scheduler completion response for worker ${worker}: ${response}" >&2
      return 1
      ;;
  esac
}

worker_record_path() {
  local kind="$1" worker="$2"
  printf '%s/worker-%s.%s.tsv\n' "${WORKER_RECORD_DIR}" "${worker}" "${kind}"
}

record_assignment() {
  local worker="$1" gpu="$2" task_id="$3" task_name="$4" chunk_indices="$5" timeout_steps="$6" chunk_steps="$7" order="$8" position="$9"
  local target
  target="$(worker_record_path assignments "${worker}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${worker}" "${gpu}" "${position}" "${order}" \
    "${task_id}" "${task_name}" "${chunk_indices}" "${timeout_steps}" >>"${target}"
}

kill_process_group() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" >/dev/null 2>&1 || return 0
  kill -TERM -- "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
  local deadline=$((SECONDS + 10))
  while kill -0 "${pid}" >/dev/null 2>&1 && (( SECONDS < deadline )); do sleep 1; done
  kill -0 "${pid}" >/dev/null 2>&1 || return 0
  kill -KILL -- "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
}

SERVER_PYTHONPATH="${PI05_REPO}/src:${PI05_REPO}/openpi/src:${CURRENT_PRETRAIN_ROOT}:${PYTHONPATH:-}"
EVAL_PYTHONPATH="${BEHAVIOR_ROOT}/OmniGibson:${BEHAVIOR_ROOT}/bddl3:${BEHAVIOR_ROOT}/joylo:${BEHAVIOR_ROOT}:${PI05_REPO}/src:${CURRENT_PRETRAIN_ROOT}:${PYTHONPATH:-}"
export CURRENT_PRETRAIN_ROOT OMNIGIBSON_DATA_PATH
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,::1"

LAUNCHED_PID=""

launch_server() {
  local gpu_id="$1" worker="$2" port="$3" log_file="$4"
  local local_worker=$((worker % NUM_GPUS))
  local -a args=(
    "${PI05_SERVER_SCRIPT}"
    --host "${PI05_SERVER_HOST}"
    --port "${port}"
    --inference-only
    --num-steps "${PI05_NUM_STEPS}"
    --dynamic-batching
    --dynamic-batch-max-size "${PI05_DYNAMIC_BATCH_MAX_SIZE}"
    --dynamic-batch-wait-ms "${PI05_DYNAMIC_BATCH_WAIT_MS}"
    --dynamic-batch-granularity "${PI05_DYNAMIC_BATCH_GRANULARITY}"
    --norm-stats-path "${PI05_NORM_STATS_PATH}"
  )
  [[ "${PI05_DISABLE_FAST_AUXILIARY}" == true ]] \
    && args+=(--disable-fast-auxiliary) \
    || args+=(--no-disable-fast-auxiliary)
  [[ "${USE_PI05_TASK_CHECKPOINT_MAPPING}" == true ]] \
    && args+=(--task-checkpoint-mapping "${PI05_TASK_CHECKPOINT_MAPPING}")
  args+=(policy:checkpoint --policy.config "${PI05_POLICY_CONFIG}" --policy.dir "${PI05_RESOLVED_POLICY_DIR}")

  (
    cd "${PI05_REPO}"
    exec env \
      OMP_NUM_THREADS="${SERVER_CPU_NUM_THREADS}" \
      MKL_NUM_THREADS="${SERVER_CPU_NUM_THREADS}" \
      OPENBLAS_NUM_THREADS="${SERVER_CPU_NUM_THREADS}" \
      NUMEXPR_NUM_THREADS="${SERVER_CPU_NUM_THREADS}" \
      TOKENIZERS_PARALLELISM=false \
      PYTHONUNBUFFERED=1 \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}" \
      XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONPATH="${SERVER_PYTHONPATH}" \
      setsid taskset -c "${SERVER_CPU_LISTS[${local_worker}]}" "${PI05_PYTHON}" "${args[@]}"
  ) >>"${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
}

wait_for_server() {
  local pid="$1" port="$2" deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    kill -0 "${pid}" >/dev/null 2>&1 || return 1
    curl --noproxy '*' --max-time 2 -fsS "http://${PI05_CLIENT_HOST}:${port}/healthz" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

launch_eval() {
  local gpu_id="$1" worker="$2" port="$3" task_name="$4" output_dir="$5" log_file="$6" instance_indices="$7"
  local local_worker=$((worker % NUM_GPUS))
  local -a chunk_indices
  read -r -a chunk_indices <<<"${instance_indices//,/ }"
  local -a args=(
    -m omnigibson.eval.eval_persistent
    --worker-dir "${PERSISTENT_WORKER_DIR}"
    --task-name "${task_name}"
    --host "${PI05_CLIENT_HOST}"
    --port "${port}"
    --mode public_test
    --instance-indices "${chunk_indices[@]}"
    --num-rollouts 1
    --num-envs "${VECTOR_ENVS_PER_PROCESS}"
    --seed "${EVAL_SEED}"
    --output-dir "${output_dir}"
    --cpu-affinity "${EVAL_CPU_LISTS[${local_worker}]}"
    --cpu-num-threads "${EVAL_CPU_NUM_THREADS}"
    --actions-to-execute "${PI05_ACTIONS_TO_EXECUTE}"
    --actions-to-keep "${PI05_ACTIONS_TO_KEEP}"
    --execute-in-n-steps "${PI05_EXECUTE_IN_N_STEPS}"
    --stage-history-len "${PI05_HISTORY_LEN}"
    --stage-votes-to-promote "${PI05_VOTES_TO_PROMOTE}"
    --pi05-proprioception-schema "${PI05_PROPRIOCEPTION_SCHEMA}"
    --pi05-base-velocity-frame "${PI05_BASE_VELOCITY_FRAME}"
  )
  [[ -z "${EVAL_MAX_STEPS}" ]] || args+=(--max-steps "${EVAL_MAX_STEPS}")
  [[ -n "${EVAL_MAX_STEPS}" ]] || args+=(--max-steps-multiplier "${EVAL_MAX_STEPS_MULTIPLIER}")
  [[ "${EVAL_WRITE_VIDEO}" == true ]] && args+=(--write-video) || args+=(--no-write-video)
  [[ "${EVAL_PARTIAL_SCENE_LOAD}" == true ]] && args+=(--partial-scene-load) || args+=(--no-partial-scene-load)
  [[ "${PI05_APPLY_EVAL_TRICKS}" == true ]] && args+=(--apply-eval-tricks) || args+=(--no-apply-eval-tricks)
  args+=(--headless --no-render-viewer-camera)

  (
    behavior_env_dir="$(dirname "$(dirname "${BEHAVIOR_PYTHON}")")"
    conda_root="$(dirname "$(dirname "${behavior_env_dir}")")"
    set +u
    source "${DRIVER_FIX_SCRIPT}"
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "${behavior_env_dir}"
    set -u
    gpu_inventory=""
    if ! command -v nvidia-smi >/dev/null 2>&1 \
      || ! gpu_inventory="$(nvidia-smi -L)" \
      || ! grep -Fq "GPU ${gpu_id}:" <<<"${gpu_inventory}"; then
      echo "GPU ${gpu_id} is unavailable after sourcing ${DRIVER_FIX_SCRIPT}" >&2
      [[ -z "${gpu_inventory}" ]] || printf '%s\n' "${gpu_inventory}" >&2
      exit 127
    fi
    cd "${BEHAVIOR_ROOT}"
    exec env \
      OMP_NUM_THREADS="${EVAL_CPU_NUM_THREADS}" \
      MKL_NUM_THREADS="${EVAL_CPU_NUM_THREADS}" \
      OPENBLAS_NUM_THREADS="${EVAL_CPU_NUM_THREADS}" \
      NUMEXPR_NUM_THREADS="${EVAL_CPU_NUM_THREADS}" \
      VECLIB_MAXIMUM_THREADS="${EVAL_CPU_NUM_THREADS}" \
      PYTHONUNBUFFERED=1 \
      PYTHONHASHSEED="${EVAL_SEED}" \
      OMNIGIBSON_HEADLESS=1 \
      OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH}" \
      PI05_CORRECTION_RULES_PATH="${PI05_REPO}/src/b1k/shared/correction_rules.py" \
      MPLCONFIGDIR="${LOG_DIR}/matplotlib-worker-${worker}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONPATH="${EVAL_PYTHONPATH}" \
      setsid taskset -c "${EVAL_CPU_LISTS[${local_worker}]}" "${BEHAVIOR_PYTHON}" "${args[@]}"
  ) >"${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
}

validate_chunk_artifacts() {
  local task_name="$1" chunk_indices="$2" attempt_dir="$3"
  local -a indices
  read -r -a indices <<<"${chunk_indices//,/ }"
  [[ -f "${attempt_dir}/evaluation_complete.json" && -d "${attempt_dir}/json" ]] || return 1
  local index json_file
  for index in "${indices[@]}"; do
    json_file="${attempt_dir}/json/${task_name}_$((301 + index))_0.json"
    [[ -f "${json_file}" ]] || return 1
  done
}

record_result() {
  local state="$1" worker="$2" gpu="$3" task_id="$4" task_name="$5" status="$6" output_dir="$7"
  local target
  if [[ "${worker}" =~ ^[0-9]+$ ]]; then
    target="$(worker_record_path results "${worker}")"
  else
    target="${RESULTS_FILE}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${state}" "${worker}" "${gpu}" \
    "${task_id}" "${task_name}" "${status}" "${output_dir}" >>"${target}"
}

record_attempt() {
  local worker="$1" gpu="$2" task_id="$3" task_name="$4" attempt="$5"
  local eval_status="$6" validation_status="$7" fatal_detected="$8" promoted="$9"
  local attempt_dir="${10}" final_dir="${11}" eval_log="${12}" validation_log="${13}"
  local target
  target="$(worker_record_path attempts "${worker}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${worker}" "${gpu}" "${task_id}" "${task_name}" \
    "${attempt}" "${eval_status}" "${validation_status}" "${fatal_detected}" "${promoted}" \
    "${attempt_dir}" "${final_dir}" "${eval_log}|${validation_log}" >>"${target}"
}

log_has_fatal_error() {
  local log_file="$1"
  grep -Eq \
    'Failed to get contact force matrix from backend|Fatal Python error|Segmentation fault|ERROR_INCOMPATIBLE_DRIVER|Failed to create any GPU devices|Dynamic batch.*(failed|error)' \
    "${log_file}"
}

stop_eval_worker() {
  local worker="$1" graceful="${2:-false}" deadline
  if [[ -n "${PERSISTENT_EVAL_PID}" ]]; then
    if [[ "${graceful}" == true ]]; then
      : >"${PERSISTENT_WORKER_DIR}/shutdown"
      deadline=$((SECONDS + EVAL_WORKER_SHUTDOWN_TIMEOUT))
      while kill -0 "${PERSISTENT_EVAL_PID}" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    fi
    kill_process_group "${PERSISTENT_EVAL_PID}"
    wait "${PERSISTENT_EVAL_PID}" 2>/dev/null || true
    rm -f "${PID_DIR}/eval_worker${worker}.pid"
    PERSISTENT_EVAL_PID=""
  fi
}

run_persistent_chunk() {
  local gpu_id="$1" worker="$2" port="$3" task_name="$4" attempt_dir="$5" log_file="$6" indices="$7" server_pid="$8" request_id="$9"
  local start_line=1 started=${SECONDS} response_id response_status process_status
  if [[ -n "${PERSISTENT_EVAL_PID}" ]] && ! kill -0 "${PERSISTENT_EVAL_PID}" 2>/dev/null; then
    stop_eval_worker "${worker}"
  fi
  if [[ -z "${PERSISTENT_EVAL_PID}" ]]; then
    PERSISTENT_GENERATION=$((PERSISTENT_GENERATION + 1))
    PERSISTENT_WORKER_DIR="${LOG_DIR}/persistent/worker-${worker}/generation-${PERSISTENT_GENERATION}"
    PERSISTENT_LOG="${PERSISTENT_WORKER_DIR}/evaluator.log"
    mkdir -p "${PERSISTENT_WORKER_DIR}"
    launch_eval "${gpu_id}" "${worker}" "${port}" "${task_name}" "${attempt_dir}" "${PERSISTENT_LOG}" "${indices}"
    PERSISTENT_EVAL_PID="${LAUNCHED_PID}"
    echo "${PERSISTENT_EVAL_PID}" >"${PID_DIR}/eval_worker${worker}.pid"
  else
    start_line=$(( $(wc -l <"${PERSISTENT_LOG}") + 1 ))
  fi
  rm -f "${PERSISTENT_WORKER_DIR}/response.tsv"
  printf '%s\t%s\t%s\t%s\n' "${request_id}" "${task_name}" "${attempt_dir}" "${indices}" \
    >"${PERSISTENT_WORKER_DIR}/request.tsv.tmp"
  mv "${PERSISTENT_WORKER_DIR}/request.tsv.tmp" "${PERSISTENT_WORKER_DIR}/request.tsv"

  PERSISTENT_STATUS=1
  while true; do
    if [[ -f "${PERSISTENT_WORKER_DIR}/response.tsv" ]]; then
      IFS=$'\t' read -r response_id response_status <"${PERSISTENT_WORKER_DIR}/response.tsv"
      if [[ "${response_id}" == "${request_id}" && "${response_status}" =~ ^[01]$ ]]; then
        PERSISTENT_STATUS="${response_status}"
      else
        PERSISTENT_STATUS=126
      fi
      break
    fi
    if ! kill -0 "${PERSISTENT_EVAL_PID}" 2>/dev/null; then
      if wait "${PERSISTENT_EVAL_PID}"; then process_status=1; else process_status=$?; fi
      PERSISTENT_STATUS="${process_status}"
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null || [[ -e "${STOP_FILE}" ]]; then
      PERSISTENT_STATUS=125
      break
    fi
    if (( EVAL_REQUEST_TIMEOUT > 0 && SECONDS - started >= EVAL_REQUEST_TIMEOUT )); then
      PERSISTENT_STATUS=124
      break
    fi
    sleep 1
  done
  if (( PERSISTENT_STATUS != 0 )); then stop_eval_worker "${worker}"; fi
  # Keep a per-attempt log for the existing fatal-pattern and promotion checks.
  tail -n +"${start_line}" "${PERSISTENT_LOG}" >"${log_file}"
  printf '\nPersistent worker request=%s status=%s process_log=%s\n' \
    "${request_id}" "${PERSISTENT_STATUS}" "${PERSISTENT_LOG}" >>"${log_file}"
}

worker_loop() {
  local worker="$1" gpu_id="$2" port="$3" server_pid="$4"
  local task_line task_id task_name chunk_indices timeout_steps chunk_steps queue_order output_dir attempt_parent attempt_dir log_file validation_log
  local assignment_position=0
  local eval_status validation_status fatal_detected promoted terminal_status attempt task_succeeded
  local PERSISTENT_EVAL_PID="" PERSISTENT_WORKER_DIR="" PERSISTENT_LOG="" PERSISTENT_GENERATION=0 PERSISTENT_STATUS=1
  local claim_status completion_status worker_status=0
  while true; do
    if task_line="$(claim_next_task "${worker}")"; then
      claim_status=0
    else
      claim_status=$?
    fi
    if (( claim_status == 1 )); then
      break
    elif (( claim_status != 0 )); then
      : >"${STOP_FILE}"
      stop_eval_worker "${worker}"
      return 1
    fi
    IFS=$'\t' read -r task_id task_name chunk_indices timeout_steps chunk_steps queue_order <<<"${task_line}"
    assignment_position=$((assignment_position + 1))
    record_assignment "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${chunk_indices}" \
      "${timeout_steps}" "${chunk_steps}" "${queue_order}" "${assignment_position}"
    output_dir="${CHUNKS_ROOT}/task-${task_id}_${task_name}/instances-${chunk_indices//,/_}"
    mkdir -p "$(dirname "${output_dir}")"
    attempt_parent="${ATTEMPTS_ROOT}/task-${task_id}_${task_name}/instances-${chunk_indices//,/_}"
    mkdir -p "${attempt_parent}"
    task_succeeded=false
    terminal_status=124

    for ((attempt = 1; attempt <= EVAL_MAX_TASK_ATTEMPTS; attempt++)); do
      attempt_dir="${attempt_parent}/attempt-${attempt}"
      log_file="${LOG_DIR}/eval_worker${worker}_gpu${gpu_id}_task${task_id}_${task_name}_instances-${chunk_indices//,/_}_attempt${attempt}.log"
      validation_log="${LOG_DIR}/validate_worker${worker}_task${task_id}_${task_name}_instances-${chunk_indices//,/_}_attempt${attempt}.log"
      echo "[worker ${worker}/gpu${gpu_id}] task=${task_id}:${task_name} chunk=${chunk_indices} attempt=${attempt}/${EVAL_MAX_TASK_ATTEMPTS}"

      if [[ -e "${attempt_dir}" || -e "${output_dir}" ]]; then
        echo "Refusing to overwrite existing task output: ${attempt_dir} or ${output_dir}" >"${validation_log}"
        record_attempt "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${attempt}" \
          126 1 false false "${attempt_dir}" "${output_dir}" "${log_file}" "${validation_log}"
        terminal_status=126
        break
      fi

      run_persistent_chunk "${gpu_id}" "${worker}" "${port}" "${task_name}" "${attempt_dir}" "${log_file}" \
        "${chunk_indices}" "${server_pid}" "${assignment_position}-${attempt}"
      eval_status="${PERSISTENT_STATUS}"

      if validate_chunk_artifacts "${task_name}" "${chunk_indices}" "${attempt_dir}" >"${validation_log}" 2>&1; then
        validation_status=0
      else
        validation_status=$?
      fi
      fatal_detected=false
      log_has_fatal_error "${log_file}" && fatal_detected=true
      promoted=false

      if (( eval_status == 0 && validation_status == 0 )) && [[ "${fatal_detected}" == false ]]; then
        if mv -T -- "${attempt_dir}" "${output_dir}"; then
          promoted=true
          terminal_status=0
          task_succeeded=true
        else
          terminal_status=126
        fi
      elif (( eval_status != 0 )); then
        terminal_status="${eval_status}"
      elif (( validation_status != 0 )); then
        terminal_status=124
      else
        terminal_status=125
      fi

      record_attempt "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${attempt}" \
        "${eval_status}" "${validation_status}" "${fatal_detected}" "${promoted}" \
        "${attempt_dir}" "${output_dir}" "${log_file}" "${validation_log}"

      if [[ "${task_succeeded}" == true ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(date '+%Y-%m-%d %H:%M:%S')" "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${chunk_indices}" \
          >>"$(worker_record_path chunks "${worker}")"
        break
      fi
      echo "[worker ${worker}] task ${task_id}:${task_name} attempt ${attempt} failed; see ${log_file} and ${validation_log}" >&2
      stop_eval_worker "${worker}"
      kill -0 "${server_pid}" >/dev/null 2>&1 || break
    done

    completion_status=failed
    [[ "${task_succeeded}" == true ]] && completion_status=success
    if [[ "${task_succeeded}" != true && "${EVAL_FAIL_FAST}" == true ]]; then
      : >"${STOP_FILE}"
    fi
    if ! complete_task_claim "${worker}" "${queue_order}" "${completion_status}"; then
      : >"${STOP_FILE}"
      stop_eval_worker "${worker}"
      return 1
    fi

    if [[ "${task_succeeded}" != true ]]; then
      worker_status=1
      record_result failed "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${terminal_status}" "${output_dir}"
      echo "[worker ${worker}] task ${task_id}:${task_name} exhausted ${EVAL_MAX_TASK_ATTEMPTS} attempt(s)." >&2
      if [[ "${EVAL_FAIL_FAST}" == true ]]; then
        : >"${STOP_FILE}"
        stop_eval_worker "${worker}"
        return 1
      fi
    fi
    kill -0 "${server_pid}" >/dev/null 2>&1 || {
      echo "Server ${server_pid} exited." >&2; stop_eval_worker "${worker}"; return 1;
    }
  done
  stop_eval_worker "${worker}" true
  return "${worker_status}"
}

WORKER_PIDS=()
SERVER_PIDS=()
CLEANUP_DONE=false

cleanup() {
  [[ "${CLEANUP_DONE}" == true ]] && return 0
  CLEANUP_DONE=true
  local pid_file pid
  echo "Cleaning up evaluator and policy server processes..."
  for pid in "${WORKER_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  while IFS= read -r pid_file; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(<"${pid_file}")"
    kill_process_group "${pid}"
    wait "${pid}" 2>/dev/null || true
  done < <(find "${PID_DIR}" -type f -name '*.pid' 2>/dev/null | sort)
  for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

launcher_exit() {
  local status=$?
  trap - INT TERM EXIT
  # Normal local completion must never stop other ranks. Unexpected exits
  # publish a failure and release peers that are still at a startup barrier.
  if (( status != 0 )); then
    : >"${STOP_FILE}"
    write_shared_marker "${RANK_STATUS_FILE}" "${status}" || true
    write_shared_marker "${RANK_DONE_FILE}" "rank=${DLC_RANK} status=${status}" || true
    if (( DLC_RANK == 0 )) && [[ ! -e "${FINAL_STATUS_FILE}" ]]; then
      write_shared_marker "${FINAL_STATUS_FILE}" 1 || true
    fi
  fi
  cleanup
  stop_dynamic_scheduler
  stop_rank_heartbeat
  exit "${status}"
}

validate_environment
if [[ "${DRY_RUN}" == true ]] && (( DLC_RANK != 0 )); then
  echo "Dry-run planning runs on rank 0; no processes started on rank ${DLC_RANK}."
  exit 0
fi
if [[ "${DRY_RUN}" != true ]]; then
  validate_gpu_runtime
fi

initialize_run_paths
mkdir -p "${GLOBAL_LOG_DIR}" "${LOG_DIR}" "${PID_DIR}" "${COORD_DIR}"

# Rank 0 alone creates or truncates shared scheduler state. Other ranks wait
# until the queue, summary tables and immutable manifest are complete.
if (( DLC_RANK == 0 )); then
  if [[ -e "${QUEUE_READY_FILE}" || -e "${START_FILE}" \
        || -e "${FINAL_STATUS_FILE}" || -e "${ONLINE_QUEUE_FILE}" \
        || -e "${SCHEDULER_ENDPOINT_FILE}" || -e "${SCHEDULER_JOURNAL_FILE}" \
        || ( -e "${RUN_MANIFEST}" && -z "${RESUME_LOG_DIR}" ) || -e "${RUN_CONFIG_FILE}" \
        || -e "${STOP_FILE}" ]]; then
    echo "Refusing to reuse non-empty multi-node run state: ${DLC_RUN_KEY}" >&2
    exit 1
  fi
fi

# Install failure propagation only after rank 0 has accepted fresh session paths.
trap launcher_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if (( DLC_RANK == 0 )); then
  check_run_configuration
  cp -- "${RUN_CONFIG_FILE}" "${SESSION_OUTPUT_ROOT}/run_config.json"
  # Rank 0 alone may convert a sharded checkpoint. Other nodes wait for the
  # published path instead of competing for a checkpoint lock on NAS.
  resolve_policy_checkpoint
  write_shared_marker "${CHECKPOINT_READY_FILE}" "${PI05_RESOLVED_POLICY_DIR}"
  mkdir -p "${RUN_OUTPUT_ROOT}" "${ATTEMPTS_ROOT}"
  build_task_queue

  TASK_COUNT="$(wc -l <"${QUEUE_FILE}" | tr -d ' ')"
  (( TASK_COUNT > 0 )) || {
    echo "No tasks queued." >&2
    exit 1
  }

  printf 'timestamp\tstate\tworker\tgpu\ttask_id\ttask_name\tstatus\toutput_dir\n' \
    >"${RESULTS_FILE}"
  printf 'timestamp\tworker\tgpu\ttask_id\ttask_name\tattempt\tevaluator_status\tvalidation_status\tfatal_log\tpromoted\tattempt_dir\tfinal_dir\tlogs\n' \
    >"${ATTEMPTS_FILE}"
  printf 'timestamp\tworker\tgpu\ttask_id\ttask_name\tinstance_indices\n' \
    >"${CHUNK_RESULTS_FILE}"
  printf 'timestamp\tworker\tgpu\tworker_position\tqueue_order\ttask_id\ttask_name\tinstance_indices\ttimeout_steps\n' \
    >"${ASSIGNMENTS_FILE}"

  mkdir -p "${WORKER_RECORD_DIR}"
  for ((worker_index = 0; worker_index < TOTAL_SCHEDULER_SLOTS; worker_index++)); do
    : >"$(worker_record_path assignments "${worker_index}")"
    : >"$(worker_record_path attempts "${worker_index}")"
    : >"$(worker_record_path results "${worker_index}")"
    : >"$(worker_record_path chunks "${worker_index}")"
  done

  manifest_video_arg=--no-write-video
  [[ "${EVAL_WRITE_VIDEO}" == true ]] && manifest_video_arg=--write-video

  if [[ -z "${RESUME_LOG_DIR}" ]]; then
    python3 "${OUTPUT_VALIDATOR}" create-manifest \
      --queue-file "${QUEUE_FILE}" \
      --output "${RUN_MANIFEST}" \
      --run-output-root "${RUN_OUTPUT_ROOT}" \
      --mode public_test \
      --instance-indices "${INSTANCE_INDEX_LIST[@]}" \
      --num-rollouts 1 \
      --num-vector-envs "${VECTOR_ENVS_PER_PROCESS}" \
      "${manifest_video_arg}"

    chmod 0444 "${RUN_MANIFEST}"
  fi

  resume_args=()
  [[ -z "${RESUME_LOG_DIR}" ]] || resume_args+=(--resume-log-dir "${RESUME_LOG_DIR}")
  [[ "${DRY_RUN}" != true ]] || resume_args+=(--dry-run)
  python3 "${RESUME_HELPER}" prepare \
    --run-root "${RUN_OUTPUT_ROOT}" --log-root "${GLOBAL_LOG_DIR}" \
    --session-root "${SESSION_OUTPUT_ROOT}" "${resume_args[@]}"

  write_shared_marker "${QUEUE_READY_FILE}" \
    "rank=0 tasks=${TASK_COUNT} instances=${#INSTANCE_INDEX_LIST[@]}"
else
  wait_for_shared_file \
    "${QUEUE_READY_FILE}" \
    "rank-0 queue initialization"
  check_run_configuration
fi

PI05_RESOLVED_POLICY_DIR="$(<"${CHECKPOINT_READY_FILE}")"
is_inference_checkpoint "${PI05_RESOLVED_POLICY_DIR}" || {
  echo "Rank ${DLC_RANK} cannot read rank-0 checkpoint: ${PI05_RESOLVED_POLICY_DIR}" >&2
  exit 1
}

for shared_file in \
    "${QUEUE_FILE}" \
    "${ONLINE_QUEUE_FILE}" \
    "${SCHEDULE_FILE}" \
    "${RESULTS_FILE}" \
    "${ATTEMPTS_FILE}" \
    "${CHUNK_RESULTS_FILE}" \
    "${ASSIGNMENTS_FILE}" \
    "${RUN_MANIFEST}"; do
  [[ -e "${shared_file}" ]] || {
    echo "Shared run file is missing after initialization: ${shared_file}" >&2
    exit 1
  }
done

TASK_COUNT="$(wc -l <"${QUEUE_FILE}" | tr -d ' ')"
(( TASK_COUNT > 0 )) || {
  echo "No tasks queued." >&2
  exit 1
}

INSTANCE_COUNT="${#INSTANCE_INDEX_LIST[@]}"
EXPECTED_RESULTS=$((TASK_COUNT * INSTANCE_COUNT))

echo "PI0.5 accelerated BEHAVIOR 2026 official evaluation:"
echo "  protocol: ${TASK_COUNT} tasks x ${INSTANCE_COUNT} public instances x 1 rollout = ${EXPECTED_RESULTS} outputs"
echo "  supported task IDs: ${TASK_IDS:-0-99}"
echo "  public instance indices: ${INSTANCE_INDEX_LIST[*]}"
echo "  timeout: ${EVAL_MAX_STEPS:-official task-specific ${EVAL_MAX_STEPS_MULTIPLIER}x mean human length}"
echo "  profile / write video: ${EVAL_PROFILE} / ${EVAL_WRITE_VIDEO}"
echo "  official dynamics: physics=120 Hz, render/action=30 Hz"
echo "  local GPUs: ${GPU_ID_LIST[*]}"
echo "  DLC topology: ${DLC_WORLD_SIZE} nodes x ${NUM_GPUS} local GPUs = ${TOTAL_SCHEDULER_SLOTS} GPU workers"
echo "  DLC rank: ${DLC_RANK}/${DLC_WORLD_SIZE}"
echo "  DLC run key / directory name: ${DLC_RUN_KEY} / ${RUN_TS}"
echo "  topology: 1 persistent server + 1 Isaac Sim process x ${VECTOR_ENVS_PER_PROCESS} vector envs per GPU; joint batch-2 policy requests"
echo "  scheduler: rank-0 central HTTP longest-task-first queue (${INSTANCE_CHUNK_SIZE} instances/chunk)"
awk -F '\t' 'NR > 1 {steps += $7; count++} END {
  printf "    pending chunks=%d estimated_steps=%d\n", count + 0, steps + 0
}' "${SCHEDULE_FILE}"
echo "  dynamic batch max/wait/granularity: ${PI05_DYNAMIC_BATCH_MAX_SIZE}/${PI05_DYNAMIC_BATCH_WAIT_MS}ms/${PI05_DYNAMIC_BATCH_GRANULARITY}"
echo "  CPU cores/threads per evaluator: ${EVAL_CPU_CORES}/${EVAL_CPU_NUM_THREADS}"
echo "  CPU threads per server: ${SERVER_CPU_NUM_THREADS}"
echo "  PI0.5 repo: ${PI05_REPO}"
echo "  server entrypoint: ${PI05_SERVER_SCRIPT}"
echo "  server lifecycle: one persistent PID per worker, reused across queued tasks"
echo "  Isaac lifecycle: persistent PID; same-task env reuse, cross-task og.clear() + env rebuild"
echo "  request timeout: ${EVAL_REQUEST_TIMEOUT}s (0 disables); worker shutdown timeout: ${EVAL_WORKER_SHUTDOWN_TIMEOUT}s"
echo "  policy config: ${PI05_POLICY_CONFIG}"
echo "  requested checkpoint: ${PI05_POLICY_DIR}"
echo "  resolved checkpoint: ${PI05_RESOLVED_POLICY_DIR}"
echo "  checkpoint auto-convert: ${PI05_AUTO_CONVERT_CKPT} (${PI05_CONVERT_SCRIPT})"
echo "  norm stats: ${PI05_NORM_STATS_PATH}"
echo "  checkpoint mapping: enabled=${USE_PI05_TASK_CHECKPOINT_MAPPING}, path=${PI05_TASK_CHECKPOINT_MAPPING}"
echo "  proprioception schema: ${PI05_PROPRIOCEPTION_SCHEMA}"
echo "  base velocity frame: ${PI05_BASE_VELOCITY_FRAME}"
echo "  environment seed: ${EVAL_SEED}"
echo "  behavior env: ${BEHAVIOR_ENV_DIR}"
echo "  GPU driver activation: ${DRIVER_FIX_SCRIPT} (sourced on evaluator process startup)"
echo "  PI0.5 env: ${PI05_ENV_DIR}"
echo "  behavior Python: ${BEHAVIOR_PYTHON}"
echo "  PI0.5 Python: ${PI05_PYTHON}"
echo "  outputs: ${RUN_OUTPUT_ROOT}"
echo "  resume log directory: ${RESUME_LOG_DIR:-none (fresh evaluation)}"
echo "  instance recovery report: ${SESSION_OUTPUT_ROOT}/resume.json"
echo "  immutable manifest: ${RUN_MANIFEST}"
echo "  attempts per instance chunk: ${EVAL_MAX_TASK_ATTEMPTS}"
echo "  scheduler preferred endpoint: ${DLC_SCHEDULER_ADVERTISE_HOST:-rank-0 auto IP}:${DLC_SCHEDULER_PORT}"
echo "  logs: ${LOG_DIR}"

if [[ "${DRY_RUN}" == true ]]; then
  echo
  echo "Dry-run central dynamic longest-task-first schedule:"
  cat "${SCHEDULE_FILE}"
  echo "Dry run complete; no server or simulator process was started."
  exit 0
fi

if (( DLC_RANK == 0 )); then
  start_dynamic_scheduler
fi
connect_dynamic_scheduler

SERVER_PORTS=()
SERVER_LOGS=()
LOCAL_WORKER_COUNT="${NUM_GPUS}"
[[ -s "${ONLINE_QUEUE_FILE}" ]] || LOCAL_WORKER_COUNT=0

for ((local_worker = 0; local_worker < LOCAL_WORKER_COUNT; local_worker++)); do
  global_worker=$((DLC_RANK * NUM_GPUS + local_worker))
  gpu_id="${GPU_ID_LIST[${local_worker}]}"
  port="$(find_free_port "$((PORT_BASE + local_worker * PORT_STRIDE))")"
  server_log="${LOG_DIR}/server_worker${global_worker}_gpu${gpu_id}_port${port}.log"

  echo "[rank ${DLC_RANK}/server ${global_worker}/gpu${gpu_id}] loading persistent checkpoint ${PI05_RESOLVED_POLICY_DIR} on port ${port}"

  launch_server \
    "${gpu_id}" \
    "${global_worker}" \
    "${port}" \
    "${server_log}"

  server_pid="${LAUNCHED_PID}"
  SERVER_PIDS+=("${server_pid}")
  SERVER_PORTS+=("${port}")
  SERVER_LOGS+=("${server_log}")
  echo "${server_pid}" >"${PID_DIR}/server${global_worker}.pid"
done

# Every node loads its local servers concurrently. Evaluators start only
# after all ranks have published healthy policy servers.
for ((local_worker = 0; local_worker < LOCAL_WORKER_COUNT; local_worker++)); do
  global_worker=$((DLC_RANK * NUM_GPUS + local_worker))

  wait_for_server \
    "${SERVER_PIDS[${local_worker}]}" \
    "${SERVER_PORTS[${local_worker}]}" || {
      echo "Server ${global_worker} failed to become ready; see ${SERVER_LOGS[${local_worker}]}" >&2
      exit 1
    }
done

write_shared_marker "${RANK_READY_FILE}" \
  "rank=${DLC_RANK} servers=${LOCAL_WORKER_COUNT} epoch=$(date +%s)"

if (( DLC_RANK == 0 )); then
  for ((rank_index = 0; rank_index < DLC_WORLD_SIZE; rank_index++)); do
    wait_for_shared_file \
      "${COORD_DIR}/rank-${rank_index}.ready" \
      "rank ${rank_index} server readiness"
  done

  evaluation_start_epoch="$(date +%s)"
  printf 'metric\tvalue\n' >"${TIMING_FILE}"
  printf 'evaluation_start_epoch\t%s\n' \
    "${evaluation_start_epoch}" >>"${TIMING_FILE}"

  write_shared_marker "${START_FILE}" "${evaluation_start_epoch}"
else
  wait_for_shared_file \
    "${START_FILE}" \
    "rank-0 evaluation start barrier"
fi

EVALUATION_START_EPOCH="$(<"${START_FILE}")"
[[ "${EVALUATION_START_EPOCH}" =~ ^[0-9]+$ ]] || {
  echo "Invalid evaluation start marker: ${START_FILE}" >&2
  exit 1
}

start_rank_heartbeat

for ((local_worker = 0; local_worker < LOCAL_WORKER_COUNT; local_worker++)); do
  global_worker=$((DLC_RANK * NUM_GPUS + local_worker))

  worker_loop \
    "${global_worker}" \
    "${GPU_ID_LIST[${local_worker}]}" \
    "${SERVER_PORTS[${local_worker}]}" \
    "${SERVER_PIDS[${local_worker}]}" &

  WORKER_PIDS+=("$!")
done

local_status=0
for worker_pid in "${WORKER_PIDS[@]}"; do
  wait "${worker_pid}" || local_status=1
done

# Stop only this rank's local evaluator/server processes before publishing its
# completion marker.
cleanup

# Heartbeats and the rank-0 scheduler remain alive through global validation.

local_server_log_error_count=0
for server_log in "${SERVER_LOGS[@]}"; do
  [[ -f "${server_log}" ]] || continue

  if log_has_fatal_error "${server_log}"; then
    echo "Fatal server log pattern found: ${server_log}" >&2
    local_server_log_error_count=$((local_server_log_error_count + 1))
  fi
done
(( local_server_log_error_count == 0 )) || local_status=1

write_shared_marker "${RANK_STATUS_FILE}" "${local_status}"
write_shared_marker "${RANK_DONE_FILE}" \
  "rank=${DLC_RANK} status=${local_status} epoch=$(date +%s)"

# Non-zero ranks never merge or validate the shared output. They remain alive
# until rank 0 publishes the final global status.
if (( DLC_RANK != 0 )); then
  wait_for_shared_file \
    "${FINAL_STATUS_FILE}" \
    "rank-0 global validation"

  final_status="$(<"${FINAL_STATUS_FILE}")"
  [[ "${final_status}" =~ ^[0-9]+$ ]] || {
    echo "Invalid final status marker: ${FINAL_STATUS_FILE}" >&2
    exit 1
  }

  echo "Rank ${DLC_RANK} finished; global status=${final_status}."
  exit "${final_status}"
fi

overall_status=0

for ((rank_index = 0; rank_index < DLC_WORLD_SIZE; rank_index++)); do
  if ! wait_for_shared_file \
      "${COORD_DIR}/rank-${rank_index}.done" \
      "rank ${rank_index} completion"; then
    overall_status=1
    continue
  fi

  rank_status_file="${COORD_DIR}/rank-${rank_index}.status"

  if ! wait_for_shared_file \
      "${rank_status_file}" \
      "rank ${rank_index} status"; then
    overall_status=1
    continue
  fi

  rank_status="$(<"${rank_status_file}")"

  if [[ ! "${rank_status}" =~ ^[0-9]+$ ]] \
      || (( rank_status != 0 )); then
    echo "Rank ${rank_index} reported status=${rank_status}." >&2
    overall_status=1
  fi
done

SCHEDULER_FINAL_STATS_FILE="${GLOBAL_LOG_DIR}/scheduler_final_stats.json"
if ! curl --noproxy '*' --max-time 10 -fsS \
    -H "X-Scheduler-Token: ${DLC_RUN_KEY}" \
    "${SCHEDULER_URL}/stats" >"${SCHEDULER_FINAL_STATS_FILE}"; then
  echo "Failed to read final dynamic scheduler state." >&2
  overall_status=1
elif ! python3 - "${SCHEDULER_FINAL_STATS_FILE}" "${SCHEDULE_FILE}" <<'PY'
import csv
import json
import sys

stats_path, schedule_path = sys.argv[1:]
with open(stats_path, encoding="utf-8") as file:
    stats = json.load(file)
with open(schedule_path, newline="", encoding="utf-8") as file:
    expected = sum(1 for _ in csv.DictReader(file, delimiter="\t"))
print(f"dynamic_scheduler_stats={stats}")
if stats["total"] != expected or stats["pending"] != 0 or stats["active"] != 0:
    raise SystemExit("Dynamic scheduler did not reach a terminal state")
if stats["completed"] != expected:
    raise SystemExit("Dynamic scheduler completion count does not match the schedule")
if stats["failed"] != 0 or stats["stopped"]:
    raise SystemExit("Dynamic scheduler reported a failed or stopped run")
PY
then
  overall_status=1
fi
stop_dynamic_scheduler

# Each worker was the sole writer of its own record files. Rank 0 now
# concatenates them into the public summary tables without any cross-node lock.
for ((worker_index = 0; worker_index < TOTAL_SCHEDULER_SLOTS; worker_index++)); do
  for record_kind in assignments attempts results chunks; do
    source_file="$(worker_record_path "${record_kind}" "${worker_index}")"
    if [[ ! -f "${source_file}" ]]; then
      echo "Worker record file is missing: ${source_file}" >&2
      overall_status=1
      continue
    fi
    case "${record_kind}" in
      assignments) cat "${source_file}" >>"${ASSIGNMENTS_FILE}" ;;
      attempts) cat "${source_file}" >>"${ATTEMPTS_FILE}" ;;
      results) cat "${source_file}" >>"${RESULTS_FILE}" ;;
      chunks) cat "${source_file}" >>"${CHUNK_RESULTS_FILE}" ;;
    esac
  done
done

# Check the exact scheduled/assigned/successful chunk multisets. Counts alone
# would not detect a duplicate replacing a missing chunk.
set +e
python3 - "${SCHEDULE_FILE}" "${ASSIGNMENTS_FILE}" "${CHUNK_RESULTS_FILE}" <<'PY'
import csv
import sys
from collections import Counter

schedule_path, assignments_path, chunks_path = sys.argv[1:]

with open(schedule_path, newline="", encoding="utf-8") as file:
    schedule_rows = list(csv.DictReader(file, delimiter="\t"))
with open(assignments_path, newline="", encoding="utf-8") as file:
    assignment_rows = list(csv.DictReader(file, delimiter="\t"))
with open(chunks_path, newline="", encoding="utf-8") as file:
    chunk_rows = list(csv.DictReader(file, delimiter="\t"))

expected_assignments = Counter(
    (
        row["order"],
        row["task_id"],
        row["task_name"],
        row["instance_indices"],
        row["timeout_steps"],
    )
    for row in schedule_rows
)
actual_assignments = Counter(
    (
        row["queue_order"],
        row["task_id"],
        row["task_name"],
        row["instance_indices"],
        row["timeout_steps"],
    )
    for row in assignment_rows
)
expected_chunks = Counter(
    (
        row["task_id"],
        row["task_name"],
        row["instance_indices"],
    )
    for row in schedule_rows
)
actual_chunks = Counter(
    (
        row["task_id"],
        row["task_name"],
        row["instance_indices"],
    )
    for row in chunk_rows
)

print(f"scheduled_chunks={sum(expected_assignments.values())}")
print(f"assigned_chunks={sum(actual_assignments.values())}")
print(f"successful_chunks={sum(actual_chunks.values())}")

if actual_assignments != expected_assignments:
    print(f"missing_assignments={list((expected_assignments - actual_assignments).elements())[:20]}")
    print(f"unexpected_assignments={list((actual_assignments - expected_assignments).elements())[:20]}")
    raise SystemExit("Dynamic assignment records do not exactly match the schedule")
if actual_chunks != expected_chunks:
    print(f"missing_successful_chunks={list((expected_chunks - actual_chunks).elements())[:20]}")
    print(f"unexpected_successful_chunks={list((actual_chunks - expected_chunks).elements())[:20]}")
    raise SystemExit("Successful chunk records do not exactly match the schedule")

print("DYNAMIC_SCHEDULE_RECORDS_OK=1")
PY
record_validation_status=$?
set -e
(( record_validation_status == 0 )) || overall_status=1

# Rank 0 alone reassembles chunk outputs into the official one-directory-per-
# task layout. The global validator then checks the merged task directories
# against the immutable manifest.
if (( overall_status == 0 )); then
  set +e
  python3 - "${QUEUE_FILE}" "${RUN_OUTPUT_ROOT}" "${INSTANCE_INDEX_LIST[*]}" "${EVAL_WRITE_VIDEO}" "${SESSION_OUTPUT_ROOT}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

queue_file, run_root_text, all_indices_text, write_video_text, session_root_text = sys.argv[1:]
run_root = Path(run_root_text)
session_root = Path(session_root_text)
all_indices = [int(value) for value in all_indices_text.split()]
write_video = write_video_text == "true"
chunks_root = session_root / ".chunks"
for line in Path(queue_file).read_text(encoding="utf-8").splitlines():
    task_id_text, task_name = line.split("\t", 1)
    task_id = int(task_id_text)
    task_prefix = f"task-{task_id}_{task_name}"
    chunk_dirs = sorted((chunks_root / task_prefix).glob("instances-*"))
    if not chunk_dirs:
        raise SystemExit(f"no chunk outputs found for {task_prefix}")
    final_dir = session_root / ".merged" / task_prefix
    final_dir.mkdir(parents=True, exist_ok=False)
    (final_dir / "json").mkdir()
    if write_video:
        (final_dir / "videos").mkdir()
    seen = set()
    for chunk in chunk_dirs:
        for source_dir_name in ("json", "videos") if write_video else ("json",):
            source_dir = chunk / source_dir_name
            if not source_dir.is_dir():
                continue
            target_dir = final_dir / source_dir_name
            for source in source_dir.iterdir():
                if source.name in seen:
                    raise SystemExit(f"duplicate merged artifact {source.name} for {task_prefix}")
                seen.add(source.name)
                shutil.copy2(source, target_dir / source.name)
    expected_json = {f"{task_name}_{301 + index}_0.json" for index in all_indices}
    actual_json = {path.name for path in (final_dir / "json").glob("*.json")}
    if actual_json != expected_json:
        raise SystemExit(f"merged JSON mismatch for {task_prefix}: expected {expected_json}, got {actual_json}")
    marker = {
        "task": task_name,
        "task_id": task_id,
        "mode": "public_test",
        "instance_ids": [301 + index for index in all_indices],
        "num_rollouts": 1,
        "num_vector_envs": 2,
        "result_count": len(all_indices),
        "write_video": write_video,
    }
    (final_dir / "evaluation_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    previous_final = run_root / task_prefix
    if previous_final.exists():
        archive = session_root / ".previous_final"
        archive.mkdir(exist_ok=True)
        previous_final.rename(archive / task_prefix)
    final_dir.rename(previous_final)
PY
  merge_status=$?
  set -e

  if (( merge_status == 0 )); then
    while IFS=$'\t' read -r task_id task_name; do
      record_result \
        ok \
        "-" \
        "-" \
        "${task_id}" \
        "${task_name}" \
        0 \
        "${RUN_OUTPUT_ROOT}/task-${task_id}_${task_name}"
    done <"${QUEUE_FILE}"
  else
    echo "Chunk-output merge failed with status=${merge_status}." >&2
    overall_status=1
  fi
fi

ok_count="$(awk -F '\t' 'NR > 1 && $2 == "ok" {n++} END {print n + 0}' "${RESULTS_FILE}")"
failed_count="$(awk -F '\t' 'NR > 1 && $2 == "failed" {n++} END {print n + 0}' "${RESULTS_FILE}")"
server_log_error_count=0
for server_log in "${GLOBAL_LOG_DIR}"/rank-*/server_worker*.log; do
  [[ -f "${server_log}" ]] || continue
  if log_has_fatal_error "${server_log}"; then
    echo "Fatal server log pattern found: ${server_log}" >&2
    server_log_error_count=$((server_log_error_count + 1))
  fi
done

(( failed_count == 0 )) || overall_status=1
(( ok_count == TASK_COUNT )) || overall_status=1
(( server_log_error_count == 0 )) || overall_status=1

RUN_VALIDATION_LOG="${GLOBAL_LOG_DIR}/run_validation.log"
global_validation_status=1
if (( overall_status == 0 )); then
  if python3 "${OUTPUT_VALIDATOR}" validate-run \
    --manifest "${RUN_MANIFEST}" \
    --run-output-root "${RUN_OUTPUT_ROOT}" \
    --completion-output "${RUN_COMPLETE}" \
    >"${RUN_VALIDATION_LOG}" 2>&1; then
    global_validation_status=0
  else
    cat "${RUN_VALIDATION_LOG}" >&2
  fi
else
  echo "Global artifact validation skipped because a worker, task, or server failed." >"${RUN_VALIDATION_LOG}"
fi
(( global_validation_status == 0 )) || overall_status=1

echo
echo "Finished with status=${overall_status}: tasks ok=${ok_count}, failed=${failed_count}"
if (( global_validation_status == 0 )); then
  echo "Globally validated artifacts: metrics=${EXPECTED_RESULTS}/${EXPECTED_RESULTS}, task markers=${TASK_COUNT}/${TASK_COUNT}"
  if [[ "${EVAL_WRITE_VIDEO}" == true ]]; then
    echo "Validated videos: ${EXPECTED_RESULTS}/${EXPECTED_RESULTS}"
  fi
  echo "Run completion certificate: ${RUN_COMPLETE}"
else
  echo "Global artifact validation failed or was skipped; no run completion certificate was issued."
fi
echo "Results table: ${RESULTS_FILE}"
echo "Attempts table: ${ATTEMPTS_FILE}"
echo "Validation log: ${RUN_VALIDATION_LOG}"
echo "Outputs: ${RUN_OUTPUT_ROOT}"
if [[ "${EVAL_WRITE_VIDEO}" != true ]]; then
  echo "Note: throughput profile omitted MP4 files; rerun with EVAL_PROFILE=submission for submit-ready outputs."
fi

evaluation_end_epoch="$(date +%s)"
evaluation_wall_seconds=$((evaluation_end_epoch - EVALUATION_START_EPOCH))
scheduler_wall_seconds=$((evaluation_end_epoch - SCRIPT_START_EPOCH))

printf 'evaluation_end_epoch\t%s\n' \
  "${evaluation_end_epoch}" >>"${TIMING_FILE}"
printf 'evaluation_wall_seconds\t%s\n' \
  "${evaluation_wall_seconds}" >>"${TIMING_FILE}"
printf 'scheduler_wall_seconds\t%s\n' \
  "${scheduler_wall_seconds}" >>"${TIMING_FILE}"

echo "Timing table: ${TIMING_FILE}"
echo "Evaluation wall time: ${evaluation_wall_seconds} seconds"
echo "Scheduler wall time: ${scheduler_wall_seconds} seconds"

write_shared_marker "${FINAL_STATUS_FILE}" "${overall_status}"
stop_rank_heartbeat
exit "${overall_status}"
