#!/usr/bin/env bash
set -euo pipefail

# Accelerated PI0.5 evaluation for all 100 tasks of the 2026 BEHAVIOR Challenge.
#
# Default protocol:
#   - 100 official tasks (2026 task IDs 0-99)
#   - public instance indices 0-9
#   - one rollout per instance
#   - official 120/30/30 Hz dynamics and task-specific 1.5x human timeout
#   - one persistent policy server + one two-slot VectorEnvironment per GPU
#
# The default throughput profile disables MP4 encoding. For submission-complete
# outputs (the 2026 challenge requires videos), use:
#   EVAL_PROFILE=submission bash run_pi05_behavior_2026_eval_chunk.sh
#
# Single-GPU smoke test:
#   GPU_IDS=0 NUM_GPUS=1 TASK_IDS=0 TASK_LIMIT=1 \
#     EVAL_INSTANCE_INDICES='0 1' EVAL_MAX_STEPS=100 \
#     bash run_pi05_behavior_2026_eval_chunk.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
PI05_REPO="${PI05_REPO:-/mnt/data_nas/wangjm/unirobot/behavior-1k-solution}"
PI05_SERVER_SCRIPT="${PI05_SERVER_SCRIPT:-${BEHAVIOR_ROOT}/serve_pi05_behavior_2026_vector.py}"
PI05_POLICY_CONFIG="${PI05_POLICY_CONFIG:-pi_behavior_b1k_2026}"
PI05_POLICY_DIR="${PI05_POLICY_DIR:-/mnt/data/ckpt/[b1k]/pi_behavior_b1k_2026/20260730_b1k_2026_full_fast_8gpu_bs2048/18000}"
PI05_INFERENCE_CKPT_SUFFIX="${PI05_INFERENCE_CKPT_SUFFIX:-_inference}"
PI05_NORM_STATS_PATH="${PI05_NORM_STATS_PATH:-${PI05_REPO}/outputs/assets/pi_behavior_b1k_2026/behavior-1k/2026-challenge-demos/norm_stats.json}"
PI05_TASK_CHECKPOINT_MAPPING="${PI05_TASK_CHECKPOINT_MAPPING:-${PI05_REPO}/task_checkpoint_mapping.json}"
USE_PI05_TASK_CHECKPOINT_MAPPING="${USE_PI05_TASK_CHECKPOINT_MAPPING:-false}"
PI05_ENV_DIR="${PI05_ENV_DIR:-${PI05_UV_PROJECT_ENVIRONMENT:-${HOME}/pi05_env}}"
PI05_PYTHON="${PI05_PYTHON:-${PI05_ENV_DIR}/bin/python}"
BEHAVIOR_ENV_DIR="${BEHAVIOR_ENV_DIR:-/root/miniconda3/envs/behavior_2026}"
BEHAVIOR_PYTHON="${BEHAVIOR_PYTHON:-${BEHAVIOR_ENV_DIR}/bin/python}"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-${BEHAVIOR_ROOT}/datasets}"
PI05_RESOLVED_POLICY_DIR=""

TASK_IDS="${TASK_IDS:-}"
TASK_LIMIT="${TASK_LIMIT:-100}"
EVAL_INSTANCE_INDICES="${EVAL_INSTANCE_INDICES:-0 1 2 3 4 5 6 7 8 9}"
EVAL_INSTANCE_INDICES="${EVAL_INSTANCE_INDICES//,/ }"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-}"
EVAL_SEED="${EVAL_SEED:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_IDS="${GPU_IDS//,/ }"
VECTOR_ENVS_PER_PROCESS="${VECTOR_ENVS_PER_PROCESS:-2}"
PORT_BASE="${PORT_BASE:-7100}"
PORT_STRIDE="${PORT_STRIDE:-100}"
PI05_SERVER_HOST="${PI05_SERVER_HOST:-localhost}"
PI05_CLIENT_HOST="${PI05_CLIENT_HOST:-localhost}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

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

EVAL_PROFILE="${EVAL_PROFILE:-throughput}"
case "${EVAL_PROFILE}" in
  throughput) PROFILE_WRITE_VIDEO=false ;;
  submission) PROFILE_WRITE_VIDEO=true ;;
  *) echo "EVAL_PROFILE must be throughput or submission, got: ${EVAL_PROFILE}" >&2; exit 2 ;;
esac
EVAL_WRITE_VIDEO="${EVAL_WRITE_VIDEO:-${PROFILE_WRITE_VIDEO}}"
EVAL_PARTIAL_SCENE_LOAD="${EVAL_PARTIAL_SCENE_LOAD:-true}"
EVAL_FAIL_FAST="${EVAL_FAIL_FAST:-true}"
EVAL_MAX_TASK_ATTEMPTS="${EVAL_MAX_TASK_ATTEMPTS:-2}"

if [[ "${EVAL_PROFILE}" == submission && "${EVAL_WRITE_VIDEO}" != true ]]; then
  echo "EVAL_PROFILE=submission requires EVAL_WRITE_VIDEO=true; use throughput for a no-video run." >&2
  exit 2
fi

CPU_RESERVE_CORES="${CPU_RESERVE_CORES:-auto}"
EVAL_CPU_CORES="${EVAL_CPU_CORES:-auto}"
EVAL_CPU_NUM_THREADS="${EVAL_CPU_NUM_THREADS:-auto}"
SERVER_CPU_NUM_THREADS="${SERVER_CPU_NUM_THREADS:-2}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${BEHAVIOR_ROOT}/logs/pi05_behavior_2026_${RUN_TS}}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${BEHAVIOR_ROOT}/logs/pi05_behavior_2026_outputs}"
RUN_OUTPUT_ROOT="${EVAL_LOG_ROOT}/${RUN_TS}"
QUEUE_FILE="${LOG_DIR}/task_queue.tsv"
QUEUE_LOCK="${LOG_DIR}/task_queue.lock"
RESULTS_FILE="${LOG_DIR}/results.tsv"
ATTEMPTS_FILE="${LOG_DIR}/attempts.tsv"
PID_DIR="${LOG_DIR}/pids"
STOP_FILE="${LOG_DIR}/stop_requested"
ATTEMPTS_ROOT="${RUN_OUTPUT_ROOT}/.attempts"
RUN_MANIFEST="${RUN_OUTPUT_ROOT}/run_manifest.json"
RUN_COMPLETE="${RUN_OUTPUT_ROOT}/run_complete.json"
OUTPUT_VALIDATOR="${BEHAVIOR_ROOT}/OmniGibson/omnigibson/eval/utils/pi05_output_validator.py"

DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: bash run_pi05_behavior_2026_eval_chunk.sh [--base-velocity-frame absolute|relative] [--dry-run] [--help]

Core overrides:
  EVAL_PROFILE              throughput (no videos) or submission (videos), default throughput.
  TASK_IDS                  Space/comma-separated 2026 task IDs in [0,99], default all 100.
  TASK_LIMIT                Limit queued tasks after selection, default 100.
  EVAL_INSTANCE_INDICES     Public split indices, default '0 1 2 3 4 5 6 7 8 9'.
  EVAL_SEED                 Fixed environment RNG seed, default 0.
  EVAL_MAX_STEPS            Optional smoke-test timeout; empty preserves official 1.5x timeout.
  EVAL_MAX_TASK_ATTEMPTS    Whole-task attempts before terminal failure, default 2.
  GPU_IDS / NUM_GPUS        GPU IDs and number of colocated env/server pairs.
  PI05_REPO                 100-task PI0.5 source checkout; must contain champion_2026 config.
  PI05_POLICY_DIR           Training or merged inference checkpoint directory.
  PI05_NORM_STATS_PATH      2026 checkpoint normalization statistics.
  PI05_BASE_VELOCITY_FRAME  Policy observation base qvel frame: absolute (legacy raw) or relative (robot-local), default absolute. Actions are always robot-local.
  PI05_ENV_DIR              PI0.5 environment directory, default ~/pi05_env.
  BEHAVIOR_ENV_DIR          2026 evaluator conda environment directory.
  LOG_DIR / EVAL_LOG_ROOT   Scheduler logs and evaluator outputs.

The default checkpoint is the 100-task 2026 PI_BEHAVIOR checkpoint. If the
requested training checkpoint is sharded, an existing sibling ending in
_inference is selected. One server is loaded once per GPU worker and reused
until that worker's task queue is empty.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

validate_seed() {
  [[ "$2" =~ ^[0-9]+$ ]] && (( $2 < 4294967296 )) \
    || { echo "$1 must be an integer in [0, 4294967296), got: $2" >&2; exit 1; }
}

is_inference_checkpoint() {
  local checkpoint_dir="$1"
  [[ -f "${checkpoint_dir}/params/_METADATA" ]] || return 1
  [[ -f "${checkpoint_dir}/params/manifest.ocdbt" && -d "${checkpoint_dir}/params/d" ]]
}

resolve_policy_checkpoint() {
  local requested="${PI05_POLICY_DIR%/}"
  local converted="${requested}${PI05_INFERENCE_CKPT_SUFFIX}"
  if is_inference_checkpoint "${requested}"; then
    PI05_RESOLVED_POLICY_DIR="${requested}"
  elif is_inference_checkpoint "${converted}"; then
    PI05_RESOLVED_POLICY_DIR="${converted}"
  else
    echo "No server-readable PI0.5 checkpoint found." >&2
    echo "Requested: ${requested}" >&2
    echo "Inference fallback: ${converted}" >&2
    echo "Use the reference merge_sharded_params_for_inference.py before evaluation." >&2
    exit 1
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
        "PI05_REPO must point to the champion_2026 checkout"
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
  validate_positive_int VECTOR_ENVS_PER_PROCESS "${VECTOR_ENVS_PER_PROCESS}"
  validate_positive_int PI05_DYNAMIC_BATCH_MAX_SIZE "${PI05_DYNAMIC_BATCH_MAX_SIZE}"
  validate_positive_int PI05_DYNAMIC_BATCH_GRANULARITY "${PI05_DYNAMIC_BATCH_GRANULARITY}"
  validate_positive_int TASK_LIMIT "${TASK_LIMIT}"
  validate_positive_int EVAL_MAX_TASK_ATTEMPTS "${EVAL_MAX_TASK_ATTEMPTS}"
  validate_seed EVAL_SEED "${EVAL_SEED}"
  validate_bool EVAL_WRITE_VIDEO "${EVAL_WRITE_VIDEO}"
  validate_bool EVAL_PARTIAL_SCENE_LOAD "${EVAL_PARTIAL_SCENE_LOAD}"
  validate_bool EVAL_FAIL_FAST "${EVAL_FAIL_FAST}"
  validate_bool PI05_APPLY_EVAL_TRICKS "${PI05_APPLY_EVAL_TRICKS}"
  validate_bool PI05_DISABLE_FAST_AUXILIARY "${PI05_DISABLE_FAST_AUXILIARY}"
  validate_bool USE_PI05_TASK_CHECKPOINT_MAPPING "${USE_PI05_TASK_CHECKPOINT_MAPPING}"
  [[ "${EVAL_WRITE_VIDEO}" != true ]] || require_command ffprobe

  (( VECTOR_ENVS_PER_PROCESS == 2 )) || {
    echo "This launcher requires VECTOR_ENVS_PER_PROCESS=2 for synchronized two-env simulation." >&2
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
  [[ -f "${PI05_SERVER_SCRIPT}" ]] || { echo "PI0.5 vector server is missing: ${PI05_SERVER_SCRIPT}" >&2; exit 1; }
  [[ -f "${BEHAVIOR_ROOT}/OmniGibson/omnigibson/eval/eval_vector.py" ]] || {
    echo "2026 vector evaluator is missing under ${BEHAVIOR_ROOT}." >&2
    exit 1
  }
  [[ -f "${OUTPUT_VALIDATOR}" ]] || {
    echo "PI0.5 output validator is missing: ${OUTPUT_VALIDATOR}" >&2
    exit 1
  }
  [[ -f "${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/metadata/B100_task_misc.csv" ]] || {
    echo "2026 challenge metadata is missing under ${OMNIGIBSON_DATA_PATH}." >&2
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

  resolve_policy_checkpoint
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
  SELECTED_TASK_IDS="${TASK_IDS}" TASK_LIMIT_VALUE="${TASK_LIMIT}" \
    python3 - "${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/metadata/B100_task_misc.csv" "${QUEUE_FILE}" <<'PY'
import csv
import os
import sys

metadata_path, output_path = sys.argv[1:]
with open(metadata_path, newline="", encoding="utf-8") as file:
    tasks = {int(row["Task ID"]): row["Task"] for row in csv.DictReader(file)}

raw_ids = os.environ.get("SELECTED_TASK_IDS", "").split()
task_ids = [int(value) for value in raw_ids] if raw_ids else list(range(100))
if len(task_ids) != len(set(task_ids)):
    raise SystemExit("TASK_IDS contains duplicates")
unsupported = [task_id for task_id in task_ids if task_id not in range(100)]
if unsupported:
    raise SystemExit(f"2026 task IDs must be in 0-99; got {unsupported}")
limit = int(os.environ["TASK_LIMIT_VALUE"])
with open(output_path, "w", encoding="utf-8") as file:
    for task_id in task_ids[:limit]:
        file.write(f"{task_id}\t{tasks[task_id]}\n")
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
  echo "${port}"
}

pop_next_task() {
  local line status
  set +e
  line="$(
    {
      flock -x 9
      [[ ! -e "${STOP_FILE}" && -s "${QUEUE_FILE}" ]] || exit 1
      IFS= read -r next_line <"${QUEUE_FILE}"
      tail -n +2 "${QUEUE_FILE}" >"${QUEUE_FILE}.tmp"
      mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
      echo "${next_line}"
    } 9>"${QUEUE_LOCK}"
  )"
  status=$?
  set -e
  (( status == 0 )) && [[ -n "${line}" ]] || return 1
  echo "${line}"
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

SERVER_PYTHONPATH="${PI05_REPO}/src:${PI05_REPO}/openpi/src:${PYTHONPATH:-}"
EVAL_PYTHONPATH="${BEHAVIOR_ROOT}/OmniGibson:${BEHAVIOR_ROOT}/bddl3:${BEHAVIOR_ROOT}/joylo:${BEHAVIOR_ROOT}:${PI05_REPO}/src:${PYTHONPATH:-}"
export OMNIGIBSON_DATA_PATH
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,::1"

LAUNCHED_PID=""

launch_server() {
  local gpu_id="$1" worker="$2" port="$3" log_file="$4"
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
      setsid taskset -c "${SERVER_CPU_LISTS[${worker}]}" "${PI05_PYTHON}" "${args[@]}"
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
  local gpu_id="$1" worker="$2" port="$3" task_name="$4" output_dir="$5" log_file="$6"
  local -a args=(
    -m omnigibson.eval.eval_vector
    --task-name "${task_name}"
    --host "${PI05_CLIENT_HOST}"
    --port "${port}"
    --mode public_test
    --instance-indices "${INSTANCE_INDEX_LIST[@]}"
    --num-rollouts 1
    --num-envs "${VECTOR_ENVS_PER_PROCESS}"
    --seed "${EVAL_SEED}"
    --output-dir "${output_dir}"
    --cpu-affinity "${EVAL_CPU_LISTS[${worker}]}"
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
  [[ "${EVAL_WRITE_VIDEO}" == true ]] && args+=(--write-video) || args+=(--no-write-video)
  [[ "${EVAL_PARTIAL_SCENE_LOAD}" == true ]] && args+=(--partial-scene-load) || args+=(--no-partial-scene-load)
  [[ "${PI05_APPLY_EVAL_TRICKS}" == true ]] && args+=(--apply-eval-tricks) || args+=(--no-apply-eval-tricks)
  args+=(--headless --no-render-viewer-camera)

  (
    behavior_env_dir="$(dirname "$(dirname "${BEHAVIOR_PYTHON}")")"
    conda_root="$(dirname "$(dirname "${behavior_env_dir}")")"
    set +u
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "${behavior_env_dir}"
    set -u
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
      setsid taskset -c "${EVAL_CPU_LISTS[${worker}]}" "${BEHAVIOR_PYTHON}" "${args[@]}"
  ) >"${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
}

record_result() {
  local state="$1" worker="$2" gpu="$3" task_id="$4" task_name="$5" status="$6" output_dir="$7"
  {
    flock -x 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${state}" "${worker}" "${gpu}" \
      "${task_id}" "${task_name}" "${status}" "${output_dir}" >>"${RESULTS_FILE}"
  } 9>"${RESULTS_FILE}.lock"
}

record_attempt() {
  local worker="$1" gpu="$2" task_id="$3" task_name="$4" attempt="$5"
  local eval_status="$6" validation_status="$7" fatal_detected="$8" promoted="$9"
  local attempt_dir="${10}" final_dir="${11}" eval_log="${12}" validation_log="${13}"
  {
    flock -x 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "${worker}" "${gpu}" "${task_id}" "${task_name}" \
      "${attempt}" "${eval_status}" "${validation_status}" "${fatal_detected}" "${promoted}" \
      "${attempt_dir}" "${final_dir}" "${eval_log}|${validation_log}" >>"${ATTEMPTS_FILE}"
  } 9>"${ATTEMPTS_FILE}.lock"
}

log_has_fatal_error() {
  local log_file="$1"
  grep -Eq \
    'Failed to get contact force matrix from backend|Fatal Python error|Segmentation fault|ERROR_INCOMPATIBLE_DRIVER|Failed to create any GPU devices|Dynamic batch.*(failed|error)' \
    "${log_file}"
}

worker_loop() {
  local worker="$1" gpu_id="$2" port="$3" server_pid="$4"
  local task_line task_id task_name output_dir attempt_parent attempt_dir log_file validation_log
  local eval_pid eval_status validation_status fatal_detected promoted terminal_status attempt task_succeeded
  while task_line="$(pop_next_task)"; do
    IFS=$'\t' read -r task_id task_name <<<"${task_line}"
    output_dir="${RUN_OUTPUT_ROOT}/task-${task_id}_${task_name}"
    attempt_parent="${ATTEMPTS_ROOT}/task-${task_id}_${task_name}"
    mkdir -p "${attempt_parent}"
    task_succeeded=false
    terminal_status=124

    for ((attempt = 1; attempt <= EVAL_MAX_TASK_ATTEMPTS; attempt++)); do
      attempt_dir="${attempt_parent}/attempt-${attempt}"
      log_file="${LOG_DIR}/eval_worker${worker}_gpu${gpu_id}_task${task_id}_${task_name}_attempt${attempt}.log"
      validation_log="${LOG_DIR}/validate_worker${worker}_task${task_id}_${task_name}_attempt${attempt}.log"
      echo "[worker ${worker}/gpu${gpu_id}] task=${task_id}:${task_name} attempt=${attempt}/${EVAL_MAX_TASK_ATTEMPTS} instances=${INSTANCE_INDEX_LIST[*]}"

      if [[ -e "${attempt_dir}" || -e "${output_dir}" ]]; then
        echo "Refusing to overwrite existing task output: ${attempt_dir} or ${output_dir}" >"${validation_log}"
        record_attempt "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${attempt}" \
          126 1 false false "${attempt_dir}" "${output_dir}" "${log_file}" "${validation_log}"
        terminal_status=126
        break
      fi

      launch_eval "${gpu_id}" "${worker}" "${port}" "${task_name}" "${attempt_dir}" "${log_file}"
      eval_pid="${LAUNCHED_PID}"
      echo "${eval_pid}" >"${PID_DIR}/eval_worker${worker}.pid"
      set +e
      wait "${eval_pid}"
      eval_status=$?
      set -e
      rm -f "${PID_DIR}/eval_worker${worker}.pid"

      if python3 "${OUTPUT_VALIDATOR}" validate-task \
        --manifest "${RUN_MANIFEST}" --task-id "${task_id}" --task-dir "${attempt_dir}" \
        >"${validation_log}" 2>&1; then
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
        record_result ok "${worker}" "${gpu_id}" "${task_id}" "${task_name}" 0 "${output_dir}"
        break
      fi
      echo "[worker ${worker}] task ${task_id}:${task_name} attempt ${attempt} failed; see ${log_file} and ${validation_log}" >&2
      kill -0 "${server_pid}" >/dev/null 2>&1 || break
    done

    if [[ "${task_succeeded}" != true ]]; then
      record_result failed "${worker}" "${gpu_id}" "${task_id}" "${task_name}" "${terminal_status}" "${output_dir}"
      echo "[worker ${worker}] task ${task_id}:${task_name} exhausted ${EVAL_MAX_TASK_ATTEMPTS} attempt(s)." >&2
      if [[ "${EVAL_FAIL_FAST}" == true ]]; then
        : >"${STOP_FILE}"
        return 1
      fi
    fi
    kill -0 "${server_pid}" >/dev/null 2>&1 || { echo "Server ${server_pid} exited." >&2; return 1; }
  done
}

WORKER_PIDS=()
SERVER_PIDS=()
CLEANUP_DONE=false

cleanup() {
  [[ "${CLEANUP_DONE}" == true ]] && return 0
  CLEANUP_DONE=true
  local pid_file pid
  echo "Cleaning up evaluator and policy server processes..."
  while IFS= read -r pid_file; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(<"${pid_file}")"
    kill_process_group "${pid}"
  done < <(find "${PID_DIR}" -type f -name '*.pid' 2>/dev/null | sort)
}

validate_environment
if [[ "${DRY_RUN}" != true ]]; then
  validate_gpu_runtime
fi
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${RUN_OUTPUT_ROOT}" "${ATTEMPTS_ROOT}"
build_task_queue
TASK_COUNT="$(wc -l <"${QUEUE_FILE}" | tr -d ' ')"
(( TASK_COUNT > 0 )) || { echo "No tasks queued." >&2; exit 1; }
INSTANCE_COUNT="${#INSTANCE_INDEX_LIST[@]}"
EXPECTED_RESULTS=$((TASK_COUNT * INSTANCE_COUNT))
printf 'timestamp\tstate\tworker\tgpu\ttask_id\ttask_name\tstatus\toutput_dir\n' >"${RESULTS_FILE}"
printf 'timestamp\tworker\tgpu\ttask_id\ttask_name\tattempt\tevaluator_status\tvalidation_status\tfatal_log\tpromoted\tattempt_dir\tfinal_dir\tlogs\n' >"${ATTEMPTS_FILE}"
manifest_video_arg=--no-write-video
[[ "${EVAL_WRITE_VIDEO}" == true ]] && manifest_video_arg=--write-video
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

echo "PI0.5 accelerated BEHAVIOR 2026 official evaluation:"
echo "  protocol: ${TASK_COUNT} tasks x ${INSTANCE_COUNT} public instances x 1 rollout = ${EXPECTED_RESULTS} outputs"
echo "  supported task IDs: ${TASK_IDS:-0-99}"
echo "  public instance indices: ${INSTANCE_INDEX_LIST[*]}"
echo "  timeout: ${EVAL_MAX_STEPS:-official task-specific 1.5x mean human length}"
echo "  profile / write video: ${EVAL_PROFILE} / ${EVAL_WRITE_VIDEO}"
echo "  official dynamics: physics=120 Hz, render/action=30 Hz"
echo "  GPUs: ${GPU_ID_LIST[*]}"
echo "  topology: 1 persistent server + 1 Isaac Sim process x 2 vector envs per GPU; joint batch-2 policy requests"
echo "  dynamic batch max/wait/granularity: ${PI05_DYNAMIC_BATCH_MAX_SIZE}/${PI05_DYNAMIC_BATCH_WAIT_MS}ms/${PI05_DYNAMIC_BATCH_GRANULARITY}"
echo "  CPU cores/threads per evaluator: ${EVAL_CPU_CORES}/${EVAL_CPU_NUM_THREADS}"
echo "  CPU threads per server: ${SERVER_CPU_NUM_THREADS}"
echo "  PI0.5 repo: ${PI05_REPO}"
echo "  server entrypoint: ${PI05_SERVER_SCRIPT}"
echo "  server lifecycle: one persistent PID per worker, reused across queued tasks"
echo "  policy config: ${PI05_POLICY_CONFIG}"
echo "  requested checkpoint: ${PI05_POLICY_DIR}"
echo "  resolved checkpoint: ${PI05_RESOLVED_POLICY_DIR}"
echo "  norm stats: ${PI05_NORM_STATS_PATH}"
echo "  checkpoint mapping: enabled=${USE_PI05_TASK_CHECKPOINT_MAPPING}, path=${PI05_TASK_CHECKPOINT_MAPPING}"
echo "  proprioception schema: ${PI05_PROPRIOCEPTION_SCHEMA}"
echo "  base velocity frame: ${PI05_BASE_VELOCITY_FRAME}"
echo "  environment seed: ${EVAL_SEED}"
echo "  behavior env: ${BEHAVIOR_ENV_DIR}"
echo "  PI0.5 env: ${PI05_ENV_DIR}"
echo "  behavior Python: ${BEHAVIOR_PYTHON}"
echo "  PI0.5 Python: ${PI05_PYTHON}"
echo "  outputs: ${RUN_OUTPUT_ROOT}"
echo "  immutable manifest: ${RUN_MANIFEST}"
echo "  whole-task attempts: ${EVAL_MAX_TASK_ATTEMPTS}"
echo "  logs: ${LOG_DIR}"

if [[ "${DRY_RUN}" == true ]]; then
  echo
  echo "Dry-run task queue:"
  nl -ba "${QUEUE_FILE}"
  echo "Dry run complete; no server or simulator process was started."
  exit 0
fi

trap cleanup INT TERM EXIT

SERVER_PORTS=()
SERVER_LOGS=()
for ((worker = 0; worker < NUM_GPUS; worker++)); do
  gpu_id="${GPU_ID_LIST[${worker}]}"
  port="$(find_free_port "$((PORT_BASE + worker * PORT_STRIDE))")"
  server_log="${LOG_DIR}/server_worker${worker}_gpu${gpu_id}_port${port}.log"
  echo "[server ${worker}/gpu${gpu_id}] loading persistent checkpoint ${PI05_RESOLVED_POLICY_DIR} on port ${port}"
  launch_server "${gpu_id}" "${worker}" "${port}" "${server_log}"
  server_pid="${LAUNCHED_PID}"
  SERVER_PIDS+=("${server_pid}")
  SERVER_PORTS+=("${port}")
  SERVER_LOGS+=("${server_log}")
  echo "${server_pid}" >"${PID_DIR}/server${worker}.pid"
done

# All checkpoint loads run concurrently. Each PID remains alive and is reused
# for every task subsequently claimed by the corresponding worker.
for ((worker = 0; worker < NUM_GPUS; worker++)); do
  wait_for_server "${SERVER_PIDS[${worker}]}" "${SERVER_PORTS[${worker}]}" || {
    echo "Server ${worker} failed to become ready; see ${SERVER_LOGS[${worker}]}" >&2
    exit 1
  }
done

for ((worker = 0; worker < NUM_GPUS; worker++)); do
  worker_loop "${worker}" "${GPU_ID_LIST[${worker}]}" "${SERVER_PORTS[${worker}]}" "${SERVER_PIDS[${worker}]}" &
  WORKER_PIDS+=("$!")
done

overall_status=0
for worker_pid in "${WORKER_PIDS[@]}"; do
  wait "${worker_pid}" || overall_status=1
done

trap - INT TERM EXIT
cleanup

ok_count="$(awk -F '\t' 'NR > 1 && $2 == "ok" {n++} END {print n + 0}' "${RESULTS_FILE}")"
failed_count="$(awk -F '\t' 'NR > 1 && $2 == "failed" {n++} END {print n + 0}' "${RESULTS_FILE}")"
server_log_error_count=0
for server_log in "${LOG_DIR}"/server_worker*.log; do
  [[ -f "${server_log}" ]] || continue
  if log_has_fatal_error "${server_log}"; then
    echo "Fatal server log pattern found: ${server_log}" >&2
    server_log_error_count=$((server_log_error_count + 1))
  fi
done

(( failed_count == 0 )) || overall_status=1
(( ok_count == TASK_COUNT )) || overall_status=1
(( server_log_error_count == 0 )) || overall_status=1

RUN_VALIDATION_LOG="${LOG_DIR}/run_validation.log"
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

exit "${overall_status}"
