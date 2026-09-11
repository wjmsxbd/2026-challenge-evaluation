#!/usr/bin/env bash
set -euo pipefail

# Two independent policy/Isaac pairs. Per-worker output directories keep the
# collector's task locks separate, and sample seeds remain independent of GPU.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
RFT_WORKER_SCRIPT="${RFT_WORKER_SCRIPT:-${SCRIPT_DIR}/run_pi05_behavior_2026_collect_rft.sh}"
RFT_GPU_IDS="${RFT_GPU_IDS:-0 1}"
RFT_INSTANCE_INDICES="${RFT_INSTANCE_INDICES:-0 1 2 3}"
RFT_PORT="${RFT_PORT:-18101}"
RFT_OUTPUT_DIR="${RFT_OUTPUT_DIR:-${BEHAVIOR_ROOT}/logs/rft_task0001_test/$(date +%Y%m%d_%H%M%S)}"
RESUME=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--output-dir needs a directory" >&2; exit 2; }
      RFT_OUTPUT_DIR="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: bash run_rft_2026_task0001_4instances_1p2x.sh [--output-dir DIR] [--resume] [--dry-run]

Task 0001 / picking_up_trash, four training instances, two GPUs concurrently.
GPU 0: instances 0,1 in two environments. GPU 1: instances 2,3 in two environments.
Each GPU has its own policy server, Isaac process, CPU set and output directory.
Default: one attempt per instance, one-success target, 1.2x timeout (6321 steps).

RFT_GPU_IDS='0 1'         Two distinct local GPU indices.
RFT_INSTANCE_INDICES     Four distinct training instance IDs, default '0 1 2 3'.
RFT_NUM_ROLLOUTS         Maximum attempts per instance (default 1).
RFT_SUCCESSES_PER_INSTANCE  Success target per instance (default 1).
RFT_CPU_THREADS          CPU threads per Isaac process (auto, up to 16).
RFT_PORT                 First server port (default 18101); the second uses +1.
RFT_MAX_STEPS            Optional absolute limit for a short smoke test.
RFT_PERTURB_POSE         Initial pose perturbation, default true.
PI05_POLICY_DIR / PI05_NORM_STATS_PATH  Same policy overrides as the reference.

Raw data: DIR/workers/gpu-<id>/data/task-0001_picking_up_trash/rollouts/.
Combined statistics: DIR/collection_summary.json after both workers finish.
--dry-run checks both worker configurations and reports GPU availability;
it can print a plan on a one-GPU machine. Real execution requires both GPUs.
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
: "${BEHAVIOR_PYTHON:?Use run_rft_2026_task0001_4instances_1p2x.sh to load the runtime defaults}"
RFT_OUTPUT_DIR="$(realpath -m -- "$RFT_OUTPUT_DIR")"
read -r -a GPUS <<<"${RFT_GPU_IDS//,/ }"
read -r -a INSTANCES <<<"${RFT_INSTANCE_INDICES//,/ }"
[[ ${#GPUS[@]} -eq 2 && ${#INSTANCES[@]} -eq 4 ]] || {
  echo "This test requires two GPU IDs and four training instance IDs" >&2; exit 2;
}
declare -A seen_gpus=() seen_instances=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ && -z "${seen_gpus[$gpu]:-}" ]] || {
    echo "GPU IDs must be distinct non-negative integers" >&2; exit 2;
  }
  seen_gpus[$gpu]=1
done
for instance in "${INSTANCES[@]}"; do
  [[ "$instance" =~ ^(0|[1-9][0-9]*)$ && -z "${seen_instances[$instance]:-}" ]] || {
    echo "Instance IDs must be distinct non-negative integers" >&2; exit 2;
  }
  seen_instances[$instance]=1
done
[[ "$RFT_PORT" =~ ^[1-9][0-9]*$ ]] && (( RFT_PORT < 65535 )) || {
  echo "RFT_PORT must be in [1, 65534]" >&2; exit 2;
}
[[ -f "$RFT_WORKER_SCRIPT" ]] || { echo "Missing RFT worker script" >&2; exit 1; }
RFT_CPU_THREADS="$("$BEHAVIOR_PYTHON" - "${RFT_CPU_THREADS:-auto}" "${SERVER_CPU_NUM_THREADS:-2}" <<'PY'
import os
import sys
requested, server_threads = sys.argv[1:]
server_threads = int(server_threads)
available = len(os.sched_getaffinity(0))
threads = min(16, (available - 2 * server_threads) // 2) if requested == "auto" else int(requested)
if min(threads, server_threads) < 1 or 2 * (threads + server_threads) > available:
    raise SystemExit("Not enough CPUs for both workers; reduce RFT_CPU_THREADS / SERVER_CPU_NUM_THREADS")
print(threads)
PY
)"
export RFT_CPU_THREADS
GPU_INVENTORY="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)" || GPU_INVENTORY=""
MISSING_GPUS=()
for gpu in "${GPUS[@]}"; do
  available=false
  while read -r visible_gpu; do [[ "$visible_gpu" != "$gpu" ]] || available=true; done <<<"$GPU_INVENTORY"
  [[ "$available" == true ]] || MISSING_GPUS+=("$gpu")
done
if (( ${#MISSING_GPUS[@]} )); then
  printf 'Requested GPU(s) unavailable in this container: %s\n' "${MISSING_GPUS[*]}" >&2
  [[ "$DRY_RUN" == true ]] || exit 1
fi
printf 'Two-GPU RFT: task=%04d, GPUs=%s, 2 environments/GPU, train instances=%s\n' \
  "${RFT_TASK_ID:-1}" "${GPUS[*]}" "${INSTANCES[*]}"
printf 'Run directory: %s\n' "$RFT_OUTPUT_DIR"
WORKER_ARGS=()
[[ "$RESUME" == false ]] || WORKER_ARGS+=(--resume)
worker_command() {
  local rank="$1"
  env RFT_GPU_ID="${GPUS[$rank]}" RFT_WORKER_INDEX="$rank" RFT_NUM_ENVS=2 \
    RFT_INSTANCE_INDICES="${INSTANCES[$((rank * 2))]} ${INSTANCES[$((rank * 2 + 1))]}" \
    RFT_PORT="$((RFT_PORT + rank))" RFT_OUTPUT_DIR="$RFT_OUTPUT_DIR/workers/gpu-${GPUS[$rank]}" \
    bash "$RFT_WORKER_SCRIPT" "${WORKER_ARGS[@]}" "${@:2}"
}
if [[ "$DRY_RUN" == true ]]; then
  worker_command 0 --dry-run
  worker_command 1 --dry-run
  exit 0
fi

mkdir -p "$RFT_OUTPUT_DIR"
exec 8>"$RFT_OUTPUT_DIR/.coordinator.lock"
flock -n 8 || { echo "Another coordinator is using this run directory" >&2; exit 1; }
if [[ -f "$RFT_OUTPUT_DIR/launch_plan.json" && "$RESUME" != true ]]; then
  echo "Run directory exists; use --resume or a new --output-dir" >&2; exit 1
fi
"$BEHAVIOR_PYTHON" - "$RFT_OUTPUT_DIR" "${GPUS[@]}" "${INSTANCES[@]}" <<'PY'
import json
from pathlib import Path
import sys
root, gpu0, gpu1, *instances = sys.argv[1:]
plan = {"num_parallel_envs": 4, "workers": [
    {"gpu": int(gpu), "instance_ids": list(map(int, instances[rank * 2:rank * 2 + 2]))}
    for rank, gpu in enumerate((gpu0, gpu1))]}
path = Path(root) / "launch_plan.json"
if path.exists() and json.loads(path.read_text()) != plan:
    raise SystemExit("GPU/instance assignments changed; use a new output directory")
path.write_text(json.dumps(plan, indent=2))
PY
WORKER_PIDS=()
cleanup() {
  local status=$? pid
  trap - EXIT INT TERM
  # Send TERM to the worker shells so their traps close only their own children.
  for pid in "${WORKER_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${WORKER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  printf '%s\n' "$status" >"$RFT_OUTPUT_DIR/exit_status.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for rank in 0 1; do
  mkdir -p "$RFT_OUTPUT_DIR/workers/gpu-${GPUS[$rank]}"
  # Execute the worker shell directly: $! must be the PID whose trap owns its children.
  env RFT_GPU_ID="${GPUS[$rank]}" RFT_WORKER_INDEX="$rank" RFT_NUM_ENVS=2 \
    RFT_INSTANCE_INDICES="${INSTANCES[$((rank * 2))]} ${INSTANCES[$((rank * 2 + 1))]}" \
    RFT_PORT="$((RFT_PORT + rank))" RFT_OUTPUT_DIR="$RFT_OUTPUT_DIR/workers/gpu-${GPUS[$rank]}" \
    setsid bash "$RFT_WORKER_SCRIPT" "${WORKER_ARGS[@]}" \
    >"$RFT_OUTPUT_DIR/workers/gpu-${GPUS[$rank]}/launcher.log" 2>&1 &
  WORKER_PIDS+=("$!")
  printf '%s\n' "$!" >"$RFT_OUTPUT_DIR/workers/gpu-${GPUS[$rank]}/worker.pid"
done
remaining=2
while (( remaining > 0 )); do
  wait -n
  remaining=$((remaining - 1))
done
WORKER_PIDS=()
"$BEHAVIOR_PYTHON" - "$RFT_OUTPUT_DIR" "${GPUS[@]}" <<'PY'
import json
from pathlib import Path
import sys
root, *gpus = sys.argv[1:]
root = Path(root)
summaries = []
for gpu in gpus:
    paths = list((root / "workers" / f"gpu-{gpu}" / "data").glob("task-*/collection_summary.json"))
    if len(paths) != 1:
        raise SystemExit(f"Expected one completed task summary for GPU {gpu}")
    summaries.append(json.loads(paths[0].read_text()))
summary = {"task": summaries[0]["task"], "task_id": summaries[0]["task_id"], "mode": "train",
           "num_gpus": 2, "num_parallel_envs": 4,
           "completed_rollouts": sum(item["completed_rollouts"] for item in summaries),
           "successful_rollouts": sum(item["successful_rollouts"] for item in summaries),
           "successes_by_instance": {key: value for item in summaries for key, value in item["successes_by_instance"].items()},
           "worker_summaries": summaries}
(root / "collection_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
