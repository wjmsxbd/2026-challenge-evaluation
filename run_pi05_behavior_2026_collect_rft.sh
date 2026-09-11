#!/usr/bin/env bash
set -euo pipefail

# Started by the current-pretrain wrapper, which supplies the policy/runtime paths.
# One local GPU hosts a persistent policy server and a vector RFT collector.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-${SCRIPT_DIR}}"
RFT_TASK_ID="${RFT_TASK_ID:-1}"
RFT_INSTANCE_INDICES="${RFT_INSTANCE_INDICES:-0 1 2 3}"
RFT_NUM_ENVS="${RFT_NUM_ENVS:-2}"
RFT_NUM_ROLLOUTS="${RFT_NUM_ROLLOUTS:-1}"
RFT_SUCCESSES_PER_INSTANCE="${RFT_SUCCESSES_PER_INSTANCE:-1}"
RFT_SAMPLE_START="${RFT_SAMPLE_START:-0}"
RFT_SEED="${RFT_SEED:-0}"
RFT_MAX_STEPS="${RFT_MAX_STEPS:-}"
RFT_MAX_STEPS_MULTIPLIER="${RFT_MAX_STEPS_MULTIPLIER:-1.2}"
RFT_PERTURB_POSE="${RFT_PERTURB_POSE:-true}"
RFT_GPU_ID="${RFT_GPU_ID:-0}"
RFT_PORT="${RFT_PORT:-18101}"
RFT_CPU_THREADS="${RFT_CPU_THREADS:-16}"
RFT_WORKER_INDEX="${RFT_WORKER_INDEX:-0}"
SERVER_CPU_NUM_THREADS="${SERVER_CPU_NUM_THREADS:-2}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
RFT_OUTPUT_DIR="${RFT_OUTPUT_DIR:-${BEHAVIOR_ROOT}/logs/rft_task0001_test/$(date +%Y%m%d_%H%M%S)}"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-${BEHAVIOR_ROOT}/datasets}"
DRIVER_FIX_SCRIPT="${DRIVER_FIX_SCRIPT:-${HOME}/driver_fix/activate.sh}"
PI05_APPLY_EVAL_TRICKS="${PI05_APPLY_EVAL_TRICKS:-true}"
PI05_NUM_STEPS="${PI05_NUM_STEPS:-20}"
PI05_BASE_VELOCITY_FRAME="${PI05_BASE_VELOCITY_FRAME:-absolute}"
RESUME=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: bash run_rft_2026_task0001_4instances_1p2x.sh [--output-dir DIR] [--resume] [--dry-run]

Defaults: task 0001 (picking_up_trash), train instances 0 1 2 3, two parallel
environments on GPU 0, one attempt per instance, 1.2x mean human-demo timeout.
The reference launcher's current-pretrain policy, inference checkpoint, action
compression (26 -> 20), correction rules and observation frame are reused.
Every action is rendered. Successful samples contain NPZ + three MP4 files.

Overrides:
  RFT_NUM_ROLLOUTS / RFT_SUCCESSES_PER_INSTANCE  Attempt budget / success target.
  RFT_INSTANCE_INDICES / RFT_SAMPLE_START / RFT_SEED  Samples to collect.
  RFT_GPU_ID / RFT_PORT / RFT_CPU_THREADS       Local resources (0 / 18101 / 16).
  RFT_MAX_STEPS                                Optional short smoke-test limit.
  RFT_PERTURB_POSE                             Initial pose perturbation (true).
  PI05_POLICY_DIR / PI05_NORM_STATS_PATH        Same overrides as the reference.
  PI05_APPLY_EVAL_TRICKS                       Apply reference corrections (true).
  RFT_OUTPUT_DIR                              Run directory; also --output-dir.

Logs and run_config.json are written to DIR. Raw episodes are under
DIR/data/task-0001_picking_up_trash/. Resume requires the original run directory.
Only an existing server-readable checkpoint (or its _inference sibling) is used.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--output-dir needs a directory" >&2; exit 2; }
      RFT_OUTPUT_DIR="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
RFT_OUTPUT_DIR="$(realpath -m -- "$RFT_OUTPUT_DIR")"

: "${PI05_PYTHON:?Use run_rft_2026_task0001_4instances_1p2x.sh to load the policy defaults}"
: "${BEHAVIOR_PYTHON:?Missing preinstalled BEHAVIOR Python}"
for value in "$RFT_NUM_ENVS" "$RFT_NUM_ROLLOUTS" "$RFT_CPU_THREADS" "$SERVER_CPU_NUM_THREADS" "$SERVER_START_TIMEOUT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Counts and timeouts must be positive integers" >&2; exit 2; }
done
for value in "$RFT_TASK_ID" "$RFT_GPU_ID" "$RFT_WORKER_INDEX" "$RFT_SAMPLE_START" "$RFT_SUCCESSES_PER_INSTANCE" "$RFT_SEED"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "IDs and sample settings must be non-negative integers" >&2; exit 2; }
done
for value in "$RFT_PERTURB_POSE" "$PI05_APPLY_EVAL_TRICKS"; do
  [[ "$value" == true || "$value" == false ]] || { echo "Boolean settings must be true or false" >&2; exit 2; }
done
[[ "$RFT_NUM_ENVS" == 2 ]] || { echo "The reference policy server requires RFT_NUM_ENVS=2" >&2; exit 2; }
[[ "${USE_PI05_TASK_CHECKPOINT_MAPPING:-false}" == false ]] || {
  echo "Set USE_PI05_TASK_CHECKPOINT_MAPPING=false and PI05_POLICY_DIR to this task's checkpoint" >&2; exit 2;
}
[[ -r "$DRIVER_FIX_SCRIPT" ]] || { echo "Missing GPU driver activation script" >&2; exit 1; }
[[ -x "$PI05_PYTHON" && -x "$BEHAVIOR_PYTHON" && -f "$PI05_SERVER_SCRIPT" && -f "$PI05_NORM_STATS_PATH" ]] || {
  echo "A required preinstalled runtime, policy server or normalization file is missing" >&2; exit 1;
}
for command in curl setsid taskset flock; do command -v "$command" >/dev/null; done

is_inference_checkpoint() {
  [[ -f "$1/params/_METADATA" && -f "$1/params/manifest.ocdbt" && -d "$1/params/d" ]]
}
if is_inference_checkpoint "${PI05_POLICY_DIR%/}"; then
  RESOLVED_CHECKPOINT="${PI05_POLICY_DIR%/}"
elif is_inference_checkpoint "${PI05_POLICY_DIR%/}${PI05_INFERENCE_CKPT_SUFFIX:-_inference}"; then
  RESOLVED_CHECKPOINT="${PI05_POLICY_DIR%/}${PI05_INFERENCE_CKPT_SUFFIX:-_inference}"
else
  echo "No readable inference checkpoint found; set PI05_POLICY_DIR to an existing inference checkpoint" >&2
  exit 1
fi

# Use stdlib only: validate task/instances and reserve disjoint CPU sets before starting Isaac.
PREFLIGHT="$("$BEHAVIOR_PYTHON" - "$OMNIGIBSON_DATA_PATH" "$RFT_TASK_ID" "$RFT_INSTANCE_INDICES" \
  "$RFT_CPU_THREADS" "$SERVER_CPU_NUM_THREADS" "$RFT_PORT" "$RFT_MAX_STEPS_MULTIPLIER" "$RFT_MAX_STEPS" "$RFT_WORKER_INDEX" <<'PY'
import json
import math
import os
from pathlib import Path
import socket
import sys

data_root, task_id, raw_ids, env_threads, server_threads, port, multiplier, max_steps, worker_index = sys.argv[1:]
rows = [json.loads(line) for line in (Path(data_root) / "2026-challenge-task-instances/metadata/task.jsonl").read_text().splitlines()]
task = next((row for row in rows if row["task_index"] == int(task_id)), None)
if task is None:
    raise SystemExit("Task ID is absent from the 2026 task metadata")
ids = [int(value) for value in raw_ids.replace(",", " ").split()]
if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
    raise SystemExit("RFT_INSTANCE_INDICES must contain unique, non-negative train instance IDs")
for instance_id in ids:
    pattern = f"*/json/*_instances/*_task_{task['task_name']}_0_{instance_id}_template-tro_state.json"
    if not any((Path(data_root) / "2026-challenge-task-instances/scenes").glob(pattern)):
        raise SystemExit(f"Training instance {instance_id} is missing for task {task_id}")
if not math.isfinite(float(multiplier)) or float(multiplier) <= 0 or (max_steps and int(max_steps) <= 0):
    raise SystemExit("RFT_MAX_STEPS_MULTIPLIER and RFT_MAX_STEPS must be positive")
cpus = sorted(os.sched_getaffinity(0))
env_count, server_count = int(env_threads), int(server_threads)
offset = int(worker_index) * (env_count + server_count)
if len(cpus) < offset + env_count + server_count:
    raise SystemExit("Not enough CPUs; reduce RFT_CPU_THREADS or SERVER_CPU_NUM_THREADS")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(port)))
print(task["task_name"])
print(",".join(map(str, cpus[offset:offset + env_count])))
print(",".join(map(str, cpus[offset + env_count:offset + env_count + server_count])))
print(int(max_steps) if max_steps else int(task["length"] * float(multiplier)))
PY
)"
mapfile -t PLAN <<<"$PREFLIGHT"
TASK_NAME="${PLAN[0]}"
EVAL_CPUS="${PLAN[1]}"
SERVER_CPUS="${PLAN[2]}"
read -r -a INSTANCE_IDS <<<"${RFT_INSTANCE_INDICES//,/ }"
SERVER_PYTHONPATH="${PI05_REPO}/src:${CURRENT_PRETRAIN_ROOT}:${PYTHONPATH:-}"
EVAL_PYTHONPATH="${BEHAVIOR_ROOT}/OmniGibson:${BEHAVIOR_ROOT}/bddl3:${BEHAVIOR_ROOT}/joylo:${BEHAVIOR_ROOT}:${SERVER_PYTHONPATH}"
export OMNIGIBSON_DATA_PATH CURRENT_PRETRAIN_ROOT
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

SERVER_ARGS=("$PI05_SERVER_SCRIPT" --host 127.0.0.1 --port "$RFT_PORT" --inference-only
  --num-steps "$PI05_NUM_STEPS" --dynamic-batching --dynamic-batch-max-size "$RFT_NUM_ENVS"
  --dynamic-batch-wait-ms 0 --dynamic-batch-granularity 1 --disable-fast-auxiliary
  --norm-stats-path "$PI05_NORM_STATS_PATH" policy:checkpoint
  --policy.config "$PI05_POLICY_CONFIG" --policy.dir "$RESOLVED_CHECKPOINT")
COLLECT_ARGS=(-m omnigibson.eval.collect_rft --task-name "$TASK_NAME" --mode train
  --instance-indices "${INSTANCE_IDS[@]}" --host 127.0.0.1 --port "$RFT_PORT" --num-envs "$RFT_NUM_ENVS"
  --policy-checkpoint "$RESOLVED_CHECKPOINT" --num-rollouts "$RFT_NUM_ROLLOUTS"
  --successes-per-instance "$RFT_SUCCESSES_PER_INSTANCE" --sample-start "$RFT_SAMPLE_START" --seed "$RFT_SEED"
  --max-steps-multiplier "$RFT_MAX_STEPS_MULTIPLIER" --output-dir "$RFT_OUTPUT_DIR/data"
  --cpu-affinity "$EVAL_CPUS" --cpu-num-threads "$RFT_CPU_THREADS"
  --actions-to-execute 26 --actions-to-keep 4 --execute-in-n-steps 20
  --pi05-base-velocity-frame "$PI05_BASE_VELOCITY_FRAME" --headless --no-render-viewer-camera --no-write-video)
[[ -z "$RFT_MAX_STEPS" ]] || COLLECT_ARGS+=(--max-steps "$RFT_MAX_STEPS")
[[ "$RFT_PERTURB_POSE" == true ]] && COLLECT_ARGS+=(--perturb-pose) || COLLECT_ARGS+=(--no-perturb-pose)
[[ "$PI05_APPLY_EVAL_TRICKS" == true ]] && COLLECT_ARGS+=(--apply-eval-tricks) || COLLECT_ARGS+=(--no-apply-eval-tricks)
[[ "$RESUME" == false ]] || COLLECT_ARGS+=(--resume)
printf 'RFT test: task=%04d (%s), train instances=%s, parallel envs=%s, GPU=%s, max steps=%s\n' \
  "$RFT_TASK_ID" "$TASK_NAME" "$RFT_INSTANCE_INDICES" "$RFT_NUM_ENVS" "$RFT_GPU_ID" "${PLAN[3]}"
printf 'Run directory: %s\n' "$RFT_OUTPUT_DIR"
if [[ "$DRY_RUN" == true ]]; then
  printf 'Collector: %q ' "$BEHAVIOR_PYTHON"; printf '%q ' "${COLLECT_ARGS[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$RFT_OUTPUT_DIR"
exec 9>"$RFT_OUTPUT_DIR/.launcher.lock"
flock -n 9 || { echo "Another launcher is using this run directory" >&2; exit 1; }
if [[ -f "$RFT_OUTPUT_DIR/run_config.json" && "$RESUME" != true ]]; then
  echo "Run directory exists; use --resume or a new --output-dir" >&2; exit 1
fi
"$BEHAVIOR_PYTHON" - "$RFT_OUTPUT_DIR" "$BEHAVIOR_ROOT" "$RFT_GPU_ID" "$RFT_NUM_ENVS" \
  "$RESOLVED_CHECKPOINT" "$PI05_REPO" "$PI05_NORM_STATS_PATH" "$RESUME" "${COLLECT_ARGS[@]}" <<'PY'
import datetime
import json
from pathlib import Path
import subprocess
import sys
run_dir, repo, gpu, envs, checkpoint, policy_repo, norm_stats, resume, *args = sys.argv[1:]
config = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              git_revision=subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
              working_tree_dirty=bool(subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True)),
              gpu=gpu, num_envs=int(envs), checkpoint=checkpoint, policy_repo=policy_repo,
              norm_stats=norm_stats, collector_args=args)
name = "run_resume_config.json" if resume == "true" else "run_config.json"
(Path(run_dir) / name).write_text(json.dumps(config, indent=2))
PY

SERVER_PID=""
COLLECTOR_PID=""
stop_process() {
  local pid="$1" first_signal="$2" deadline
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
  kill -"$first_signal" -- "-$pid" 2>/dev/null || kill -"$first_signal" "$pid" 2>/dev/null || true
  deadline=$((SECONDS + 30))
  while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_process "$COLLECTOR_PID" INT
  stop_process "$SERVER_PID" TERM
  printf '%s\n' "$status" >"$RFT_OUTPUT_DIR/exit_status.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(
  cd "$PI05_REPO"
  exec env OMP_NUM_THREADS="$SERVER_CPU_NUM_THREADS" MKL_NUM_THREADS="$SERVER_CPU_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$SERVER_CPU_NUM_THREADS" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}" XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    CUDA_VISIBLE_DEVICES="$RFT_GPU_ID" PYTHONPATH="$SERVER_PYTHONPATH" \
    setsid taskset -c "$SERVER_CPUS" "$PI05_PYTHON" "${SERVER_ARGS[@]}"
) >>"$RFT_OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$RFT_OUTPUT_DIR/server.pid"
deadline=$((SECONDS + SERVER_START_TIMEOUT))
until curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$RFT_PORT/healthz" >/dev/null 2>&1; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "Policy server exited; see server.log" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "Policy server startup timed out; see server.log" >&2; exit 1; }
  sleep 5
done
echo "Policy server ready; starting parallel collection."
(
  behavior_env_dir="$(dirname "$(dirname "$BEHAVIOR_PYTHON")")"
  conda_root="$(dirname "$(dirname "$behavior_env_dir")")"
  set +u
  source "$DRIVER_FIX_SCRIPT"
  source "$conda_root/etc/profile.d/conda.sh"
  conda activate "$behavior_env_dir"
  set -u
  cd "$BEHAVIOR_ROOT/OmniGibson"
  exec env OMP_NUM_THREADS="$RFT_CPU_THREADS" MKL_NUM_THREADS="$RFT_CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$RFT_CPU_THREADS" NUMEXPR_NUM_THREADS="$RFT_CPU_THREADS" \
    PYTHONUNBUFFERED=1 PYTHONHASHSEED="$RFT_SEED" OMNIGIBSON_HEADLESS=1 \
    PI05_SKIP_ACTION_CHUNK_RENDERING=false PI05_CORRECTION_RULES_PATH="$PI05_REPO/src/b1k/shared/correction_rules.py" \
    MPLCONFIGDIR="$RFT_OUTPUT_DIR/matplotlib" CUDA_VISIBLE_DEVICES="$RFT_GPU_ID" PYTHONPATH="$EVAL_PYTHONPATH" \
    setsid taskset -c "$EVAL_CPUS" "$BEHAVIOR_PYTHON" "${COLLECT_ARGS[@]}"
) >>"$RFT_OUTPUT_DIR/collector.log" 2>&1 &
COLLECTOR_PID=$!
printf '%s\n' "$COLLECTOR_PID" >"$RFT_OUTPUT_DIR/collector.pid"
wait "$COLLECTOR_PID"
COLLECTOR_PID=""
# Isaac shutdown can return zero after an exception; require the collection summary too.
printf -v TASK_DIR 'task-%04d_%s' "$RFT_TASK_ID" "$TASK_NAME"
[[ -f "$RFT_OUTPUT_DIR/data/$TASK_DIR/collection_summary.json" ]] || {
  echo "Collection did not finish; see collector.log" >&2; exit 1;
}
"$BEHAVIOR_PYTHON" - "$RFT_OUTPUT_DIR/data/$TASK_DIR" "$RFT_SAMPLE_START" "$RFT_NUM_ROLLOUTS" \
  "$RFT_SUCCESSES_PER_INSTANCE" "${INSTANCE_IDS[@]}" <<'PY'
import json
from pathlib import Path
import sys
root, start, attempts, target, *instances = sys.argv[1:]
root = Path(root)
for instance in map(int, instances):
    complete = successes = 0
    for sample in range(int(start), int(start) + int(attempts)):
        path = root / "rollouts" / f"instance-{instance:04d}" / f"sample-{sample:06d}" / "episode.json"
        if path.is_file():
            episode = json.loads(path.read_text())
            if episode.get("complete") is not True:
                raise SystemExit(f"Incomplete episode: {path}")
            complete += 1
            successes += int(episode["success"])
    if complete != int(attempts) and not (int(target) and successes >= int(target)):
        raise SystemExit(f"Instance {instance} did not finish its attempt budget or reach its success target")
print("Validated: every requested instance finished its attempts or reached its success target.")
PY
cat "$RFT_OUTPUT_DIR/data/$TASK_DIR/collection_summary.json"
