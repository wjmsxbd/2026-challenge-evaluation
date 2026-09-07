#!/usr/bin/env bash
set -euo pipefail

# Tasks 50-99, public instances 0-3, one rollout each (200 rollouts).
# Reuse Isaac processes with a task-specific 1.2x mean human-demo timeout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf -v TASK_IDS '%s ' {50..99}
export TASK_IDS
export TASK_LIMIT=50
export EVAL_INSTANCE_INDICES='0 1 2 3'
export EVAL_MAX_STEPS_MULTIPLIER=1.2
# An inherited absolute step limit would override the multiplier.
unset EVAL_MAX_STEPS

# Fixed fast-profile optimization: skip rendering only inside each action
# chunk's steps 2..10 (1-based). Rendering resumes at step 1 and step 11.
export PI05_SKIP_ACTION_CHUNK_RENDERING=true

exec bash "${SCRIPT_DIR}/run_eval_2026_persistent.sh" "$@"
