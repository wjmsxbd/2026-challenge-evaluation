"""Single-request filesystem mailbox for a persistent evaluator process.

The scheduler publishes request.tsv using rename, then waits for response.tsv.
Each request has a unique ID; a failed request terminates this worker so that
the scheduler can retry in a fresh process. This module needs no simulator.
"""

import os
from pathlib import Path
import sys
import time
import traceback


def serve_requests(worker_dir, run_request, poll_interval=0.2):
    """Run requests until shutdown or failure; return a process exit status.

    Requests contain ID, task name, output directory, and comma-separated
    public instance indices. Responses contain the same ID and an exit status.
    """
    worker_dir = Path(worker_dir)
    request_path = worker_dir / "request.tsv"
    while not (worker_dir / "shutdown").exists():
        if not request_path.exists():
            time.sleep(poll_interval)
            continue
        request_id = "invalid"
        status = 0
        try:
            fields = request_path.read_text(encoding="utf-8").rstrip("\n").split("\t")
            request_path.unlink()
            request_id, task_name, output_dir, indices_text = fields
            indices = [int(value) for value in indices_text.split(",")]
            if not request_id or not task_name or not output_dir or not indices or len(set(indices)) != len(indices):
                raise ValueError("Invalid persistent evaluation request")
            if Path(output_dir).exists():
                raise FileExistsError(f"Refusing to overwrite existing attempt: {output_dir}")
            print(f"Persistent request start: id={request_id} pid={os.getpid()} task={task_name}", flush=True)
            run_request(task_name, indices, output_dir)
        except Exception:
            traceback.print_exc()
            status = 1
        print(f"Persistent request finish: id={request_id} pid={os.getpid()} status={status}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        temporary = worker_dir / "response.tsv.tmp"
        temporary.write_text(f"{request_id}\t{status}\n", encoding="utf-8")
        temporary.replace(worker_dir / "response.tsv")
        if status:
            return status
    return 0
