"""Central task broker for multi-node PI0.5 evaluation.

Only rank 0 runs this HTTP service. Every GPU worker claims one task chunk at
a time, then acknowledges it before claiming another. The broker is the sole
owner of mutable scheduling state, avoiding unreliable cross-node NAS locks.
"""

import argparse
import hmac
import json
import logging
import os
import threading
import urllib.parse
from collections import deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


LOGGER = logging.getLogger("pi05_dynamic_scheduler")


@dataclass(frozen=True)
class TaskChunk:
    task_id: int
    task_name: str
    instance_indices: str
    timeout_steps: int
    chunk_steps: int
    queue_order: int

    @classmethod
    def from_tsv(cls, line: str) -> "TaskChunk":
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 6:
            raise ValueError(f"Expected 6 tab-separated task fields, got {len(fields)}: {line!r}")
        task_id, task_name, instance_indices, timeout_steps, chunk_steps, queue_order = fields
        return cls(
            task_id=int(task_id),
            task_name=task_name,
            instance_indices=instance_indices,
            timeout_steps=int(timeout_steps),
            chunk_steps=int(chunk_steps),
            queue_order=int(queue_order),
        )

    def to_tsv(self) -> str:
        return (
            f"{self.task_id}\t{self.task_name}\t{self.instance_indices}\t"
            f"{self.timeout_steps}\t{self.chunk_steps}\t{self.queue_order}"
        )


class SchedulerState:
    """Thread-safe, idempotent task-claim state."""

    def __init__(self, tasks: list[TaskChunk], stop_file: Path, journal_path: Path):
        queue_orders = [task.queue_order for task in tasks]
        if len(queue_orders) != len(set(queue_orders)):
            raise ValueError("Queue order values must be unique")
        self._pending = deque(tasks)
        self._active: dict[int, TaskChunk] = {}
        self._completed: dict[int, tuple[int, str]] = {}
        self._stop_file = stop_file
        self._lock = threading.Lock()
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._journal = journal_path.open("a", encoding="utf-8", buffering=1)
        self._total = len(tasks)
        self._write_event("start", total=self._total)

    def close(self) -> None:
        with self._lock:
            self._write_event("shutdown", **self._stats_unlocked())
            self._journal.close()

    def claim(self, worker: int) -> tuple[str, TaskChunk | None]:
        with self._lock:
            if worker in self._active:
                return "assigned", self._active[worker]
            if self._stop_file.exists():
                return "stopped", None
            if not self._pending:
                return "empty", None
            task = self._pending.popleft()
            self._active[worker] = task
            self._write_event("claim", worker=worker, task=asdict(task))
            return "assigned", task

    def complete(self, worker: int, queue_order: int, status: str) -> str:
        if status not in {"success", "failed"}:
            raise ValueError(f"Invalid completion status: {status}")
        with self._lock:
            task = self._active.get(worker)
            if task is None:
                previous = self._completed.get(queue_order)
                if previous == (worker, status):
                    return "already_complete"
                raise ValueError(f"Worker {worker} has no active claim")
            if task.queue_order != queue_order:
                raise ValueError(
                    f"Worker {worker} owns queue order {task.queue_order}, not {queue_order}"
                )
            del self._active[worker]
            self._completed[queue_order] = (worker, status)
            self._write_event("complete", worker=worker, queue_order=queue_order, status=status)
            return "complete"

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> dict[str, int | bool]:
        failed = sum(status == "failed" for _, status in self._completed.values())
        return {
            "total": self._total,
            "pending": len(self._pending),
            "active": len(self._active),
            "completed": len(self._completed),
            "failed": failed,
            "stopped": self._stop_file.exists(),
        }

    def _write_event(self, event: str, **fields) -> None:
        self._journal.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")


class SchedulerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, state: SchedulerState, token: str):
        super().__init__(address, SchedulerHandler)
        self.scheduler_state = state
        self.scheduler_token = token


class SchedulerHandler(BaseHTTPRequestHandler):
    server: SchedulerHTTPServer

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok", **self.server.scheduler_state.stats()})
            return
        if path == "/stats":
            if not self._authorized():
                return
            self._send_json(HTTPStatus.OK, self.server.scheduler_state.stats())
            return
        self._send_text(HTTPStatus.NOT_FOUND, "not_found\n")

    def do_POST(self) -> None:
        if not self._authorized():
            return
        path = urllib.parse.urlsplit(self.path).path
        try:
            form = self._read_form()
            worker = self._nonnegative_int(form, "worker")
            if path == "/claim":
                state, task = self.server.scheduler_state.claim(worker)
                body = state if task is None else f"assigned\t{task.to_tsv()}"
                self._send_text(HTTPStatus.OK, body + "\n")
                return
            if path == "/complete":
                queue_order = self._positive_int(form, "queue_order")
                status = self._one_value(form, "status")
                result = self.server.scheduler_state.complete(worker, queue_order, status)
                self._send_text(HTTPStatus.OK, result + "\n")
                return
            self._send_text(HTTPStatus.NOT_FOUND, "not_found\n")
        except ValueError as error:
            self._send_text(HTTPStatus.CONFLICT, f"error\t{error}\n")

    def log_message(self, format_string: str, *args) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format_string % args)

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-Scheduler-Token", "")
        if hmac.compare_digest(supplied, self.server.scheduler_token):
            return True
        self._send_text(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return False

    def _read_form(self) -> dict[str, list[str]]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if not 0 <= content_length <= 8192:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(content_length).decode("utf-8")
        return urllib.parse.parse_qs(raw, keep_blank_values=True, strict_parsing=True)

    @staticmethod
    def _one_value(form: dict[str, list[str]], key: str) -> str:
        values = form.get(key, [])
        if len(values) != 1:
            raise ValueError(f"Expected exactly one {key}")
        return values[0]

    @classmethod
    def _nonnegative_int(cls, form: dict[str, list[str]], key: str) -> int:
        value = int(cls._one_value(form, key))
        if value < 0:
            raise ValueError(f"{key} must be nonnegative")
        return value

    @classmethod
    def _positive_int(cls, form: dict[str, list[str]], key: str) -> int:
        value = int(cls._one_value(form, key))
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload, sort_keys=True) + "\n", "application/json")

    def _send_text(self, status: HTTPStatus, body: str) -> None:
        self._send(status, body, "text/plain; charset=utf-8")

    def _send(self, status: HTTPStatus, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            LOGGER.warning("Client disconnected before receiving the response")


def load_tasks(path: Path) -> list[TaskChunk]:
    return [TaskChunk.from_tsv(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--queue-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tasks = load_tasks(args.queue_file)
    if not tasks:
        raise SystemExit(f"Task queue is empty: {args.queue_file}")
    state = SchedulerState(tasks, args.stop_file, args.journal)
    server = SchedulerHTTPServer((args.host, args.port), state, args.token)
    LOGGER.info("Serving %d task chunks on %s:%d", len(tasks), args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
