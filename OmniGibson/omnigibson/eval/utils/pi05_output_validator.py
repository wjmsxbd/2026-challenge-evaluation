#!/usr/bin/env python3
"""Validate PI0.5 BEHAVIOR 2026 evaluator artifacts without importing OmniGibson.

The launcher writes every evaluator attempt outside the final task namespace.  This
module validates an attempt against an immutable run manifest before the launcher
promotes it, and validates the exact final directory set before declaring a run
complete.
"""

# SPEEDUP_EVAL: keep artifact validation independent from Isaac Sim so a failed
# simulator attempt can be rejected/retried without importing or relaunching it.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn


SCHEMA_VERSION = 1
PUBLIC_INSTANCE_ID_START = 301
PUBLIC_INSTANCE_COUNT = 20
VIDEO_CODEC = "h264"
VIDEO_WIDTH = 672
VIDEO_HEIGHT = 448
VIDEO_FPS = Fraction(30, 1)
DISTANCE_KEYS = ("base", "left", "right")
TIME_KEYS = ("simulator_steps", "simulator_time", "normalized_time")


class ValidationError(RuntimeError):
    """Raised when an artifact does not match the declared evaluation protocol."""


def _fail(message: str) -> NoReturn:
    raise ValidationError(message)


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant {value!r}")


def _load_json_bytes(path: Path) -> tuple[Any, bytes]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"), parse_constant=_reject_json_constant)
        return value, encoded
    except ValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse JSON {path}: {exc}")


def _load_json(path: Path) -> Any:
    return _load_json_bytes(path)[0]


def _atomic_json_dump(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                _fail(f"refusing to overwrite existing file: {path}")
            temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _require_exact_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{label} must be {expected}, got {value!r}")


def _require_finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be a finite number, got {value!r}")


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be a positive integer, got {value!r}")
    return value


def _validate_manifest_structure(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> None:
    _require_exact_int(manifest.get("schema_version"), SCHEMA_VERSION, "manifest.schema_version")
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        _fail("manifest.created_at must be a non-empty string")
    if manifest.get("mode") != "public_test":
        _fail(f"manifest.mode must be 'public_test', got {manifest.get('mode')!r}")
    if not isinstance(manifest.get("write_video"), bool):
        _fail("manifest.write_video must be boolean")
    _require_positive_int(manifest.get("num_vector_envs"), "manifest.num_vector_envs")
    _require_exact_int(manifest.get("num_rollouts"), 1, "manifest.num_rollouts")
    if manifest.get("rollout_ids") != [0]:
        _fail(f"manifest.rollout_ids must be [0], got {manifest.get('rollout_ids')!r}")

    indices = manifest.get("public_instance_indices")
    if not isinstance(indices, list) or not indices:
        _fail("manifest.public_instance_indices must be a non-empty list")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        _fail("manifest.public_instance_indices must contain only integers")
    if len(indices) != len(set(indices)):
        _fail("manifest.public_instance_indices contains duplicates")
    invalid_indices = [index for index in indices if index < 0 or index >= PUBLIC_INSTANCE_COUNT]
    if invalid_indices:
        _fail(
            f"manifest.public_instance_indices must be in [0,{PUBLIC_INSTANCE_COUNT - 1}], "
            f"got {invalid_indices}"
        )
    expected_instance_ids = [PUBLIC_INSTANCE_ID_START + index for index in indices]
    if manifest.get("instance_ids") != expected_instance_ids:
        _fail(
            f"manifest.instance_ids must be derived from public indices as {expected_instance_ids}, "
            f"got {manifest.get('instance_ids')!r}"
        )

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        _fail("manifest.tasks must be a non-empty list")
    seen_ids: set[int] = set()
    seen_dirs: set[str] = set()
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, dict):
            _fail(f"manifest.tasks[{index}] must be an object")
        task_id = raw_task.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id not in range(100):
            _fail(f"manifest.tasks[{index}].task_id must be in [0,99], got {task_id!r}")
        task_name = raw_task.get("task")
        if (
            not isinstance(task_name, str)
            or not task_name
            or task_name in {".", ".."}
            or "/" in task_name
            or "\0" in task_name
        ):
            _fail(f"manifest.tasks[{index}].task is unsafe: {task_name!r}")
        expected_dir = f"task-{task_id}_{task_name}"
        if raw_task.get("output_dir") != expected_dir:
            _fail(
                f"manifest.tasks[{index}].output_dir must be {expected_dir!r}, "
                f"got {raw_task.get('output_dir')!r}"
            )
        if task_id in seen_ids or expected_dir in seen_dirs:
            _fail(f"manifest contains duplicate task entry for ID {task_id}")
        seen_ids.add(task_id)
        seen_dirs.add(expected_dir)

    _require_exact_int(manifest.get("task_count"), len(tasks), "manifest.task_count")
    expected_results = len(tasks) * len(expected_instance_ids)
    _require_exact_int(
        manifest.get("expected_result_count"), expected_results, "manifest.expected_result_count"
    )
    expected_videos = expected_results if manifest["write_video"] else 0
    _require_exact_int(
        manifest.get("expected_video_count"), expected_videos, "manifest.expected_video_count"
    )

    declared_root = manifest.get("run_output_root")
    if not isinstance(declared_root, str) or not Path(declared_root).is_absolute():
        _fail(f"manifest.run_output_root must be an absolute path, got {declared_root!r}")
    declared_manifest_path = manifest.get("manifest_path")
    if not isinstance(declared_manifest_path, str) or not Path(declared_manifest_path).is_absolute():
        _fail(f"manifest.manifest_path must be an absolute path, got {declared_manifest_path!r}")
    expected_manifest_path = Path(declared_root).resolve() / "run_manifest.json"
    if Path(declared_manifest_path).resolve() != expected_manifest_path:
        _fail(
            f"manifest must be stored at {expected_manifest_path}, got {Path(declared_manifest_path).resolve()}"
        )
    if manifest_path is not None and Path(declared_manifest_path).resolve() != manifest_path.resolve():
        _fail(
            f"manifest declares path {declared_manifest_path}, but was loaded from {manifest_path.resolve()}"
        )


def _parse_queue(queue_file: Path) -> list[dict[str, Any]]:
    try:
        lines = queue_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(f"cannot read task queue {queue_file}: {exc}")
    tasks: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_output_dirs: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 2:
            _fail(f"{queue_file}:{line_number}: expected task_id<TAB>task_name")
        raw_task_id, task_name = fields
        try:
            task_id = int(raw_task_id)
        except ValueError:
            _fail(f"{queue_file}:{line_number}: invalid task ID {raw_task_id!r}")
        if task_id < 0 or task_id > 99:
            _fail(f"{queue_file}:{line_number}: task ID must be in [0,99], got {task_id}")
        if not task_name or task_name in {".", ".."} or "/" in task_name or "\0" in task_name:
            _fail(f"{queue_file}:{line_number}: unsafe task name {task_name!r}")
        output_dir = f"task-{task_id}_{task_name}"
        if task_id in seen_ids:
            _fail(f"duplicate task ID in queue: {task_id}")
        if output_dir in seen_output_dirs:
            _fail(f"duplicate task output directory in queue: {output_dir}")
        seen_ids.add(task_id)
        seen_output_dirs.add(output_dir)
        tasks.append({"task_id": task_id, "task": task_name, "output_dir": output_dir})
    if not tasks:
        _fail("task queue is empty")
    return tasks


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    # SPEEDUP_EVAL: freeze the expected task/instance/video set before workers
    # start; later validation compares every attempt against this immutable plan.
    if args.output.exists():
        _fail(f"refusing to overwrite immutable manifest: {args.output}")
    if args.mode != "public_test":
        _fail(f"the 2026 launcher requires public_test mode, got {args.mode!r}")
    indices = args.instance_indices
    if not indices:
        _fail("at least one public instance index is required")
    if len(indices) != len(set(indices)):
        _fail("public instance indices contain duplicates")
    invalid = [index for index in indices if index < 0 or index >= PUBLIC_INSTANCE_COUNT]
    if invalid:
        _fail(f"public instance indices must be in [0,{PUBLIC_INSTANCE_COUNT - 1}], got {invalid}")
    if args.num_rollouts != 1:
        _fail("the 2026 launcher protocol currently requires exactly one rollout")
    if args.num_vector_envs <= 0:
        _fail("num-vector-envs must be positive")

    tasks = _parse_queue(args.queue_file)
    instance_ids = [PUBLIC_INSTANCE_ID_START + index for index in indices]
    result_count = len(tasks) * len(instance_ids) * args.num_rollouts
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "mode": args.mode,
        "write_video": args.write_video,
        "num_vector_envs": args.num_vector_envs,
        "num_rollouts": args.num_rollouts,
        "rollout_ids": list(range(args.num_rollouts)),
        "public_instance_indices": indices,
        "instance_ids": instance_ids,
        "task_count": len(tasks),
        "expected_result_count": result_count,
        "expected_video_count": result_count if args.write_video else 0,
        "run_output_root": str(args.run_output_root.resolve()),
        "manifest_path": str(args.output.resolve()),
        "tasks": tasks,
    }
    _validate_manifest_structure(manifest, manifest_path=args.output)
    _atomic_json_dump(args.output, manifest, replace=False)
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    # SPEEDUP_EVAL: hash the exact manifest bytes so workers cannot silently use
    # a different task set or instance selection during a long distributed run.
    raw_manifest, encoded = _load_json_bytes(path)
    manifest = _require_dict(raw_manifest, f"manifest {path}")
    _validate_manifest_structure(manifest, manifest_path=path)
    manifest["_manifest_path"] = str(path.resolve())
    manifest["_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _task_from_manifest(manifest: dict[str, Any], task_id: int) -> dict[str, Any]:
    matches = [task for task in manifest["tasks"] if task.get("task_id") == task_id]
    if len(matches) != 1:
        _fail(f"manifest contains {len(matches)} entries for task ID {task_id}")
    return _require_dict(matches[0], f"manifest task {task_id}")


def _expected_names(task: dict[str, Any], manifest: dict[str, Any], suffix: str) -> set[str]:
    return {
        f"{task['task']}_{instance_id}_{rollout_id}.{suffix}"
        for instance_id in manifest["instance_ids"]
        for rollout_id in manifest["rollout_ids"]
    }


def _actual_file_names(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        _fail(f"required directory is missing: {directory}")
    return {entry.name for entry in directory.iterdir() if entry.is_file() and entry.suffix == f".{suffix}"}


def _validate_exact_names(actual: set[str], expected: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        _fail(f"{label} filename set mismatch; missing={missing}, extra={extra}")


def _validate_metric_dict(result: dict[str, Any], key: str, required_keys: tuple[str, ...]) -> None:
    metric = _require_dict(result.get(key), key)
    for metric_key in required_keys:
        if metric_key not in metric:
            _fail(f"{key}.{metric_key} is missing")
        _require_finite_number(metric[metric_key], f"{key}.{metric_key}")


def _validate_result_json(
    path: Path,
    *,
    task_name: str,
    instance_id: int,
    rollout_id: int,
) -> int:
    result = _require_dict(_load_json(path), f"result {path}")
    if result.get("task") != task_name:
        _fail(f"{path}: task must be {task_name!r}, got {result.get('task')!r}")
    _require_exact_int(result.get("instance_id"), instance_id, f"{path}: instance_id")
    _require_exact_int(result.get("rollout_id"), rollout_id, f"{path}: rollout_id")
    steps = result.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        _fail(f"{path}: steps must be a positive integer, got {steps!r}")
    if not isinstance(result.get("success"), bool):
        _fail(f"{path}: success must be boolean")
    _validate_metric_dict(result, "agent_distance", DISTANCE_KEYS)
    _validate_metric_dict(result, "normalized_agent_distance", DISTANCE_KEYS)
    q_score = _require_dict(result.get("q_score"), "q_score")
    _require_finite_number(q_score.get("final"), "q_score.final")
    _validate_metric_dict(result, "time", TIME_KEYS)
    _require_exact_int(result["time"]["simulator_steps"], steps, f"{path}: time.simulator_steps")
    return steps


def _select_frame_rate(stream: dict[str, Any], path: Path) -> Fraction:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if not isinstance(value, str):
            continue
        try:
            rate = Fraction(value)
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    _fail(f"{path}: ffprobe did not report a valid positive frame rate")


def _parse_probe_output(completed: subprocess.CompletedProcess[str], path: Path) -> dict[str, Any]:
    if completed.returncode != 0 or completed.stderr.strip():
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        _fail(f"ffprobe rejected {path}: {detail}")
    try:
        probe = json.loads(completed.stdout, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValidationError) as exc:
        _fail(f"invalid ffprobe JSON for {path}: {exc}")
    if not isinstance(probe, dict):
        _fail(f"{path}: ffprobe output must be an object")
    return probe


def _probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"cannot run ffprobe for {path}: {exc}")
    streams = _parse_probe_output(completed, path).get("streams")
    if not isinstance(streams, list) or any(not isinstance(stream, dict) for stream in streams):
        _fail(f"{path}: ffprobe streams must be a list of objects")
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        _fail(f"{path}: expected exactly one video stream, got {len(video_streams)}")
    return video_streams[0]


def _count_video_frames(path: Path, ffprobe: str) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"cannot count video frames for {path}: {exc}")
    streams = _parse_probe_output(completed, path).get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        _fail(f"{path}: frame-count probe must return exactly one video stream")
    try:
        return int(streams[0].get("nb_read_frames"))
    except (TypeError, ValueError):
        _fail(f"{path}: nb_read_frames is missing or invalid")


def _validate_video(path: Path, expected_frames: int, ffprobe: str) -> None:
    stream = _probe_video(path, ffprobe)
    if stream.get("codec_name") != VIDEO_CODEC:
        _fail(f"{path}: codec must be {VIDEO_CODEC}, got {stream.get('codec_name')!r}")
    _require_exact_int(stream.get("width"), VIDEO_WIDTH, f"{path}: width")
    _require_exact_int(stream.get("height"), VIDEO_HEIGHT, f"{path}: height")
    frame_rate = _select_frame_rate(stream, path)
    if frame_rate != VIDEO_FPS:
        _fail(f"{path}: frame rate must be 30 fps, got {frame_rate}")
    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = _count_video_frames(path, ffprobe)
    if frame_count != expected_frames:
        _fail(f"{path}: frame count {frame_count} does not equal result steps {expected_frames}")


def validate_instance(
    directory: Path, task_name: str, instance_id: int, *, write_video: bool, ffprobe: str = "ffprobe"
) -> int:
    """Validate one finished rollout, including one saved by an interrupted pair."""
    name = f"{task_name}_{instance_id}_0"
    steps = _validate_result_json(
        directory / "json" / f"{name}.json", task_name=task_name, instance_id=instance_id, rollout_id=0
    )
    if write_video:
        _validate_video(directory / "videos" / f"{name}.mp4", steps, ffprobe)
    return steps


def validate_task(
    manifest: dict[str, Any],
    task_id: int,
    task_dir: Path,
    *,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    task = _task_from_manifest(manifest, task_id)
    if not task_dir.is_dir():
        _fail(f"task output directory is missing: {task_dir}")

    marker_path = task_dir / "evaluation_complete.json"
    if not marker_path.is_file():
        _fail(f"completion marker is missing: {marker_path}")
    marker = _require_dict(_load_json(marker_path), f"completion marker {marker_path}")
    expected_marker = {
        "task": task["task"],
        "task_id": task_id,
        "mode": manifest["mode"],
        "instance_ids": manifest["instance_ids"],
        "num_rollouts": manifest["num_rollouts"],
        "num_vector_envs": manifest["num_vector_envs"],
        "result_count": len(manifest["instance_ids"]) * len(manifest["rollout_ids"]),
        "write_video": manifest["write_video"],
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected or type(marker.get(key)) is not type(expected):
            _fail(f"{marker_path}: {key} must be {expected!r}, got {marker.get(key)!r}")

    json_dir = task_dir / "json"
    expected_json_names = _expected_names(task, manifest, "json")
    _validate_exact_names(_actual_file_names(json_dir, "json"), expected_json_names, str(json_dir))
    steps_by_tuple: dict[tuple[int, int], int] = {}
    for instance_id in manifest["instance_ids"]:
        for rollout_id in manifest["rollout_ids"]:
            name = f"{task['task']}_{instance_id}_{rollout_id}.json"
            steps_by_tuple[(instance_id, rollout_id)] = _validate_result_json(
                json_dir / name,
                task_name=task["task"],
                instance_id=instance_id,
                rollout_id=rollout_id,
            )

    video_dir = task_dir / "videos"
    expected_video_names = _expected_names(task, manifest, "mp4") if manifest["write_video"] else set()
    if manifest["write_video"]:
        actual_video_names = _actual_file_names(video_dir, "mp4")
    else:
        actual_video_names = _actual_file_names(video_dir, "mp4") if video_dir.exists() else set()
    _validate_exact_names(actual_video_names, expected_video_names, str(video_dir))
    if manifest["write_video"]:
        for instance_id in manifest["instance_ids"]:
            for rollout_id in manifest["rollout_ids"]:
                name = f"{task['task']}_{instance_id}_{rollout_id}.mp4"
                _validate_video(video_dir / name, steps_by_tuple[(instance_id, rollout_id)], ffprobe)

    return {
        "task_id": task_id,
        "task": task["task"],
        "result_count": len(expected_json_names),
        "video_count": len(expected_video_names),
    }


def validate_run(
    manifest: dict[str, Any],
    run_output_root: Path,
    *,
    completion_output: Path,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    if not run_output_root.is_dir():
        _fail(f"run output root is missing: {run_output_root}")
    resolved_root = run_output_root.resolve()
    declared_root = Path(manifest["run_output_root"]).resolve()
    if declared_root != resolved_root:
        _fail(
            f"run output root does not match manifest: expected {declared_root}, got {resolved_root}"
        )
    expected_completion = resolved_root / "run_complete.json"
    if completion_output.resolve() != expected_completion:
        _fail(
            f"completion certificate must be {expected_completion}, got {completion_output.resolve()}"
        )
    manifest_path = Path(manifest.get("_manifest_path", manifest["manifest_path"])).resolve()
    if manifest_path == expected_completion:
        _fail("manifest and completion certificate paths must be different")
    # Never leave a stale success marker behind after a later failed validation. This path is constrained above.
    completion_output.unlink(missing_ok=True)
    expected_dirs = {task["output_dir"] for task in manifest["tasks"]}
    actual_dirs = {
        entry.name
        for entry in run_output_root.iterdir()
        if entry.name.startswith("task-")
    }
    _validate_exact_names(actual_dirs, expected_dirs, f"final task directories under {run_output_root}")

    summaries = []
    tuples: set[tuple[int, int, int]] = set()
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        summaries.append(
            validate_task(manifest, task_id, run_output_root / task["output_dir"], ffprobe=ffprobe)
        )
        for instance_id in manifest["instance_ids"]:
            for rollout_id in manifest["rollout_ids"]:
                value = (task_id, instance_id, rollout_id)
                if value in tuples:
                    _fail(f"duplicate result tuple: {value}")
                tuples.add(value)

    result_count = sum(summary["result_count"] for summary in summaries)
    video_count = sum(summary["video_count"] for summary in summaries)
    _require_exact_int(result_count, manifest["expected_result_count"], "global result count")
    _require_exact_int(video_count, manifest["expected_video_count"], "global video count")
    _require_exact_int(len(tuples), manifest["expected_result_count"], "global unique result tuple count")
    manifest_sha256 = manifest.get("_manifest_sha256")
    if isinstance(manifest_sha256, str) and "_manifest_path" in manifest:
        try:
            current_manifest_sha256 = hashlib.sha256(Path(manifest["_manifest_path"]).read_bytes()).hexdigest()
        except OSError as exc:
            _fail(f"cannot re-read manifest before certification: {exc}")
        if current_manifest_sha256 != manifest_sha256:
            _fail("manifest changed while run artifacts were being validated")
    if not isinstance(manifest_sha256, str):
        canonical_manifest = {
            key: value for key, value in manifest.items() if not key.startswith("_")
        }
        encoded_manifest = json.dumps(
            canonical_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        manifest_sha256 = hashlib.sha256(encoded_manifest).hexdigest()
    completion = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "status": "complete",
        "run_output_root": str(resolved_root),
        "task_count": len(summaries),
        "task_ids": [task["task_id"] for task in manifest["tasks"]],
        "instance_ids": manifest["instance_ids"],
        "result_count": result_count,
        "video_count": video_count,
        "write_video": manifest["write_video"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }
    _atomic_json_dump(completion_output, completion)
    return completion


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--queue-file", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--run-output-root", type=Path, required=True)
    create.add_argument("--mode", default="public_test")
    create.add_argument("--instance-indices", type=int, nargs="+", required=True)
    create.add_argument("--num-rollouts", type=int, default=1)
    create.add_argument("--num-vector-envs", type=int, required=True)
    video_group = create.add_mutually_exclusive_group(required=True)
    video_group.add_argument("--write-video", dest="write_video", action="store_true")
    video_group.add_argument("--no-write-video", dest="write_video", action="store_false")

    task = subparsers.add_parser("validate-task")
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--task-id", type=int, required=True)
    task.add_argument("--task-dir", type=Path, required=True)
    task.add_argument("--ffprobe", default="ffprobe")

    run = subparsers.add_parser("validate-run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--run-output-root", type=Path, required=True)
    run.add_argument("--completion-output", type=Path)
    run.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create-manifest":
            summary = create_manifest(args)
            print(
                f"Created manifest: tasks={summary['task_count']} "
                f"results={summary['expected_result_count']} videos={summary['expected_video_count']}"
            )
        elif args.command == "validate-task":
            manifest = load_manifest(args.manifest)
            summary = validate_task(manifest, args.task_id, args.task_dir, ffprobe=args.ffprobe)
            print(
                f"Validated task {summary['task_id']}:{summary['task']}: "
                f"results={summary['result_count']} videos={summary['video_count']}"
            )
        else:
            manifest = load_manifest(args.manifest)
            completion_output = args.completion_output or args.run_output_root / "run_complete.json"
            summary = validate_run(
                manifest,
                args.run_output_root,
                completion_output=completion_output,
                ffprobe=args.ffprobe,
            )
            print(
                f"Validated complete run: tasks={summary['task_count']} "
                f"results={summary['result_count']} videos={summary['video_count']}"
            )
        return 0
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
