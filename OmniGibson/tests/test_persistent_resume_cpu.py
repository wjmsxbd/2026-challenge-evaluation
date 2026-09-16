"""Recover explicitly selected evaluations at episode boundaries without Isaac."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import shutil
import time

import pytest


UTILS = Path(__file__).resolve().parents[1] / "omnigibson/eval/utils"


@pytest.fixture
def resume(monkeypatch):
    monkeypatch.syspath_prepend(str(UTILS))
    return importlib.import_module("pi05_persistent_resume")


def _run(tmp_path, name, *, seed="0", write_video=False):
    from pi05_output_validator import create_manifest

    root = tmp_path / "outputs" / name
    logs = tmp_path / "logs" / f"pi05_behavior_2026_persistent_{name}"
    root.mkdir(parents=True)
    (logs / "task_queues").mkdir(parents=True)
    (logs / "coord").mkdir()
    config = {
        "task_ids": "0",
        "task_limit": "1",
        "instance_indices": "0 1 2 3",
        "seed": seed,
        "write_video": str(write_video).lower(),
        "dry_run": "false",
        "world_size": "2",
        "num_gpus": "1",
        "run_output_root": str(root),
    }
    (root / "run_config.json").write_text(json.dumps(config))
    (logs / "coord/run_config.json").write_text(json.dumps(config))
    queue = logs / "task_queue.tsv"
    queue.write_text("0\ttask_a\n")
    create_manifest(
        argparse.Namespace(
            queue_file=queue,
            output=root / "run_manifest.json",
            run_output_root=root,
            mode="public_test",
            instance_indices=[0, 1, 2, 3],
            num_rollouts=1,
            num_vector_envs=2,
            write_video=write_video,
        )
    )
    (logs / "task_schedule.tsv").write_text(
        "order\tworker\ttask_id\ttask_name\tinstance_indices\ttimeout_steps\tchunk_steps\testimated_steps\tspeed_source\n"
        "1\tpending\t0\ttask_a\t0,1\t100\t200\t200\tstep_lpt_online\n"
        "2\tpending\t0\ttask_a\t2,3\t100\t200\t200\tstep_lpt_online\n"
    )
    (logs / "task_queues/online.tsv").write_text("0\ttask_a\t0,1\t100\t200\t1\n0\ttask_a\t2,3\t100\t200\t2\n")
    return root, logs


def _session(root, logs, name="20260909_110000", **overrides):
    session = root / ".sessions" / name
    session_logs = logs / ".sessions" / name
    session.mkdir(parents=True)
    (session_logs / "task_queues").mkdir(parents=True)
    config = {**json.loads((root / "run_config.json").read_text()), **overrides}
    (session / "run_config.json").write_text(json.dumps(config))
    shutil.copy2(logs / "task_schedule.tsv", session_logs / "task_schedule.tsv")
    shutil.copy2(logs / "task_queues/online.tsv", session_logs / "task_queues/online.tsv")
    return session, session_logs


def _result(directory, index):
    (directory / "json").mkdir(parents=True, exist_ok=True)
    path = directory / "json" / f"task_a_{301 + index}_0.json"
    path.write_text(
        json.dumps(
            {
                "task": "task_a",
                "instance_id": 301 + index,
                "rollout_id": 0,
                "steps": 10,
                "success": False,
                "agent_distance": {"base": 0, "left": 0, "right": 0},
                "normalized_agent_distance": {"base": 0, "left": 0, "right": 0},
                "q_score": {"final": 0},
                "time": {"simulator_steps": 10, "simulator_time": 10 / 30, "normalized_time": 1},
            }
        )
    )
    return path


def test_partial_pair_keeps_finished_instance_and_reruns_only_unfinished(resume, tmp_path):
    root, logs = _run(tmp_path, "20260909_100000")
    complete = root / ".chunks/task-0_task_a/instances-0_1"
    _result(complete, 0)
    _result(complete, 1)
    attempt = root / ".attempts/task-0_task_a/instances-2_3/attempt-1"
    saved = _result(attempt, 2)
    saved_bytes = saved.read_bytes()
    (attempt / "json/task_a_304_0.json").write_text('{"task":')
    session, session_logs = _session(root, logs)
    report = resume.prepare_resume(root, session_logs, session, logs)
    assert report["source_run"] == str(root)
    assert report["reused_instances"] == 3
    assert report["pending_instances"] == report["pending_chunks"] == 1
    assert (session_logs / "task_queues/online.tsv").read_text() == "0\ttask_a\t3\t100\t100\t1\n"
    assert len(list((session / ".chunks").glob("*/instances-*/json/*.json"))) == 3
    assert (session / ".chunks/task-0_task_a/instances-2/json/task_a_303_0.json").read_bytes() == saved_bytes
    assert saved.read_bytes() == saved_bytes
    assert not (attempt / "evaluation_complete.json").exists()


def test_no_directory_means_fresh_even_when_a_matching_unfinished_run_exists(resume, tmp_path):
    previous, _ = _run(tmp_path, "20260909_100000")
    _result(previous / ".chunks/task-0_task_a/instances-0_1", 0)
    current, logs = _run(tmp_path, "20260909_110000")
    report = resume.prepare_resume(current, logs, current)
    assert report["source_run"] is None
    assert report["reused_instances"] == 0
    assert report["pending_instances"] == 4


def test_explicit_resume_rejects_configuration_mismatch(resume, tmp_path):
    root, logs = _run(tmp_path, "20260909_100000")
    saved = _result(root / ".chunks/task-0_task_a/instances-0_1", 0)
    original = saved.read_bytes()
    session, session_logs = _session(root, logs, seed="99")
    with pytest.raises(ValueError, match="Resume configuration differs"):
        resume.prepare_resume(root, session_logs, session, logs)
    assert saved.read_bytes() == original
    assert not (session / ".chunks").exists()


def test_only_specified_directory_is_used_and_missing_directory_is_rejected(resume, tmp_path):
    root, logs = _run(tmp_path, "20260909_100000")
    other, _ = _run(tmp_path, "20260909_105000")
    _result(other / ".chunks/task-0_task_a/instances-0_1", 0)
    session, session_logs = _session(root, logs)
    assert resume.prepare_resume(root, session_logs, session, logs)["reused_instances"] == 0
    with pytest.raises(ValueError, match="existing evaluation log directory"):
        resume.source_configuration(tmp_path / "missing")


def test_repeated_resume_uses_progress_from_prior_sessions_and_all_done_leaves_empty_queue(resume, tmp_path):
    root, logs = _run(tmp_path, "20260909_100000")
    for index in range(3):
        _result(root / ".attempts/task-0_task_a/instances-0_1/attempt-1", index)
    session, session_logs = _session(root, logs)
    report = resume.prepare_resume(root, session_logs, session, logs)
    assert report["reused_instances"] == 3
    _result(session / ".attempts/task-0_task_a/instances-3/attempt-1", 3)
    # Reuse the ORIGINAL log directory, even after a resume was itself interrupted.
    latest, latest_logs = _session(root, logs, "20260909_120000")
    again = resume.prepare_resume(root, latest_logs, latest, logs)
    assert again["reused_instances"] == 4
    assert again["pending_instances"] == again["pending_chunks"] == 0
    assert (latest_logs / "task_queues/online.tsv").read_text() == ""


def test_legacy_log_configuration_and_completed_run_can_be_resumed_explicitly(resume, tmp_path):
    root, logs = _run(tmp_path, "old-dlc-job-id")
    _result(root / ".attempts/task-0_task_a/instances-0_1/attempt-1", 0)
    session, session_logs = _session(root, logs)
    (root / "run_config.json").unlink()
    (root / "run_complete.json").write_text("{}")
    report = resume.prepare_resume(root, session_logs, session, logs)
    assert report["reused_instances"] == 1
    assert not (root / "run_complete.json").exists()
    assert (session / "previous_run_complete.json").is_file()


def test_submission_requires_finalized_video_as_well_as_metrics(resume, tmp_path, monkeypatch):
    import pi05_output_validator as validator

    root, logs = _run(tmp_path, "20260909_100000", write_video=True)
    attempt = root / ".attempts/task-0_task_a/instances-0_1/attempt-1"
    for index in range(2):
        _result(attempt, index)
    (attempt / "videos").mkdir()
    (attempt / "videos/task_a_301_0.mp4").write_bytes(b"finished-video")
    checked = []

    def validate_video(path, expected_frames, ffprobe):
        checked.append((path.name, expected_frames))
        if not path.is_file():
            raise validator.ValidationError("unfinished video")

    monkeypatch.setattr(validator, "_validate_video", validate_video)
    session, session_logs = _session(root, logs)
    report = resume.prepare_resume(root, session_logs, session, logs)
    assert checked == [("task_a_301_0.mp4", 10), ("task_a_302_0.mp4", 10)]
    assert report["reused_instances"] == 1
    assert report["pending_instances"] == 3
    assert (session / ".chunks/task-0_task_a/instances-0/videos/task_a_301_0.mp4").read_bytes() == b"finished-video"


def test_restarting_same_dlc_job_shares_a_fresh_date_with_peer_first(resume, tmp_path):
    timestamps = []
    for generation in range(2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            ack = tmp_path / ".dlc_runs/same-job/rank-1.json"
            old_nonce = resume.read_json(ack).get("nonce")
            peer = pool.submit(resume.launch_timestamp, tmp_path, "same-job", 1, 2, 5, False)
            if generation:
                deadline = time.monotonic() + 3
                while resume.read_json(ack).get("nonce") == old_nonce and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert not peer.done(), "A fresh peer must not consume the previous launch result"
            leader = pool.submit(resume.launch_timestamp, tmp_path, "same-job", 0, 2, 5, False)
            timestamp = leader.result(timeout=6)
            assert peer.result(timeout=6) == timestamp
            time.strptime(timestamp, "%Y%m%d_%H%M%S")
            timestamps.append(timestamp)
    assert len(set(timestamps)) == 2


def test_peer_times_out_when_rank_zero_is_absent(resume, tmp_path):
    with pytest.raises(TimeoutError, match="coordinating"):
        resume.launch_timestamp(tmp_path, "absent-leader", 1, 2, 0.1, False)
