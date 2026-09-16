"""Date-named launches and instance-level recovery, without importing Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from pi05_output_validator import ValidationError, _atomic_json_dump, load_manifest, validate_instance


# Older launchers did not record these options. Only their historical defaults
# can be recovered from a legacy run configuration.
LEGACY_DEFAULTS = {
    "actions_to_execute": "26",
    "actions_to_keep": "4",
    "execute_in_n_steps": "20",
    "history_len": "3",
    "votes_to_promote": "2",
    "num_steps": "20",
    "disable_fast_auxiliary": "true",
    "proprioception_schema": "r1pro_v3_61",
    "use_task_checkpoint_mapping": "false",
    "task_checkpoint_mapping": "",
    "skip_action_chunk_rendering": "false",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def reserve_timestamp(output_root: Path) -> str:
    output_root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            (output_root / timestamp).mkdir()
            return timestamp
        except FileExistsError:
            time.sleep(0.2)


def launch_timestamp(output_root: Path, key: str, rank: int, world_size: int, timeout: float, dry_run: bool) -> str:
    if world_size == 1 or dry_run:
        return reserve_timestamp(output_root)

    # Fresh challenges and per-process nonces prevent a restarted DLC job from
    # consuming the previous launch's paths, even when peers start before rank 0.
    rendezvous = output_root / ".dlc_runs" / key
    challenge_path = rendezvous / "challenge.json"
    result_path = rendezvous / "launch.json"
    nonce = uuid.uuid4().hex
    deadline = time.monotonic() + timeout
    if rank == 0:
        _atomic_json_dump(challenge_path, {"generation": nonce})
        while time.monotonic() < deadline:
            peers = {str(peer): read_json(rendezvous / f"rank-{peer}.json") for peer in range(1, world_size)}
            if all(peer.get("generation") == nonce and peer.get("nonce") for peer in peers.values()):
                timestamp = reserve_timestamp(output_root)
                _atomic_json_dump(
                    result_path,
                    {
                        "generation": nonce,
                        "timestamp": timestamp,
                        "peers": {peer: value["nonce"] for peer, value in peers.items()},
                    },
                )
                return timestamp
            time.sleep(0.1)
    else:
        last_generation = None
        while time.monotonic() < deadline:
            generation = read_json(challenge_path).get("generation")
            if generation and generation != last_generation:
                _atomic_json_dump(rendezvous / f"rank-{rank}.json", {"generation": generation, "nonce": nonce})
                last_generation = generation
            result = read_json(result_path)
            if result.get("generation") == generation and result.get("peers", {}).get(str(rank)) == nonce:
                return result["timestamp"]
            time.sleep(0.1)
    raise TimeoutError(f"Timed out coordinating the date-named launch for {key}: {rendezvous}")


def protocol(config: dict) -> dict:
    ignored = {"world_size", "num_gpus", "fail_fast", "run_output_root", "resume"}
    result = {key: value for key, value in config.items() if key not in ignored}
    for key, value in LEGACY_DEFAULTS.items():
        result.setdefault(key, value)
    task_ids = str(result.get("task_ids", "")).replace(",", " ").split()
    selected = [int(value) for value in task_ids] if task_ids else list(range(100))
    result["task_ids"] = sorted(selected[: int(result.pop("task_limit", 100))])
    result["instance_indices"] = sorted(int(value) for value in str(result.get("instance_indices", "")).split())
    for key in ("policy_dir", "policy_repo", "policy_server", "norm_stats", "task_checkpoint_mapping"):
        if result.get(key):
            result[key] = str(Path(result[key]).resolve())
    return result


def source_configuration(log_dir: Path) -> dict:
    config_path = log_dir / "coord/run_config.json"
    config = read_json(config_path)
    if not config.get("run_output_root"):
        raise ValueError(f"Resume requires an existing evaluation log directory with {config_path}")
    load_manifest(Path(config["run_output_root"]) / "run_manifest.json")
    return config


def instance_sources(source_root: Path, task_dir: str, session_root: Path) -> list[Path]:
    # Promoted outputs take priority. An interrupted pair may also have a fully
    # completed instance in its attempt directory without a chunk-level marker.
    sessions = sorted(
        (
            path
            for path in (source_root / ".sessions").glob("*")
            if not path.name.startswith(".") and path != session_root
        ),
        reverse=True,
    )
    sources = []
    for base in sessions + [source_root]:
        sources.extend(sorted((base / ".chunks" / task_dir).glob("instances-*")))
        sources.append(base / task_dir)
        sources.extend(sorted((base / ".attempts" / task_dir).glob("instances-*/attempt-*")))
    return sources


def copy_instance(source: Path, target: Path, task: dict, instance_id: int, manifest: dict) -> None:
    target.mkdir(parents=True)
    for directory, suffix in [("json", "json")] + ([("videos", "mp4")] if manifest["write_video"] else []):
        (target / directory).mkdir()
        name = f"{task['task']}_{instance_id}_0.{suffix}"
        # Independent copies also isolate the recovered run from later edits to
        # the original outputs. Only validated, finished instances get copied.
        shutil.copy2(source / directory / name, target / directory / name)
    _atomic_json_dump(
        target / "evaluation_complete.json",
        {
            "task": task["task"],
            "task_id": task["task_id"],
            "mode": manifest["mode"],
            "instance_ids": [instance_id],
            "num_rollouts": 1,
            "num_vector_envs": manifest["num_vector_envs"],
            "result_count": 1,
            "write_video": manifest["write_video"],
        },
    )


def prepare_resume(
    run_root: Path, log_root: Path, session_root: Path, resume_log_dir: Path | None = None, dry_run: bool = False
) -> dict:
    config = read_json(session_root / "run_config.json")
    if not config:
        raise ValueError(f"Missing run configuration: {run_root}")
    manifest = load_manifest(run_root / "run_manifest.json")
    source_root = None
    if resume_log_dir is not None:
        previous = source_configuration(resume_log_dir)
        if Path(previous["run_output_root"]).resolve() != run_root.resolve():
            raise ValueError("The specified log directory refers to a different output root")
        expected, requested = protocol(previous), protocol(config)
        differences = {
            key: (expected.get(key), requested.get(key))
            for key in expected.keys() | requested.keys()
            if expected.get(key) != requested.get(key)
        }
        # A dry run previews the requested protocol without changing results.
        differences.pop("dry_run", None)
        if differences:
            raise ValueError(f"Resume configuration differs (previous, requested): {differences}")
        if not dry_run:
            source_root = run_root
    recovered = set()
    records = []
    if source_root is not None:
        print(f"Resume source: {source_root}", flush=True)
        certificate = run_root / "run_complete.json"
        if certificate.exists():
            certificate.rename(session_root / "previous_run_complete.json")
        for task in manifest["tasks"]:
            sources = instance_sources(source_root, task["output_dir"], session_root)
            for instance_id in manifest["instance_ids"]:
                for source in sources:
                    if not (source / "json" / f"{task['task']}_{instance_id}_0.json").is_file():
                        continue
                    try:
                        validate_instance(source, task["task"], instance_id, write_video=manifest["write_video"])
                    except ValidationError as error:
                        print(
                            f"Resume rejected instance {task['task_id']}:{instance_id} at {source}: {error}", flush=True
                        )
                        continue
                    index = instance_id - 301
                    target = session_root / ".chunks" / task["output_dir"] / f"instances-{index}"
                    copy_instance(source, target, task, instance_id, manifest)
                    recovered.add((task["task_id"], index))
                    records.append({"task_id": task["task_id"], "instance_index": index, "source": str(source)})
                    break

    schedule_path = log_root / "task_schedule.tsv"
    shutil.copy2(schedule_path, log_root / "full_task_schedule.tsv")
    with schedule_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    pending_rows = []
    for row in rows:
        indices = [int(value) for value in row["instance_indices"].split(",")]
        pending = [index for index in indices if (int(row["task_id"]), index) not in recovered]
        if not pending:
            continue
        row["instance_indices"] = ",".join(map(str, pending))
        row["order"] = str(len(pending_rows) + 1)
        row["chunk_steps"] = row["estimated_steps"] = str(int(row["timeout_steps"]) * len(pending))
        pending_rows.append(row)
    with schedule_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pending_rows)
    with (log_root / "task_queues/online.tsv").open("w", encoding="utf-8") as file:
        for row in pending_rows:
            file.write(
                "\t".join(
                    row[key]
                    for key in ("task_id", "task_name", "instance_indices", "timeout_steps", "chunk_steps", "order")
                )
                + "\n"
            )
    report = {
        "source_run": str(source_root) if source_root else None,
        "resume_log_dir": str(resume_log_dir) if resume_log_dir else None,
        "reused_instances": len(recovered),
        "pending_instances": manifest["expected_result_count"] - len(recovered),
        "pending_chunks": len(pending_rows),
        "instances": records,
    }
    _atomic_json_dump(session_root / "resume.json", report)
    print(f"Resume: reused={report['reused_instances']} pending={report['pending_instances']} instances", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("timestamp")
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--key", required=True)
    launch.add_argument("--rank", type=int, required=True)
    launch.add_argument("--world-size", type=int, required=True)
    launch.add_argument("--timeout", type=float, required=True)
    launch.add_argument("--dry-run", action="store_true")
    source = commands.add_parser("source-root")
    source.add_argument("--log-dir", type=Path, required=True)
    resume = commands.add_parser("prepare")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument("--log-root", type=Path, required=True)
    resume.add_argument("--session-root", type=Path, required=True)
    resume.add_argument("--resume-log-dir", type=Path)
    resume.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "timestamp":
        print(launch_timestamp(args.output_root, args.key, args.rank, args.world_size, args.timeout, args.dry_run))
    elif args.command == "source-root":
        print(Path(source_configuration(args.log_dir)["run_output_root"]).resolve())
    else:
        prepare_resume(args.run_root, args.log_root, args.session_root, args.resume_log_dir, args.dry_run)


if __name__ == "__main__":
    main()
