"""CPU affinity and thread-pool configuration for evaluation processes."""

import os
import re
from dataclasses import dataclass
from typing import Mapping


CPU_AFFINITY_ENV = "EVAL_CPU_AFFINITY"
CPU_CORES_PER_ENV_ENV = "EVAL_CPU_CORES_PER_ENV"
CPU_WORKER_INDEX_ENV = "EVAL_CPU_WORKER_INDEX"
CPU_NUM_THREADS_ENV = "EVAL_CPU_NUM_THREADS"

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class CPUConfig:
    """Resolved CPU resources for one evaluator process."""

    affinity: tuple[int, ...] | None
    num_threads: int | None
    worker_index: int | None


def parse_cpu_affinity(value: str) -> tuple[int, ...]:
    """Parse a Linux CPU list such as ``0-3,8,10-11``."""
    value = value.strip()
    if not value:
        raise ValueError("CPU affinity cannot be empty.")

    cpus = set()
    for item in value.split(","):
        item = item.strip()
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError(f"Invalid CPU affinity item {item!r} in {value!r}.")
        if "-" not in item:
            cpus.add(int(item))
            continue

        start_text, end_text = item.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if start > end:
            raise ValueError(f"Invalid descending CPU range {item!r}.")
        cpus.update(range(start, end + 1))

    return tuple(sorted(cpus))


def format_cpu_affinity(cpus: tuple[int, ...] | list[int] | set[int]) -> str:
    """Format CPU ids as a compact Linux CPU list."""
    ordered = sorted(set(cpus))
    if not ordered:
        return ""

    ranges = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _resolve_int(
    cli_value: int | None,
    env_name: str,
    environ: Mapping[str, str],
    *,
    minimum: int,
) -> int | None:
    raw_value = cli_value if cli_value is not None else environ.get(env_name)
    if raw_value is None or raw_value == "":
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw_value!r}.") from exc
    if value < minimum:
        raise ValueError(f"{env_name} must be >= {minimum}, got {value}.")
    return value


def resolve_cpu_config(
    *,
    cpu_affinity: str | None = None,
    cpu_cores_per_env: int | None = None,
    cpu_worker_index: int | None = None,
    cpu_num_threads: int | None = None,
    environ: Mapping[str, str] | None = None,
    allowed_cpus: set[int] | None = None,
) -> CPUConfig:
    """Resolve CLI/environment CPU settings against the process cpuset."""
    environ = os.environ if environ is None else environ
    cli_auto_assignment = cpu_cores_per_env is not None or cpu_worker_index is not None
    if cpu_affinity is not None:
        affinity_text = cpu_affinity
        auto_environ = {}
    elif cli_auto_assignment:
        # Selecting auto assignment on the CLI overrides a stale explicit-affinity environment setting.
        affinity_text = None
        auto_environ = environ
    else:
        affinity_text = environ.get(CPU_AFFINITY_ENV) or None
        auto_environ = environ
    cores_per_env = _resolve_int(
        cpu_cores_per_env,
        CPU_CORES_PER_ENV_ENV,
        auto_environ,
        minimum=1,
    )
    worker_index = _resolve_int(
        cpu_worker_index,
        CPU_WORKER_INDEX_ENV,
        auto_environ,
        minimum=0,
    )
    num_threads = _resolve_int(
        cpu_num_threads,
        CPU_NUM_THREADS_ENV,
        environ,
        minimum=1,
    )

    if affinity_text is not None and (cores_per_env is not None or worker_index is not None):
        raise ValueError(
            f"{CPU_AFFINITY_ENV}/--cpu-affinity cannot be combined with "
            "--cpu-cores-per-env or --cpu-worker-index."
        )
    if worker_index is not None and cores_per_env is None:
        raise ValueError("--cpu-worker-index requires --cpu-cores-per-env.")

    if allowed_cpus is None:
        if not hasattr(os, "sched_getaffinity"):
            raise ValueError("CPU affinity configuration requires Linux os.sched_getaffinity().")
        try:
            allowed_cpus = set(os.sched_getaffinity(0))
        except OSError as exc:
            raise ValueError(f"Failed to read the current process CPU affinity: {exc}") from exc
    else:
        allowed_cpus = set(allowed_cpus)
    if not allowed_cpus:
        raise ValueError("The current process has no allowed CPUs.")

    affinity = None
    if affinity_text is not None:
        affinity = parse_cpu_affinity(affinity_text)
        unavailable = set(affinity) - allowed_cpus
        if unavailable:
            raise ValueError(
                "Requested CPUs are outside the current process cpuset: "
                f"{format_cpu_affinity(unavailable)}; allowed: {format_cpu_affinity(allowed_cpus)}."
            )
    elif cores_per_env is not None:
        worker_index = 0 if worker_index is None else worker_index
        ordered_cpus = sorted(allowed_cpus)
        start = worker_index * cores_per_env
        end = start + cores_per_env
        if end > len(ordered_cpus):
            raise ValueError(
                f"Worker {worker_index} needs CPU slots {start}-{end - 1}, but only "
                f"{len(ordered_cpus)} CPUs are allowed ({format_cpu_affinity(allowed_cpus)})."
            )
        affinity = tuple(ordered_cpus[start:end])

    if num_threads is None and affinity is not None:
        num_threads = len(affinity)

    return CPUConfig(affinity=affinity, num_threads=num_threads, worker_index=worker_index)


def apply_cpu_config(config: CPUConfig) -> None:
    """Apply affinity and native-library thread limits to the current process."""
    # SPEEDUP_EVAL: isolate each evaluator/server worker's CPU set and cap native
    # thread pools to prevent multi-GPU runs from oversubscribing the host CPU.
    if config.num_threads is not None:
        thread_count = str(config.num_threads)
        for env_name in THREAD_ENV_VARS:
            os.environ[env_name] = thread_count

    if config.affinity is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU affinity configuration requires Linux os.sched_setaffinity().")
        try:
            os.sched_setaffinity(0, set(config.affinity))
        except OSError as exc:
            raise RuntimeError(f"Failed to apply CPU affinity {format_cpu_affinity(config.affinity)}: {exc}") from exc


def configure_runtime_thread_pools(num_threads: int | None) -> None:
    """Apply thread limits for libraries that expose runtime configuration APIs."""
    if num_threads is None:
        return

    import cv2
    import torch

    # SPEEDUP_EVAL: OpenCV and Torch can otherwise create independent pools in
    # every vector worker, causing simulator/policy scheduling jitter.
    cv2.setNumThreads(num_threads)
    torch.set_num_threads(num_threads)
