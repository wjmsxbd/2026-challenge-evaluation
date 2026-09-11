"""Stream successful RFT episodes to Comet-style NPZ and three camera videos."""

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import av
import numpy as np

CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
FORMAT_VERSION = "behavior_rft_raw_v1"


def numpy_copy(value, dtype=None) -> np.ndarray:
    """Detach a value from simulator-owned CPU/GPU buffers."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def write_json(path: Path, data: dict) -> None:
    """Publish JSON after flushing its temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2, allow_nan=False, default=float)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def collection_directory(path: Path, settings: dict, *, resume: bool):
    """Lock one task's output and reject accidental mixing of collection configurations."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another collector is writing {path}; use a separate output directory") from exc
        manifest_path = path / "collection.json"
        manifest = {"format": FORMAT_VERSION, "settings": settings}
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Collection exists at {path}; pass --resume to continue it")
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Collection settings changed at {path}; use a new output directory")
        else:
            if any(item.name != ".collection.lock" for item in path.iterdir()):
                raise FileExistsError(f"Output directory has no collection manifest and is not empty: {path}")
            write_json(manifest_path, manifest)
        yield path


def episode_path(task_dir: Path, instance_id: int, sample_id: int) -> Path:
    """Return a stable sample path, independent of batching and worker assignment."""
    return task_dir / "rollouts" / f"instance-{instance_id:04d}" / f"sample-{sample_id:06d}"


def completed_episode(path: Path, *, expected_identity: dict | None = None) -> dict | None:
    """Read a committed sample, checking that successful payloads are still present."""
    if not path.exists():
        return None
    metadata = json.loads((path / "episode.json").read_text())
    if metadata.get("format") != FORMAT_VERSION or metadata.get("complete") is not True:
        raise ValueError(f"Incomplete or unrecognized RFT episode: {path}")
    if type(metadata.get("success")) is not bool:
        raise ValueError(f"Invalid success label: {path}")
    if expected_identity is not None and any(metadata.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"Episode identity does not match the requested sample: {path}")
    if metadata["success"]:
        for filename in ("state_action.npz", *(f"{role}.mp4" for role in CAMERA_ROLES)):
            payload = path / filename
            if not payload.is_file() or payload.stat().st_size == 0:
                raise ValueError(f"Missing successful trajectory payload: {payload}")
        with np.load(path / "state_action.npz", allow_pickle=True) as archive:
            data = archive["arr_0"].item()
            frames = metadata["num_frames"]
            if data["state"].shape != (frames, 61) or data["action"].shape != (frames, 23):
                raise ValueError(f"State/action shape does not match episode metadata: {path}")
        for role in CAMERA_ROLES:
            with av.open(str(path / f"{role}.mp4")) as video:
                stream = video.streams.video[0]
                if stream.frames != frames or stream.average_rate != metadata["fps"]:
                    raise ValueError(f"Video frame count/rate does not match episode metadata: {path / role}")
    return metadata


class RFTRecorder:
    """Record (observation_t, executed_action_t) without buffering whole RGB trajectories.

    ``set_observation`` snapshots one observation. ``before_step`` writes that snapshot
    with the action about to be executed. ``after_step`` records the outcome. Only
    ``finish`` publishes the directory, after all video encoders have been flushed.
    """

    def __init__(self, path: Path, metadata: dict, *, fps: int):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite RFT episode: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        self.metadata = dict(metadata)
        self.fps = fps
        self._writers = {}
        self._camera_shapes = {}
        self._observation = None
        self._states = []
        self._actions = []
        self._base_qpos = []
        self._base_qvel = []
        self._rewards = []
        self._terminated = []
        self._truncated = []
        self._closed = False

    def set_observation(self, state, images: dict, *, base_qpos, base_qvel) -> None:
        """Snapshot native 61-D state, three RGB cameras, and canonical base joint coordinates."""
        state = numpy_copy(state, np.float32)
        base_qpos = numpy_copy(base_qpos, np.float32)
        base_qvel = numpy_copy(base_qvel, np.float32)
        if state.shape != (61,) or base_qpos.shape != (3,) or base_qvel.shape != (3,):
            raise ValueError("Expected state (61,), base_qpos (3,), and base_qvel (3,)")
        if not all(np.isfinite(value).all() for value in (state, base_qpos, base_qvel)):
            raise ValueError("Cannot record non-finite robot state")
        frames = {}
        for role in CAMERA_ROLES:
            frame = numpy_copy(images[role])
            if frame.ndim != 3 or frame.shape[-1] not in (3, 4) or frame.dtype != np.uint8:
                raise ValueError(f"Expected HWC uint8 RGB/RGBA for {role}, got {frame.shape}, {frame.dtype}")
            if frame.shape[0] % 2 or frame.shape[1] % 2:
                raise ValueError(f"Video dimensions must be even: {role} {frame.shape}")
            frame = np.ascontiguousarray(frame[..., :3])
            if role in self._camera_shapes and self._camera_shapes[role] != list(frame.shape):
                raise ValueError(f"Camera shape changed within episode: {role}")
            self._camera_shapes[role] = list(frame.shape)
            frames[role] = frame
        self._observation = (state, frames, base_qpos, base_qvel)

    def before_step(self, action) -> None:
        """Write the observation preceding this action before the simulator can mutate it."""
        if self._closed or self._observation is None or len(self._actions) != len(self._rewards):
            raise RuntimeError("Recorder needs an observation and a completed previous transition")
        action = numpy_copy(action, np.float32)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite, executed 23-D action")
        state, images, base_qpos, base_qvel = self._observation
        for role, image in images.items():
            if role not in self._writers:
                container = av.open(str(self.staging / f"{role}.mp4"), mode="w")
                # Register before configuring the encoder so abort() also closes failed encoders.
                self._writers[role] = (container, None)
                stream = container.add_stream("libx264", rate=self.fps)
                self._writers[role] = (container, stream)
                stream.height, stream.width = image.shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
            container, stream = self._writers[role]
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        self._states.append(state)
        self._actions.append(action)
        self._base_qpos.append(base_qpos)
        self._base_qvel.append(base_qvel)
        self._observation = None

    def after_step(self, reward, terminated, truncated) -> None:
        """Complete exactly one action after a successful simulator step."""
        if len(self._actions) != len(self._rewards) + 1 or not np.isfinite(float(reward)):
            raise RuntimeError("Unpaired action or non-finite reward in RFT recording")
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))

    def _close_writers(self) -> None:
        writers, self._writers = self._writers, {}
        errors = []
        for container, stream in writers.values():
            try:
                if stream is not None:
                    for packet in stream.encode():
                        container.mux(packet)
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    container.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError("Failed to finalize RFT videos") from errors[0]

    def finish(self, result: dict) -> dict:
        """Commit a complete successful payload, or just metadata for a failed rollout."""
        frames = len(self._actions)
        if self._closed or frames == 0 or len(self._rewards) != frames:
            raise RuntimeError("Cannot finalize an empty, closed, or incomplete trajectory")
        if frames != int(result["steps"]):
            raise ValueError("Recorded frames do not match the number of executed environment steps")
        if not (self._terminated[-1] or self._truncated[-1]):
            raise RuntimeError("Only terminal episodes can be committed")
        self._close_writers()
        success = bool(result["success"])
        if success:
            data = {
                "state": np.stack(self._states),
                "action": np.stack(self._actions),
                "timestamp": np.arange(frames, dtype=np.float64) / self.fps,
                "base_qpos": np.stack(self._base_qpos),
                "base_qvel": np.stack(self._base_qvel),
                "reward": np.asarray(self._rewards, dtype=np.float32),
                "terminated": np.asarray(self._terminated, dtype=bool),
                "truncated": np.asarray(self._truncated, dtype=bool),
            }
            # Keep Comet's arr_0.item()["state" / "action"] convention.
            np.savez_compressed(self.staging / "state_action.npz", data)
        else:
            for role in CAMERA_ROLES:
                (self.staging / f"{role}.mp4").unlink(missing_ok=True)
        metadata = {
            **self.metadata,
            "format": FORMAT_VERSION,
            "complete": True,
            "success": success,
            "num_frames": frames,
            "fps": self.fps,
            "camera_shapes": self._camera_shapes,
            "state_schema": "r1pro_v3_61",
            "state_base_velocity_frame": "robot_local",
            "base_qpos_layout": ["x", "y", "yaw"],
            "base_qvel_frame": "articulation_canonical",
            "action_dim": 23,
            "action_base_velocity_frame": "robot_local",
            "alignment": "observation_t, executed_action_t, reward_t_plus_1",
            "result": result,
        }
        write_json(self.staging / "episode.json", metadata)
        if self.path.exists():
            raise FileExistsError(f"RFT episode was concurrently created: {self.path}")
        self.staging.rename(self.path)
        self._closed = True
        self._observation = None
        return metadata

    def abort(self) -> None:
        """Close encoders and discard only this recorder's unpublished temporary files."""
        if self._closed:
            return
        try:
            self._close_writers()
        finally:
            shutil.rmtree(self.staging)
            self._closed = True
            self._observation = None
