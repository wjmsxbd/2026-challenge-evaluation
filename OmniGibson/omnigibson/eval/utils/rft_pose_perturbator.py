"""Deterministic planar start-pose perturbations for RFT collection."""

import numpy as np
from scipy.spatial.transform import Rotation


def sample_seed(seed: int, task_id: int, instance_id: int, sample_id: int) -> int:
    """Derive an environment seed independently of worker, slot, and scheduling order."""
    values = [seed, task_id, instance_id, sample_id]
    if any(value < 0 for value in values):
        raise ValueError("Seed, task, instance and sample IDs must be non-negative")
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


def perturb_pose(position, orientation, *, seed: int, translation: float, yaw_degrees: float) -> tuple:
    """Apply local XY translation and yaw to an XYZW robot pose, leaving the source unchanged.

    The returned metadata contains the sampled local offsets and both world poses.
    Call on the freshly loaded instance pose on every attempt, before the scene is snapshotted.
    """
    if not np.isfinite([translation, yaw_degrees]).all() or min(translation, yaw_degrees) < 0:
        raise ValueError("Pose perturbation bounds must be finite and non-negative")
    position = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    if position.shape != (3,) or orientation.shape != (4,):
        raise ValueError("Expected position (3,) and XYZW orientation (4,)")
    if not np.isfinite(position).all() or not np.isfinite(orientation).all():
        raise ValueError("Robot pose must be finite")
    rotation = Rotation.from_quat(orientation)
    rng = np.random.default_rng(seed)
    dx, dy = rng.uniform(-translation, translation, size=2)
    yaw = float(rng.uniform(-yaw_degrees, yaw_degrees))
    new_position = position + rotation.apply([dx, dy, 0.0])
    new_orientation = (rotation * Rotation.from_euler("z", yaw, degrees=True)).as_quat()
    metadata = {
        "translation_frame": "robot_local",
        "translation_xy": [float(dx), float(dy)],
        "yaw_degrees": yaw,
        "original_position": position.tolist(),
        "original_orientation_xyzw": orientation.tolist(),
        "position": new_position.tolist(),
        "orientation_xyzw": new_orientation.tolist(),
    }
    return new_position.tolist(), new_orientation.tolist(), metadata
