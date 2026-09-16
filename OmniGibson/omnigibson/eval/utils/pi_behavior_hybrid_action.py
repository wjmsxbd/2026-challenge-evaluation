"""Pure NumPy/SciPy adapter for PiBehavior's hybrid R1Pro control path."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


PI_BEHAVIOR_JOINT_ACTION_DIM = 23
PI_BEHAVIOR_EEF_ACTION_DIM = 20
R1PRO_HYBRID_ACTION_DIM = 21
R1PRO_PROPRIO_DIM = 61

_LEFT_EEF_PROPRIO_SLICE = np.s_[17:24]
_RIGHT_EEF_PROPRIO_SLICE = np.s_[42:49]
_ROT6D_EPS = 1e-8


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Project first-two-column rotation representations onto SO(3)."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rot6d last dimension 6, got {rot6d.shape}")
    if not np.isfinite(rot6d).all():
        raise ValueError("EEF rotation6d values must be finite")

    first = rot6d[..., :3]
    second = rot6d[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= _ROT6D_EPS):
        raise ValueError("EEF rotation6d first axis must be non-zero")
    first = first / first_norm

    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm <= _ROT6D_EPS):
        raise ValueError("EEF rotation6d axes must not be collinear")
    second = second / second_norm
    third = np.cross(first, second, axis=-1)
    return np.stack((first, second, third), axis=-1)


def _anchor_eef_poses(raw_proprio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_proprio = np.asarray(raw_proprio, dtype=np.float64)
    if raw_proprio.shape != (R1PRO_PROPRIO_DIM,):
        raise ValueError(f"Expected raw R1Pro proprio shape ({R1PRO_PROPRIO_DIM},), got {raw_proprio.shape}")
    if not np.isfinite(raw_proprio).all():
        raise ValueError("Raw R1Pro proprioception must contain only finite values")

    blocks = np.stack(
        (raw_proprio[_LEFT_EEF_PROPRIO_SLICE], raw_proprio[_RIGHT_EEF_PROPRIO_SLICE]),
        axis=0,
    )
    quaternion_norms = np.linalg.norm(blocks[:, 3:7], axis=-1, keepdims=True)
    if np.any(quaternion_norms <= _ROT6D_EPS):
        raise ValueError("Raw R1Pro EEF quaternions must be non-zero")
    rotations = Rotation.from_quat(blocks[:, 3:7] / quaternion_norms).as_matrix()
    return blocks[:, :3], rotations


def compose_pi_behavior_hybrid_action_chunk(
    joint_action_chunk: np.ndarray,
    eef_action_chunk: np.ndarray,
    raw_proprio: np.ndarray,
) -> np.ndarray:
    """Compose a 21D OmniGibson chunk from PiBehavior's joint and EEF heads.

    ``joint_action_chunk`` contains the server-restored 23D controls. Only its
    base velocity and absolute torso targets (the first seven values) are used.
    Each 10D block in ``eef_action_chunk`` contains anchor-relative xyz,
    anchor-relative rotation6d, and an absolute gripper command. The anchor is
    the raw 61D R1Pro proprioception used for the same policy request.

    The returned OmniGibson action order is base(3), torso(4), left absolute IK
    pose(6), left gripper(1), right absolute IK pose(6), right gripper(1).
    Absolute IK orientation is encoded as an axis-angle rotation vector.
    """
    joint_actions = np.asarray(joint_action_chunk)
    eef_actions = np.asarray(eef_action_chunk)
    if joint_actions.ndim != 2 or joint_actions.shape[1] != PI_BEHAVIOR_JOINT_ACTION_DIM:
        raise ValueError(
            f"Expected joint action chunk [H,{PI_BEHAVIOR_JOINT_ACTION_DIM}], got {joint_actions.shape}"
        )
    if eef_actions.ndim != 2 or eef_actions.shape[1] != PI_BEHAVIOR_EEF_ACTION_DIM:
        raise ValueError(f"Expected EEF action chunk [H,{PI_BEHAVIOR_EEF_ACTION_DIM}], got {eef_actions.shape}")
    if joint_actions.shape[0] != eef_actions.shape[0]:
        raise ValueError(
            "Joint and EEF chunks must have the same horizon, got "
            f"{joint_actions.shape[0]} and {eef_actions.shape[0]}"
        )
    if not np.isfinite(joint_actions).all() or not np.isfinite(eef_actions).all():
        raise ValueError("Joint and EEF action chunks must contain only finite values")

    anchor_positions, anchor_rotations = _anchor_eef_poses(raw_proprio)
    horizon = joint_actions.shape[0]
    hybrid = np.empty((horizon, R1PRO_HYBRID_ACTION_DIM), dtype=np.float32)
    hybrid[:, :7] = joint_actions[:, :7]

    for arm_index, (eef_start, output_start) in enumerate(((0, 7), (10, 14))):
        block = np.asarray(eef_actions[:, eef_start : eef_start + 10], dtype=np.float64)
        target_positions = anchor_positions[arm_index] + block[:, :3]
        delta_rotations = _rot6d_to_matrix(block[:, 3:9])
        # Training applies the same fixed local-axis adjustment to state and
        # target before forming R_target @ R_anchor.T, so it cancels here.
        target_rotations = delta_rotations @ anchor_rotations[arm_index]
        target_rotvecs = Rotation.from_matrix(target_rotations).as_rotvec()

        hybrid[:, output_start : output_start + 3] = target_positions
        hybrid[:, output_start + 3 : output_start + 6] = target_rotvecs
        hybrid[:, output_start + 6] = block[:, 9]

    if not np.isfinite(hybrid).all():
        raise ValueError("Composed hybrid action chunk contains non-finite values")
    return hybrid


def validate_pi_behavior_hybrid_config(
    action_dim: int,
    *,
    proprioception_schema: str,
    apply_eval_tricks: bool,
    enable_action_chunk_maintenance: bool,
    enable_compression: bool,
) -> bool:
    """Validate evaluator options and report whether hybrid EEF control is active."""
    if action_dim not in {PI_BEHAVIOR_JOINT_ACTION_DIM, R1PRO_HYBRID_ACTION_DIM}:
        raise ValueError(
            "PIBehavior evaluation requires the standard 23D or hybrid EEF 21D robot action space, "
            f"got {action_dim}"
        )
    if action_dim == PI_BEHAVIOR_JOINT_ACTION_DIM:
        return False

    if proprioception_schema != "r1pro_v3_61":
        raise ValueError("Hybrid EEF control requires proprioception_schema='r1pro_v3_61' for its pose anchor")
    incompatible = [
        name
        for name, enabled in (
            ("apply_eval_tricks", apply_eval_tricks),
            ("enable_action_chunk_maintenance", enable_action_chunk_maintenance),
            ("enable_compression", enable_compression),
        )
        if enabled
    ]
    if incompatible:
        raise ValueError(
            "Hybrid EEF control requires correction tricks, action-chunk maintenance, and compression to be "
            f"disabled; enabled options: {', '.join(incompatible)}"
        )
    return True


__all__ = [
    "PI_BEHAVIOR_EEF_ACTION_DIM",
    "PI_BEHAVIOR_JOINT_ACTION_DIM",
    "R1PRO_HYBRID_ACTION_DIM",
    "R1PRO_PROPRIO_DIM",
    "compose_pi_behavior_hybrid_action_chunk",
    "validate_pi_behavior_hybrid_config",
]
