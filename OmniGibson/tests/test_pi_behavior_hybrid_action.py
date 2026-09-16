import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

MODULE_PATH = Path(__file__).parents[1] / "omnigibson" / "eval" / "utils" / "pi_behavior_hybrid_action.py"
SPEC = importlib.util.spec_from_file_location("pi_behavior_hybrid_action", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HYBRID_ACTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HYBRID_ACTION)

compose_pi_behavior_hybrid_action_chunk = HYBRID_ACTION.compose_pi_behavior_hybrid_action_chunk
validate_pi_behavior_hybrid_config = HYBRID_ACTION.validate_pi_behavior_hybrid_config


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def _raw_proprio(left_position, left_rotation, right_position, right_rotation) -> np.ndarray:
    proprio = np.zeros(61, dtype=np.float32)
    proprio[17:20] = left_position
    proprio[20:24] = Rotation.from_matrix(left_rotation).as_quat()
    proprio[42:45] = right_position
    proprio[45:49] = Rotation.from_matrix(right_rotation).as_quat()
    return proprio


def test_hybrid_action_uses_joint_base_torso_and_eef_absolute_poses():
    identity = np.eye(3)
    raw_proprio = _raw_proprio([0.4, -0.2, 1.0], identity, [-0.3, 0.5, 0.8], identity)
    joint_actions = np.arange(46, dtype=np.float32).reshape(2, 23) / 10
    eef_actions = np.zeros((2, 20), dtype=np.float32)
    eef_actions[:, 3:9] = _matrix_to_rot6d(identity)
    eef_actions[:, 13:19] = _matrix_to_rot6d(identity)
    eef_actions[:, 0:3] = [[0.1, 0.2, -0.3], [-0.2, 0.0, 0.4]]
    eef_actions[:, 10:13] = [[-0.1, 0.3, 0.2], [0.4, -0.5, 0.1]]
    eef_actions[:, 9] = [-1.0, 0.25]
    eef_actions[:, 19] = [0.5, 1.0]

    hybrid = compose_pi_behavior_hybrid_action_chunk(joint_actions, eef_actions, raw_proprio)

    assert hybrid.shape == (2, 21)
    assert hybrid.dtype == np.float32
    np.testing.assert_array_equal(hybrid[:, :7], joint_actions[:, :7])
    np.testing.assert_allclose(hybrid[:, 7:10], raw_proprio[17:20] + eef_actions[:, 0:3])
    np.testing.assert_allclose(hybrid[:, 10:13], 0.0, atol=1e-7)
    np.testing.assert_array_equal(hybrid[:, 13], eef_actions[:, 9])
    np.testing.assert_allclose(hybrid[:, 14:17], raw_proprio[42:45] + eef_actions[:, 10:13])
    np.testing.assert_allclose(hybrid[:, 17:20], 0.0, atol=1e-7)
    np.testing.assert_array_equal(hybrid[:, 20], eef_actions[:, 19])


def test_hybrid_action_premultiplies_anchor_orientation_by_predicted_delta():
    left_anchor = Rotation.from_euler("z", 70, degrees=True).as_matrix()
    right_anchor = Rotation.from_euler("y", -35, degrees=True).as_matrix()
    left_delta = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    right_delta = Rotation.from_euler("zy", [40, 15], degrees=True).as_matrix()
    raw_proprio = _raw_proprio([0, 0, 0], left_anchor, [0, 0, 0], right_anchor)
    eef_actions = np.zeros((1, 20), dtype=np.float32)
    eef_actions[0, 3:9] = _matrix_to_rot6d(left_delta)
    eef_actions[0, 13:19] = _matrix_to_rot6d(right_delta)

    hybrid = compose_pi_behavior_hybrid_action_chunk(
        np.zeros((1, 23), dtype=np.float32),
        eef_actions,
        raw_proprio,
    )

    actual_left = Rotation.from_rotvec(hybrid[0, 10:13]).as_matrix()
    actual_right = Rotation.from_rotvec(hybrid[0, 17:20]).as_matrix()
    np.testing.assert_allclose(actual_left, left_delta @ left_anchor, atol=1e-6)
    np.testing.assert_allclose(actual_right, right_delta @ right_anchor, atol=1e-6)


def test_hybrid_config_is_opt_in_and_rejects_joint_only_postprocessing():
    assert not validate_pi_behavior_hybrid_config(
        23,
        proprioception_schema="r1pro_v2_256",
        apply_eval_tricks=True,
        enable_action_chunk_maintenance=True,
        enable_compression=True,
    )
    assert validate_pi_behavior_hybrid_config(
        21,
        proprioception_schema="r1pro_v3_61",
        apply_eval_tricks=False,
        enable_action_chunk_maintenance=False,
        enable_compression=False,
    )

    with pytest.raises(ValueError, match="enabled options: apply_eval_tricks"):
        validate_pi_behavior_hybrid_config(
            21,
            proprioception_schema="r1pro_v3_61",
            apply_eval_tricks=True,
            enable_action_chunk_maintenance=False,
            enable_compression=False,
        )
    with pytest.raises(ValueError, match="r1pro_v3_61"):
        validate_pi_behavior_hybrid_config(
            21,
            proprioception_schema="r1pro_v2_256",
            apply_eval_tricks=False,
            enable_action_chunk_maintenance=False,
            enable_compression=False,
        )


def test_hybrid_action_rejects_degenerate_rotation6d():
    identity = np.eye(3)
    raw_proprio = _raw_proprio([0, 0, 0], identity, [0, 0, 0], identity)
    eef_actions = np.zeros((1, 20), dtype=np.float32)
    eef_actions[0, 3:9] = [1, 0, 0, 2, 0, 0]
    eef_actions[0, 13:19] = _matrix_to_rot6d(identity)

    with pytest.raises(ValueError, match="must not be collinear"):
        compose_pi_behavior_hybrid_action_chunk(
            np.zeros((1, 23), dtype=np.float32),
            eef_actions,
            raw_proprio,
        )


def test_hybrid_robot_config_uses_absolute_pose_ik_for_both_arms():
    path = Path(__file__).parents[1] / "omnigibson" / "eval" / "r1pro_hybrid_eef.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert len(config["reset_joint_pos"]) == 28
    for arm in ("arm_left", "arm_right"):
        controller = config["controller_config"][arm]
        assert controller["name"] == "InverseKinematicsController"
        assert controller["mode"] == "absolute_pose"
        assert controller["command_input_limits"] is None
        assert controller["command_output_limits"] is None
