import json

import pytest

from harness.mine_world import (
    ROVER_CAMERA_PATH,
    MineRoverExperiment,
    RoverDriveCommand,
    select_wheel_dofs,
    write_reactor_seed_manifest,
)


def test_select_wheel_dofs_uses_loaded_names_not_a_hardcoded_robot_layout():
    left, right = select_wheel_dofs(
        ["caster_joint", "rear_right_wheel_joint", "front_left_wheel_joint", "front_right_wheel_joint", "rear_left_wheel_joint"]
    )
    assert left == ["front_left_wheel_joint", "rear_left_wheel_joint"]
    assert right == ["front_right_wheel_joint", "rear_right_wheel_joint"]
    with pytest.raises(ValueError, match="left and right"):
        select_wheel_dofs(["front_wheel_joint"])


def test_rover_experiment_has_hard_bounds():
    assert MineRoverExperiment("run-1", 7, RoverDriveCommand()).camera_resolution == (180, 320)
    with pytest.raises(ValueError, match="linear_velocity"):
        RoverDriveCommand(linear_velocity_mps=1.0)
    with pytest.raises(ValueError, match="run_id"):
        MineRoverExperiment("../unsafe", 7, RoverDriveCommand())


def test_reactor_seed_manifest_references_a_real_isaac_image_and_derived_layer(tmp_path):
    image = tmp_path / "frames" / "rgb_000001.png"
    image.parent.mkdir()
    image.write_bytes(b"real-frame-integration-test")
    layer = tmp_path / "session.usda"
    layer.write_text("#usda 1.0\n", encoding="utf-8")
    experiment = MineRoverExperiment("run-1", 7, RoverDriveCommand())

    output = write_reactor_seed_manifest(
        tmp_path, experiment, seed_image=image, source_stage="assets/worlds/mine_v1/mine_world.usda",
        session_layer=layer, rover_pose_before=[0, 0, 0], rover_pose_after=[1, 0, 0],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["camera_prim"] == ROVER_CAMERA_PATH
    assert payload["seed_image_path"] == str(image)
    assert payload["source_authority"] == "physics_grounded_isaac_rgb"
    assert "not physical ground truth" in payload["usage"]
