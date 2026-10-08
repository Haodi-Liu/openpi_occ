import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.rlbench import enablement_sim_worker as worker


def test_relative_pose_uses_rotated_base_and_quaternion_sign():
    q = np.sqrt(0.5)
    base = np.array([[1, 2, 3, 0, 0, q, q], [1, 2, 3, 0, 0, -q, -q]])
    tip = np.array([[2, 2, 3, 0, 0, q, q], [2, 2, 3, 0, 0, q, q]])
    result = worker.relative_pose(base, tip)
    np.testing.assert_allclose(result[:, :3], [[0, -1, 0], [0, -1, 0]], atol=1e-12)
    np.testing.assert_allclose(np.abs(result[:, 6]), 1)
    with pytest.raises(ValueError, match="quaternion"):
        worker.relative_pose(np.zeros(7), tip[0])


@pytest.mark.parametrize("grasper", ["left", "right"])
def test_command_holds_targets_for_gt_span_and_only_grasper_attaches(grasper):
    events = []
    steps = 0

    def arm(side):
        return SimpleNamespace(set_joint_target_positions=lambda value: events.append((side, "target", value.copy())))

    def actuate(target, velocity, side):
        events.append((side, "drive", target, velocity))
        return steps >= 20  # Closing takes twenty physical advances, then reports done.

    def step():
        nonlocal steps
        events.append(("step",))
        steps += 1

    robot = SimpleNamespace(
        left_arm=arm("left"),
        right_arm=arm("right"),
        actutate_gripper=actuate,
        release_gripper=lambda side: events.append((side, "release")),
        grasp=lambda obj, side: events.append((side, "grasp", obj)),
    )
    scene = SimpleNamespace(robot=robot, step=step, task=SimpleNamespace(get_graspable_objects=lambda: ["target"]))
    command = np.r_[np.arange(7), 0, np.arange(7) + 10, 0].astype(float)
    control = worker.ControlState(previous={"left": 0.0, "right": 0.0})
    assert worker.execute_command(scene, command, control, physics_steps=np.int64(21), grasper=grasper) == 21
    assert steps == 21
    assert sum(e[1] == "drive" for e in events if len(e) > 1) == 42
    assert sum(e[1] == "target" for e in events if len(e) > 1) == 2
    assert [e for e in events if len(e) > 1 and e[1] == "grasp"] == [(grasper, "grasp", "target")]
    assert events[-2:] == [("step",), (grasper, "grasp", "target")]
    np.testing.assert_array_equal(events[0][2], command[8:15])
    np.testing.assert_array_equal(events[1][2], command[:7])

    # The next arm target is only applied after all 21 steps, even for repeated close.
    events.clear()
    command[0] += 1
    assert worker.execute_command(scene, command, control, physics_steps=1, grasper=grasper) == 1
    assert steps == 22
    assert events[-2:] == [("step",), (grasper, "grasp", "target")]

    events.clear()
    command[[7, 15]] = 1
    worker.execute_command(scene, command, control, physics_steps=2, grasper=grasper)
    assert [e[:2] for e in events] == [
        ("right", "target"),
        ("left", "target"),
        ("right", "release"),
        ("right", "drive"),
        ("left", "release"),
        ("left", "drive"),
        ("step",),
        ("right", "release"),
        ("left", "release"),
        ("step",),
    ]
    assert control.previous == {"left": 1.0, "right": 1.0}
    assert control.open_done == {"left": True, "right": True}
    command[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        worker.execute_command(scene, command, control, physics_steps=1, grasper=grasper)
    assert steps == 24


@pytest.mark.parametrize("physics_steps", [0, -1, True, np.bool_(1), 1.5, float("nan"), "21"])
def test_invalid_physics_span_is_rejected_before_touching_scene(physics_steps):
    with pytest.raises(ValueError, match="positive integer"):
        worker.execute_command(None, np.zeros(16), {}, physics_steps=physics_steps, grasper="right")


def test_invalid_gt_grasper_is_rejected_before_touching_scene():
    with pytest.raises(ValueError, match="GT Scheme"):
        worker.execute_command(None, np.zeros(16), {}, physics_steps=1, grasper="both")


def test_opening_completion_survives_command_boundaries_and_rearms_after_closing():
    calls = []
    steps = 0

    def actuate(target, velocity, side):
        calls.append((side, target))
        # Right opening finishes first; left needs another physical step.
        return side == "right" or steps > 0

    def step():
        nonlocal steps
        steps += 1

    arm = SimpleNamespace(set_joint_target_positions=lambda value: None)
    scene = SimpleNamespace(
        robot=SimpleNamespace(
            left_arm=arm,
            right_arm=arm,
            actutate_gripper=actuate,
            release_gripper=lambda side: None,
            grasp=lambda obj, side: None,
        ),
        step=step,
        task=SimpleNamespace(get_graspable_objects=list),
    )
    control = worker.ControlState(previous={"left": 1.0, "right": 1.0})
    command = np.zeros(16)
    command[[7, 15]] = 1
    worker.execute_command(scene, command, control, physics_steps=1, grasper="right")
    assert control.open_done == {"left": False, "right": True}
    # The next command completes the unfinished side only.
    worker.execute_command(scene, command, control, physics_steps=2, grasper="right")
    assert calls == [("right", 1.0), ("left", 1.0), ("left", 1.0)]
    assert control.open_done == {"left": True, "right": True}
    calls.clear()
    # Even if contact pushes fingers inward, there is no new open actuation.
    scene.robot.actutate_gripper = lambda *args: pytest.fail("Completed opening was driven again")
    command[0] = 0.5
    worker.execute_command(scene, command, control, physics_steps=3, grasper="right")
    assert steps == 6
    scene.robot.actutate_gripper = actuate
    command[7] = 0
    worker.execute_command(scene, command, control, physics_steps=2, grasper="right")
    assert control.open_done == {"left": False, "right": True}
    assert calls == [("left", 0.0), ("left", 0.0)]
    command[7] = 1
    worker.execute_command(scene, command, control, physics_steps=2, grasper="right")
    assert calls[-1] == ("left", 1.0)
    assert len(calls) == 3


@pytest.mark.parametrize("steps", [0, np.int64(20)])
def test_initial_settling_only_advances_physics(steps):
    calls = []
    scene = SimpleNamespace(step=lambda: calls.append("physics"))
    assert worker.settle_initial_state(scene, steps) == steps
    assert calls == ["physics"] * steps


@pytest.mark.parametrize("steps", [-1, True, np.bool_(0), 1.5, float("nan"), "20", None])
def test_invalid_initial_settling_is_rejected_before_scene_or_output(steps, tmp_path):
    with pytest.raises(ValueError, match="non-negative integer"):
        worker.settle_initial_state(None, steps)
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="non-negative integer"):
        worker.run_replay(None, output, settle_steps=steps)
    assert not output.exists()


def test_supplied_phases_are_preserved_without_recleaning():
    runs = [
        {"phase": 1, "start": 0, "end": 2},
        {"phase": 2, "start": 2, "end": 3},
        {"phase": 3, "start": 3, "end": 5},
        {"phase": 4, "start": 5, "end": 8},
    ]
    # A supplied one-node phase stays one node, independently of automatic P=6.
    np.testing.assert_array_equal(worker.phases_from_runs(runs, 8), [1, 1, 2, 3, 3, 4, 4, 4])
    with pytest.raises(ValueError, match="cover"):
        worker.phases_from_runs(runs[1:], 8)


def test_edge_gt_geometry_uses_original_nodes_and_accepts_singleton_axis():
    objects = [
        (SimpleNamespace(get_name=lambda: "box_edge"), None),
        (SimpleNamespace(get_name=lambda: "phone_edge"), None),
    ]
    scene = SimpleNamespace(task=SimpleNamespace(_initial_objs_in_scene=objects))
    base = [0, 0, 0, 0, 0, 0, 1]
    demo = [SimpleNamespace(task_low_dim_state=np.array([[*base, 0, y, 0, 0, 0, 0, 1]])) for y in (-0.1, -9.0, -0.2)]
    inputs = SimpleNamespace(
        demo=demo, index={"raw_observation_row": np.array([0, 2])}, metadata={"source_task_name": "bimanual_edge_phone"}
    )
    geometry, names = worker.ground_truth_geometry(scene, inputs)
    np.testing.assert_allclose(geometry, [0.1, 0.2])
    assert names == ["box_edge", "phone_edge"]


def test_initial_clearance_cannot_stand_in_for_withdrawal_evidence():
    evidence = [
        {"condition_status": {"1": True, "3": clear}, "gt_grasper_attached": held, "assistant_target_contact": contact}
        for clear, held, contact in [
            (True, False, False),
            (False, True, True),
            (True, True, False),
            (True, False, False),
        ]
    ]
    summary = worker.summarize_evidence(evidence, np.array([1, 2, 3, 4]))
    assert summary["clear_path_in_gt_clear_or_lift"] == {"count": 2, "first_node": 2, "last_node": 3}
    assert summary["attached_at_or_after_first_clear"]["count"] == 1
    assert summary["nodes_at_or_after_first_clear"] == 2
    assert summary["attached_gt_lift_nodes"] == 0
    assert summary["recovery_status"] == "pending"


def _hold_point():
    return {
        "condition_status": {"1": True, "3": True},
        "gt_grasper_attached": True,
        "evaluator_grasp_held": True,
        "assistant_target_contact": False,
    }


def test_withdrawal_hold_rejects_misaligned_evidence():
    with pytest.raises(ValueError, match="matching lengths"):
        worker.withdrawal_hold_complete([_hold_point()], np.array([3, 4]))


def test_withdrawal_hold_requires_ten_later_commands_and_gt_phase_four():
    phases = np.array([1, 2, 3] + [4] * 12)
    evidence = [_hold_point() for _ in phases]
    # Initial clearance is irrelevant; the first eligible baseline is node 2.
    assert not worker.withdrawal_hold_complete(evidence[:12], phases[:12])
    assert worker.withdrawal_hold_complete(evidence[:13], phases[:13])
    # Even a long clear/held stretch must not cut off the remaining GT withdrawal.
    assert not worker.withdrawal_hold_complete(evidence, np.minimum(phases, 3))
    summary = worker.summarize_evidence(evidence[:13], phases[:13])
    assert summary["withdrawal_hold_complete"]
    assert summary["recovery_status"] == "pending"


@pytest.mark.parametrize(
    ("node", "field", "value"),
    [
        (3, "gt_grasper_attached", False),  # Earlier loss is not erased by a later regrasp.
        (3, "evaluator_grasp_held", False),  # Conflicting evidence also disables early stopping.
        (13, "clear", False),
        (13, "assistant_target_contact", True),
    ],
)
def test_withdrawal_hold_does_not_accept_loss_conflict_or_incomplete_clearance(node, field, value):
    phases = np.array([1, 2, 3] + [4] * 12)
    evidence = [_hold_point() for _ in phases]
    if field == "clear":
        evidence[node]["condition_status"]["3"] = value
    else:
        evidence[node][field] = value
    assert not worker.withdrawal_hold_complete(evidence, phases)


@pytest.mark.parametrize(
    ("max_steps", "loss_node", "included", "expected_stop", "reason"),
    [
        (None, None, True, 16, "withdrawal hold confirmed"),
        (8, None, True, 8, "requested prefix"),
        (30, None, True, 30, "end of episode"),
        (None, 6, True, 30, "end of episode"),
        (None, None, False, 30, "end of episode"),
    ],
)
def test_replay_stop_preserves_timing_evidence_and_full_gt_geometry(
    monkeypatch, tmp_path, max_steps, loss_node, included, expected_stop, reason
):
    physics = 0
    node = 0
    updates = []
    commands = []
    shutdown = []
    roles = {"grasper": "right", "pusher": "left"}
    scene = SimpleNamespace(
        task=SimpleNamespace(get_role_assignment=lambda: roles),
        pyrep=SimpleNamespace(get_simulation_timestep=lambda: 0.05),
    )

    def step():
        nonlocal physics
        physics += 1

    scene.step = step
    task_env = SimpleNamespace(_scene=scene, reset_to_demo=lambda *a, **k: None)
    environment = SimpleNamespace(
        launch=lambda: None, get_task=lambda task: task_env, shutdown=lambda: shutdown.append(True)
    )
    # Exercise the actual replay loop and file writing without a simulator installation.
    modules = {
        "pyrep.backend": {"sim": SimpleNamespace(simGetSimulationTime=lambda: physics * 0.05)},
        "rlbench.action_modes.action_mode": {"BimanualJointPositionActionMode": lambda *a: None},
        "rlbench.action_modes.arm_action_modes": {"BimanualJointPosition": lambda: None},
        "rlbench.action_modes.gripper_action_modes": {"BimanualDiscrete": lambda: None},
        "rlbench.backend.utils": {"task_file_to_task_class": lambda *a, **k: None},
        "rlbench.environment": {"Environment": lambda **k: environment},
        "rlbench.observation_config": {"ObservationConfig": lambda **k: None},
    }
    for name, exports in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**exports))
    monkeypatch.setattr(worker, "initialize_control", lambda *a: None)
    monkeypatch.setattr(worker, "resolve_points", lambda *a: {})
    monkeypatch.setattr(worker, "robot_contact_handles", lambda *a: {})
    monkeypatch.setattr(worker, "ground_truth_geometry", lambda *a: (np.arange(31, dtype=float), []))
    monkeypatch.setattr(worker, "read_completed_state", lambda *a: {"geometry": float(node)})

    def execute(scene, command, control, *, physics_steps, grasper):
        nonlocal node
        node = int(command[0])
        commands.append(node)
        for _ in range(physics_steps):
            scene.step()
        return physics_steps

    def read_evidence(*a, advance_evaluator):
        updates.append(advance_evaluator)
        point = _hold_point()
        point["condition_status"]["3"] = node >= 5
        point["assistant_target_contact"] = node < 5
        point["gt_grasper_attached"] = point["evaluator_grasp_held"] = node >= 2 and node != loss_node
        return point

    monkeypatch.setattr(worker, "execute_command", execute)
    monkeypatch.setattr(worker, "read_functional_evidence", read_evidence)
    commanded = np.zeros((31, 16))
    commanded[:, 0] = np.arange(31)
    spans = np.r_[0, np.tile([1, 3], 15)]
    source_steps = np.cumsum(spans)
    inputs = worker.ReplayInputs(
        demo=object(),
        metadata={"source_task_name": "bimanual_pick_plate"},
        index={
            "raw_observation_row": np.arange(31) * 2,
            "raw_command_row": np.r_[0, np.arange(30) * 2 + 1],
            "physics_step": source_steps,
            "command_kind": np.array(["initial"] + ["step"] * 30),
        },
        commanded_joint16=commanded,
        clean_phase=np.array([1] * 2 + [2] * 2 + [3] * 12 + [4] * 15),
        roles=roles,
        identity={"sim_dt": 0.05, "sidecar_status": "included" if included else "excluded"},
    )
    output = tmp_path / "replay"
    report = worker.run_replay(inputs, output, max_steps, settle_steps=3)
    assert report["stop_reason"] == reason
    assert report["post_clear_hold_commands"] == 10
    assert report["num_executed_commands"] == expected_stop
    assert report["full_episode_replayed"] == (expected_stop == 30)
    assert report["num_executed_physics_steps"] == source_steps[expected_stop]
    assert report["settle_elapsed_sim_time"] == pytest.approx(0.15)
    assert report["summary"]["recovery_status"] == "pending"
    assert commands == list(range(1, expected_stop + 1))
    assert updates == [False] + [True] * expected_stop
    assert shutdown == [True]
    assert json.loads((output / "replay.json").read_text()) == report
    with np.load(output / "replay.npz", allow_pickle=False) as arrays:
        for key in arrays.files:
            assert len(arrays[key]) == (31 if key == "gt_geometry_full" else expected_stop + 1)
        np.testing.assert_array_equal(arrays["gt_geometry_full"], np.arange(31))
        np.testing.assert_array_equal(arrays["gt_geometry"], np.arange(expected_stop + 1))
        np.testing.assert_array_equal(arrays["executed_physics_steps"], source_steps[: expected_stop + 1])
        np.testing.assert_allclose(arrays["elapsed_sim_time"], source_steps[: expected_stop + 1] * 0.05)


def test_contact_handles_do_not_mix_shared_dual_panda_roots():
    def branch(handle, children):
        return SimpleNamespace(
            get_handle=lambda: handle,
            get_objects_in_tree=lambda: [SimpleNamespace(get_handle=lambda h=h: h) for h in children],
        )

    left_arm, right_arm = branch(1, [10, 20]), branch(1, [10, 20])
    left_arm.joints = [branch(10, [11, 12])]
    right_arm.joints = [branch(20, [21, 22])]
    robot = SimpleNamespace(
        left_arm=left_arm, right_arm=right_arm, left_gripper=branch(11, [12]), right_gripper=branch(21, [22])
    )
    assert worker.robot_contact_handles(SimpleNamespace(robot=robot)) == {
        "left": {10, 11, 12},
        "right": {20, 21, 22},
    }
