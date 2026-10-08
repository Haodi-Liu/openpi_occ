"""Replay one effective RLBench demonstration in a fresh ppi process.

Only ``run-replay`` is implemented here. It records recovery evidence for the
reference-preparation step; it does not decide admission or run a model.
NumPy/SciPy helpers can be imported without RLBench, JAX, or LeRobot.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

EXECUTION_RULE = "joint16_gt_spans_gt_grasper_open_latch_settle_v3"
POST_CLEAR_HOLD_COMMANDS = 10
SIDES = ("left", "right")
INDEX_KEYS = ("raw_observation_row", "raw_command_row", "physics_step", "command_kind")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _finite(values: Any, name: str, shape: tuple | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: shape={result.shape}, expected={shape} or non-finite values.")
    return result


def _pose(values: Any, name: str) -> np.ndarray:
    result = _finite(values, name)
    if result.ndim < 1 or result.shape[-1] != 7 or np.any(np.linalg.norm(result[..., 3:], axis=-1) == 0):
        raise ValueError(f"Invalid pose7/quaternion: {name}.")
    return result


def relative_pose(base: Any, tip: Any) -> np.ndarray:
    """Express tip in base coordinates; inputs are (..., 7) world poses, xyzw."""
    base, tip = _pose(base, "base"), _pose(tip, "tip")
    if base.shape != tip.shape:
        raise ValueError("Relative poses must have matching shapes.")
    shape = base.shape
    base, tip = base.reshape(-1, 7), tip.reshape(-1, 7)
    inverse = Rotation.from_quat(base[:, 3:]).inv()
    position = inverse.apply(tip[:, :3] - base[:, :3])
    rotation = inverse * Rotation.from_quat(tip[:, 3:])
    return np.concatenate((position, rotation.as_quat()), axis=-1).reshape(shape)


def hysteresis(value: float, previous: float) -> float:
    """Decode a candidate gripper value; expert commands are already binary."""
    if not np.isfinite(value) or previous not in (0.0, 1.0):
        raise ValueError("Expected a finite gripper value and binary previous command.")
    if value <= 0.4:
        return 0.0
    if value >= 0.6:
        return 1.0
    return previous


def _joint16(command: Any) -> np.ndarray:
    command = _finite(command, "joint16 command", (16,))
    if not np.isin(command[[7, 15]], (0.0, 1.0)).all():
        raise ValueError("Execution requires binary gripper commands; decode candidates first.")
    return command


@dataclasses.dataclass
class ControlState:
    """Carry gripper commands and completed opening through prefix/candidate boundaries."""

    previous: dict[str, float]
    open_done: dict[str, bool] = dataclasses.field(default_factory=lambda: dict.fromkeys(SIDES, False))


def execute_command(
    scene: Any,
    command: Any,
    control: ControlState,
    *,
    physics_steps: int,
    grasper: str,
) -> int:
    """Hold one command for its GT duration; only the GT grasper may attach."""
    command = _joint16(command)
    if (
        isinstance(physics_steps, (bool, np.bool_))
        or not isinstance(physics_steps, (int, np.integer))
        or physics_steps <= 0
    ):
        raise ValueError("physics_steps must be a positive integer.")
    if grasper not in SIDES:
        raise ValueError("grasper must be the left or right arm from the GT Scheme.")
    sides = {"left": command[:8], "right": command[8:]}
    for side in ("right", "left"):
        getattr(scene.robot, side + "_arm").set_joint_target_positions(sides[side][:7])
        target = float(sides[side][7])
        if target != control.previous[side]:
            control.open_done[side] = False
        control.previous[side] = target
    for _ in range(physics_steps):
        ready_to_grasp = False
        for side in ("right", "left"):
            target = float(sides[side][7])
            if target == 1.0:
                scene.robot.release_gripper(side)
                if not control.open_done[side]:
                    control.open_done[side] = bool(scene.robot.actutate_gripper(target, 0.04, side))
            else:
                done = scene.robot.actutate_gripper(target, 0.04, side)
                if side == grasper and done:
                    ready_to_grasp = True
        scene.step()
        if ready_to_grasp:
            for obj in scene.task.get_graspable_objects():
                scene.robot.grasp(obj, grasper)
    return int(physics_steps)


def initialize_control(scene: Any, command: Any) -> ControlState:
    """Set initial holding targets and clear controller history without stepping."""
    command = _joint16(command)
    for side, offset in (("left", 0), ("right", 8)):
        getattr(scene.robot, side + "_arm").set_joint_target_positions(command[offset : offset + 7])
        gripper = getattr(scene.robot, side + "_gripper")
        gripper._prev_positions = [None] * len(gripper.joints)  # noqa: SLF001
        gripper._prev_vels = [None] * len(gripper.joints)  # noqa: SLF001
        gripper.set_joint_target_velocities([0.0] * len(gripper.joints))
    # The first open command confirms completion; subsequent open commands retain it.
    return ControlState(previous={"left": float(command[7]), "right": float(command[15])})


def validate_settle_steps(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("settle_steps must be a non-negative integer.")
    return int(value)


def settle_initial_state(scene: Any, steps: int) -> int:
    """Hold initial targets before node zero, without actuating grippers or evaluating conditions."""
    steps = validate_settle_steps(steps)
    for _ in range(steps):
        scene.step()
    return steps


def phases_from_runs(runs: list[dict], num_nodes: int) -> np.ndarray:
    """Consume the supplied annotation; do not rerun automatic phase cleaning."""
    phase = np.empty(num_nodes, dtype=np.int8)
    cursor = 0
    for run in runs:
        start, end, value = run["start"], run["end"], run["phase"]
        if start != cursor or not start < end <= num_nodes or value not in (1, 2, 3, 4):
            raise ValueError("Phase runs must cover every effective node exactly once.")
        phase[start:end] = value
        cursor = end
    if cursor != num_nodes:
        raise ValueError("Phase runs do not cover the effective sequence.")
    return phase


@dataclasses.dataclass
class ReplayInputs:
    demo: Any
    metadata: dict
    index: dict[str, np.ndarray]
    commanded_joint16: np.ndarray
    clean_phase: np.ndarray
    roles: dict[str, str]
    identity: dict


def _validate_repo_binding(root: Path, manifest: dict, audit: dict, metadata: dict) -> None:
    """Check the selected episode's portable metadata without importing LeRobot."""
    info = _read_json(root / "meta/info.json")
    if info["total_frames"] != manifest["num_dataset_rows"]:
        raise ValueError("LeRobot row count differs from the supplied sidecar.")
    offset = 0
    found = False
    with (root / "meta/episodes.jsonl").open() as stream:
        for line in stream:
            episode = json.loads(line)
            if episode["episode_index"] == audit["lerobot_episode_index"]:
                found = True
                if (
                    episode["length"] != metadata["num_transitions"]
                    or offset != audit["global_start_index"]
                    or episode["tasks"] != [metadata["task"]]
                ):
                    raise ValueError("LeRobot episode identity differs from the export/sidecar.")
                break
            offset += episode["length"]
    if not found:
        raise ValueError("The sidecar episode is missing from LeRobot metadata.")


def load_replay_inputs(export_episode_dir: Path, annotations_dir: Path, lerobot_root: Path) -> ReplayInputs:
    """Bind current source files, effective nodes, supplied phases, and repo metadata."""
    # Exporter owns command/state extraction. RLBench is needed only to read raw pickles.
    if __package__:
        from . import export_rlbench_split as exporter
    else:
        import export_rlbench_split as exporter

    metadata = _read_json(export_episode_dir / "meta.json")
    if metadata["action_semantics"] != exporter.EFFECTIVE_ACTION_SEMANTICS:
        raise ValueError("Replay requires the effective_v2 export.")
    source_dir = Path(metadata["source_episode_dir"])
    demo = exporter.load_pickle(source_dir / "low_dim_obs.pkl")
    num_nodes = metadata["num_observations"]
    if len(demo) != metadata["raw_num_observations"] or num_nodes != metadata["num_transitions"] + 1:
        raise ValueError("Source/export lengths changed; regenerate the dependent artifacts.")
    with np.load(export_episode_dir / metadata["effective_index_file"], allow_pickle=False) as archive:
        index = {key: archive[key] for key in INDEX_KEYS}
    for key in INDEX_KEYS[:3]:
        if index[key].shape != (num_nodes,) or not np.issubdtype(index[key].dtype, np.integer):
            raise ValueError(f"Invalid effective index: {key}.")
    obs_rows, command_rows, steps = (index[key] for key in INDEX_KEYS[:3])
    if (
        command_rows[0] != 0
        or obs_rows[-1] != len(demo) - 1
        or np.any(command_rows < 0)
        or np.any(obs_rows >= len(demo))
        or np.any(command_rows > obs_rows)
        or not np.array_equal(command_rows[1:], obs_rows[:-1] + 1)
        or steps[0] != 0
        or np.any(np.diff(steps) <= 0)
        or not np.array_equal(index["command_kind"], ["initial"] + ["step"] * (num_nodes - 1))
    ):
        raise ValueError("Invalid effective node boundaries or timing.")
    commands = np.stack([exporter.extract_action(demo[int(row)].misc) for row in command_rows])
    # The exported index is authoritative. Verify its groups against the current
    # source; never silently rebuild it after a recollected demo replaces a file.
    # Shapes were checked above; omit strict= for the Python 3.8 simulator environment.
    for n, (start, end) in enumerate(zip(command_rows, obs_rows)):  # noqa: B905
        for row in range(int(start), int(end) + 1):
            if demo[row].misc.get("physics_step") != steps[n] or not np.array_equal(
                exporter.extract_action(demo[row].misc), commands[n]
            ):
                raise ValueError(f"Source no longer matches effective node {n}; regenerate the export.")
    observations = [demo[int(row)] for row in obs_rows]
    source_dts = _finite([obs.misc["sim_dt"] for obs in observations], "source sim_dt", (num_nodes,))
    if source_dts[0] <= 0 or not np.allclose(source_dts, source_dts[0], rtol=1e-6, atol=0):
        raise ValueError("Source sim_dt must be constant and positive.")
    phases = np.array([exporter.extract_phase(obs.misc) for obs in observations], dtype=np.int8)
    expected = {
        "state": np.stack([exporter.extract_state(obs) for obs in observations[:-1]]),
        "actions": commands[1:],
        "phase_before_action": phases[:-1],
        "phase_after_action": phases[1:],
    }
    for key, array in expected.items():
        if not np.array_equal(array, np.load(export_episode_dir / f"{key}.npy", allow_pickle=False)):
            raise ValueError(f"Current source disagrees with exported {key}.")

    manifest = _read_json(annotations_dir / "manifest.json")
    quality_path = annotations_dir / "quality_report.json"
    digest = "sha256:" + hashlib.sha256(quality_path.read_bytes()).hexdigest()
    if digest != manifest["quality_report_sha256"]:
        raise ValueError("Sidecar quality report does not match its manifest.")
    matches = [
        audit
        for audit in _read_json(quality_path)["episodes"]
        if audit["source_task_name"] == metadata["source_task_name"]
        and audit["source_episode_number"] == metadata["source_episode_number"]
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching source episode in the supplied sidecar.")
    audit = matches[0]
    if (
        audit["num_actions"] != num_nodes - 1
        or audit["action_semantics"] != metadata["action_semantics"]
        or not np.array_equal(phases_from_runs(audit["raw_runs"], num_nodes), phases)
    ):
        raise ValueError("Sidecar source phases/identity do not match the effective export.")
    clean_phase = phases_from_runs(audit["clean_runs"], num_nodes)
    _validate_repo_binding(lerobot_root, manifest, audit, metadata)

    scheme_files = list(source_dir.glob("scheme_info_*.pkl"))
    if len(scheme_files) != 1:
        raise ValueError("Expected one GT scheme_info_*.pkl in the source episode.")
    scheme = exporter.load_pickle(scheme_files[0])
    roles = scheme.get("role_assignment")
    if roles not in ({"grasper": "left", "pusher": "right"}, {"grasper": "right", "pusher": "left"}):
        raise ValueError("GT Scheme must assign distinct grasping and assisting arms.")
    if scheme.get("active_scheme") != roles["grasper"] + "_grasper":
        raise ValueError("GT scheme name disagrees with its roles.")
    variation = int(exporter.load_pickle(source_dir / "variation_number.pkl"))
    if variation != metadata["variation_number"]:
        raise ValueError("Source/export variations disagree.")
    demo.initial_observation = deepcopy(demo[0])
    demo.initial_arm_scheme = scheme
    demo.variation_number = variation
    identity = {
        "export_episode_dir": str(export_episode_dir),
        "source_episode_dir": str(source_dir),
        "annotations_dir": str(annotations_dir),
        "lerobot_root": str(lerobot_root),
        "repo_id": manifest["repo_id"],
        "sidecar_manifest_digest": manifest["manifest_digest"],
        "source_task_name": metadata["source_task_name"],
        "source_episode_number": metadata["source_episode_number"],
        "lerobot_episode_index": audit["lerobot_episode_index"],
        "global_start_index": audit["global_start_index"],
        "sidecar_status": audit["status"],
        "variation_number": variation,
        "scheme": scheme,
        "action_semantics": metadata["action_semantics"],
        "sim_dt": float(source_dts[0]),
    }
    return ReplayInputs(demo, metadata, index, commands, clean_phase, dict(roles), identity)


def resolve_points(scene: Any, inputs: ReplayInputs) -> dict:
    """Resolve task-specific objects from source names, never from episode numbers."""
    objects = {obj.get_name(): obj for obj, _ in scene.task._initial_objs_in_scene}  # noqa: SLF001
    misc = inputs.demo[0].misc
    names = {"grasp": "grasp_pt", "contact": misc["contact_source"]}
    if misc["has_affordance"]:
        names["affordance"] = misc["affordance_source"]
    if inputs.metadata["source_task_name"] == "bimanual_edge_phone":
        names["phone_edge"] = "phone_edge"
        if names.get("affordance") != "box_edge":
            raise ValueError("edge_phone geometry requires phone_edge relative to box_edge.")
    for row in inputs.index["raw_observation_row"]:
        current = inputs.demo[int(row)].misc
        if any(current.get(key) != misc.get(key) for key in ("contact_source", "affordance_source", "has_affordance")):
            raise ValueError("Keypoint sources change within this episode.")
    return {key: objects[name] for key, name in names.items()}


def read_completed_state(scene: Any, roles: Mapping[str, str], points: dict, task_name: str) -> dict:
    """Read actual world poses and finger joints after the command, without stepping."""
    tip = np.stack([_pose(getattr(scene.robot, side + "_arm").get_tip().get_pose(), side + " tip") for side in SIDES])
    finger = np.stack(
        [
            _finite(getattr(scene.robot, side + "_gripper").get_joint_positions(), side + " fingers", (2,))
            for side in SIDES
        ]
    )
    grasp, contact = (_pose(points[name].get_pose(), name) for name in ("grasp", "contact"))
    gi, ci = SIDES.index(roles["grasper"]), SIDES.index(roles["pusher"])
    result = {
        "joint_positions": np.stack([getattr(scene.robot, side + "_arm").get_joint_positions() for side in SIDES]),
        "tip_pose": tip,
        "finger": finger,
        "grasp_pose": grasp,
        "contact_pose": contact,
        "target_pose": _pose(scene.task.target_object.get_pose(), "target"),
        "target_velocity": np.asarray(scene.task.target_object.get_velocity()),
        "geometry": -points["phone_edge"].get_position(relative_to=points["affordance"])[1]
        if task_name == "bimanual_edge_phone"
        else grasp[2],
        "grasp_relative": relative_pose(grasp, tip[gi]),
        "contact_relative": relative_pose(contact, tip[ci]),
        "grasp_finger": finger[gi],
        "contact_finger": finger[ci],
    }
    if "affordance" in points:
        affordance = _pose(points["affordance"].get_pose(), "affordance")
        result.update(affordance_pose=affordance, environment_relative=relative_pose(affordance, contact))
    for key, value in result.items():
        _finite(value, key)
    return result


def ground_truth_geometry(scene: Any, inputs: ReplayInputs) -> tuple[np.ndarray, list[str]]:
    """Read original GT geometry; the replay supplies only the task-object ordering."""
    names = [obj.get_name() for obj, _ in scene.task._initial_objs_in_scene]  # noqa: SLF001
    observations = [inputs.demo[int(row)] for row in inputs.index["raw_observation_row"]]
    if inputs.metadata["source_task_name"] == "bimanual_edge_phone":
        states = _finite(
            [np.asarray(obs.task_low_dim_state).reshape(-1) for obs in observations],
            "GT task state",
            (len(observations), len(names) * 7),
        ).reshape(len(observations), len(names), 7)
        geometry = -relative_pose(states[:, names.index("box_edge")], states[:, names.index("phone_edge")])[:, 1]
    else:
        geometry = _finite([obs.misc["grasp_position"][2] for obs in observations], "GT grasp height")
    return geometry, names


def robot_contact_handles(scene: Any) -> dict[str, set[int]]:
    """Cache robot handles before attachments can change the scene tree."""
    result = {}
    for side in SIDES:
        handles = set()
        # PandaLeft/PandaRight share the DualPanda model root. Its tree contains
        # both arms; only the first joint's subtree identifies this side.
        arm = getattr(scene.robot, side + "_arm")
        for component in (arm.joints[0], getattr(scene.robot, side + "_gripper")):
            handles.add(component.get_handle())
            handles.update(obj.get_handle() for obj in component.get_objects_in_tree())
        result[side] = handles
    if result["left"] & result["right"]:
        raise ValueError("Left/right contact attribution must use disjoint robot subtrees.")
    return result


def read_functional_evidence(scene: Any, roles: dict, handles: dict, *, advance_evaluator: bool) -> dict:
    """Update conditions once at a completed point, then read their cached results."""
    task = scene.task
    if advance_evaluator:
        # Current four tasks do not evaluate conditions inside task.step().
        # Do not also call success() or individual stateful conditions here.
        task.evaluate_phase_and_get_labels()
    progress = task.phased_evaluator.get_phase_progress()
    target = task.target_object
    contacts = target.get_contact()
    contact_counts = {
        side: sum(bool(set(item["contact_handles"]) & handles[side]) for item in contacts) for side in SIDES
    }
    attached = {
        side: [obj.get_name() for obj in getattr(scene.robot, side + "_gripper").get_grasped_objects()]
        for side in SIDES
    }
    return {
        "condition_status": {str(k): bool(v) for k, v in progress["condition_status"].items()},
        "evaluator_phase": int(progress["current_phase"]),
        "evaluator_grasp_held": bool(progress["grasp_held"]),
        "attached_objects": attached,
        "gt_grasper_attached": target.get_name() in attached[roles["grasper"]],
        "target_contact_count_by_arm": contact_counts,
        "assistant_target_contact": contact_counts[roles["pusher"]] > 0,
        "target_contacts": contacts,
    }


def withdrawal_hold_complete(evidence: list[dict], clean_phase: np.ndarray) -> bool:
    """Finish GT withdrawal, then require a clear/held baseline plus ten later command endpoints."""
    if len(evidence) != len(clean_phase):
        raise ValueError("Evidence and clean_phase must have matching lengths.")
    points = POST_CLEAR_HOLD_COMMANDS + 1
    if len(evidence) < points or clean_phase[-1] != 4:
        return False
    # A loss or disagreement anywhere during withdrawal needs the remaining replay for review.
    if any(
        phase >= 3 and not (point["gt_grasper_attached"] and point["evaluator_grasp_held"])
        for phase, point in zip(clean_phase, evidence)  # noqa: B905 - lengths checked above; Python 3.8.
    ):
        return False
    return all(
        phase >= 3 and point["condition_status"]["3"] and not point["assistant_target_contact"]
        for phase, point in zip(clean_phase[-points:], evidence[-points:])  # noqa: B905 - equal slices; Python 3.8.
    )


def summarize_evidence(evidence: list[dict], clean_phase: np.ndarray) -> dict:
    """Report observations only; functional pass/fail belongs to the later review."""

    def nodes(predicate):
        return [i for i, point in enumerate(evidence) if predicate(point)]

    def span(indices):
        return {
            "count": len(indices),
            "first_node": indices[0] if indices else None,
            "last_node": indices[-1] if indices else None,
        }

    held = nodes(lambda p: p["gt_grasper_attached"])
    # Clearance can already be true while the assistant is far away at reset.
    # The relevant withdrawal evidence starts in the supplied GT phase 3/4.
    clear = [i for i, point in enumerate(evidence) if clean_phase[i] >= 3 and point["condition_status"]["3"]]
    after_clear = [i for i in held if clear and i >= clear[0]]
    gt_lift = np.flatnonzero(clean_phase == 4).tolist()
    return {
        "space_condition_met": span(nodes(lambda p: p["condition_status"]["1"])),
        "assistant_target_contact": span(nodes(lambda p: p["assistant_target_contact"])),
        "gt_grasper_attached": span(held),
        "clear_path_in_gt_clear_or_lift": span(clear),
        "attached_at_or_after_first_clear": span(after_clear),
        "nodes_at_or_after_first_clear": len(evidence) - clear[0] if clear else 0,
        "gt_lift_nodes": len(gt_lift),
        "attached_gt_lift_nodes": sum(i in held for i in gt_lift),
        "withdrawal_hold_complete": withdrawal_hold_complete(evidence, clean_phase),
        "recovery_status": "pending",
        "note": "Execution evidence only; later review decides functional admission.",
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def run_replay(inputs: ReplayInputs, output_dir: Path, max_steps: int | None = None, *, settle_steps: int = 0) -> dict:
    """Replay through withdrawal/hold evidence, or an explicitly requested command prefix."""
    settle_steps = validate_settle_steps(settle_steps)
    from pyrep.backend import sim
    from rlbench.action_modes.action_mode import BimanualJointPositionActionMode
    from rlbench.action_modes.arm_action_modes import BimanualJointPosition
    from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    num_steps = len(inputs.commanded_joint16) - 1
    stop = num_steps if max_steps is None else max_steps
    if not 0 <= stop <= num_steps:
        raise ValueError(f"max_steps must be within [0, {num_steps}].")
    stop_reason = "end of episode" if stop == num_steps else "requested prefix"
    auto_stop = max_steps is None and inputs.identity["sidecar_status"] == "included"
    output_dir.mkdir(parents=True, exist_ok=False)
    config = ObservationConfig(
        camera_configs={}, gripper_joint_positions=True, task_low_dim_state=True, record_bimanual_action_commands=False
    )
    environment = Environment(
        action_mode=BimanualJointPositionActionMode(BimanualJointPosition(), BimanualDiscrete()),
        obs_config=config,
        robot_setup="dual_panda",
        headless=True,
    )
    try:
        environment.launch()
        task_name = inputs.metadata["source_task_name"]
        task_env = environment.get_task(task_file_to_task_class(task_name, bimanual=True))
        task_env.reset_to_demo(inputs.demo, restore_scene=True)
        scene = task_env._scene  # noqa: SLF001
        if scene.task.get_role_assignment() != inputs.roles:
            raise ValueError("Restored task roles differ from GT Scheme.")
        control = initialize_control(scene, inputs.commanded_joint16[0])
        points = resolve_points(scene, inputs)
        gt_geometry, object_names = ground_truth_geometry(scene, inputs)
        handles = robot_contact_handles(scene)
        dt = float(scene.pyrep.get_simulation_timestep())
        if not np.isclose(dt, inputs.identity["sim_dt"], rtol=1e-6, atol=0):
            raise ValueError(f"Simulator timestep {dt} differs from source timestep {inputs.identity['sim_dt']}.")
        before_settle = float(sim.simGetSimulationTime())
        settle_initial_state(scene, settle_steps)
        origin = float(sim.simGetSimulationTime())
        settle_elapsed = origin - before_settle
        if not np.isclose(settle_elapsed, settle_steps * dt, atol=max(1e-5, settle_steps * dt * 1e-6), rtol=0):
            raise RuntimeError("Initial settling advanced an unexpected amount of simulation time.")
        source_steps = inputs.index["physics_step"]
        command_steps = np.diff(source_steps, prepend=source_steps[0])
        states, evidence, times, executed_steps = [], [], [], []
        total_physics_steps = 0
        for node in range(stop + 1):
            if node:
                total_physics_steps += execute_command(
                    scene,
                    inputs.commanded_joint16[node],
                    control,
                    physics_steps=int(command_steps[node]),
                    grasper=inputs.roles["grasper"],
                )
            expected_steps = int(source_steps[node] - source_steps[0])
            if total_physics_steps != expected_steps:
                raise RuntimeError(f"Unexpected physical step count at node {node}.")
            elapsed = float(sim.simGetSimulationTime()) - origin
            expected_time = expected_steps * dt
            if not np.isclose(elapsed, expected_time, atol=max(1e-5, expected_time * 1e-6), rtol=0):
                raise RuntimeError(f"Unexpected physical advancement at node {node}: {elapsed} vs {expected_time}.")
            states.append(read_completed_state(scene, inputs.roles, points, task_name))
            point = read_functional_evidence(scene, inputs.roles, handles, advance_evaluator=node > 0)
            point.update(effective_node=node, clean_phase=int(inputs.clean_phase[node]))
            evidence.append(point)
            times.append(elapsed)
            executed_steps.append(total_physics_steps)
            if auto_stop and node < stop and withdrawal_hold_complete(evidence, inputs.clean_phase[: node + 1]):
                stop_reason = "withdrawal hold confirmed"
                break
        stop = len(evidence) - 1
        arrays = {key: np.stack([state[key] for state in states]) for key in states[0]}
        arrays.update({key: value[: stop + 1] for key, value in inputs.index.items()})
        arrays.update(
            commanded_joint16=inputs.commanded_joint16[: stop + 1],
            clean_phase=inputs.clean_phase[: stop + 1],
            gt_geometry=gt_geometry[: stop + 1],
            # Reference preparation still needs all N+1 GT values, even after an early replay stop.
            gt_geometry_full=gt_geometry,
            effective_node=np.arange(stop + 1),
            physics_steps_per_command=command_steps[: stop + 1],
            executed_physics_steps=np.asarray(executed_steps, dtype=np.int64),
            elapsed_sim_time=np.asarray(times),
        )
        report = {
            "status": "complete",
            "execution_rule": EXECUTION_RULE,
            "post_clear_hold_commands": POST_CLEAR_HOLD_COMMANDS,
            "stop_reason": stop_reason,
            "source": inputs.identity,
            "num_available_commands": num_steps,
            "num_executed_commands": stop,
            "num_executed_physics_steps": total_physics_steps,
            "settle_steps": settle_steps,
            "settle_elapsed_sim_time": settle_elapsed,
            "full_episode_replayed": stop == num_steps,
            "sim_dt": dt,
            "trajectory_file": "replay.npz",
            "task_object_names": object_names,
            "point_names": {key: obj.get_name() for key, obj in points.items()},
            "summary": summarize_evidence(evidence, inputs.clean_phase[: stop + 1]),
            "evidence": evidence,
        }
        np.savez(output_dir / "replay.npz", **arrays)
        (output_dir / "replay.json").write_text(
            json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n"
        )
        return report
    finally:
        environment.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    replay = sub.add_parser("run-replay", help="Record one expert replay; no model or persistent worker.")
    replay.add_argument(
        "--request-file", type=Path, help="JSON with the three input paths and optional max_steps/settle_steps."
    )
    replay.add_argument("--export-episode-dir", type=Path)
    replay.add_argument("--annotations-dir", type=Path)
    replay.add_argument("--lerobot-root", type=Path)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument(
        "--max-steps", type=int, help="Exact prefix length in effective commands; disables automatic stopping."
    )
    replay.add_argument(
        "--settle-steps", type=int, help="Initial physical settling steps before node zero (default: 0)."
    )
    args = parser.parse_args()
    keys = ("export_episode_dir", "annotations_dir", "lerobot_root")
    if args.request_file:
        if any(getattr(args, key) is not None for key in (*keys, "max_steps", "settle_steps")):
            parser.error("Use a request file or explicit input flags, not both.")
        request = _read_json(args.request_file)
        base = args.request_file.resolve().parent
    else:
        request = {key: getattr(args, key) for key in (*keys, "max_steps")}
        request["settle_steps"] = args.settle_steps if args.settle_steps is not None else 0
        base = Path.cwd()
    if any(not request.get(key) for key in keys):
        parser.error("export_episode_dir, annotations_dir, and lerobot_root are required.")
    try:
        settle_steps = validate_settle_steps(request.get("settle_steps", 0))
    except ValueError as exc:
        parser.error(str(exc))
    paths = {key: (base / Path(request[key]).expanduser()).resolve() for key in keys}
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}; choose a new directory.")
    inputs = load_replay_inputs(**paths)
    report = run_replay(inputs, args.output_dir.resolve(), request.get("max_steps"), settle_steps=settle_steps)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "num_executed_commands": report["num_executed_commands"],
                "num_executed_physics_steps": report["num_executed_physics_steps"],
                "settle_steps": report["settle_steps"],
                "full_episode_replayed": report["full_episode_replayed"],
                "stop_reason": report["stop_reason"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
