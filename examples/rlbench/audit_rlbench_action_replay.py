"""Replay selected RLBench command transitions through the production action mode.

This is a no-model sanity check for the intermediate RLBench export. It:

1. verifies the exported left-first state/action arrays against the raw demo;
2. selects one ordinary transition, the maximum raw tracking-error window, and
   every gripper-command change window;
3. restores the recorded episode seed and reconstructs each selected fragment;
4. routes actions through the production OpenPI-to-OCC layout converter and
   ``BimanualJointPositionActionMode``;
5. writes a JSON report with routing, joint-tracking, and gripper checks.

Repeated close events are intentionally outside this check because a persistent
16D gripper state cannot represent them.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from rlbench.action_modes.action_mode import BimanualJointPositionActionMode
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

ACTION_SEMANTICS = "executed_joint_target_commanded_gripper"
GRIPPER_INDICES = (7, 15)
DEFAULT_TASKS = (
    "bimanual_edge_phone",
    "bimanual_pick_fork",
    "bimanual_pick_plate",
    "bimanual_pivot_phone",
)


class AuditedBimanualJointPosition(BimanualJointPosition):
    """Production arm mode with read-only capture of its last input."""

    def __init__(self) -> None:
        super().__init__()
        self.last_action: np.ndarray | None = None

    def action_pre_step(self, scene, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        super().action_pre_step(scene, action)


class AuditedBimanualDiscrete(BimanualDiscrete):
    """Production gripper mode with read-only capture around actuation."""

    def __init__(self) -> None:
        super().__init__()
        self.last_action: np.ndarray | None = None
        self.before_open_amount: np.ndarray | None = None
        self.after_open_amount: np.ndarray | None = None

    def action(self, scene, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.before_open_amount = _physical_gripper_open_amount(scene)
        super().action(scene, action)
        self.after_open_amount = _physical_gripper_open_amount(scene)


def _physical_gripper_open_amount(scene) -> np.ndarray:
    return np.asarray(
        [
            scene.robot.right_gripper.get_open_amount()[0],
            scene.robot.left_gripper.get_open_amount()[0],
        ],
        dtype=np.float32,
    )


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def observed_state(obs: Any) -> np.ndarray:
    state = np.asarray(
        [
            *obs.left.joint_positions,
            obs.left.gripper_open,
            *obs.right.joint_positions,
            obs.right.gripper_open,
        ],
        dtype=np.float32,
    )
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid observed state: {state!r}.")
    return state


def recorded_action(misc: dict[str, Any]) -> np.ndarray:
    action = np.asarray(
        [
            *misc["left_executed_demo_joint_position_action"],
            misc["left_commanded_gripper_state"],
            *misc["right_executed_demo_joint_position_action"],
            misc["right_commanded_gripper_state"],
        ],
        dtype=np.float32,
    )
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError(f"Invalid recorded action: {action!r}.")
    if not np.isin(action[list(GRIPPER_INDICES)], (0.0, 1.0)).all():
        raise ValueError(f"Non-binary gripper command: {action!r}.")
    return action


def load_production_converter(
    occ_models_dir: Path,
) -> Callable[..., np.ndarray]:
    """Load the exact converter used by the OCC OpenPI evaluation agent."""
    occ_models_dir = occ_models_dir.resolve()
    if not occ_models_dir.is_dir():
        raise FileNotFoundError(f"OCC models directory not found: {occ_models_dir}")
    sys.path.insert(0, str(occ_models_dir))
    try:
        module = importlib.import_module("agents.openpi_policy.agent")
    finally:
        sys.path.pop(0)
    return module._openpi_joint16_to_occ_joint16  # noqa: SLF001


def validate_export(
    demo: Any,
    episode_export_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    states = np.load(episode_export_dir / "state.npy")
    actions = np.load(episode_export_dir / "actions.npy")
    metadata = json.loads((episode_export_dir / "meta.json").read_text())

    expected_states = np.asarray(
        [observed_state(demo[t]) for t in range(len(demo) - 1)],
        dtype=np.float32,
    )
    expected_actions = np.asarray(
        [recorded_action(demo[t + 1].misc) for t in range(len(demo) - 1)],
        dtype=np.float32,
    )

    if states.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError(f"Expected float32 arrays, got state={states.dtype}, action={actions.dtype}.")
    if states.shape != expected_states.shape or not np.array_equal(states, expected_states):
        raise ValueError("Exported states do not exactly match demo[t].")
    if actions.shape != expected_actions.shape or not np.array_equal(actions, expected_actions):
        raise ValueError("Exported actions do not exactly match demo[t + 1].misc.")
    if metadata.get("action_semantics") != ACTION_SEMANTICS:
        raise ValueError(f"Unexpected action_semantics: {metadata.get('action_semantics')!r}.")
    return states, actions, metadata


def select_fragments(
    demo: Any,
    states: np.ndarray,
    actions: np.ndarray,
    window_radius: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    next_joints = np.asarray(
        [
            np.r_[
                demo[t + 1].left.joint_positions,
                demo[t + 1].right.joint_positions,
            ]
            for t in range(len(demo) - 1)
        ],
        dtype=np.float32,
    )
    target_joints = np.c_[actions[:, :7], actions[:, 8:15]]
    tracking_error = np.max(
        np.abs(target_joints - next_joints),
        axis=1,
    )

    initial_command = np.asarray(
        [
            demo[0].misc["left_commanded_gripper_state"],
            demo[0].misc["right_commanded_gripper_state"],
        ],
        dtype=np.float32,
    )
    gripper_commands = actions[:, list(GRIPPER_INDICES)]
    prior_commands = np.vstack([initial_command, gripper_commands[:-1]])
    command_change = np.any(gripper_commands != prior_commands, axis=1)
    ordinary_indices = np.flatnonzero(~command_change)
    if ordinary_indices.size == 0:
        raise ValueError("Demo has no ordinary transition.")

    ordinary_errors = tracking_error[ordinary_indices]
    ordinary_index = int(ordinary_indices[np.argmin(np.abs(ordinary_errors - np.median(ordinary_errors)))])
    max_tracking_index = int(np.argmax(tracking_error))
    change_indices = np.flatnonzero(command_change).tolist()

    def window(label: str, center: int, radius: int) -> dict[str, Any]:
        return {
            "label": label,
            "center": int(center),
            "start": max(0, int(center) - radius),
            "end": min(len(actions) - 1, int(center) + radius),
        }

    fragments = [
        window("representative_ordinary", ordinary_index, 0),
        window(
            "maximum_raw_tracking_error",
            max_tracking_index,
            window_radius,
        ),
    ]
    fragments.extend(window(f"gripper_command_change_{index}", index, window_radius) for index in change_indices)
    return fragments, tracking_error


def reconstruct_fragment_start(task_env, demo: Any, start: int) -> dict[str, Any]:
    """Restore the episode and then place the robot/object at demo[start]."""
    _, reset_obs = task_env.reset_to_demo(demo)
    raw_obs = demo[start]
    robot = task_env._robot  # noqa: SLF001
    robot.release_gripper()

    robot.right_arm.set_joint_positions(
        raw_obs.right.joint_positions,
        disable_dynamics=True,
    )
    robot.left_arm.set_joint_positions(
        raw_obs.left.joint_positions,
        disable_dynamics=True,
    )
    robot.right_gripper.set_joint_positions(
        raw_obs.right.gripper_joint_positions,
        disable_dynamics=True,
    )
    robot.left_gripper.set_joint_positions(
        raw_obs.left.gripper_joint_positions,
        disable_dynamics=True,
    )

    task = task_env._task  # noqa: SLF001
    target_object = getattr(task, "target_object", None)
    if target_object is not None and raw_obs.object_6d_pose is not None:
        target_object.set_matrix(raw_obs.object_6d_pose["matrix"])

    attachment_attempted = False
    attachment_succeeded = False
    scheme = None
    if hasattr(task, "get_active_scheme"):
        scheme = task.get_active_scheme()
    if target_object is not None and scheme in ("right_grasper", "left_grasper"):
        side = "right" if scheme == "right_grasper" else "left"
        command = float(raw_obs.misc[f"{side}_commanded_gripper_state"])
        if command == 0.0:
            attachment_attempted = True
            gripper = robot.right_gripper if side == "right" else robot.left_gripper
            attachment_succeeded = bool(gripper.grasp(target_object))

    reset_joint_error = float(
        np.max(np.abs(observed_state(reset_obs)[np.r_[0:7, 8:15]] - observed_state(demo[0])[np.r_[0:7, 8:15]]))
    )
    return {
        "reset_joint_max_abs_error": reset_joint_error,
        "scheme": scheme,
        "attachment_attempted": attachment_attempted,
        "attachment_succeeded": attachment_succeeded,
    }


def replay_fragment(
    task_env,
    demo: Any,
    actions: np.ndarray,
    fragment: dict[str, Any],
    converter: Callable[..., np.ndarray],
    arm_mode: AuditedBimanualJointPosition,
    gripper_mode: AuditedBimanualDiscrete,
) -> dict[str, Any]:
    reconstruction = reconstruct_fragment_start(
        task_env,
        demo,
        fragment["start"],
    )
    transition_results = []

    for index in range(fragment["start"], fragment["end"] + 1):
        before = task_env.get_observation()
        left_first_action = actions[index]
        left_gripper = float(left_first_action[7])
        right_gripper = float(left_first_action[15])
        occ_action = converter(
            left_first_action,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )

        after, _, _ = task_env.step(occ_action)
        expected_arm_input = np.r_[
            left_first_action[8:15],
            left_first_action[:7],
        ].astype(np.float32)
        expected_gripper_input = np.asarray(
            [right_gripper, left_gripper],
            dtype=np.float32,
        )
        if arm_mode.last_action is None or gripper_mode.last_action is None:
            raise RuntimeError("Action-mode audit hooks did not capture inputs.")

        target_joints = np.r_[
            left_first_action[:7],
            left_first_action[8:15],
        ]
        before_joints = np.r_[
            before.left.joint_positions,
            before.right.joint_positions,
        ]
        after_joints = np.r_[
            after.left.joint_positions,
            after.right.joint_positions,
        ]
        raw_next_joints = np.r_[
            demo[index + 1].left.joint_positions,
            demo[index + 1].right.joint_positions,
        ]
        prior_command = np.asarray(
            [
                demo[index].misc["left_commanded_gripper_state"],
                demo[index].misc["right_commanded_gripper_state"],
            ],
            dtype=np.float32,
        )
        command = np.asarray(
            [left_gripper, right_gripper],
            dtype=np.float32,
        )
        command_changed = command != prior_command
        observed_gripper_after = np.asarray(
            [after.left.gripper_open, after.right.gripper_open],
            dtype=np.float32,
        )
        physical_gripper_before = gripper_mode.before_open_amount[[1, 0]]
        physical_gripper_after = gripper_mode.after_open_amount[[1, 0]]
        changed_physical_direction_passed = True
        for side_index in np.flatnonzero(command_changed):
            if command[side_index] == 0.0:
                changed_physical_direction_passed &= bool(
                    physical_gripper_after[side_index] <= physical_gripper_before[side_index] + 1e-4
                    and physical_gripper_after[side_index] < 0.95
                )
            else:
                changed_physical_direction_passed &= bool(
                    physical_gripper_after[side_index] >= physical_gripper_before[side_index] - 1e-4
                    and physical_gripper_after[side_index] > 0.95
                )

        before_target_error = float(np.max(np.abs(before_joints - target_joints)))
        after_target_error = float(np.max(np.abs(after_joints - target_joints)))
        result = {
            "index": int(index),
            "routing_arm_max_abs_error": float(np.max(np.abs(arm_mode.last_action - expected_arm_input))),
            "routing_gripper_max_abs_error": float(np.max(np.abs(gripper_mode.last_action - expected_gripper_input))),
            "joint_error_to_target_before": before_target_error,
            "joint_error_to_target_after": after_target_error,
            "joint_target_error_nonincreasing": bool(after_target_error <= before_target_error + 1e-4),
            "joint_max_abs_error_vs_raw_next": float(np.max(np.abs(after_joints - raw_next_joints))),
            "command_left_right": command.tolist(),
            "command_changed_left_right": command_changed.tolist(),
            "observed_gripper_after_left_right": observed_gripper_after.tolist(),
            "changed_gripper_matches_command": bool(
                np.all(observed_gripper_after[command_changed] == command[command_changed])
            ),
            "changed_gripper_physical_direction_passed": changed_physical_direction_passed,
            "physical_gripper_before_right_left": gripper_mode.before_open_amount.tolist(),
            "physical_gripper_after_right_left": gripper_mode.after_open_amount.tolist(),
            "finite_after": bool(np.isfinite(observed_state(after)).all()),
        }
        transition_results.append(result)

    return {
        **fragment,
        **reconstruction,
        "transitions": transition_results,
    }


def audit_task(
    environment: Environment,
    task_name: str,
    raw_dir: Path,
    export_dir: Path,
    converter: Callable[..., np.ndarray],
    arm_mode: AuditedBimanualJointPosition,
    gripper_mode: AuditedBimanualDiscrete,
    window_radius: int,
) -> dict[str, Any]:
    raw_episode_dir = raw_dir / f"{task_name}.train" / "all_variations" / "episodes" / "episode0"
    export_episode_dir = export_dir / "train" / task_name / "episode0"
    demo = load_pickle(raw_episode_dir / "low_dim_obs.pkl")
    variation = int(load_pickle(raw_episode_dir / "variation_number.pkl"))
    states, actions, metadata = validate_export(
        demo,
        export_episode_dir,
    )
    fragments, raw_tracking_error = select_fragments(
        demo,
        states,
        actions,
        window_radius,
    )

    task_class = task_file_to_task_class(task_name, bimanual=True)
    task_env = environment.get_task(task_class)
    task_env.set_variation(variation)
    # The collector called reset once before get_demos(). Prime the task in the
    # same way; reset_to_demo() then restores the seed captured before the
    # collector's second reset.
    task_env.reset()

    fragment_results = [
        replay_fragment(
            task_env,
            demo,
            actions,
            fragment,
            converter,
            arm_mode,
            gripper_mode,
        )
        for fragment in fragments
    ]
    transitions = [transition for fragment in fragment_results for transition in fragment["transitions"]]
    routing_passed = all(
        transition["routing_arm_max_abs_error"] == 0.0 and transition["routing_gripper_max_abs_error"] == 0.0
        for transition in transitions
    )
    finite_passed = all(transition["finite_after"] for transition in transitions)
    gripper_passed = all(
        transition["changed_gripper_matches_command"] and transition["changed_gripper_physical_direction_passed"]
        for transition in transitions
    )
    reset_passed = all(fragment["reset_joint_max_abs_error"] <= 5e-3 for fragment in fragment_results)
    target_nonincreasing_count = sum(transition["joint_target_error_nonincreasing"] for transition in transitions)
    command_change_count = sum(any(transition["command_changed_left_right"]) for transition in transitions)
    maximum_tracking_fragment = next(
        fragment for fragment in fragment_results if fragment["label"] == "maximum_raw_tracking_error"
    )

    return {
        "task": task_name,
        "variation": variation,
        "num_demo_observations": len(demo),
        "num_demo_transitions": len(actions),
        "action_semantics": metadata["action_semantics"],
        "export_time_alignment_exact": True,
        "raw_max_tracking_error": float(np.max(raw_tracking_error)),
        "raw_max_tracking_error_index": int(np.argmax(raw_tracking_error)),
        "num_replayed_transition_checks": len(transitions),
        "routing_passed": routing_passed,
        "finite_passed": finite_passed,
        "gripper_changes_passed": gripper_passed,
        "reset_reconstruction_passed": reset_passed,
        "num_gripper_command_change_checks": command_change_count,
        "joint_target_nonincreasing_count": target_nonincreasing_count,
        "joint_target_check_count": len(transitions),
        "max_tracking_fragment_abs_error_vs_raw_next": max(
            transition["joint_max_abs_error_vs_raw_next"] for transition in maximum_tracking_fragment["transitions"]
        ),
        "max_joint_abs_error_vs_raw_next": max(
            transition["joint_max_abs_error_vs_raw_next"] for transition in transitions
        ),
        "fragments": fragment_results,
        "passed": bool(routing_passed and finite_passed and gripper_passed and reset_passed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root containing <task>.train raw demo directories.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Root containing the stage-two intermediate export.",
    )
    parser.add_argument(
        "--occ-models-dir",
        type=Path,
        required=True,
        help="occ_grasp_models root containing the production OpenPI agent.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=1,
        help="Context transitions on each side of max-error/change centers.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.window_radius < 0:
        parser.error("--window-radius must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    converter = load_production_converter(args.occ_models_dir)
    arm_mode = AuditedBimanualJointPosition()
    gripper_mode = AuditedBimanualDiscrete()
    obs_config = ObservationConfig()
    obs_config.camera_configs = {}
    environment = Environment(
        action_mode=BimanualJointPositionActionMode(
            arm_mode,
            gripper_mode,
        ),
        obs_config=obs_config,
        robot_setup="dual_panda",
        headless=args.headless,
    )

    task_results = []
    environment.launch()
    try:
        for task_name in args.tasks:
            print(f"[{task_name}] replaying selected command fragments...")
            task_result = audit_task(
                environment,
                task_name,
                args.raw_dir.resolve(),
                args.export_dir.resolve(),
                converter,
                arm_mode,
                gripper_mode,
                args.window_radius,
            )
            task_results.append(task_result)
            print(
                f"[{task_name}] passed={task_result['passed']} "
                f"checks={task_result['num_replayed_transition_checks']} "
                f"raw-next-max="
                f"{task_result['max_joint_abs_error_vs_raw_next']:.6f}"
            )
    finally:
        environment.shutdown()

    report = {
        "action_semantics": ACTION_SEMANTICS,
        "raw_dir": str(args.raw_dir.resolve()),
        "export_dir": str(args.export_dir.resolve()),
        "production_converter": ("agents.openpi_policy.agent._openpi_joint16_to_occ_joint16"),
        "production_action_mode": "BimanualJointPositionActionMode",
        "window_radius": args.window_radius,
        "tasks": task_results,
        "passed": all(task["passed"] for task in task_results),
        "scope_note": (
            "Persistent 16D gripper state cannot represent repeated close "
            "events; repeated-close equivalence was not required."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Stage-three replay passed={report['passed']}; report={args.output_json.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
