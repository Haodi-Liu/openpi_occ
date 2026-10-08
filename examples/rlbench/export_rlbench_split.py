"""Export RLBench raw demos into an intermediate numpy/json format.

This script is intended to run in an environment where `import rlbench` works
so that RLBench's pickled Demo/BimanualObservation objects can be deserialized.

The output format is intentionally simple:

<output_dir>/
  <split>/
    <task>/
      episode0/
        state.npy
        actions.npy
        phase_before_action.npy
        phase_after_action.npy
        meta.json
        effective_index.npz  # Only with --effective_commands.

Where:
- `T` is the number of observation nodes: raw frames by default, or recorded
  boundaries after physical advancement (including the initial state) with `--effective_commands`.
- `state.npy` has shape `(T - 1, 16)` and stores the current-step
  left-first joint state:
  `[left_joint_positions7, left_gripper, right_joint_positions7, right_gripper]`.
- `actions.npy` has shape `(T - 1, 16)` and stores the command applied before
  the next observation:
  `[left_joint_target7, left_commanded_gripper_state,
    right_joint_target7, right_commanded_gripper_state]`.
- `phase_before_action.npy` and `phase_after_action.npy` have shape `(T - 1,)`
  and store the integer phase labels from the observations immediately before
  and after each action.
- `meta.json` stores the language instruction, variation number, and absolute
  image paths for the three RGB cameras used by the downstream pipeline.
- `effective_index.npz` maps all effective nodes to their raw observation and
  command rows, physics steps, and command kinds. Zero-step groups use their
  last observation row and first command row. Every transition advances physical
  time; its physics-step increment may exceed one. No intermediate frames are added.
- `export_report.json` at the output root lists exported counts and episodes
  with ambiguous timing. Those episodes are not exported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np

try:
    import rlbench  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised in real environment
    raise SystemExit(
        "This script must run in an environment where `import rlbench` works so RLBench pickles can be deserialized."
    ) from exc


CAMERA_DIRS = ("front_rgb", "wrist_left_rgb", "wrist_right_rgb")
EPISODE_RE = re.compile(r"episode(\d+)$")
ACTION_SEMANTICS = "executed_joint_target_commanded_gripper"
EFFECTIVE_ACTION_SEMANTICS = "executed_joint_target_commanded_gripper_effective_v2"
PHASE_VALUES = frozenset({1, 2, 3, 4})
PHASE_SEMANTICS = "observation_phase_type_before_after_action"


class TimingAmbiguityError(ValueError):
    """Raw observations cannot be ordered into recorded command boundaries."""


def _extract_gripper_scalar(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (1,):
        raise ValueError(f"Expected {name} to contain exactly one value, got shape {array.shape}.")
    return array


def extract_joint8(arm_obs: Any) -> np.ndarray:
    """Build one arm's `[joint_positions7, gripper_open]` vector."""
    joints = np.asarray(arm_obs.joint_positions, dtype=np.float32).reshape(-1)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError(f"Expected finite joint_positions with shape (7,), got {joints!r}.")

    gripper = _extract_gripper_scalar(arm_obs.gripper_open, "gripper_open")
    if not np.isfinite(gripper).all():
        raise ValueError(f"Expected finite gripper_open, got {gripper!r}.")

    joint8 = np.concatenate([joints, gripper], axis=0)
    if joint8.shape != (8,):
        raise ValueError(f"Expected joint8 with shape (8,), got {joint8.shape}.")
    return joint8.astype(np.float32, copy=False)


def extract_state(obs: Any) -> np.ndarray:
    """Build the 16D left-first observed joint state."""
    state = np.concatenate(
        [
            extract_joint8(obs.left),
            extract_joint8(obs.right),
        ],
        axis=0,
    )
    if state.shape != (16,):
        raise ValueError(f"Expected a 16D joint state vector, got shape {state.shape}.")
    return state.astype(np.float32, copy=False)


def extract_action(misc: dict[str, Any]) -> np.ndarray:
    """Build the 16D left-first persistent command action."""
    values = []
    for side in ("left", "right"):
        target_key = f"{side}_executed_demo_joint_position_action"
        gripper_key = f"{side}_commanded_gripper_state"
        if target_key not in misc or gripper_key not in misc:
            raise ValueError(f"Missing {side} action command.")

        target = np.asarray(misc[target_key], dtype=np.float32)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"Invalid {target_key}: {target!r}.")

        gripper = _extract_gripper_scalar(misc[gripper_key], gripper_key)
        if not np.isfinite(gripper).all() or float(gripper[0]) not in (0.0, 1.0):
            raise ValueError(f"Invalid {gripper_key}: {gripper!r}.")
        values.extend((target, gripper))

    return np.concatenate(values).astype(np.float32, copy=False)


def extract_phase(misc: dict[str, Any]) -> int:
    """Read one strict integer phase label from an observation."""
    if "phase_type" not in misc:
        raise ValueError("Missing phase_type.")
    value = np.asarray(misc["phase_type"])
    scalar = value.reshape(-1)[0] if value.size == 1 else None
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, np.integer)):
        raise ValueError("phase_type must be one scalar integer in 1..4.")
    phase = int(scalar)
    if phase not in PHASE_VALUES:
        raise ValueError(f"Invalid phase_type: {misc['phase_type']!r}.")
    return phase


def build_effective_index(demo: Any) -> dict[str, np.ndarray]:
    """Keep every recorded boundary with physical advancement; merge zero-step records."""
    values = [obs.misc.get("physics_step") for obs in demo]
    if len(values) < 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values
    ):
        raise TimingAmbiguityError("Missing integer physics_step.")
    steps = np.asarray(values, dtype=np.int64)
    if steps[0] != 0 or np.any(np.diff(steps) < 0):
        raise TimingAmbiguityError("Invalid physics-step sequence.")
    commands = np.stack([extract_action(obs.misc) for obs in demo])
    obs_rows, command_rows, kinds = [0], [0], ["initial"]
    for raw in range(1, len(demo)):
        previous = command_rows[-1]
        delta = int(steps[raw] - steps[obs_rows[-1]])
        if delta == 0:
            if not np.array_equal(commands[raw], commands[previous]):
                raise TimingAmbiguityError(f"Different commands at one physics step: {raw}.")
            obs_rows[-1] = raw  # Keep the last observation, including phase updates.
            continue
        obs_rows.append(raw)
        command_rows.append(raw)
        kinds.append("step")
    if len(obs_rows) < 2:
        raise TimingAmbiguityError("No physical command.")
    return {
        "raw_observation_row": np.asarray(obs_rows, dtype=np.int64),
        "raw_command_row": np.asarray(command_rows, dtype=np.int64),
        "physics_step": steps[obs_rows],
        "command_kind": np.asarray(kinds),  # initial / step
    }


def extract_transitions(
    demo: Any, index: dict[str, np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align state/phase-before with the command and phase-after."""
    if len(demo) < 2:
        raise ValueError("Demo must contain at least two observations.")
    obs_rows = np.arange(len(demo)) if index is None else index["raw_observation_row"]
    command_rows = np.arange(len(demo)) if index is None else index["raw_command_row"]
    states = np.asarray(
        [extract_state(demo[int(row)]) for row in obs_rows[:-1]],
        dtype=np.float32,
    )
    actions = np.asarray(
        [extract_action(demo[int(row)].misc) for row in command_rows[1:]],
        dtype=np.float32,
    )
    phase_before = np.asarray(
        [extract_phase(demo[int(row)].misc) for row in obs_rows[:-1]],
        dtype=np.int8,
    )
    phase_after = np.asarray(
        [extract_phase(demo[int(row)].misc) for row in obs_rows[1:]],
        dtype=np.int8,
    )
    if not np.array_equal(phase_before[1:], phase_after[:-1]):
        raise ValueError("Inconsistent consecutive phase labels.")
    return states, actions, phase_before, phase_after


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = EPISODE_RE.match(path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def normalize_task_prompt(descriptions: Any, task_name: str) -> str:
    """Extract a single language instruction string from variation_descriptions.pkl."""
    if isinstance(descriptions, bytes):
        return descriptions.decode("utf-8")
    if isinstance(descriptions, str):
        return descriptions
    if isinstance(descriptions, np.ndarray):
        descriptions = descriptions.tolist()
    if isinstance(descriptions, (list, tuple)):
        if not descriptions:
            return task_name
        first = descriptions[0]
        if isinstance(first, bytes):
            return first.decode("utf-8")
        return str(first)
    return str(descriptions)


def coerce_int(value: Any) -> int:
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def build_image_path(ep_dir: Path, camera_dir: str, step_idx: int) -> Path:
    image_path = (ep_dir / camera_dir / f"rgb_{step_idx:04d}.png").resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image for step {step_idx}: {image_path}")
    return image_path


def export_episode(ep_dir: Path, out_dir: Path, task_name: str, *, effective_commands: bool = False) -> dict[str, Any]:
    """Export one RLBench episode into the intermediate format."""
    episode_match = EPISODE_RE.fullmatch(ep_dir.name)
    if episode_match is None:
        raise ValueError(f"Invalid source episode name: {ep_dir.name!r}.")

    demo = load_pickle(ep_dir / "low_dim_obs.pkl")
    descriptions = load_pickle(ep_dir / "variation_descriptions.pkl")
    variation_number = load_pickle(ep_dir / "variation_number.pkl")

    if len(demo) < 2:
        raise ValueError(f"Episode {ep_dir} has only {len(demo)} observations; need at least 2.")

    task_prompt = normalize_task_prompt(descriptions, task_name)

    index = build_effective_index(demo) if effective_commands else None
    states, actions, phase_before, phase_after = extract_transitions(demo, index)
    obs_rows = np.arange(len(demo)) if index is None else index["raw_observation_row"]
    image_paths: dict[str, list[str]] = {camera_dir: [] for camera_dir in CAMERA_DIRS}

    for step_idx in range(len(states)):
        for camera_dir in CAMERA_DIRS:
            image_paths[camera_dir].append(str(build_image_path(ep_dir, camera_dir, int(obs_rows[step_idx]))))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "state.npy", states)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "phase_before_action.npy", phase_before)
    np.save(out_dir / "phase_after_action.npy", phase_after)
    if index is not None:
        np.savez(out_dir / "effective_index.npz", **index)

    metadata = {
        "task": task_prompt,
        "source_task_name": task_name,
        "source_episode_number": int(episode_match.group(1)),
        "variation_number": coerce_int(variation_number),
        "action_semantics": EFFECTIVE_ACTION_SEMANTICS if index is not None else ACTION_SEMANTICS,
        "phase_semantics": PHASE_SEMANTICS,
        "source_episode_dir": str(ep_dir.resolve()),
        "num_observations": len(obs_rows),
        "raw_num_observations": len(demo),
        "effective_index_file": "effective_index.npz" if index is not None else None,
        "num_transitions": len(states),
        "front_rgb": image_paths["front_rgb"],
        "wrist_left_rgb": image_paths["wrist_left_rgb"],
        "wrist_right_rgb": image_paths["wrist_right_rgb"],
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    return {
        "episode": ep_dir.name,
        "num_observations": len(obs_rows),
        "num_transitions": len(states),
        "task_prompt": task_prompt,
    }


def export_task(
    data_dir: Path, split: str, output_dir: Path, task_name: str, *, effective_commands: bool = False
) -> dict[str, Any]:
    task_episode_root = data_dir / f"{task_name}.{split}" / "all_variations" / "episodes"
    if not task_episode_root.is_dir():
        raise FileNotFoundError(f"Task split directory not found: {task_episode_root}")

    episode_dirs = sorted(
        [path for path in task_episode_root.iterdir() if path.is_dir() and path.name.startswith("episode")],
        key=episode_sort_key,
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under: {task_episode_root}")

    task_output_root = output_dir / split / task_name
    total_transitions = 0
    exported_count = 0
    pending = []
    print(f"[{task_name}] exporting {len(episode_dirs)} episodes from {task_episode_root}")

    for episode_dir in episode_dirs:
        try:
            summary = export_episode(
                episode_dir, task_output_root / episode_dir.name, task_name, effective_commands=effective_commands
            )
        except TimingAmbiguityError as exc:
            pending.append({"source_episode_dir": str(episode_dir.resolve()), "reason": str(exc)})
            print(f"[{task_name}] pending {episode_dir.name}: {exc}")
            continue
        exported_count += 1
        total_transitions += summary["num_transitions"]

    print(f"[{task_name}] done: {exported_count} episodes, {total_transitions} transitions, {len(pending)} pending")
    return {
        "task_name": task_name,
        "num_episodes": exported_count,
        "pending": pending,
        "num_transitions": total_transitions,
        "output_dir": str(task_output_root.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Root directory containing <task>.<split> folders.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--effective_commands",
        action="store_true",
        help="Keep recorded command boundaries and merge zero-step duplicates.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Root output directory for the exported intermediate format."
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names without the .train/.val/.test suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    task_summaries = [
        export_task(data_dir, args.split, output_dir, task_name, effective_commands=args.effective_commands)
        for task_name in args.tasks
    ]

    total_episodes = sum(item["num_episodes"] for item in task_summaries)
    total_transitions = sum(item["num_transitions"] for item in task_summaries)
    report = {
        "split": args.split,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "tasks": task_summaries,
        "total_episodes": total_episodes,
        "total_transitions": total_transitions,
        "total_pending_episodes": sum(len(item["pending"]) for item in task_summaries),
    }
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    (output_dir / "export_report.json").write_text(report_json + "\n")
    print(report_json)


if __name__ == "__main__":
    main()
