"""Build a LeRobot dataset from the intermediate RLBench joint export format.

Expected input layout (produced by `examples/rlbench/export_rlbench_split.py`):

<input_dir>/
  <split>/
    <task>/
      episode0/
        state.npy
        actions.npy
        meta.json
        effective_index.npz  # effective_v2 exports only

This script writes a LeRobot dataset that can be consumed by the OpenPI
training pipeline. The vector layout is the 16D left-first joint-control layout:

    [left_joint_positions(7), left_gripper_open(1),
     right_joint_positions(7), right_gripper_open(1)]

Each exported transition becomes one LeRobot frame: its mapped current images
and measured state, plus the next retained node's commanded action. Effective
physics_step gaps are not resampled; fps defines logical row timestamps only.
Stage cleanup and training-row selection belong to the later oracle sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro

IMAGE_FEATURES = {
    "front_rgb": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_left_rgb": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_right_rgb": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
}

JOINT16_DIM_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper_open",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper_open",
]

VECTOR_FEATURES = {
    "state": {
        "dtype": "float32",
        "shape": (16,),
        "names": [JOINT16_DIM_NAMES],
    },
    "actions": {
        "dtype": "float32",
        "shape": (16,),
        "names": [JOINT16_DIM_NAMES],
    },
}

JOINT16_GRIPPER_IDXS = (7, 15)
GRIPPER_VALUE_ATOL = 1e-5
RAW_ACTION_SEMANTICS = "executed_joint_target_commanded_gripper"
EFFECTIVE_ACTION_SEMANTICS = "executed_joint_target_commanded_gripper_effective_v2"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape != (256, 256, 3):
        raise ValueError(f"Expected image shape (256, 256, 3), got {array.shape} for {path}")
    return array


def sorted_episode_dirs(split_dir: Path) -> list[Path]:
    # The oracle sidecar uses this same lexical order to recover episode_index.
    return sorted(
        [path for path in split_dir.glob("*/*") if path.is_dir()],
        key=lambda path: (path.parent.name, path.name),
    )


def validate_export_semantics(episode_dirs: list[Path]) -> str:
    """Reject mixed or obsolete exports before creating the output repository."""
    semantics = {
        json.loads((episode_dir / "meta.json").read_text()).get("action_semantics") for episode_dir in episode_dirs
    }
    if len(semantics) != 1:
        raise ValueError(f"Cannot mix action_semantics in one LeRobot repository: {semantics}")
    action_semantics = semantics.pop()
    if action_semantics not in (RAW_ACTION_SEMANTICS, EFFECTIVE_ACTION_SEMANTICS):
        raise ValueError(f"Unsupported action_semantics: {action_semantics!r}; regenerate the export")
    return action_semantics


def validate_effective_index(episode_dir: Path, metadata: dict, num_frames: int) -> None:
    """Check the effective node count and timing without expanding physical gaps."""
    num_nodes = num_frames + 1
    if metadata.get("num_transitions") != num_frames or metadata.get("num_observations") != num_nodes:
        raise ValueError(f"Effective export must have {num_frames} transitions and {num_nodes} nodes: {episode_dir}")
    index_file = metadata.get("effective_index_file")
    if not index_file:
        raise ValueError(f"Missing effective_index_file in {episode_dir / 'meta.json'}")
    with np.load(episode_dir / index_file, allow_pickle=False) as index:
        for key in ("raw_observation_row", "raw_command_row", "physics_step"):
            values = index[key]
            if values.shape != (num_nodes,) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"Expected {key} to contain {num_nodes} integer node entries: {episode_dir}")
        physics_step = index["physics_step"]
        if physics_step[0] != 0 or np.any(physics_step[1:] <= physics_step[:-1]):
            raise ValueError(f"Effective physics_step must start at 0 and strictly increase: {episode_dir}")


def validate_joint16_array(array: np.ndarray, path: Path) -> None:
    """Validate the joint16 contract expected from export_rlbench_split.py."""
    if array.ndim != 2 or array.shape[1] != 16:
        raise ValueError(f"Expected {path.name} shape (T, 16), got {array.shape} in {path.parent}")
    if len(array) == 0:
        raise ValueError(f"Expected at least one transition in {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Found non-finite values in {path}")

    gripper_values = array[:, JOINT16_GRIPPER_IDXS]
    if gripper_values.min() < -GRIPPER_VALUE_ATOL or gripper_values.max() > 1.0 + GRIPPER_VALUE_ATOL:
        raise ValueError(
            f"Expected gripper values in {path} to be within [0, 1], "
            f"got min {float(gripper_values.min()):.6f}, max {float(gripper_values.max()):.6f}."
        )


def validate_episode_artifacts(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    state_path = episode_dir / "state.npy"
    actions_path = episode_dir / "actions.npy"
    meta_path = episode_dir / "meta.json"

    if not state_path.is_file():
        raise FileNotFoundError(f"Missing state.npy: {state_path}")
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions.npy: {actions_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    states = np.asarray(np.load(state_path, allow_pickle=False), dtype=np.float32)
    actions = np.asarray(np.load(actions_path, allow_pickle=False), dtype=np.float32)
    metadata = json.loads(meta_path.read_text())

    validate_joint16_array(states, state_path)
    validate_joint16_array(actions, actions_path)
    if len(states) != len(actions):
        raise ValueError(f"state/action length mismatch in {episode_dir}: {len(states)} vs {len(actions)}")

    for key in ("front_rgb", "wrist_left_rgb", "wrist_right_rgb"):
        if key not in metadata:
            raise KeyError(f"Missing '{key}' in {meta_path}")
        if len(metadata[key]) != len(states):
            raise ValueError(
                f"Metadata image count mismatch for {key} in {episode_dir}: {len(metadata[key])} vs {len(states)}"
            )

    if "task" not in metadata:
        raise KeyError(f"Missing 'task' in {meta_path}")

    if metadata["action_semantics"] == EFFECTIVE_ACTION_SEMANTICS:
        validate_effective_index(episode_dir, metadata, len(states))

    return states, actions, metadata


def create_dataset(
    repo_id: str,
    fps: int,
    *,
    output_root: str | None = None,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
) -> tuple[LeRobotDataset, Path]:
    dataset_root = (Path(output_root).expanduser().resolve() / repo_id) if output_root else (HF_LEROBOT_HOME / repo_id)
    if dataset_root.exists():
        raise FileExistsError(f"Dataset already exists: {dataset_root}. Choose a new repo_id or output_root.")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=fps,
        features={**IMAGE_FEATURES, **VECTOR_FEATURES},
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    return dataset, dataset_root


def main(
    input_dir: str,
    split: str,
    repo_id: str,
    fps: int = 20,
    output_root: str | None = None,
    push_to_hub: bool = False,  # noqa: FBT001, FBT002 - Tyro CLI option.
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
) -> None:
    input_root = Path(input_dir).expanduser().resolve()
    split_dir = input_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    episode_dirs = sorted_episode_dirs(split_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under: {split_dir}")
    action_semantics = validate_export_semantics(episode_dirs)

    dataset, dataset_root = create_dataset(
        repo_id,
        fps,
        output_root=output_root,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    total_frames = 0
    task_names = set()
    try:
        for episode_dir in episode_dirs:
            states, actions, metadata = validate_episode_artifacts(episode_dir)
            task_prompt = str(metadata["task"])
            task_names.add(task_prompt)

            for idx in range(len(states)):
                dataset.add_frame(
                    {
                        "front_rgb": load_rgb(Path(metadata["front_rgb"][idx])),
                        "wrist_left_rgb": load_rgb(Path(metadata["wrist_left_rgb"][idx])),
                        "wrist_right_rgb": load_rgb(Path(metadata["wrist_right_rgb"][idx])),
                        "state": states[idx],
                        "actions": actions[idx],
                        "task": task_prompt,
                    }
                )
            dataset.save_episode()
            total_frames += len(states)
    finally:
        dataset.stop_image_writer()

    if push_to_hub:
        dataset.push_to_hub(tags=["rlbench", "bimanual", "joint-control"], private=False)

    print(
        json.dumps(
            {
                "input_dir": str(input_root),
                "split": split,
                "repo_id": repo_id,
                "dataset_root": str(dataset_root),
                "num_episodes": len(episode_dirs),
                "num_frames": total_frames,
                "num_tasks": len(task_names),
                "action_semantics": action_semantics,
                "vector_layout": JOINT16_DIM_NAMES,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
