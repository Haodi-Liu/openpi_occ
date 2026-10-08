import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from examples.rlbench import build_rlbench_lerobot as builder


def make_effective_episode(episode_dir: Path, offset: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    episode_dir.mkdir(parents=True)
    observation_rows = np.array([1, 2, 5, 6])
    command_rows = np.array([0, 2, 4, 6])
    raw_states = np.arange(7 * 16, dtype=np.float32).reshape(7, 16) / 100 + offset
    raw_states[:, [7, 15]] = 0.4
    raw_commands = -raw_states.copy()
    raw_commands[:, [7, 15]] = 1.0
    states = raw_states[observation_rows[:-1]]
    actions = raw_commands[command_rows[1:]]
    np.save(episode_dir / "state.npy", states)
    np.save(episode_dir / "actions.npy", actions)
    np.savez(
        episode_dir / "effective_index.npz",
        raw_observation_row=observation_rows,
        raw_command_row=command_rows,
        physics_step=np.array([0, 2, 9, 10]),
        command_kind=np.array(["initial", "step", "step", "step"]),
    )
    metadata = {
        "task": "pick the plate",
        "action_semantics": builder.EFFECTIVE_ACTION_SEMANTICS,
        "num_observations": 4,
        "num_transitions": 3,
        "effective_index_file": "effective_index.npz",
    }
    for camera_index, key in enumerate(builder.IMAGE_FEATURES):
        paths = []
        for raw_row in observation_rows:
            path = episode_dir / f"{key}_{raw_row}.png"
            pixels = np.full((256, 256, 3), offset + camera_index * 20 + raw_row, dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
        metadata[key] = paths[:-1]
    (episode_dir / "meta.json").write_text(json.dumps(metadata))
    return states, actions, metadata


def test_effective_export_roundtrip_preserves_rows_and_logical_timing(tmp_path):
    export_root = tmp_path / "export"
    task_dir = export_root / "train" / "bimanual_pick_plate"
    episode2 = make_effective_episode(task_dir / "episode2", offset=60)
    episode10 = make_effective_episode(task_dir / "episode10", offset=10)
    output_root = tmp_path / "lerobot"
    repo_id = "local/effective_test"
    builder.main(
        input_dir=str(export_root),
        split="train",
        repo_id=repo_id,
        output_root=str(output_root),
        image_writer_threads=0,
        image_writer_processes=0,
    )

    dataset = builder.LeRobotDataset(
        repo_id,
        root=output_root / repo_id,
        delta_timestamps={"actions": [0.0, 0.05, 0.1]},
    )
    assert len(dataset) == 6  # Three transitions per episode; no terminal row or gap expansion.
    for episode_index, (states, actions, metadata) in enumerate((episode10, episode2)):
        for frame_index in range(3):
            row = dataset.hf_dataset[episode_index * 3 + frame_index]
            assert row["episode_index"].item() == episode_index
            assert row["frame_index"].item() == frame_index
            assert row["timestamp"].item() == pytest.approx(frame_index / 20)
            np.testing.assert_array_equal(row["state"].numpy(), states[frame_index])
            np.testing.assert_array_equal(row["actions"].numpy(), actions[frame_index])
            for key in builder.IMAGE_FEATURES:
                expected = builder.load_rgb(Path(metadata[key][frame_index]))
                actual = row[key].numpy().transpose(1, 2, 0) * 255
                np.testing.assert_allclose(actual, expected, atol=1e-5)
        sample = dataset[episode_index * 3]
        assert sample["task"] == metadata["task"]
        np.testing.assert_array_equal(sample["actions"].numpy(), actions)


@pytest.mark.parametrize(
    "other_semantics", [builder.RAW_ACTION_SEMANTICS, "executed_joint_target_commanded_gripper_effective_v1"]
)
def test_rejects_incompatible_semantics_before_creating_repo(tmp_path, other_semantics):
    task_dir = tmp_path / "export" / "train" / "bimanual_pick_plate"
    make_effective_episode(task_dir / "episode0")
    _, _, metadata = make_effective_episode(task_dir / "episode1")
    metadata["action_semantics"] = other_semantics
    (task_dir / "episode1" / "meta.json").write_text(json.dumps(metadata))
    output_root = tmp_path / "lerobot"
    with pytest.raises(ValueError, match="Cannot mix action_semantics"):
        builder.main(str(tmp_path / "export"), "train", "local/test", output_root=str(output_root))
    assert not output_root.exists()


def test_effective_index_rejects_zero_step_boundary(tmp_path):
    episode_dir = tmp_path / "episode0"
    make_effective_episode(episode_dir)
    index_path = episode_dir / "effective_index.npz"
    with np.load(index_path) as index:
        arrays = dict(index)
    arrays["physics_step"] = np.array([0, 2, 2, 10])
    np.savez(index_path, **arrays)
    with pytest.raises(ValueError, match="strictly increase"):
        builder.validate_episode_artifacts(episode_dir)


def test_existing_repository_is_preserved(tmp_path):
    repo_root = tmp_path / "local" / "existing"
    repo_root.mkdir(parents=True)
    sentinel = repo_root / "keep.txt"
    sentinel.write_text("existing data")
    with pytest.raises(FileExistsError, match="Dataset already exists"):
        builder.create_dataset("local/existing", 20, output_root=str(tmp_path))
    assert sentinel.read_text() == "existing data"
