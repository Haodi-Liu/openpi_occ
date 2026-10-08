import dataclasses
from typing import ClassVar

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import rlbench_timing
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import oracle_phase_sidecar as _oracle_phase_sidecar
from scripts import generate_pi05_rlbench_oracle_sidecar as _sidecar_generator


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_can_keep_final_partial_batch():
    config = pi0_config.Pi0Config(action_dim=8, action_horizon=3, max_token_len=8)
    dataset = _data_loader.FakeDataset(config, 5)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        drop_last=False,
    )

    batches = list(loader)

    assert len(batches) == 2
    assert batches[0]["actions"].shape[0] == 4
    assert batches[1]["actions"].shape[0] == 1


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMetadata:
        fps = 20
        tasks: ClassVar = {0: "overall task"}

    class FakeRawDataset:
        def __init__(self, repo_id, delta_timestamps=None):
            self.repo_id = repo_id
            self.delta_timestamps = delta_timestamps
            self.hf_dataset = object()

        def __getitem__(self, index):
            return {"task_index": 0, "payload": index}

        def __len__(self):
            return 4

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda repo_id: FakeMetadata())
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeRawDataset)


def test_subtask_fields_select_only_oracle_anchor_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lerobot(monkeypatch)
    captured = {}

    class FakeOracleView:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __getitem__(self, index):
            return {"payload": index, "prompt": "oracle"}

        def __len__(self):
            return 1

    monkeypatch.setattr(_oracle_phase_sidecar, "OraclePhaseAnchorDataset", FakeOracleView)
    data_config = _config.DataConfig(
        repo_id="local/oracle",
        prompt_from_task=True,
        subtask_annotations_dir="/sealed/sidecar",
        subtask_replan_steps=10,
    )
    model_config = pi0_config.Pi0Config(action_horizon=20)

    dataset = _data_loader.create_torch_dataset(data_config, 20, model_config)

    assert isinstance(dataset, FakeOracleView)
    assert captured["annotations_dir"] == "/sealed/sidecar"
    assert captured["action_horizon"] == 20
    assert captured["replan_steps"] == 10
    assert captured["repo_id"] == "local/oracle"
    assert captured["task_prompts"] == {0: "overall task"}
    assert isinstance(captured["dataset"], _data_loader.TransformedDataset)


def test_subtask_fields_reject_partial_pair_or_missing_task_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lerobot(monkeypatch)
    model_config = pi0_config.Pi0Config(action_horizon=20)

    with pytest.raises(ValueError, match="subtask_annotations_dir is required with the oracle sidecar"):
        _data_loader.create_torch_dataset(
            _config.DataConfig(repo_id="local/oracle", prompt_from_task=True, subtask_replan_steps=10),
            20,
            model_config,
        )
    with pytest.raises(ValueError, match="prompt_from_task is required for oracle dataset binding"):
        _data_loader.create_torch_dataset(
            _config.DataConfig(
                repo_id="local/oracle",
                subtask_annotations_dir="/sealed/sidecar",
                subtask_replan_steps=10,
            ),
            20,
            model_config,
        )


def test_b0_and_b1_share_action_chunks_but_select_different_rows_and_prompts(tmp_path, monkeypatch):
    repo_id = "local/rlbench_view_test"
    repo_home = tmp_path / "lerobot"
    monkeypatch.setattr(_data_loader.lerobot_dataset, "HF_LEROBOT_HOME", repo_home)
    cameras = ("front_rgb", "wrist_left_rgb", "wrist_right_rgb")
    features = {key: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]} for key in cameras}
    features.update({key: {"dtype": "float32", "shape": (16,), "names": None} for key in ("state", "actions")})
    recorded = _data_loader.lerobot_dataset.LeRobotDataset.create(
        repo_id, fps=20, root=repo_home / repo_id, robot_type="panda", features=features
    )
    task = "bimanual_pick_plate"
    prompt = "Pick up the plate."
    # Exclude the first episode so B1 position zero must map to a later base-repo row.
    phases_by_episode = (np.ones(44, dtype=np.int8), np.repeat([1, 2, 3, 4], [8, 12, 12, 12]).astype(np.int8))
    audits, records, action_arrays = [], [], []
    global_start = 0
    for episode_index, phases in enumerate(phases_by_episode):
        num_actions = len(phases) - 1
        states = np.repeat(np.arange(global_start, global_start + num_actions, dtype=np.float32)[:, None], 16, axis=1)
        states /= 100
        actions = states + np.float32(0.1)
        action_arrays.append(actions)
        for frame_index in range(num_actions):
            recorded.add_frame(
                {
                    **{key: np.full((8, 8, 3), global_start + frame_index, dtype=np.uint8) for key in cameras},
                    "state": states[frame_index],
                    "actions": actions[frame_index],
                    "task": prompt,
                }
            )
        recorded.save_episode()
        decision = _oracle_phase_sidecar.assess_episode(phases, num_actions, replan_steps=10)
        kwargs = {
            "source_task_name": task,
            "source_episode_number": episode_index,
            "lerobot_episode_index": episode_index,
            "global_start_index": global_start,
            "overall_instruction": prompt,
            "raw_phases": phases,
            "decision": decision,
        }
        audits.append(
            _oracle_phase_sidecar.make_episode_audit(
                **kwargs, replan_steps=10, action_semantics="executed_joint_target_commanded_gripper_effective_v2"
            )
        )
        records.extend(_oracle_phase_sidecar.make_annotation_records(**kwargs))
        global_start += num_actions
    recorded.stop_image_writer()

    annotations_dir = tmp_path / "oracle"
    manifest = _oracle_phase_sidecar.make_unsealed_manifest(
        repo_id=repo_id,
        hf_dataset_fingerprint=recorded.hf_dataset._fingerprint,  # noqa: SLF001
        num_dataset_rows=global_start,
        action_horizon=40,
        replan_steps=10,
        action_horizon_source=rlbench_timing.ACTION_HORIZON_SOURCE,
        replan_steps_source=rlbench_timing.REPLAN_STEPS_SOURCE,
        episodes=audits,
        records=records,
    )
    _sidecar_generator._write_sealed_sidecar(  # noqa: SLF001
        annotations_dir, records, _oracle_phase_sidecar.make_quality_report(audits), manifest
    )

    base = _config.get_config("pi05_rlbench")
    b0_config = dataclasses.replace(base.data.base_config, repo_id=repo_id)
    b1_config = dataclasses.replace(b0_config, subtask_annotations_dir=str(annotations_dir), subtask_replan_steps=10)
    b0 = _data_loader.create_torch_dataset(b0_config, 40, base.model)
    b1 = _data_loader.create_torch_dataset(b1_config, 40, base.model)

    assert len(b0) == 86
    assert len(b1) == 43
    assert b0[0]["prompt"] == prompt
    assert b0[0]["episode_index"].item() == 0
    for position, subtask_type in ((0, "1"), (8, "1_to_2"), (42, "4")):
        sample = b1[position]
        base_sample = b0[43 + position]
        assert sample["index"].item() == 43 + position
        assert sample["prompt"] == _oracle_phase_sidecar.SUBTASK_TEXTS[task][subtask_type]
        assert base_sample["prompt"] == prompt
        for key in (*cameras, "state", "actions", "actions_is_pad"):
            np.testing.assert_array_equal(sample[key], base_sample[key])
        expected_rows = np.minimum(np.arange(position, position + 40), 42)
        np.testing.assert_array_equal(sample["actions"], action_arrays[1][expected_rows])
        np.testing.assert_array_equal(sample["actions_is_pad"], np.arange(position, position + 40) > 42)
        assert not {"source_task_name", "raw_phase_before", "clean_phase", "subtask_type", "subtask"}.intersection(
            sample
        )
