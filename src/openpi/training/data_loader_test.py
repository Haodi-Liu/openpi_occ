import dataclasses
from typing import ClassVar

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import oracle_phase_sidecar as _oracle_phase_sidecar


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
