"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import json
import math
import pathlib

from lerobot.common.datasets import lerobot_dataset
import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
from openpi.shared import checkpoint_fingerprint
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        if max_frames < 1:
            raise ValueError("max_frames must be positive when provided.")
        dataset = torch.utils.data.Subset(dataset, range(max_frames))
        num_batches = math.ceil(max_frames / batch_size)
        shuffle = True
        drop_last = False
    else:
        num_batches = math.ceil(len(dataset) / batch_size)
        shuffle = False
        drop_last = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        drop_last=drop_last,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def update_action_stats(stats: normalize.RunningStats, batch: dict) -> None:
    actions = np.asarray(batch["actions"])
    if "actions_is_pad" not in batch:
        stats.update(actions)
        return

    actions_is_pad = np.asarray(batch["actions_is_pad"], dtype=np.bool_)
    if actions_is_pad.shape != actions.shape[:-1]:
        raise ValueError(
            "Expected actions_is_pad shape to match actions without the action dimension, "
            f"got {actions_is_pad.shape} vs {actions.shape[:-1]}."
        )

    valid_actions = actions[~actions_is_pad]
    if len(valid_actions) == 0:
        raise ValueError("actions_is_pad masked out every action timestep in a norm-stats batch.")
    stats.update(valid_actions)


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    asset_id: str | None = None,
    assets_base_dir: str | None = None,
    output_assets_dir: str | None = None,
    batch_size: int | None = None,
):
    config = _config.get_config(config_name)
    if repo_id is not None or asset_id is not None:
        data = config.data
        if repo_id is not None:
            data = dataclasses.replace(data, repo_id=repo_id)
        if asset_id is not None:
            data = dataclasses.replace(data, assets=dataclasses.replace(data.assets, asset_id=asset_id))
        config = dataclasses.replace(config, data=data)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    num_samples_processed = 0

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        num_samples_processed += int(np.asarray(batch["state"]).shape[0])
        stats["state"].update(np.asarray(batch["state"]))
        update_action_stats(stats["actions"], batch)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_id = data_config.asset_id or data_config.repo_id
    if output_id is None:
        raise ValueError("Data config must have an asset_id or repo_id")

    output_root = pathlib.Path(output_assets_dir).resolve() if output_assets_dir is not None else config.assets_dirs
    output_path = output_root / output_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    if data_config.rlds_data_dir is None and data_config.repo_id not in {None, "fake"}:
        raw_dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
        dataset_fingerprint = getattr(raw_dataset.hf_dataset, "_fingerprint", None)
        if dataset_fingerprint is None:
            raise ValueError("HuggingFace dataset fingerprint is unavailable; cannot seal norm provenance.")
        provenance = checkpoint_fingerprint.seal_provenance(
            {
                "schema_version": 1,
                "repo_id": data_config.repo_id,
                "hf_dataset_fingerprint": str(dataset_fingerprint),
                "action_horizon": config.model.action_horizon,
                "num_dataset_rows": len(raw_dataset),
                "num_samples_processed": num_samples_processed,
                "complete_dataset": max_frames is None and num_samples_processed == len(raw_dataset),
                "norm_stats_sha256": checkpoint_fingerprint.sha256_file(output_path / "norm_stats.json"),
            }
        )
        provenance_path = output_path / "norm_stats_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Writing norm provenance to: {provenance_path}")


if __name__ == "__main__":
    tyro.cli(main)
