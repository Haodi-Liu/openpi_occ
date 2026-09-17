from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
import pathlib
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
from openpi.shared import checkpoint_fingerprint
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.oracle_phase_sidecar as oracle
import openpi.training.utils as training_utils


def validate_local_orbax_step(
    checkpoint_dir: pathlib.Path | str,
    *,
    require_train_state: bool,
) -> pathlib.Path:
    """Validate a finalized local composite Orbax step without restoring arrays."""
    step_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    if not step_dir.is_dir():
        raise FileNotFoundError(f"Orbax checkpoint step directory not found: {step_dir}")
    if not step_dir.name.isdigit():
        raise ValueError(f"Orbax checkpoint step directory must have a numeric name: {step_dir}")
    if not ocp.utils.is_checkpoint_finalized(step_dir):
        raise ValueError(f"Orbax checkpoint step is not finalized: {step_dir}")

    step_metadata_path = step_dir / "_CHECKPOINT_METADATA"
    if not step_metadata_path.is_file():
        raise FileNotFoundError(f"Orbax step metadata not found: {step_metadata_path}")
    try:
        step_metadata = json.loads(step_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Orbax step metadata JSON: {step_metadata_path}") from exc
    if not isinstance(step_metadata, dict):
        raise ValueError(f"Orbax step metadata must be a JSON object: {step_metadata_path}")

    commit_timestamp = step_metadata.get("commit_timestamp_nsecs")
    if isinstance(commit_timestamp, bool) or not isinstance(commit_timestamp, int) or commit_timestamp <= 0:
        raise ValueError(f"Orbax checkpoint step has no valid commit timestamp: {step_dir}")

    required_items = {"assets", "params"}
    if require_train_state:
        required_items.add("train_state")
    item_handlers = step_metadata.get("item_handlers")
    if not isinstance(item_handlers, dict) or not required_items.issubset(item_handlers):
        missing = sorted(required_items - set(item_handlers if isinstance(item_handlers, dict) else ()))
        raise ValueError(f"Orbax checkpoint step is missing required items {missing}: {step_dir}")

    if not (step_dir / "assets").is_dir():
        raise FileNotFoundError(f"Orbax checkpoint assets item not found: {step_dir / 'assets'}")
    pytree_items = ("params", "train_state") if require_train_state else ("params",)
    for item in pytree_items:
        item_dir = step_dir / item
        if not item_dir.is_dir() or not (item_dir / "_METADATA").is_file():
            raise FileNotFoundError(f"Orbax checkpoint {item} item is incomplete: {item_dir}")

    return step_dir


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    # Fingerprint the same inference params that are exported (EMA when enabled).
    with at.disable_typechecking():
        train_state, params = _split_params(state)

    data_config = data_loader.data_config()
    provenance = data_config.checkpoint_provenance
    checkpoint_provenance = None
    if provenance is not None:
        # Training checks still use the current, unaveraged parameters.
        frozen_vlm_fingerprint = checkpoint_fingerprint.fingerprint_frozen_vlm(state.params)
        action_fingerprint = checkpoint_fingerprint.fingerprint_trainable_action(state.params)
        checkpoint_provenance = {
            **provenance,
            "checkpoint_step": step,
            "checkpoint_action_fingerprint": (
                checkpoint_fingerprint.fingerprint_trainable_action(params)
                if state.ema_params is not None
                else action_fingerprint
            ),
        }
        protocol = provenance.get("action_trainable_protocol")
        if protocol == oracle.ACTION_TRAINABLE_PROTOCOL:
            if frozen_vlm_fingerprint != provenance["source_vlm_fingerprint"]:
                raise ValueError("Frozen source VLM changed during action training; refusing to save.")
            if action_fingerprint == provenance["source_action_fingerprint"]:
                raise ValueError("No allowlisted action parameter changed during training; refusing to save.")
        elif protocol == oracle.FULL_MODEL_TRAINABLE_PROTOCOL:
            checkpoint_provenance["checkpoint_vlm_fingerprint"] = (
                checkpoint_fingerprint.fingerprint_frozen_vlm(params)
                if state.ema_params is not None
                else frozen_vlm_fingerprint
            )
            logging.info(
                "Full-model checkpoint audit: vlm_changed=%s, action_changed=%s.",
                frozen_vlm_fingerprint != provenance["source_vlm_fingerprint"],
                action_fingerprint != provenance["source_action_fingerprint"],
            )
        else:
            raise ValueError(f"Unsupported action_trainable_protocol: {protocol!r}.")

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        if checkpoint_provenance is not None:
            checkpoint_fingerprint.write_checkpoint_provenance(directory, checkpoint_provenance)

    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
