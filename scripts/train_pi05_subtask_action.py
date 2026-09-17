"""Train the π0.5 action path on deterministic oracle-phase text."""

from __future__ import annotations

import dataclasses
import gc
import importlib
import importlib.metadata
import json
import logging
import pathlib
import re
from typing import Any, Literal

import flax.nnx as nnx
import jax
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import rlbench_policy
from openpi.shared import checkpoint_fingerprint
from openpi.shared import nnx_utils
from openpi.shared import rlbench_timing
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import oracle_phase_sidecar as oracle
from openpi.training import weight_loaders

_train = importlib.import_module("scripts.train" if __package__ else "train")

NORM_PROVENANCE_FILENAME = "norm_stats_provenance.json"
_DYNAMIC_CHECKPOINT_FIELDS = frozenset(
    {
        "checkpoint_step",
        "checkpoint_vlm_fingerprint",
        "checkpoint_action_fingerprint",
        "provenance_digest",
    }
)
_EXTENDABLE_RESUME_FIELDS = frozenset({"num_train_steps"})
_AUDIT_ONLY_FIELDS = frozenset(
    {
        "source_task_name",
        "source_episode_number",
        "raw_phase_before",
        "clean_phase",
        "subtask_type",
        "subtask",
    }
)


@dataclasses.dataclass(frozen=True)
class Args:
    exp_name: str
    source_checkpoint_dir: str
    norm_assets_dir: str
    annotations_dir: str
    train_scope: Literal["action_only", "full_model"] = "action_only"
    num_train_steps: int | None = None
    batch_size: int | None = None
    save_interval: int | None = None
    keep_period: int | None = None
    fsdp_devices: int = 4
    wandb_enabled: bool = False
    preflight_only: bool = False
    resume: bool = False


@dataclasses.dataclass(frozen=True)
class VerifiedOfficialCheckpointWeightLoader(weight_loaders.WeightLoader):
    """Load the official source and verify both parameter partitions."""

    params_path: str
    expected_vlm_fingerprint: str
    expected_action_fingerprint: str

    def load(self, params):
        loaded = weight_loaders.CheckpointWeightLoader(self.params_path).load(params)
        actual_vlm = checkpoint_fingerprint.fingerprint_frozen_vlm(loaded)
        if actual_vlm != self.expected_vlm_fingerprint:
            raise ValueError("Initialized VLM differs from the independently verified official source.")
        actual_action = checkpoint_fingerprint.fingerprint_trainable_action(loaded)
        if actual_action != self.expected_action_fingerprint:
            raise ValueError("Initialized action path differs from the independently verified official source.")
        return loaded


def action_only_freeze_filter() -> nnx.filterlib.Filter:
    """Freeze everything except the existing π0.5 action allowlist."""
    trainable = nnx.Any(
        nnx_utils.PathRegex(r"PaliGemma/llm/.*_1(?:/.*)?"),
        nnx_utils.PathRegex(r"(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*"),
    )
    return nnx.Not(trainable)


def _training_contract(train_scope: str) -> tuple[int, str]:
    if train_scope == "action_only":
        return 1, oracle.ACTION_TRAINABLE_PROTOCOL
    if train_scope == "full_model":
        return 2, oracle.FULL_MODEL_TRAINABLE_PROTOCOL
    raise ValueError(f"Unsupported train_scope: {train_scope!r}.")


def _optimization_contract(base: _config.TrainConfig) -> dict[str, Any]:
    """Seal the actual optimizer, LR schedule, and EMA setting used by both train scopes."""
    return {
        "optimizer_type": type(base.optimizer).__name__,
        "optimizer": dataclasses.asdict(base.optimizer),
        "lr_schedule_type": type(base.lr_schedule).__name__,
        "lr_schedule": dataclasses.asdict(base.lr_schedule),
        "ema_decay": base.ema_decay,
    }


def resolve_base_train_config(args: Args) -> _config.TrainConfig:
    """Load pi05_rlbench and apply only explicitly provided budget overrides."""
    base = _config.get_config(rlbench_timing.RLBENCH_CONFIG_NAME)
    overrides = {
        field: value
        for field in ("num_train_steps", "batch_size", "save_interval", "keep_period")
        if (value := getattr(args, field)) is not None
    }
    return dataclasses.replace(base, **overrides)


def validate_inputs_and_build_provenance(
    args: Args,
    timing: rlbench_timing.TimingContract,
    base: _config.TrainConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every immutable input and return the manifest and sealed provenance."""
    _validate_args(args, base)
    manifest, records, quality = oracle.load_and_validate_sidecar(args.annotations_dir)
    rlbench_timing.validate_manifest_timing(manifest, timing)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(manifest["repo_id"])
    raw_dataset = lerobot_dataset.LeRobotDataset(manifest["repo_id"])
    dataset_fingerprint = getattr(raw_dataset.hf_dataset, "_fingerprint", None)
    if dataset_fingerprint is None or not str(dataset_fingerprint):
        raise ValueError("HuggingFace dataset fingerprint is unavailable.")
    oracle.validate_dataset_binding(
        manifest,
        records,
        quality,
        repo_id=manifest["repo_id"],
        fingerprint=str(dataset_fingerprint),
        episode_indices=raw_dataset.hf_dataset["episode_index"],
        frame_indices=raw_dataset.hf_dataset["frame_index"],
        task_indices=raw_dataset.hf_dataset["task_index"],
        task_prompts=dataset_meta.tasks,
    )
    if len(raw_dataset) != manifest["num_dataset_rows"]:
        raise ValueError(
            f"LeRobot row count mismatch: live={len(raw_dataset)}, sidecar={manifest['num_dataset_rows']}."
        )
    if len(records) != manifest["included_anchor_count"]:
        raise ValueError("Oracle record count differs from included_anchor_count.")
    del raw_dataset, dataset_meta
    gc.collect()

    norm_root = pathlib.Path(args.norm_assets_dir).expanduser().resolve()
    norm_dir = norm_root / manifest["repo_id"]
    norm_stats_path = norm_dir / "norm_stats.json"
    norm_provenance_path = norm_dir / NORM_PROVENANCE_FILENAME
    if not norm_stats_path.is_file() or not norm_provenance_path.is_file():
        raise FileNotFoundError(
            "Oracle RLBench norm_stats.json and norm_stats_provenance.json are both required at "
            f"{norm_dir}; recompute them from the complete oracle repo."
        )
    norm_stats_sha256 = checkpoint_fingerprint.sha256_file(norm_stats_path)
    norm_provenance = _load_norm_provenance(norm_provenance_path)
    validate_norm_provenance(
        norm_provenance,
        repo_id=manifest["repo_id"],
        hf_dataset_fingerprint=manifest["hf_dataset_fingerprint"],
        action_horizon=timing.action_horizon,
        norm_stats_sha256=norm_stats_sha256,
        num_dataset_rows=manifest["num_dataset_rows"],
    )

    source_dir = checkpoint_fingerprint.validate_official_source(
        args.source_checkpoint_dir,
        checkpoint_fingerprint.OFFICIAL_PI05_BASE_ID,
    )
    logging.info("Fingerprinting the official pi05_base parameters on host memory.")
    source_params = _model.restore_params(source_dir / "params", restore_type=np.ndarray)
    source_vlm_fingerprint = checkpoint_fingerprint.fingerprint_frozen_vlm(source_params)
    source_action_fingerprint = checkpoint_fingerprint.fingerprint_trainable_action(source_params)
    del source_params
    gc.collect()

    policy_metadata = {
        "action_layout": rlbench_policy.RLBENCH_ACTION_LAYOUT,
        "language_condition": oracle.LANGUAGE_CONDITION,
        "subtask_protocol": oracle.PROTOCOL,
        "action_horizon": timing.action_horizon,
        "replan_steps": timing.replan_steps,
        "action_loss_protocol": oracle.ACTION_LOSS_PROTOCOL,
        "sidecar_manifest_digest": manifest["manifest_digest"],
    }
    checkpoint_schema, trainable_protocol = _training_contract(args.train_scope)
    provenance = checkpoint_fingerprint.seal_provenance(
        {
            "schema_version": checkpoint_schema,
            **policy_metadata,
            "policy_metadata": policy_metadata,
            "repo_id": manifest["repo_id"],
            "hf_dataset_fingerprint": manifest["hf_dataset_fingerprint"],
            "annotations_sha256": manifest["annotations_sha256"],
            "quality_report_sha256": manifest["quality_report_sha256"],
            "num_dataset_rows": manifest["num_dataset_rows"],
            "included_episode_count": manifest["included_episode_count"],
            "included_anchor_count": manifest["included_anchor_count"],
            "subtask_texts_sha256": manifest["subtask_texts_sha256"],
            "sampling_protocol": oracle.SAMPLING_PROTOCOL,
            "norm_asset_id": manifest["repo_id"],
            "norm_stats_sha256": norm_stats_sha256,
            "norm_provenance_digest": norm_provenance["provenance_digest"],
            "source_checkpoint_id": checkpoint_fingerprint.OFFICIAL_PI05_BASE_ID,
            "source_checkpoint_resolved": str(source_dir),
            "source_vlm_fingerprint": source_vlm_fingerprint,
            "source_action_fingerprint": source_action_fingerprint,
            "action_trainable_protocol": trainable_protocol,
            "optimization_contract": _optimization_contract(base),
            "num_train_steps": base.num_train_steps,
            "batch_size": base.batch_size,
            "save_interval": base.save_interval,
            "keep_period": base.keep_period,
            "fsdp_devices": args.fsdp_devices,
            "lerobot_version": importlib.metadata.version("lerobot"),
            **timing.to_metadata(),
        }
    )
    logging.info(
        "Validated oracle inputs: rows=%d, episodes=%d/%d included, anchors=%d, texts=%d, H=%d, K=%d, loss=%s",
        manifest["num_dataset_rows"],
        manifest["included_episode_count"],
        manifest["candidate_episode_count"],
        manifest["included_anchor_count"],
        sum(len(task_texts) for task_texts in manifest["subtask_texts"].values()),
        timing.action_horizon,
        timing.replan_steps,
        oracle.ACTION_LOSS_PROTOCOL,
    )
    return manifest, provenance


def build_train_config(
    args: Args,
    timing: rlbench_timing.TimingContract,
    base: _config.TrainConfig,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> _config.TrainConfig:
    """Build the one oracle action-policy config while reusing the standard trainer."""
    if base.model.action_horizon != timing.action_horizon:
        raise ValueError("pi05_rlbench action_horizon differs from the runtime timing contract.")
    source_dir = checkpoint_fingerprint.validate_official_source(
        args.source_checkpoint_dir,
        checkpoint_fingerprint.OFFICIAL_PI05_BASE_ID,
    )
    base_data = base.data.base_config or _config.DataConfig()
    oracle_data = dataclasses.replace(
        base.data,
        repo_id=manifest["repo_id"],
        assets=_config.AssetsConfig(
            assets_dir=str(pathlib.Path(args.norm_assets_dir).expanduser().resolve()),
            asset_id=manifest["repo_id"],
        ),
        base_config=dataclasses.replace(
            base_data,
            prompt_from_task=True,
            subtask_annotations_dir=str(pathlib.Path(args.annotations_dir).expanduser().resolve()),
            subtask_replan_steps=timing.replan_steps,
            checkpoint_provenance=provenance,
        ),
    )
    return dataclasses.replace(
        base,
        name=oracle.TRAIN_CONFIG_NAME,
        exp_name=args.exp_name,
        data=oracle_data,
        weight_loader=VerifiedOfficialCheckpointWeightLoader(
            str(source_dir / "params"),
            expected_vlm_fingerprint=provenance["source_vlm_fingerprint"],
            expected_action_fingerprint=provenance["source_action_fingerprint"],
        ),
        freeze_filter=action_only_freeze_filter() if args.train_scope == "action_only" else nnx.Nothing(),
        ema_decay=base.ema_decay,
        fsdp_devices=args.fsdp_devices,
        wandb_enabled=args.wandb_enabled,
        resume=args.resume,
        overwrite=False,
        policy_metadata=dict(provenance["policy_metadata"]),
    )


def validate_trainable_parameter_paths(
    config: _config.TrainConfig,
    train_scope: str,
) -> tuple[str, ...]:
    """Materialize shapes only and prove the trainable tree matches the selected scope."""
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable = nnx.state(abstract_model, config.trainable_filter).flat_state()
    paths = tuple(sorted("/".join(map(str, path)) for path in trainable))
    if not paths:
        raise ValueError("The selected train scope exposed no trainable parameters.")

    if train_scope == "full_model":
        all_params = nnx.state(abstract_model, nnx.Param).flat_state()
        all_paths = tuple(sorted("/".join(map(str, path)) for path in all_params))
        if paths != all_paths:
            raise ValueError("full_model did not expose exactly all nnx.Param leaves.")
        logging.info("Verified all %d model parameter leaves are trainable.", len(paths))
        return paths

    if train_scope != "action_only":
        raise ValueError(f"Unsupported train_scope: {train_scope!r}.")
    allowed = re.compile(
        r"(?:PaliGemma/llm/.*_1(?:/.*)?|(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*)"
    )
    unexpected = [path for path in paths if allowed.fullmatch(path) is None]
    if unexpected:
        raise ValueError(f"Freeze filter exposed non-action parameters: {unexpected}.")
    required_families = (
        "PaliGemma/llm/",
        "action_in_proj/",
        "action_out_proj/",
        "time_mlp_in/",
        "time_mlp_out/",
    )
    missing = [family for family in required_families if not any(path.startswith(family) for path in paths)]
    if missing:
        raise ValueError(f"Freeze filter omitted required action parameter families: {missing}.")
    forbidden = ("PaliGemma/img/", "PaliGemma/llm/embedder/")
    if any(path.startswith(forbidden) for path in paths):
        raise ValueError("Freeze filter exposed the vision encoder or tied embedder.")
    logging.info("Verified %d trainable action leaves.", len(paths))
    return paths


def validate_one_oracle_batch(
    config: _config.TrainConfig,
    *,
    expected_rows: int,
    allowed_prompts: frozenset[str],
) -> None:
    """Build the dense included-frame dataset and one transformed batch before training."""
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("Oracle normalization stats were not loaded.")
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if len(dataset) != expected_rows:
        raise ValueError(f"Oracle dataset length mismatch: live={len(dataset)}, expected={expected_rows}.")

    for position in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        sample = dataset[position]
        prompt = sample.get("prompt")
        if not isinstance(prompt, str) or prompt not in allowed_prompts:
            raise ValueError(f"Oracle sample {position} has an unsealed prompt: {prompt!r}.")
        leaked = sorted(_AUDIT_ONLY_FIELDS.intersection(sample))
        if leaked:
            raise ValueError(f"Oracle audit-only fields leaked into model sample {position}: {leaked}.")
        actions = np.asarray(sample["actions"])
        actions_is_pad = np.asarray(sample["actions_is_pad"])
        if actions.shape != (config.model.action_horizon, rlbench_policy.RLBENCH_ACTION_DIM):
            raise ValueError(f"Raw action chunk has unexpected shape: {actions.shape}.")
        if actions_is_pad.shape != (config.model.action_horizon,):
            raise ValueError(f"Raw action padding mask has unexpected shape: {actions_is_pad.shape}.")

    transformed = _data_loader.transform_dataset(dataset, data_config)
    loader = _data_loader.DataLoaderImpl(
        data_config,
        _data_loader.TorchDataLoader(
            transformed,
            local_batch_size=config.batch_size,
            num_batches=1,
            num_workers=0,
            shuffle=False,
        ),
    )
    observation, actions = next(iter(loader))
    expected_action_shape = (config.batch_size, config.model.action_horizon, config.model.action_dim)
    if tuple(actions.shape) != expected_action_shape:
        raise ValueError(f"Transformed action batch shape mismatch: {actions.shape} vs {expected_action_shape}.")
    if observation.actions_is_pad is None or tuple(observation.actions_is_pad.shape) != expected_action_shape[:-1]:
        raise ValueError("Transformed action padding mask is missing or has the wrong shape.")
    if observation.tokenized_prompt is None or observation.tokenized_prompt.shape[0] != config.batch_size:
        raise ValueError("Oracle prompts were not tokenized into the model batch.")
    if not np.isfinite(np.asarray(actions)).all() or not np.isfinite(np.asarray(observation.state)).all():
        raise ValueError("Oracle preflight batch contains non-finite state or action values.")
    logging.info("Validated one oracle batch with action shape %s and full-H padding mask.", actions.shape)


def validate_resume_provenance(config: _config.TrainConfig, expected: dict[str, Any]) -> pathlib.Path | None:
    """Require matching immutable provenance and a nondecreasing training budget."""
    if not config.resume:
        return None
    checkpoint_dir = config.checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume; checkpoint directory does not exist: {checkpoint_dir}")
    complete_steps = []
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        try:
            _checkpoints.validate_local_orbax_step(path, require_train_state=True)
        except (FileNotFoundError, ValueError) as exc:
            logging.warning("Ignoring incomplete local Orbax step %s: %s", path, exc)
            continue
        if (path / "assets" / checkpoint_fingerprint.CHECKPOINT_PROVENANCE_PATH).is_file():
            complete_steps.append(path)
    complete_steps.sort(key=lambda path: int(path.name))
    if not complete_steps:
        raise FileNotFoundError(f"Cannot resume; no complete oracle checkpoint exists in {checkpoint_dir}.")
    latest = complete_steps[-1]
    stored = checkpoint_fingerprint.load_checkpoint_provenance(latest)
    if stored.get("checkpoint_step") != int(latest.name):
        raise ValueError("Latest checkpoint directory and checkpoint_step provenance disagree.")
    required_checkpoint_fingerprints = {"checkpoint_action_fingerprint"}
    if expected["action_trainable_protocol"] == oracle.FULL_MODEL_TRAINABLE_PROTOCOL:
        required_checkpoint_fingerprints.add("checkpoint_vlm_fingerprint")
    missing = sorted(required_checkpoint_fingerprints - stored.keys())
    if missing:
        raise ValueError(f"Latest checkpoint provenance is missing fingerprints: {missing}.")
    comparison_exclusions = _DYNAMIC_CHECKPOINT_FIELDS | _EXTENDABLE_RESUME_FIELDS
    expected_immutable = {key: value for key, value in expected.items() if key not in comparison_exclusions}
    stored_immutable = {key: value for key, value in stored.items() if key not in comparison_exclusions}
    if stored_immutable != expected_immutable:
        differing = sorted(
            key
            for key in set(stored_immutable) | set(expected_immutable)
            if stored_immutable.get(key) != expected_immutable.get(key)
        )
        raise ValueError(f"Resume provenance differs from the current immutable inputs: {differing}.")

    stored_train_steps = stored.get("num_train_steps")
    requested_train_steps = expected.get("num_train_steps")
    for label, value in (
        ("stored num_train_steps", stored_train_steps),
        ("requested num_train_steps", requested_train_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Resume provenance has invalid {label}: {value!r}.")
    latest_step = int(latest.name)
    if latest_step >= stored_train_steps:
        raise ValueError("Latest checkpoint step lies outside its sealed training budget.")
    if requested_train_steps < stored_train_steps:
        raise ValueError(
            "Cannot decrease num_train_steps when resuming: "
            f"checkpoint={stored_train_steps}, requested={requested_train_steps}."
        )
    if requested_train_steps <= latest_step + 1:
        raise ValueError(
            "Requested num_train_steps leaves no training step after the latest checkpoint: "
            f"latest={latest_step}, requested={requested_train_steps}."
        )
    if requested_train_steps > stored_train_steps:
        logging.info(
            "Extending the sealed training budget from %d to %d at checkpoint %s.",
            stored_train_steps,
            requested_train_steps,
            latest,
        )
    else:
        logging.info("Resume provenance exactly matches latest complete checkpoint %s.", latest)
    return latest


def _validate_args(args: Args, base: _config.TrainConfig) -> None:
    name = args.exp_name.strip()
    if not name or name != args.exp_name or pathlib.PurePath(name).name != name or name in {".", ".."}:
        raise ValueError("exp_name must be a non-empty single path component without surrounding whitespace.")
    positive = {
        "num_train_steps": base.num_train_steps,
        "batch_size": base.batch_size,
        "save_interval": base.save_interval,
        "keep_period": base.keep_period,
        "fsdp_devices": args.fsdp_devices,
    }
    for field, value in positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    if base.keep_period % base.save_interval != 0:
        raise ValueError("keep_period must be an integer multiple of save_interval.")
    device_count = jax.device_count()
    if base.batch_size % device_count != 0:
        raise ValueError(f"batch_size={base.batch_size} must be divisible by device_count={device_count}.")
    if args.fsdp_devices > device_count or device_count % args.fsdp_devices != 0:
        raise ValueError(f"fsdp_devices={args.fsdp_devices} must divide device_count={device_count}.")


def _load_norm_provenance(path: pathlib.Path) -> dict[str, Any]:
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid norm provenance JSON: {path}") from exc
    if not isinstance(provenance, dict) or "provenance_digest" not in provenance:
        raise ValueError(f"Norm provenance is missing its digest: {path}")
    if checkpoint_fingerprint.seal_provenance(provenance)["provenance_digest"] != provenance["provenance_digest"]:
        raise ValueError(f"Norm provenance digest mismatch: {path}")
    return provenance


def validate_norm_provenance(
    provenance: dict[str, Any],
    *,
    repo_id: str,
    hf_dataset_fingerprint: str,
    action_horizon: int,
    norm_stats_sha256: str,
    num_dataset_rows: int,
) -> None:
    expected = {
        "schema_version": 1,
        "repo_id": repo_id,
        "hf_dataset_fingerprint": hf_dataset_fingerprint,
        "action_horizon": action_horizon,
        "num_dataset_rows": num_dataset_rows,
        "num_samples_processed": num_dataset_rows,
        "complete_dataset": True,
        "norm_stats_sha256": norm_stats_sha256,
    }
    expected_keys = set(expected) | {"provenance_digest"}
    if set(provenance) != expected_keys:
        raise ValueError(
            f"Norm provenance fields mismatch: got={sorted(provenance)}, expected={sorted(expected_keys)}."
        )
    for key, value in expected.items():
        if provenance[key] != value:
            raise ValueError(f"Norm provenance mismatch for {key}: stats={provenance[key]!r}, expected={value!r}.")


def main(args: Args) -> None:
    timing = rlbench_timing.load_runtime_contract()
    base = resolve_base_train_config(args)
    manifest, provenance = validate_inputs_and_build_provenance(args, timing, base)
    config = build_train_config(args, timing, base, manifest, provenance)
    validate_trainable_parameter_paths(config, args.train_scope)
    validate_resume_provenance(config, provenance)
    allowed_prompts = frozenset(
        text for task_texts in manifest["subtask_texts"].values() for text in task_texts.values()
    )
    validate_one_oracle_batch(
        config,
        expected_rows=manifest["included_anchor_count"],
        allowed_prompts=allowed_prompts,
    )
    if args.preflight_only:
        logging.info("Oracle action-policy preflight completed; training was not started.")
        return
    _train.main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
