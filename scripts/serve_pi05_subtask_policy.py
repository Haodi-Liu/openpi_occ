"""Serve and mechanically verify the oracle-text π0.5 action policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import logging
import socket
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import tyro

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import rlbench_policy
from openpi.serving import websocket_policy_server
from openpi.shared import checkpoint_fingerprint
from openpi.shared import rlbench_timing
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import oracle_phase_sidecar as oracle


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: str
    annotations_dir: str
    port: int = 8000
    counterfactual_samples: int = 8
    check_only: bool = False


@dataclasses.dataclass(frozen=True)
class RequirePromptInOracleTable(transforms.DataTransformFn):
    """Normalize a scalar prompt and reject text outside the sealed oracle vocabulary."""

    allowed_prompts: frozenset[str]

    def __call__(self, data: dict) -> dict:
        prompt: Any = data.get("prompt")
        if isinstance(prompt, np.ndarray):
            if prompt.size != 1:
                raise ValueError(f"Oracle prompt must be scalar, got shape {prompt.shape}.")
            prompt = prompt.item()
        if isinstance(prompt, bytes | np.bytes_):
            prompt = bytes(prompt).decode("utf-8")
        if isinstance(prompt, np.str_):
            prompt = str(prompt)
        if not isinstance(prompt, str) or prompt not in self.allowed_prompts:
            raise ValueError(f"Prompt is not in the sealed oracle text table: {prompt!r}.")
        return {**data, "prompt": prompt}


def _checkpoint_train_scope(provenance: Mapping[str, Any]) -> str:
    contract = (
        provenance.get("schema_version"),
        provenance.get("action_trainable_protocol"),
    )
    if contract == (1, oracle.ACTION_TRAINABLE_PROTOCOL):
        return "action_only"
    if contract == (2, oracle.FULL_MODEL_TRAINABLE_PROTOCOL):
        return "full_model"
    raise ValueError(f"Unsupported checkpoint training contract: {contract!r}.")


def validate_checkpoint_provenance(
    provenance: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    timing: rlbench_timing.TimingContract,
) -> str:
    """Require the exact oracle protocols and all immutable artifact bindings."""
    train_scope = _checkpoint_train_scope(provenance)
    schema_version = 1 if train_scope == "action_only" else 2
    trainable_protocol = (
        oracle.ACTION_TRAINABLE_PROTOCOL if train_scope == "action_only" else oracle.FULL_MODEL_TRAINABLE_PROTOCOL
    )
    policy_metadata = {
        "action_layout": rlbench_policy.RLBENCH_ACTION_LAYOUT,
        "language_condition": oracle.LANGUAGE_CONDITION,
        "subtask_protocol": oracle.PROTOCOL,
        "action_horizon": timing.action_horizon,
        "replan_steps": timing.replan_steps,
        "action_loss_protocol": oracle.ACTION_LOSS_PROTOCOL,
        "sidecar_manifest_digest": manifest["manifest_digest"],
    }
    required = {
        "schema_version": schema_version,
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
        "source_checkpoint_id": checkpoint_fingerprint.OFFICIAL_PI05_BASE_ID,
        "action_trainable_protocol": trainable_protocol,
        **timing.to_metadata(),
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"Checkpoint provenance mismatch for {key}: checkpoint={provenance.get(key)!r}, expected={expected!r}."
            )

    forbidden = {"action_loss_horizon", "subtask_replan_steps"}
    forbidden.update(key for key in provenance if key.startswith("high_level_"))
    if forbidden.intersection(provenance):
        raise ValueError(
            f"Checkpoint contains retired generated-subtask fields: {sorted(forbidden.intersection(provenance))}."
        )
    expected_fields = set(required) | {
        "norm_stats_sha256",
        "norm_provenance_digest",
        "source_checkpoint_resolved",
        "source_vlm_fingerprint",
        "source_action_fingerprint",
        "checkpoint_action_fingerprint",
        "checkpoint_step",
        "num_train_steps",
        "batch_size",
        "save_interval",
        "keep_period",
        "fsdp_devices",
        "lerobot_version",
        "optimization_contract",
        "provenance_digest",
    }
    if train_scope == "full_model":
        expected_fields.add("checkpoint_vlm_fingerprint")
    if set(provenance) != expected_fields:
        raise ValueError(
            f"Checkpoint provenance fields mismatch: got={sorted(provenance)}, expected={sorted(expected_fields)}."
        )
    digest_fields = (
        "provenance_digest",
        "norm_stats_sha256",
        "norm_provenance_digest",
        "source_vlm_fingerprint",
        "source_action_fingerprint",
        "checkpoint_action_fingerprint",
    )
    if train_scope == "full_model":
        digest_fields += ("checkpoint_vlm_fingerprint",)
    for key in digest_fields:
        value = provenance.get(key)
        if not isinstance(value, str) or not value.startswith(checkpoint_fingerprint.FINGERPRINT_PREFIX):
            raise ValueError(f"Checkpoint provenance is missing a valid {key}.")
    string_fields = ("source_checkpoint_resolved", "lerobot_version")
    for key in string_fields:
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise ValueError(f"Checkpoint provenance is missing {key}.")
    optimization_contract = provenance.get("optimization_contract")
    if not isinstance(optimization_contract, dict) or set(optimization_contract) != {
        "optimizer_type",
        "optimizer",
        "lr_schedule_type",
        "lr_schedule",
        "ema_decay",
    }:
        raise ValueError("Checkpoint provenance has an invalid optimization_contract.")
    if not isinstance(optimization_contract["optimizer"], dict) or not isinstance(
        optimization_contract["lr_schedule"], dict
    ):
        raise ValueError("Checkpoint optimization settings must be mappings.")
    positive_integer_fields = (
        "num_train_steps",
        "batch_size",
        "save_interval",
        "keep_period",
        "fsdp_devices",
    )
    for key in positive_integer_fields:
        value = provenance.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Checkpoint provenance has invalid {key}: {value!r}.")
    checkpoint_step = provenance.get("checkpoint_step")
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step < 0:
        raise ValueError(f"Checkpoint provenance has invalid checkpoint_step: {checkpoint_step!r}.")
    if checkpoint_step >= provenance["num_train_steps"]:
        raise ValueError("checkpoint_step lies outside the sealed training budget.")
    if (
        train_scope == "action_only"
        and provenance["checkpoint_action_fingerprint"] == provenance["source_action_fingerprint"]
    ):
        raise ValueError("Checkpoint action parameters are unchanged from the official source.")
    return train_scope


def _build_server_metadata(
    provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    actual_vlm: str,
    actual_action: str,
) -> dict[str, Any]:
    """Expose the verified action and oracle-text contracts to the OCC client."""
    return {
        **provenance["policy_metadata"],
        "train_scope": _checkpoint_train_scope(provenance),
        "action_trainable_protocol": provenance["action_trainable_protocol"],
        "checkpoint_step": provenance["checkpoint_step"],
        "source_vlm_fingerprint": provenance["source_vlm_fingerprint"],
        "checkpoint_vlm_fingerprint": actual_vlm,
        "source_action_fingerprint": provenance["source_action_fingerprint"],
        "checkpoint_action_fingerprint": actual_action,
        "checkpoint_provenance_digest": provenance["provenance_digest"],
        "norm_stats_sha256": provenance["norm_stats_sha256"],
        "subtask_texts": manifest["subtask_texts"],
        "subtask_texts_sha256": manifest["subtask_texts_sha256"],
    }


def create_verified_oracle_action_policy(
    args: Args,
    timing: rlbench_timing.TimingContract,
) -> tuple[_policy.Policy, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Verify one local Orbax checkpoint and construct a pure action policy."""
    if args.port < 1 or args.port > 65_535:
        raise ValueError(f"port must be in [1, 65535], got {args.port}.")
    if isinstance(args.counterfactual_samples, bool) or args.counterfactual_samples < 0:
        raise ValueError("counterfactual_samples must be a non-negative integer.")

    manifest, records, quality = oracle.load_and_validate_sidecar(args.annotations_dir)
    rlbench_timing.validate_manifest_timing(manifest, timing)
    checkpoint_dir = _checkpoints.validate_local_orbax_step(args.checkpoint_dir, require_train_state=False)
    if (checkpoint_dir / "model.safetensors").exists():
        raise ValueError("Oracle action serving accepts only the JAX Orbax checkpoint produced by this trainer.")

    provenance = checkpoint_fingerprint.load_checkpoint_provenance(checkpoint_dir)
    train_scope = validate_checkpoint_provenance(provenance, manifest=manifest, timing=timing)
    if provenance["checkpoint_step"] != int(checkpoint_dir.name):
        raise ValueError("Checkpoint directory name and checkpoint_step provenance disagree.")

    base = _config.get_config(rlbench_timing.RLBENCH_CONFIG_NAME)
    if base.model.action_horizon != timing.action_horizon:
        raise ValueError("pi05_rlbench action_horizon differs from the runtime timing contract.")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = base.model.load(params)
    model_state = nnx.state(model)
    actual_vlm = checkpoint_fingerprint.fingerprint_frozen_vlm(model_state)
    actual_action = checkpoint_fingerprint.fingerprint_trainable_action(model_state)
    expected_vlm = (
        provenance["source_vlm_fingerprint"]
        if train_scope == "action_only"
        else provenance["checkpoint_vlm_fingerprint"]
    )
    if actual_vlm != expected_vlm:
        raise ValueError("Checkpoint VLM fingerprint differs from checkpoint provenance.")
    if actual_action != provenance["checkpoint_action_fingerprint"]:
        raise ValueError("Checkpoint action fingerprint differs from checkpoint provenance.")
    if train_scope == "action_only" and actual_action == provenance["source_action_fingerprint"]:
        raise ValueError("Checkpoint action path is unchanged from pi05_base.")

    checkpoint_assets = checkpoint_dir / "assets"
    norm_stats_path = checkpoint_assets / manifest["repo_id"] / "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"Checkpoint normalization stats not found: {norm_stats_path}")
    if checkpoint_fingerprint.sha256_file(norm_stats_path) != provenance["norm_stats_sha256"]:
        raise ValueError("Checkpoint norm stats digest differs from training provenance.")
    data_factory = dataclasses.replace(
        base.data,
        repo_id=manifest["repo_id"],
        assets=_config.AssetsConfig(
            assets_dir=str(checkpoint_assets),
            asset_id=manifest["repo_id"],
        ),
    )
    data_config = data_factory.create(checkpoint_assets, base.model)
    if data_config.asset_id != manifest["repo_id"] or data_config.norm_stats is None:
        raise ValueError("Checkpoint normalization stats were not loaded under the sealed repo asset ID.")

    allowed_prompts = frozenset(
        text for task_texts in manifest["subtask_texts"].values() for text in task_texts.values()
    )
    server_metadata = _build_server_metadata(provenance, manifest, actual_vlm, actual_action)
    action_policy = _policy.Policy(
        model,
        transforms=[
            RequirePromptInOracleTable(allowed_prompts),
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=server_metadata,
    )
    logging.info(
        "Verified oracle action checkpoint step=%d, H=%d, K=%d, anchors=%d.",
        provenance["checkpoint_step"],
        timing.action_horizon,
        timing.replan_steps,
        manifest["included_anchor_count"],
    )
    return action_policy, manifest, records, quality


def select_wrong_subtask(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> tuple[str, str]:
    """Choose a distant pure phase from the same source task."""
    task = str(record["source_task_name"])
    clean_phase = int(record["clean_phase"])
    wrong_type = "4" if clean_phase <= 2 else "1"
    wrong_text = str(manifest["subtask_texts"][task][wrong_type])
    if wrong_text == record["subtask"]:
        raise ValueError("Counterfactual selector produced the original subtask text.")
    return wrong_type, wrong_text


def select_counterfactual_records(
    records: Sequence[Mapping[str, Any]],
    count: int,
    tasks: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Select deterministic, task-balanced records spread over each task trajectory."""
    if isinstance(count, bool) or count < 0:
        raise ValueError("Counterfactual count must be a non-negative integer.")
    if count == 0:
        return []
    if tasks is None:
        record_tasks = {str(record["source_task_name"]) for record in records}
        tasks = tuple(task for task in oracle.TASKS if task in record_tasks)
    else:
        tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in oracle.TASKS for task in tasks):
        raise ValueError("Counterfactual tasks must be a non-empty unique subset of supported oracle tasks.")
    by_task = {task: [record for record in records if record["source_task_name"] == task] for task in tasks}
    if any(not task_records for task_records in by_task.values()):
        raise ValueError("Counterfactual records do not cover every task sealed in the oracle manifest.")
    selected: list[Mapping[str, Any]] = []
    for index in range(count):
        task_index = index % len(tasks)
        task = tasks[task_index]
        task_records = by_task[task]
        occurrence = index // len(tasks)
        total_for_task = (count - task_index + len(tasks) - 1) // len(tasks)
        position = min(
            len(task_records) - 1,
            ((2 * occurrence + 1) * len(task_records)) // (2 * total_for_task),
        )
        selected.append(task_records[position])
    return selected


def check_counterfactual_text(
    policy: _policy.Policy,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    *,
    count: int,
    timing: rlbench_timing.TimingContract,
) -> dict[str, Any] | None:
    """Compare correct/wrong same-task text with identical observation and flow noise."""
    selected = select_counterfactual_records(records, count, tasks=manifest["tasks"])
    if not selected:
        logging.info("Counterfactual text check skipped because count=0.")
        return None

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(manifest["repo_id"])
    dataset = lerobot_dataset.LeRobotDataset(manifest["repo_id"])
    fingerprint = getattr(dataset.hf_dataset, "_fingerprint", None)
    if fingerprint is None or not str(fingerprint):
        raise ValueError("HuggingFace dataset fingerprint is unavailable for counterfactual validation.")
    oracle.validate_dataset_binding(
        manifest,
        records,
        quality,
        repo_id=manifest["repo_id"],
        fingerprint=str(fingerprint),
        episode_indices=dataset.hf_dataset["episode_index"],
        frame_indices=dataset.hf_dataset["frame_index"],
        task_indices=dataset.hf_dataset["task_index"],
        task_prompts=dataset_meta.tasks,
    )

    action_dim = _config.get_config(rlbench_timing.RLBENCH_CONFIG_NAME).model.action_dim
    reports = []
    language_sensitive = False
    for record in selected:
        global_index = int(record["global_index"])
        item = dataset[global_index]
        base_observation = {
            "observation/state": np.asarray(item["state"], dtype=np.float32),
            "observation/front_rgb": np.asarray(item["front_rgb"]),
            "observation/wrist_left_rgb": np.asarray(item["wrist_left_rgb"]),
            "observation/wrist_right_rgb": np.asarray(item["wrist_right_rgb"]),
        }
        noise = np.random.default_rng(global_index).standard_normal(
            (timing.action_horizon, action_dim), dtype=np.float32
        )
        correct_observation = {**base_observation, "prompt": str(record["subtask"])}
        wrong_type, wrong_text = select_wrong_subtask(manifest, record)
        wrong_observation = {**base_observation, "prompt": wrong_text}
        correct_actions = np.asarray(policy.infer(correct_observation, noise=noise)["actions"], dtype=np.float32)
        wrong_actions = np.asarray(policy.infer(wrong_observation, noise=noise)["actions"], dtype=np.float32)
        expected_shape = (timing.action_horizon, rlbench_policy.RLBENCH_ACTION_DIM)
        if correct_actions.shape != expected_shape or wrong_actions.shape != expected_shape:
            raise ValueError(
                f"Counterfactual action shape mismatch at global_index={global_index}: "
                f"{correct_actions.shape}, {wrong_actions.shape}, expected={expected_shape}."
            )
        if not np.isfinite(correct_actions).all() or not np.isfinite(wrong_actions).all():
            raise ValueError(f"Counterfactual actions are non-finite at global_index={global_index}.")
        delta = np.abs(correct_actions - wrong_actions)
        first_k_delta = delta[: timing.replan_steps]
        language_sensitive = language_sensitive or bool(np.any(first_k_delta != 0.0))
        reports.append(
            {
                "global_index": global_index,
                "source_task_name": record["source_task_name"],
                "subtask_type": record["subtask_type"],
                "wrong_subtask_type": wrong_type,
                "first_k": {
                    "mean_abs_delta": float(np.mean(first_k_delta)),
                    "max_abs_delta": float(np.max(first_k_delta)),
                },
                "full_h": {
                    "mean_abs_delta": float(np.mean(delta)),
                    "max_abs_delta": float(np.max(delta)),
                },
            }
        )
    summary = {
        "num_samples": len(reports),
        "first_k_mean_abs_delta": float(np.mean([report["first_k"]["mean_abs_delta"] for report in reports])),
        "first_k_max_abs_delta": float(np.max([report["first_k"]["max_abs_delta"] for report in reports])),
        "full_h_mean_abs_delta": float(np.mean([report["full_h"]["mean_abs_delta"] for report in reports])),
        "full_h_max_abs_delta": float(np.max([report["full_h"]["max_abs_delta"] for report in reports])),
        "samples": reports,
    }
    logging.info("Oracle text counterfactual report:\n%s", json.dumps(summary, indent=2, sort_keys=True))
    if not language_sensitive:
        raise ValueError("All first-K actions are exactly identical under correct and wrong oracle text.")
    return summary


def main(args: Args) -> None:
    timing = rlbench_timing.load_runtime_contract()
    policy, manifest, records, quality = create_verified_oracle_action_policy(args, timing)
    check_counterfactual_text(
        policy,
        manifest,
        records,
        quality,
        count=args.counterfactual_samples,
        timing=timing,
    )
    if args.check_only:
        logging.info("Oracle action-policy checkpoint and counterfactual checks completed.")
        return

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating oracle action server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
