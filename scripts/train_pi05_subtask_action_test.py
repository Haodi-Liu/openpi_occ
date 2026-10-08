from __future__ import annotations

import dataclasses
import json

import pytest

from openpi.shared import checkpoint_fingerprint
from openpi.shared import rlbench_timing
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import oracle_phase_sidecar as oracle
from scripts import train_pi05_subtask_action as trainer


@pytest.fixture(autouse=True)
def _isolated_occ_timing(tmp_path, monkeypatch):
    config_path = tmp_path / "OPENPI_POLICY.yaml"
    config_path.write_text("replan_steps: 10\n", encoding="utf-8")
    monkeypatch.setenv(rlbench_timing.OCC_TIMING_FILE_ENV, str(config_path))


def _args(tmp_path, source_dir) -> trainer.Args:
    return trainer.Args(
        exp_name="oracle_test",
        source_checkpoint_dir=str(source_dir),
        norm_assets_dir=str(tmp_path / "norm"),
        annotations_dir=str(tmp_path / "sidecar"),
        fsdp_devices=1,
    )


def _timing() -> rlbench_timing.TimingContract:
    return rlbench_timing.load_runtime_contract()


def _write_finalized_local_orbax_step(step_dir, *, include_train_state: bool = True) -> None:
    item_handlers = {
        "assets": "openpi.training.checkpoints.CallbackHandler",
        "params": "orbax.checkpoint.PyTreeCheckpointHandler",
    }
    if include_train_state:
        item_handlers["train_state"] = "orbax.checkpoint.PyTreeCheckpointHandler"
    step_dir.mkdir(parents=True)
    (step_dir / "_CHECKPOINT_METADATA").write_text(
        json.dumps({"item_handlers": item_handlers, "commit_timestamp_nsecs": 1}),
        encoding="utf-8",
    )
    (step_dir / "assets").mkdir()
    for item in item_handlers.keys() - {"assets"}:
        item_dir = step_dir / item
        item_dir.mkdir()
        (item_dir / "_METADATA").write_text("{}\n", encoding="utf-8")


def test_norm_provenance_requires_the_complete_dataset() -> None:
    action_horizon = _config.get_config(rlbench_timing.RLBENCH_CONFIG_NAME).model.action_horizon
    expected = {
        "schema_version": 1,
        "repo_id": "local/oracle",
        "hf_dataset_fingerprint": "fingerprint",
        "action_horizon": action_horizon,
        "num_dataset_rows": 100,
        "num_samples_processed": 100,
        "complete_dataset": True,
        "norm_stats_sha256": "sha256:norm",
    }
    provenance = checkpoint_fingerprint.seal_provenance(expected)
    trainer.validate_norm_provenance(
        provenance,
        repo_id="local/oracle",
        hf_dataset_fingerprint="fingerprint",
        action_horizon=action_horizon,
        norm_stats_sha256="sha256:norm",
        num_dataset_rows=100,
    )

    incomplete = checkpoint_fingerprint.seal_provenance({**expected, "num_samples_processed": 20})
    with pytest.raises(ValueError, match="num_samples_processed"):
        trainer.validate_norm_provenance(
            incomplete,
            repo_id="local/oracle",
            hf_dataset_fingerprint="fingerprint",
            action_horizon=action_horizon,
            norm_stats_sha256="sha256:norm",
            num_dataset_rows=100,
        )
    extra = checkpoint_fingerprint.seal_provenance({**expected, "unexpected": True})
    with pytest.raises(ValueError, match="fields mismatch"):
        trainer.validate_norm_provenance(
            extra,
            repo_id="local/oracle",
            hf_dataset_fingerprint="fingerprint",
            action_horizon=action_horizon,
            norm_stats_sha256="sha256:norm",
            num_dataset_rows=100,
        )


@pytest.mark.parametrize("train_scope", ["action_only", "full_model"])
def test_build_train_config_inherits_base_budget_and_allows_explicit_smoke_overrides(tmp_path, train_scope) -> None:
    source_dir = tmp_path / "pi05_base"
    (source_dir / "params").mkdir(parents=True)
    args = dataclasses.replace(_args(tmp_path, source_dir), train_scope=train_scope)
    timing = _timing()
    manifest = {"repo_id": "local/oracle"}
    provenance = {
        "source_vlm_fingerprint": "sha256:vlm",
        "source_action_fingerprint": "sha256:action",
        "policy_metadata": {"action_loss_protocol": oracle.ACTION_LOSS_PROTOCOL},
    }

    configured_base = _config.get_config(rlbench_timing.RLBENCH_CONFIG_NAME)
    base = trainer.resolve_base_train_config(args)
    config = trainer.build_train_config(args, timing, base, manifest, provenance)

    assert config.name == oracle.TRAIN_CONFIG_NAME
    assert config.model.action_horizon == timing.action_horizon
    for field in ("num_train_steps", "batch_size", "save_interval", "keep_period"):
        assert getattr(base, field) == getattr(configured_base, field)
        assert getattr(config, field) == getattr(configured_base, field)
    assert base.checkpoint_base_dir == configured_base.checkpoint_base_dir
    assert config.checkpoint_base_dir == configured_base.checkpoint_base_dir
    assert config.ema_decay == base.ema_decay == configured_base.ema_decay == 0.999
    assert config.data.repo_id == "local/oracle"
    assert config.data.assets.asset_id == "local/oracle"
    assert configured_base.data.base_config.subtask_annotations_dir is None
    assert configured_base.data.base_config.subtask_replan_steps is None
    assert config.data.base_config.prompt_from_task
    assert config.data.base_config.subtask_annotations_dir == str((tmp_path / "sidecar").resolve())
    assert config.data.base_config.subtask_replan_steps == timing.replan_steps
    assert trainer._training_contract("action_only") == (1, oracle.ACTION_TRAINABLE_PROTOCOL)  # noqa: SLF001
    assert trainer._training_contract("full_model") == (  # noqa: SLF001
        2,
        oracle.FULL_MODEL_TRAINABLE_PROTOCOL,
    )
    assert trainer._optimization_contract(base) == {  # noqa: SLF001
        "optimizer_type": type(base.optimizer).__name__,
        "optimizer": dataclasses.asdict(base.optimizer),
        "lr_schedule_type": type(base.lr_schedule).__name__,
        "lr_schedule": dataclasses.asdict(base.lr_schedule),
        "ema_decay": base.ema_decay,
    }

    smoke_args = dataclasses.replace(
        args,
        num_train_steps=2,
        batch_size=16,
        save_interval=1,
        keep_period=1,
    )
    smoke_base = trainer.resolve_base_train_config(smoke_args)
    assert smoke_base.num_train_steps == 2
    assert smoke_base.batch_size == 16
    assert smoke_base.save_interval == 1
    assert smoke_base.keep_period == 1


def test_resume_requires_exact_immutable_provenance_and_allows_budget_extension(tmp_path) -> None:
    config = dataclasses.replace(
        _config.get_config("debug"),
        name=oracle.TRAIN_CONFIG_NAME,
        exp_name="resume_test",
        checkpoint_base_dir=str(tmp_path),
        overwrite=False,
        resume=True,
    )
    expected = checkpoint_fingerprint.seal_provenance(
        {
            "schema_version": 1,
            "repo_id": "local/oracle",
            "num_train_steps": 18_000,
            "source_action_fingerprint": "sha256:source",
            "action_trainable_protocol": oracle.ACTION_TRAINABLE_PROTOCOL,
            "optimization_contract": {"optimizer": "sealed"},
        }
    )
    step_dir = config.checkpoint_dir / "1999"
    _write_finalized_local_orbax_step(step_dir)
    checkpoint_fingerprint.write_checkpoint_provenance(
        step_dir / "assets",
        {
            **expected,
            "checkpoint_step": 1999,
            "checkpoint_action_fingerprint": "sha256:trained",
        },
    )

    assert trainer.validate_resume_provenance(config, expected) == step_dir

    extended = checkpoint_fingerprint.seal_provenance({**expected, "num_train_steps": 20_000})
    assert trainer.validate_resume_provenance(config, extended) == step_dir

    decreased = checkpoint_fingerprint.seal_provenance({**expected, "num_train_steps": 17_000})
    with pytest.raises(ValueError, match="Cannot decrease num_train_steps"):
        trainer.validate_resume_provenance(config, decreased)

    changed = checkpoint_fingerprint.seal_provenance({**expected, "repo_id": "local/other"})
    with pytest.raises(ValueError, match="repo_id"):
        trainer.validate_resume_provenance(config, changed)

    changed_optimization = checkpoint_fingerprint.seal_provenance(
        {**expected, "optimization_contract": {"optimizer": "changed"}}
    )
    with pytest.raises(ValueError, match="optimization_contract"):
        trainer.validate_resume_provenance(config, changed_optimization)

    full_model = checkpoint_fingerprint.seal_provenance(
        {
            **expected,
            "schema_version": 2,
            "action_trainable_protocol": oracle.FULL_MODEL_TRAINABLE_PROTOCOL,
        }
    )
    with pytest.raises(ValueError, match="checkpoint_vlm_fingerprint"):
        trainer.validate_resume_provenance(config, full_model)


def test_resume_budget_must_leave_a_step_after_the_latest_checkpoint(tmp_path) -> None:
    config = dataclasses.replace(
        _config.get_config("debug"),
        name=oracle.TRAIN_CONFIG_NAME,
        exp_name="completed_resume_test",
        checkpoint_base_dir=str(tmp_path),
        overwrite=False,
        resume=True,
    )
    expected = checkpoint_fingerprint.seal_provenance(
        {
            "schema_version": 1,
            "repo_id": "local/oracle",
            "num_train_steps": 2_000,
            "source_action_fingerprint": "sha256:source",
            "action_trainable_protocol": oracle.ACTION_TRAINABLE_PROTOCOL,
            "optimization_contract": {"optimizer": "sealed"},
        }
    )
    step_dir = config.checkpoint_dir / "1999"
    _write_finalized_local_orbax_step(step_dir)
    checkpoint_fingerprint.write_checkpoint_provenance(
        step_dir / "assets",
        {
            **expected,
            "checkpoint_step": 1999,
            "checkpoint_action_fingerprint": "sha256:trained",
        },
    )

    with pytest.raises(ValueError, match="leaves no training step"):
        trainer.validate_resume_provenance(config, expected)

    extended = checkpoint_fingerprint.seal_provenance({**expected, "num_train_steps": 2_001})
    assert trainer.validate_resume_provenance(config, extended) == step_dir


def test_local_orbax_step_requires_commit_metadata_and_required_items(tmp_path) -> None:
    complete = tmp_path / "1"
    _write_finalized_local_orbax_step(complete)
    assert _checkpoints.validate_local_orbax_step(complete, require_train_state=True) == complete.resolve()

    missing_train_state = tmp_path / "2"
    _write_finalized_local_orbax_step(missing_train_state, include_train_state=False)
    assert (
        _checkpoints.validate_local_orbax_step(missing_train_state, require_train_state=False)
        == missing_train_state.resolve()
    )
    with pytest.raises(ValueError, match="train_state"):
        _checkpoints.validate_local_orbax_step(missing_train_state, require_train_state=True)

    uncommitted = tmp_path / "3"
    _write_finalized_local_orbax_step(uncommitted)
    metadata_path = uncommitted / "_CHECKPOINT_METADATA"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["commit_timestamp_nsecs"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="commit timestamp"):
        _checkpoints.validate_local_orbax_step(uncommitted, require_train_state=True)

    missing_params = tmp_path / "4"
    _write_finalized_local_orbax_step(missing_params)
    (missing_params / "params" / "_METADATA").unlink()
    with pytest.raises(FileNotFoundError, match="params"):
        _checkpoints.validate_local_orbax_step(missing_params, require_train_state=False)
