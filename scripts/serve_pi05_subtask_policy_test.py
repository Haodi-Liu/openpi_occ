from __future__ import annotations

import copy

import numpy as np
import pytest

from openpi.policies import rlbench_policy
from openpi.shared import checkpoint_fingerprint
from openpi.shared import rlbench_timing
from openpi.training import oracle_phase_sidecar as oracle
from scripts import serve_pi05_subtask_policy as server


@pytest.fixture(autouse=True)
def _isolated_occ_timing(tmp_path, monkeypatch):
    config_path = tmp_path / "OPENPI_POLICY.yaml"
    config_path.write_text("replan_steps: 10\n", encoding="utf-8")
    monkeypatch.setenv(rlbench_timing.OCC_TIMING_FILE_ENV, str(config_path))


def _timing() -> rlbench_timing.TimingContract:
    return rlbench_timing.load_runtime_contract()


def _manifest() -> dict:
    return {
        "manifest_digest": "sha256:manifest",
        "repo_id": "local/oracle",
        "hf_dataset_fingerprint": "dataset-fingerprint",
        "annotations_sha256": "sha256:annotations",
        "quality_report_sha256": "sha256:quality",
        "num_dataset_rows": 100,
        "included_episode_count": 4,
        "included_anchor_count": 20,
        "tasks": list(oracle.TASKS),
        "subtask_texts_sha256": oracle.subtask_texts_sha256(),
        "subtask_texts": copy.deepcopy(oracle.SUBTASK_TEXTS),
    }


def _provenance(
    manifest: dict,
    timing: rlbench_timing.TimingContract,
    *,
    train_scope: str = "action_only",
) -> dict:
    if train_scope == "action_only":
        schema_version = 1
        trainable_protocol = oracle.ACTION_TRAINABLE_PROTOCOL
    elif train_scope == "full_model":
        schema_version = 2
        trainable_protocol = oracle.FULL_MODEL_TRAINABLE_PROTOCOL
    else:
        raise ValueError(train_scope)
    policy_metadata = {
        "action_layout": rlbench_policy.RLBENCH_ACTION_LAYOUT,
        "language_condition": oracle.LANGUAGE_CONDITION,
        "subtask_protocol": oracle.PROTOCOL,
        "action_horizon": timing.action_horizon,
        "replan_steps": timing.replan_steps,
        "action_loss_protocol": oracle.ACTION_LOSS_PROTOCOL,
        "sidecar_manifest_digest": manifest["manifest_digest"],
    }
    provenance = {
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
        "norm_stats_sha256": "sha256:norm",
        "norm_provenance_digest": "sha256:norm-provenance",
        "source_checkpoint_id": checkpoint_fingerprint.OFFICIAL_PI05_BASE_ID,
        "source_checkpoint_resolved": "/checkpoints/pi05_base",
        "source_vlm_fingerprint": "sha256:source-vlm",
        "source_action_fingerprint": "sha256:source-action",
        "checkpoint_action_fingerprint": "sha256:trained-action",
        "checkpoint_step": 17_999,
        "action_trainable_protocol": trainable_protocol,
        "optimization_contract": {
            "optimizer_type": "AdamW",
            "optimizer": {"clip_gradient_norm": 1.0},
            "lr_schedule_type": "CosineDecaySchedule",
            "lr_schedule": {"peak_lr": 5e-5},
            "ema_decay": 0.999,
        },
        "num_train_steps": 18_000,
        "batch_size": 32,
        "save_interval": 2_000,
        "keep_period": 6_000,
        "fsdp_devices": 4,
        "lerobot_version": "0.test",
        **timing.to_metadata(),
    }
    if train_scope == "full_model":
        provenance["checkpoint_vlm_fingerprint"] = "sha256:trained-vlm"
    return checkpoint_fingerprint.seal_provenance(provenance)


def test_oracle_checkpoint_provenance_requires_exact_protocol_and_digests() -> None:
    manifest = _manifest()
    timing = _timing()
    provenance = _provenance(manifest, timing)
    assert server.validate_checkpoint_provenance(provenance, manifest=manifest, timing=timing) == "action_only"

    for key, bad_value in (
        ("language_condition", f"{oracle.LANGUAGE_CONDITION}__mismatch"),
        ("annotations_sha256", "sha256:wrong"),
        ("sidecar_manifest_digest", "sha256:wrong"),
    ):
        changed = copy.deepcopy(provenance)
        changed[key] = bad_value
        with pytest.raises(ValueError, match=f"mismatch for {key}"):
            server.validate_checkpoint_provenance(changed, manifest=manifest, timing=timing)

    changed = copy.deepcopy(provenance)
    changed["high_level_prompt_protocol"] = "retired"
    with pytest.raises(ValueError, match="retired generated-subtask fields"):
        server.validate_checkpoint_provenance(changed, manifest=manifest, timing=timing)

    changed = copy.deepcopy(provenance)
    changed["tokenizer_model"] = "retired"
    with pytest.raises(ValueError, match="provenance fields mismatch"):
        server.validate_checkpoint_provenance(changed, manifest=manifest, timing=timing)


def test_oracle_checkpoint_requires_full_action_chunk_loss_protocol() -> None:
    manifest = _manifest()
    timing = _timing()
    provenance = _provenance(manifest, timing)
    provenance["action_loss_protocol"] = f"{oracle.ACTION_LOSS_PROTOCOL}__mismatch"
    with pytest.raises(ValueError, match="action_loss_protocol"):
        server.validate_checkpoint_provenance(provenance, manifest=manifest, timing=timing)


def test_full_model_checkpoint_uses_schema_two_and_its_own_vlm_fingerprint() -> None:
    manifest = _manifest()
    timing = _timing()
    provenance = _provenance(manifest, timing, train_scope="full_model")

    assert server.validate_checkpoint_provenance(provenance, manifest=manifest, timing=timing) == "full_model"

    changed = copy.deepcopy(provenance)
    changed["action_trainable_protocol"] = oracle.ACTION_TRAINABLE_PROTOCOL
    with pytest.raises(ValueError, match="Unsupported checkpoint training contract"):
        server.validate_checkpoint_provenance(changed, manifest=manifest, timing=timing)

    missing_vlm = copy.deepcopy(provenance)
    del missing_vlm["checkpoint_vlm_fingerprint"]
    with pytest.raises(ValueError, match="provenance fields mismatch"):
        server.validate_checkpoint_provenance(missing_vlm, manifest=manifest, timing=timing)


def test_checkpoint_requires_structured_optimization_contract() -> None:
    manifest = _manifest()
    timing = _timing()
    provenance = _provenance(manifest, timing)
    provenance["optimization_contract"] = {"optimizer": {}}

    with pytest.raises(ValueError, match="invalid optimization_contract"):
        server.validate_checkpoint_provenance(provenance, manifest=manifest, timing=timing)


def test_wrong_text_is_from_same_task_and_different_phase() -> None:
    manifest = _manifest()
    task = oracle.TASKS[1]
    for phase in oracle.PHASE_VALUES:
        record = {
            "source_task_name": task,
            "clean_phase": phase,
            "subtask": oracle.SUBTASK_TEXTS[task][str(phase)],
        }
        wrong_type, wrong_text = server.select_wrong_subtask(manifest, record)
        assert wrong_type == ("4" if phase <= 2 else "1")
        assert wrong_text == oracle.SUBTASK_TEXTS[task][wrong_type]
        assert wrong_text != record["subtask"]


def test_server_prompt_gate_accepts_only_the_sealed_text_vocabulary() -> None:
    text_entries = [text for task_texts in oracle.SUBTASK_TEXTS.values() for text in task_texts.values()]
    allowed = frozenset(text_entries)
    assert len(text_entries) == 28
    assert len(allowed) == 24
    gate = server.RequirePromptInOracleTable(allowed)
    prompt = next(iter(allowed))

    assert gate({"prompt": prompt})["prompt"] == prompt
    assert gate({"prompt": prompt.encode()})["prompt"] == prompt
    assert gate({"prompt": np.asarray(prompt)})["prompt"] == prompt
    with pytest.raises(ValueError, match="not in the sealed oracle text table"):
        gate({"prompt": "overall task prompt"})
    with pytest.raises(ValueError, match="must be scalar"):
        gate({"prompt": np.asarray([prompt, prompt])})


def test_server_metadata_exposes_the_verified_subtask_table() -> None:
    manifest = _manifest()
    provenance = _provenance(manifest, _timing())

    metadata = server._build_server_metadata(  # noqa: SLF001
        provenance,
        manifest,
        actual_vlm="sha256:actual-vlm",
        actual_action="sha256:actual-action",
    )

    assert metadata["action_layout"] == rlbench_policy.RLBENCH_ACTION_LAYOUT
    assert metadata["train_scope"] == "action_only"
    assert metadata["action_trainable_protocol"] == oracle.ACTION_TRAINABLE_PROTOCOL
    assert metadata["checkpoint_step"] == provenance["checkpoint_step"]
    assert metadata["source_vlm_fingerprint"] == provenance["source_vlm_fingerprint"]
    assert metadata["checkpoint_vlm_fingerprint"] == "sha256:actual-vlm"
    assert metadata["source_action_fingerprint"] == provenance["source_action_fingerprint"]
    assert metadata["checkpoint_action_fingerprint"] == "sha256:actual-action"
    assert metadata["checkpoint_provenance_digest"] == provenance["provenance_digest"]
    assert metadata["norm_stats_sha256"] == provenance["norm_stats_sha256"]
    assert metadata["subtask_texts"] == manifest["subtask_texts"]
    assert metadata["subtask_texts_sha256"] == manifest["subtask_texts_sha256"]


def test_server_metadata_exposes_full_model_vlm_identity() -> None:
    manifest = _manifest()
    provenance = _provenance(manifest, _timing(), train_scope="full_model")

    metadata = server._build_server_metadata(  # noqa: SLF001
        provenance,
        manifest,
        actual_vlm=provenance["checkpoint_vlm_fingerprint"],
        actual_action=provenance["checkpoint_action_fingerprint"],
    )

    assert metadata["train_scope"] == "full_model"
    assert metadata["action_trainable_protocol"] == oracle.FULL_MODEL_TRAINABLE_PROTOCOL
    assert metadata["source_vlm_fingerprint"] == provenance["source_vlm_fingerprint"]
    assert metadata["checkpoint_vlm_fingerprint"] == provenance["checkpoint_vlm_fingerprint"]


def test_counterfactual_record_selection_is_task_balanced() -> None:
    records = [
        {"source_task_name": task, "global_index": task_index * 100 + index}
        for task_index, task in enumerate(oracle.TASKS)
        for index in range(10)
    ]
    selected = server.select_counterfactual_records(records, 8)
    assert [record["source_task_name"] for record in selected] == list(oracle.TASKS) * 2


def test_counterfactual_record_selection_uses_single_manifest_task() -> None:
    task = "bimanual_pick_plate"
    records = [{"source_task_name": task, "global_index": index} for index in range(10)]

    selected = server.select_counterfactual_records(records, 4, tasks=[task])

    assert len(selected) == 4
    assert {record["source_task_name"] for record in selected} == {task}
