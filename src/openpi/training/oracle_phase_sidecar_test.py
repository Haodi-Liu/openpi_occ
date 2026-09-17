from __future__ import annotations

import json
import pathlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.shared import rlbench_timing
from openpi.training import oracle_phase_sidecar as oracle
from scripts import generate_pi05_rlbench_oracle_sidecar as generator


def _phases(*runs: tuple[int, int]) -> np.ndarray:
    return np.concatenate([np.full(length, phase, dtype=np.int8) for phase, length in runs])


def _run_values(phases: np.ndarray) -> tuple[int, ...]:
    return tuple(run.phase for run in oracle.run_length_encode(phases))


def _write_sidecar(
    directory: pathlib.Path,
    episodes: list[dict],
    records: list[dict],
) -> dict:
    directory.mkdir()
    annotations_path = directory / oracle.ANNOTATIONS_FILENAME
    quality_path = directory / oracle.QUALITY_REPORT_FILENAME
    manifest_path = directory / oracle.MANIFEST_FILENAME
    annotations_path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    quality = oracle.make_quality_report(episodes)
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unsealed = oracle.make_unsealed_manifest(
        repo_id="local/synthetic_oracle",
        hf_dataset_fingerprint="synthetic-fingerprint",
        num_dataset_rows=sum(episode["num_actions"] for episode in episodes),
        action_horizon=20,
        replan_steps=10,
        action_horizon_source=rlbench_timing.ACTION_HORIZON_SOURCE,
        replan_steps_source=rlbench_timing.REPLAN_STEPS_SOURCE,
        episodes=episodes,
        records=records,
    )
    manifest = oracle.seal_manifest(unsealed, annotations_path, quality_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


@pytest.fixture(scope="module")
def sealed_sidecar(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("oracle") / "sidecar"
    raw = _phases((1, 8), (2, 10), (3, 10), (4, 8))
    decision = oracle.assess_episode(raw, len(raw) - 1, replan_steps=10)
    assert decision.included
    episodes = []
    records = []
    global_cursor = 0
    episode_index = 0
    for task in oracle.TASKS:
        for episode_number in sorted(
            range(oracle.LEGACY_EXPECTED_EPISODES_PER_TASK), key=lambda value: f"episode{value}"
        ):
            prompt = f"Synthetic overall instruction for {task}."
            episodes.append(
                oracle.make_episode_audit(
                    source_task_name=task,
                    source_episode_number=episode_number,
                    lerobot_episode_index=episode_index,
                    global_start_index=global_cursor,
                    overall_instruction=prompt,
                    action_semantics="next_observed_joint_position_observed_gripper",
                    raw_phases=raw,
                    decision=decision,
                    replan_steps=10,
                )
            )
            records.extend(
                oracle.make_annotation_records(
                    source_task_name=task,
                    source_episode_number=episode_number,
                    lerobot_episode_index=episode_index,
                    global_start_index=global_cursor,
                    overall_instruction=prompt,
                    raw_phases=raw,
                    decision=decision,
                )
            )
            global_cursor += len(raw) - 1
            episode_index += 1
    _write_sidecar(directory, episodes, records)
    return directory


def _copy_sidecar(source: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    destination = tmp_path / "sidecar"
    shutil.copytree(source, destination)
    return destination


def _reseal_manifest(directory: pathlib.Path) -> None:
    manifest_path = directory / oracle.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["annotations_sha256"] = oracle.sha256_file(directory / oracle.ANNOTATIONS_FILENAME)
    manifest["quality_report_sha256"] = oracle.sha256_file(directory / oracle.QUALITY_REPORT_FILENAME)
    manifest["manifest_digest"] = oracle.compute_manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_short_aba_island_is_suppressed() -> None:
    raw = _phases((1, 8), (2, 3), (1, 8), (2, 8), (3, 8), (4, 8))
    result = oracle.clean_phases(raw, persistence=6)
    assert _run_values(result.clean) == (1, 2, 3, 4)
    assert result.suppressed[0].stable_phase == 1
    assert result.suppressed[0].candidate_phase == 2
    assert result.suppressed[0].length == 3


def test_dense_transition_type_covers_exactly_the_previous_k_window() -> None:
    clean = _phases((1, 8), (2, 20), (3, 12), (4, 20))
    num_actions = len(clean) - 1
    anchors = oracle.classify_anchors(clean, num_actions, replan_steps=10)
    types = {anchor.frame_index: anchor.subtask_type for anchor in anchors}

    assert [anchor.frame_index for anchor in anchors] == list(range(num_actions))
    assert [index for index, value in types.items() if value == "1_to_2"] == list(range(8, 18))
    assert [index for index, value in types.items() if value == "2_to_3"] == list(range(28, 38))
    assert [index for index, value in types.items() if value == "3_to_4"] == list(range(40, 50))
    assert types[7] == "1"
    assert types[18] == "2"


def test_dense_sliding_window_rejects_transitions_split_across_fixed_periods() -> None:
    raw = _phases((1, 7), (2, 6), (3, 12), (4, 10))
    decision = oracle.assess_episode(raw, len(raw) - 1, replan_steps=10)

    assert not decision.included
    assert "multiple_transitions_in_lookback_window" in decision.reasons


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (
            _phases((1, 8), (2, 12), (1, 12), (2, 12), (3, 12), (4, 8)),
            "non_monotonic_phase_path",
        ),
        (_phases((1, 8), (3, 12), (4, 8)), "non_adjacent_transition"),
        (_phases((1, 6), (2, 6), (3, 6), (4, 8)), "multiple_transitions_in_lookback_window"),
        (
            _phases(
                (1, 8),
                (2, 1),
                (3, 1),
                (2, 1),
                (3, 1),
                (2, 1),
                (3, 1),
                (2, 1),
                (3, 1),
                (2, 1),
                (3, 1),
                (1, 6),
                (2, 10),
                (3, 10),
                (4, 10),
            ),
            "long_unconfirmed_oscillation",
        ),
        (_phases((1, 8), (2, 8), (3, 8)), "does_not_end_in_phase_4"),
    ],
)
def test_unmappable_episode_is_excluded(raw: np.ndarray, reason: str) -> None:
    decision = oracle.assess_episode(raw, len(raw) - 1, replan_steps=10)
    assert not decision.included
    assert reason in decision.reasons


@pytest.mark.parametrize(
    ("task", "number", "boundaries"),
    [
        ("bimanual_pick_plate", 16, (90, 242, 266)),
        ("bimanual_pick_fork", 0, (77, 190, 228)),
        ("bimanual_pivot_phone", 66, (102, 200, 263)),
    ],
)
def test_manual_review_recovers_all_frames_with_exact_transition_starts(
    task: str, number: int, boundaries: tuple[int, int, int]
) -> None:
    raw = _phases((1, 20), (2, 20), (1, 20), (2, 80), (3, 80), (4, 140))
    original = raw.copy()
    assert not oracle.assess_episode(raw, len(raw) - 1, 10).included

    decision = oracle.assess_source_episode(raw, len(raw) - 1, 10, source_task_name=task, source_episode_number=number)

    assert decision.included
    assert not decision.reasons
    assert decision.boundaries == boundaries
    assert [anchor.frame_index for anchor in decision.anchors] == list(range(len(raw) - 1))
    for phase, boundary in enumerate(boundaries, start=1):
        assert decision.cleaned.clean[boundary - 1] == phase
        assert decision.cleaned.clean[boundary] == phase + 1
        assert [a.frame_index for a in decision.anchors if a.subtask_type == f"{phase}_to_{phase + 1}"] == list(
            range(boundary, boundary + 10)
        )
        assert decision.anchors[boundary + 10].subtask_type == str(phase + 1)
    np.testing.assert_array_equal(raw, original)


def test_manual_review_leaves_automatically_included_episode_unchanged() -> None:
    raw = _phases((1, 8), (2, 10), (3, 10), (4, 8))
    decision = oracle.assess_source_episode(
        raw, len(raw) - 1, 10, source_task_name="bimanual_pick_plate", source_episode_number=16
    )
    assert decision.included
    assert decision.boundaries == (8, 18, 28)
    np.testing.assert_array_equal(decision.cleaned.clean, raw)


@pytest.mark.parametrize(
    ("task", "number"),
    [
        ("bimanual_edge_phone", 144),
        ("bimanual_pick_fork", 4),
        ("bimanual_pick_plate", 15),
        ("bimanual_pick_plate", 92),
        ("bimanual_pick_plate", 148),
        ("bimanual_pick_plate", 0),  # An unreviewed source episode must not inherit fork episode 0's review.
    ],
)
def test_bad_and_unreviewed_episodes_remain_excluded(task: str, number: int) -> None:
    raw = _phases((1, 400))
    decision = oracle.assess_source_episode(raw, len(raw) - 1, 10, source_task_name=task, source_episode_number=number)
    assert not decision.included
    assert not decision.anchors
    assert decision.reasons == ("does_not_end_in_phase_4", "missing_phase")


@pytest.mark.parametrize(("num_actions", "replan_steps", "error"), [(200, 10, "outside"), (350, 30, "quality checks")])
def test_manual_review_rejects_boundaries_incompatible_with_episode_or_timing(
    num_actions: int, replan_steps: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        oracle.assess_source_episode(
            _phases((1, num_actions + 1)),
            num_actions,
            replan_steps,
            source_task_name="bimanual_pick_plate",
            source_episode_number=16,
        )


@pytest.fixture
def manually_reviewed_sidecar(tmp_path: pathlib.Path) -> pathlib.Path:
    raw = _phases((1, 20), (2, 3), (1, 20), (2, 20), (1, 20), (2, 60), (3, 80), (4, 140))
    episodes = []
    records = []
    # The LeRobot index 0 intentionally differs from the source episode number 16.
    for episode_index, number in enumerate((16, 92)):
        decision = oracle.assess_source_episode(
            raw, len(raw) - 1, 10, source_task_name="bimanual_pick_plate", source_episode_number=number
        )
        kwargs = {
            "source_task_name": "bimanual_pick_plate",
            "source_episode_number": number,
            "lerobot_episode_index": episode_index,
            "global_start_index": episode_index * (len(raw) - 1),
            "overall_instruction": "Pick up the plate.",
            "raw_phases": raw,
            "decision": decision,
        }
        episodes.append(
            oracle.make_episode_audit(
                **kwargs, action_semantics="next_observed_joint_position_observed_gripper", replan_steps=10
            )
        )
        records.extend(oracle.make_annotation_records(**kwargs))
    directory = tmp_path / "reviewed_sidecar"
    _write_sidecar(directory, episodes, records)
    return directory


def test_manual_review_sidecar_round_trips_with_raw_label_audit(manually_reviewed_sidecar: pathlib.Path) -> None:
    manifest, records, quality = oracle.load_and_validate_sidecar(manually_reviewed_sidecar)
    assert manifest["included_episode_count"] == manifest["excluded_episode_count"] == 1
    assert manifest["included_anchor_count"] == len(records) == 362
    assert {record["source_episode_number"] for record in records} == {16}
    assert records[90]["raw_phase_before"] == 2
    assert records[90]["clean_phase"] == 2
    assert records[90]["subtask_type"] == "1_to_2"
    assert records[20]["raw_phase_before"] == 2
    assert records[20]["clean_phase"] == 1
    audit = quality["episodes"][0]
    assert audit["boundaries"] == [90, 242, 266]
    assert audit["raw_runs"] != audit["clean_runs"]
    assert audit["suppressed_segments"] == [
        {"stable_phase": 1, "candidate_phase": 2, "start": 20, "end": 23, "length": 3}
    ]
    assert audit["longest_unconfirmed_deviation"] == 3


def test_validator_rejects_resealed_wrong_manual_boundary(manually_reviewed_sidecar: pathlib.Path) -> None:
    quality_path = manually_reviewed_sidecar / oracle.QUALITY_REPORT_FILENAME
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    audit = quality["episodes"][0]
    audit["boundaries"][0] = 91
    audit["clean_runs"][0].update(end=91, length=91)
    audit["clean_runs"][1].update(start=91, length=151)
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    _reseal_manifest(manually_reviewed_sidecar)
    with pytest.raises(ValueError, match="Clean phase sequence mismatch"):
        oracle.load_and_validate_sidecar(manually_reviewed_sidecar)


def test_phase_continuity_mismatch_is_rejected() -> None:
    before = np.asarray([1, 1, 2], dtype=np.int8)
    after = np.asarray([1, 3, 3], dtype=np.int8)
    with pytest.raises(ValueError, match="continuity mismatch"):
        oracle.reconstruct_observation_phases(before, after)


def test_dense_anchors_cover_every_real_action_row_and_not_the_terminal_observation() -> None:
    clean = _phases((1, 8), (2, 10), (3, 10), (4, 8))
    anchors = oracle.classify_anchors(clean, num_actions=35, replan_steps=10)
    assert [anchor.frame_index for anchor in anchors] == list(range(35))


def test_fixed_text_table_has_all_28_entries_and_stable_digest() -> None:
    assert tuple(oracle.SUBTASK_TEXTS) == oracle.TASKS
    assert all(tuple(texts) == oracle.SUBTASK_TYPES for texts in oracle.SUBTASK_TEXTS.values())
    assert sum(len(texts) for texts in oracle.SUBTASK_TEXTS.values()) == 28
    assert oracle.subtask_texts_sha256() == "sha256:61d3dab6bdf0a78c2c152648c7b4193309405b666f542dc9307dbdcdcc6662f6"


def test_complete_synthetic_sidecar_round_trips(sealed_sidecar: pathlib.Path) -> None:
    manifest, records, quality = oracle.load_and_validate_sidecar(sealed_sidecar)
    assert manifest["candidate_episode_count"] == 600
    assert manifest["included_episode_count"] == 600
    assert manifest["excluded_episode_count"] == 0
    assert len(records) == manifest["included_anchor_count"] == 21_000
    assert len(quality["episodes"]) == 600


def test_single_task_variable_episode_sidecar_round_trips(tmp_path: pathlib.Path) -> None:
    directory = tmp_path / "single_task_sidecar"
    raw = _phases((1, 8), (2, 10), (3, 10), (4, 8))
    decision = oracle.assess_episode(raw, len(raw) - 1, replan_steps=10)
    task = "bimanual_pick_plate"
    episodes = []
    records = []
    global_cursor = 0
    for episode_index, episode_number in enumerate((0, 12, 3)):
        prompt = f"Synthetic overall instruction for {task}."
        episodes.append(
            oracle.make_episode_audit(
                source_task_name=task,
                source_episode_number=episode_number,
                lerobot_episode_index=episode_index,
                global_start_index=global_cursor,
                overall_instruction=prompt,
                action_semantics="next_observed_joint_position_observed_gripper",
                raw_phases=raw,
                decision=decision,
                replan_steps=10,
            )
        )
        records.extend(
            oracle.make_annotation_records(
                source_task_name=task,
                source_episode_number=episode_number,
                lerobot_episode_index=episode_index,
                global_start_index=global_cursor,
                overall_instruction=prompt,
                raw_phases=raw,
                decision=decision,
            )
        )
        global_cursor += len(raw) - 1

    _write_sidecar(directory, episodes, records)
    manifest, loaded_records, quality = oracle.load_and_validate_sidecar(directory)

    assert manifest["tasks"] == [task]
    assert manifest["expected_episodes_per_task"] is None
    assert manifest["candidate_episode_count"] == 3
    assert manifest["subtask_texts"] == {task: oracle.SUBTASK_TEXTS[task]}
    assert manifest["subtask_texts_sha256"] == oracle.subtask_texts_sha256((task,))
    assert loaded_records == records
    assert quality["expected_episodes_per_task"] is None


def test_generator_accepts_one_supported_task_with_any_episode_set(tmp_path: pathlib.Path) -> None:
    split_dir = tmp_path / "train"
    task_dir = split_dir / "bimanual_edge_phone"
    for episode_name in ("episode0", "episode12", "episode3"):
        (task_dir / episode_name).mkdir(parents=True)

    episode_dirs = generator._validate_source_layout(split_dir)  # noqa: SLF001

    assert [path.name for path in episode_dirs] == ["episode0", "episode12", "episode3"]


@pytest.mark.parametrize(
    "source_action_metadata",
    [
        {"action_semantics": "next_observed_joint_position_observed_gripper"},
        {"action_semantics": "executed_joint_target_commanded_gripper"},
        {"action_semantics": "custom_action_definition"},
        {"action_semantics": ""},
        {"action_semantics": "  "},
        {"action_semantics": None},
        {"action_semantics": 123},
        {"action_semantics": {"custom": ["any", "format"]}},
        {},
    ],
    ids=["observed", "executed", "custom", "empty", "whitespace", "null", "integer", "object", "missing"],
)
def test_generator_preserves_source_action_semantics_through_sealing_and_loading(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source_action_metadata: dict
) -> None:
    source_dir = tmp_path / "export"
    task = "bimanual_edge_phone"
    episode_dir = source_dir / "train" / task / "episode0"
    episode_dir.mkdir(parents=True)
    raw = _phases((1, 8), (2, 10), (3, 10), (4, 8))
    num_actions = len(raw) - 1
    observations = np.arange(len(raw) * 16, dtype=np.float32).reshape(-1, 16)
    states, actions = observations[:-1], observations[1:]
    for name, values in (
        ("state", states),
        ("actions", actions),
        ("phase_before_action", raw[:-1]),
        ("phase_after_action", raw[1:]),
    ):
        np.save(episode_dir / f"{name}.npy", values)
    prompt = "Pick up the phone."
    metadata = {
        "task": prompt,
        "source_task_name": task,
        "source_episode_number": 0,
        "phase_semantics": oracle.PHASE_SEMANTICS,
        "num_observations": len(raw),
        "num_transitions": num_actions,
        **source_action_metadata,
    }
    (episode_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    class FakeHfDataset(dict):
        _fingerprint = "synthetic-fingerprint"

        def __len__(self):
            return num_actions

    hf_dataset = FakeHfDataset(
        state=states,
        actions=actions,
        episode_index=np.zeros(num_actions, dtype=np.int64),
        frame_index=np.arange(num_actions),
        task_index=np.zeros(num_actions, dtype=np.int64),
    )
    dataset_root = tmp_path / "lerobot"
    monkeypatch.setattr(
        generator, "LeRobotDatasetMetadata", lambda _: SimpleNamespace(root=dataset_root, tasks={0: prompt})
    )
    monkeypatch.setattr(
        generator, "LeRobotDataset", lambda _: SimpleNamespace(root=dataset_root, hf_dataset=hf_dataset)
    )
    timing = rlbench_timing.TimingContract(
        action_horizon=20,
        replan_steps=10,
        action_horizon_source=rlbench_timing.ACTION_HORIZON_SOURCE,
        replan_steps_source=rlbench_timing.REPLAN_STEPS_SOURCE,
    )
    monkeypatch.setattr(rlbench_timing, "load_runtime_contract", lambda: timing)
    output_dir = tmp_path / "sidecar"

    generator.main(
        generator.Args(repo_id="local/synthetic_oracle", source_export_dir=str(source_dir), output_dir=str(output_dir))
    )

    manifest, records, quality = oracle.load_and_validate_sidecar(output_dir)
    assert manifest["num_dataset_rows"] == len(records) == num_actions
    assert [audit["action_semantics"] for audit in quality["episodes"]] == [
        source_action_metadata.get("action_semantics")
    ]


def test_validator_accepts_omitted_action_semantics(
    manually_reviewed_sidecar: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _, records, quality = oracle.load_and_validate_sidecar(manually_reviewed_sidecar)
    for audit in quality["episodes"]:
        audit.pop("action_semantics")
    directory = tmp_path / "sidecar_without_action_semantics"
    _write_sidecar(directory, quality["episodes"], records)

    _, loaded_records, loaded_quality = oracle.load_and_validate_sidecar(directory)

    assert loaded_records == records
    assert all("action_semantics" not in audit for audit in loaded_quality["episodes"])


def test_validator_rejects_extra_file(sealed_sidecar: pathlib.Path, tmp_path: pathlib.Path) -> None:
    directory = _copy_sidecar(sealed_sidecar, tmp_path)
    (directory / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="files mismatch"):
        oracle.load_and_validate_sidecar(directory)


def test_validator_rejects_resealed_wrong_anchor(sealed_sidecar: pathlib.Path, tmp_path: pathlib.Path) -> None:
    directory = _copy_sidecar(sealed_sidecar, tmp_path)
    annotations_path = directory / oracle.ANNOTATIONS_FILENAME
    records = oracle.read_annotation_records(annotations_path)
    records[0]["frame_index"] = 1
    annotations_path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    _reseal_manifest(directory)
    with pytest.raises(ValueError, match="Annotation content mismatch"):
        oracle.load_and_validate_sidecar(directory)


def test_validator_rejects_changed_fixed_text_even_if_resealed(
    sealed_sidecar: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    directory = _copy_sidecar(sealed_sidecar, tmp_path)
    manifest_path = directory / oracle.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["subtask_texts"][oracle.TASKS[0]]["1"] += " changed"
    manifest["subtask_texts_sha256"] = oracle.canonical_sha256(manifest["subtask_texts"])
    manifest["manifest_digest"] = oracle.compute_manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixed subtask text table mismatch"):
        oracle.load_and_validate_sidecar(directory)


def test_validator_rejects_content_tampering(sealed_sidecar: pathlib.Path, tmp_path: pathlib.Path) -> None:
    directory = _copy_sidecar(sealed_sidecar, tmp_path)
    annotations_path = directory / oracle.ANNOTATIONS_FILENAME
    annotations_path.write_text(annotations_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotations digest mismatch"):
        oracle.load_and_validate_sidecar(directory)


def test_dataset_binding_rejects_wrong_episode_index(sealed_sidecar: pathlib.Path) -> None:
    manifest, records, quality = oracle.load_and_validate_sidecar(sealed_sidecar)
    row_count = manifest["num_dataset_rows"]
    episode_indices = np.empty(row_count, dtype=np.int64)
    frame_indices = np.empty(row_count, dtype=np.int64)
    task_indices = np.empty(row_count, dtype=np.int64)
    task_prompts = {index: f"Synthetic overall instruction for {task}." for index, task in enumerate(oracle.TASKS)}
    task_to_index = {task: index for index, task in enumerate(oracle.TASKS)}
    for audit in quality["episodes"]:
        start = audit["global_start_index"]
        end = start + audit["num_actions"]
        episode_indices[start:end] = audit["lerobot_episode_index"]
        frame_indices[start:end] = np.arange(audit["num_actions"])
        task_indices[start:end] = task_to_index[audit["source_task_name"]]

    oracle.validate_dataset_binding(
        manifest,
        records,
        quality,
        repo_id=manifest["repo_id"],
        fingerprint=manifest["hf_dataset_fingerprint"],
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        task_indices=task_indices,
        task_prompts=task_prompts,
    )
    episode_indices[0] = 1
    with pytest.raises(ValueError, match="episode_index mismatch"):
        oracle.validate_dataset_binding(
            manifest,
            records,
            quality,
            repo_id=manifest["repo_id"],
            fingerprint=manifest["hf_dataset_fingerprint"],
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            task_indices=task_indices,
            task_prompts=task_prompts,
        )


def _synthetic_binding(
    manifest: dict,
    quality: dict,
) -> tuple[dict[str, np.ndarray | str], dict[int, str]]:
    row_count = manifest["num_dataset_rows"]
    episode_indices = np.empty(row_count, dtype=np.int64)
    frame_indices = np.empty(row_count, dtype=np.int64)
    task_indices = np.empty(row_count, dtype=np.int64)
    task_prompts = {index: f"Synthetic overall instruction for {task}." for index, task in enumerate(oracle.TASKS)}
    task_to_index = {task: index for index, task in enumerate(oracle.TASKS)}
    for audit in quality["episodes"]:
        start = audit["global_start_index"]
        end = start + audit["num_actions"]
        episode_indices[start:end] = audit["lerobot_episode_index"]
        frame_indices[start:end] = np.arange(audit["num_actions"])
        task_indices[start:end] = task_to_index[audit["source_task_name"]]

    class FakeHfDataset(dict):
        _fingerprint = manifest["hf_dataset_fingerprint"]

    return (
        FakeHfDataset(
            episode_index=episode_indices,
            frame_index=frame_indices,
            task_index=task_indices,
        ),
        task_prompts,
    )


def test_oracle_anchor_dataset_exposes_only_included_records_and_subtask_prompt(
    sealed_sidecar: pathlib.Path,
) -> None:
    manifest, records, quality = oracle.load_and_validate_sidecar(sealed_sidecar)
    hf_dataset, task_prompts = _synthetic_binding(manifest, quality)

    class BaseDataset:
        def __getitem__(self, index):
            return {"payload": index, "prompt": "overall task"}

    dataset = oracle.OraclePhaseAnchorDataset(
        dataset=BaseDataset(),
        hf_dataset=hf_dataset,
        annotations_dir=sealed_sidecar,
        action_horizon=20,
        replan_steps=10,
        repo_id=manifest["repo_id"],
        task_prompts=task_prompts,
    )

    assert len(dataset) == manifest["included_anchor_count"]
    for position in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[position]
        record = records[position]
        assert sample == {"payload": record["global_index"], "prompt": record["subtask"]}
        assert not {"source_task_name", "raw_phase_before", "clean_phase", "subtask_type", "subtask"}.intersection(
            sample
        )


def test_oracle_anchor_dataset_allows_runtime_fingerprint_drift_but_rejects_wrong_timing(
    sealed_sidecar: pathlib.Path,
) -> None:
    manifest, _, quality = oracle.load_and_validate_sidecar(sealed_sidecar)
    hf_dataset, task_prompts = _synthetic_binding(manifest, quality)
    hf_dataset.__dict__["_fingerprint"] = "wrong-fingerprint"
    kwargs = {
        "dataset": [],
        "hf_dataset": hf_dataset,
        "annotations_dir": sealed_sidecar,
        "action_horizon": 20,
        "replan_steps": 10,
        "repo_id": manifest["repo_id"],
        "task_prompts": task_prompts,
    }
    oracle.OraclePhaseAnchorDataset(**kwargs)

    hf_dataset.__dict__["_fingerprint"] = manifest["hf_dataset_fingerprint"]
    with pytest.raises(ValueError, match="Timing metadata mismatch"):
        oracle.OraclePhaseAnchorDataset(**{**kwargs, "action_horizon": 19})


def test_validator_rejects_retired_generated_protocol(sealed_sidecar: pathlib.Path, tmp_path: pathlib.Path) -> None:
    directory = _copy_sidecar(sealed_sidecar, tmp_path)
    manifest_path = directory / oracle.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "generated_subtask"
    manifest["manifest_digest"] = oracle.compute_manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported oracle protocol"):
        oracle.load_and_validate_sidecar(directory)


def test_lerobot_identity_rejects_wrong_action() -> None:
    states = np.zeros((2, 16), dtype=np.float32)
    source_actions = np.zeros((2, 16), dtype=np.float32)
    dataset_actions = source_actions.copy()
    dataset_actions[1, 3] = 1.0
    with pytest.raises(ValueError, match="action values mismatch"):
        generator.verify_lerobot_episode_identity(
            global_start_index=0,
            expected_episode_index=0,
            source_states=states,
            source_actions=source_actions,
            task_prompt="prompt",
            dataset_states=states.copy(),
            dataset_actions=dataset_actions,
            episode_indices=[0, 0],
            frame_indices=[0, 1],
            task_indices=[0, 0],
            task_prompts={0: "prompt"},
        )
