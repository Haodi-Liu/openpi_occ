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
    episodes = []
    records = []
    global_cursor = 0
    episode_index = 0
    for task in oracle.TASKS:
        for episode_number in sorted(
            range(oracle.LEGACY_EXPECTED_EPISODES_PER_TASK), key=lambda value: f"episode{value}"
        ):
            prompt = f"Synthetic overall instruction for {task}."
            rows = np.arange(len(raw))
            if manual := oracle.MANUAL_RAW_BOUNDARIES.get((task, episode_number)):
                rows = np.concatenate(
                    [
                        np.arange(8),
                        np.arange(manual[0], manual[0] + 10),
                        np.arange(manual[1], manual[1] + 10),
                        np.arange(manual[2], manual[2] + 8),
                    ]
                )
            decision = oracle.assess_source_episode(
                raw,
                len(raw) - 1,
                10,
                source_task_name=task,
                source_episode_number=episode_number,
                raw_observation_row=rows,
            )
            assert decision.included
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
                    raw_observation_row=rows,
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
    "raw",
    [
        _phases((1, 20), (2, 20), (1, 10), (2, 10), (3, 20), (2, 10), (3, 10), (4, 20), (3, 10)),
        # The first entries into phases 3 and 4 both follow a regression to phase 1.
        _phases((1, 20), (2, 20), (1, 20), (3, 20), (1, 20), (4, 20), (3, 10)),
        # A later adjacent 2->3 or 3->4 must not replace the earlier phase entry.
        _phases((1, 20), (2, 20), (1, 20), (3, 10), (2, 10), (3, 10), (2, 10), (4, 10), (3, 10), (4, 10)),
    ],
)
def test_first_transitions_repair_excluded_episode_and_round_trip(tmp_path: pathlib.Path, raw: np.ndarray) -> None:
    automatic = oracle.assess_episode(raw, len(raw) - 1, 10)
    assert not automatic.included
    decision = oracle.assess_source_episode(
        raw,
        len(raw) - 1,
        10,
        source_task_name="bimanual_pick_fork",
        source_episode_number=91,
    )
    assert decision.included
    assert decision.boundaries == (20, 60, 100)
    assert decision.cleaned.clean[40] == 2  # Ignore the later return to phase 1.
    assert decision.cleaned.clean[120] == 4  # Keep phase 4 after its first transition.
    assert decision.phase_source == "first_transitions"
    assert decision.automatic_exclusion_reasons == automatic.reasons
    kwargs = {
        "source_task_name": "bimanual_pick_fork",
        "source_episode_number": 91,
        "lerobot_episode_index": 0,
        "global_start_index": 0,
        "overall_instruction": "Pick up the fork.",
        "raw_phases": raw,
        "decision": decision,
    }
    audit = oracle.make_episode_audit(**kwargs, replan_steps=10)
    records = oracle.make_annotation_records(**kwargs)
    _write_sidecar(tmp_path / "sidecar", [audit], records)
    _, loaded_records, quality = oracle.load_and_validate_sidecar(tmp_path / "sidecar")
    assert loaded_records == records
    assert quality["episodes"][0]["automatic_exclusion_reasons"] == list(automatic.reasons)
    assert loaded_records[40]["raw_phase_before"] == 1
    assert loaded_records[40]["clean_phase"] == 2


@pytest.mark.parametrize(
    "raw",
    [
        _phases((1, 20), (3, 20), (4, 20)),  # A genuinely missing phase must not be invented.
        _phases((1, 20), (2, 20), (4, 20), (3, 20), (4, 20), (2, 20), (3, 20)),  # First hits out of order.
        _phases((1, 20), (2, 6), (3, 20), (2, 20), (3, 20), (4, 20)),  # Still ambiguous in a K-step window.
    ],
)
def test_first_transition_fallback_keeps_unusable_episodes_excluded(raw: np.ndarray) -> None:
    decision = oracle.assess_source_episode(
        raw,
        len(raw) - 1,
        10,
        source_task_name="bimanual_pick_fork",
        source_episode_number=91,
    )
    assert not decision.included
    assert not decision.anchors


@pytest.mark.parametrize(
    ("task", "episode_number", "expected"),
    [
        ("bimanual_pick_fork", 90, (40, 72, 85)),
        ("bimanual_pick_plate", 12, (33, 71, 82)),
        ("bimanual_pick_plate", 49, (37, 70, 86)),
        ("bimanual_pick_plate", 68, (27, 70, 85)),
    ],
)
def test_manual_raw_boundaries_map_to_effective_nodes_and_round_trip(
    tmp_path: pathlib.Path,
    task: str,
    episode_number: int,
    expected: tuple[int, ...],
) -> None:
    rows = np.arange(0, 300, 3)  # Several manual boundary frames lie inside merged zero-step groups.
    raw = np.ones(len(rows), dtype=np.int8)
    decision = oracle.assess_source_episode(
        raw,
        len(raw) - 1,
        10,
        source_task_name=task,
        source_episode_number=episode_number,
        raw_observation_row=rows,
    )
    assert decision.included
    assert decision.phase_source == "manual_raw_boundaries"
    assert decision.boundaries == expected
    for phase, boundary in enumerate(expected, start=2):
        assert decision.cleaned.clean[boundary - 1] == phase - 1
        assert decision.cleaned.clean[boundary] == phase
    kwargs = {
        "source_task_name": task,
        "source_episode_number": episode_number,
        "lerobot_episode_index": 0,
        "global_start_index": 0,
        "overall_instruction": "Pick up the object.",
        "raw_phases": raw,
        "decision": decision,
    }
    audit = oracle.make_episode_audit(**kwargs, replan_steps=10, raw_observation_row=rows)
    records = oracle.make_annotation_records(**kwargs)
    _write_sidecar(tmp_path / "sidecar", [audit], records)
    _, loaded_records, quality = oracle.load_and_validate_sidecar(tmp_path / "sidecar")
    assert loaded_records == records
    assert quality["episodes"][0]["raw_observation_row"] == rows.tolist()


def test_manual_boundaries_override_an_automatic_pass() -> None:
    raw = _phases((1, 50), (2, 50), (3, 50), (4, 150))
    assert oracle.assess_episode(raw, len(raw) - 1, 10).included
    decision = oracle.assess_source_episode(
        raw,
        len(raw) - 1,
        10,
        source_task_name="bimanual_pick_fork",
        source_episode_number=90,
    )
    assert decision.included
    assert decision.boundaries == (120, 214, 253)
    assert decision.phase_source == "manual_raw_boundaries"


@pytest.fixture
def automatically_cleaned_sidecar(tmp_path: pathlib.Path) -> pathlib.Path:
    valid_raw = _phases((1, 20), (2, 3), (1, 20), (2, 30), (3, 30), (4, 30))
    episodes = []
    records = []
    for episode_index, raw in enumerate((valid_raw, np.ones_like(valid_raw))):
        decision = oracle.assess_episode(raw, len(raw) - 1, 10)
        kwargs = {
            "source_task_name": "bimanual_pick_plate",
            "source_episode_number": episode_index,
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
    directory = tmp_path / "cleaned_sidecar"
    _write_sidecar(directory, episodes, records)
    return directory


def test_automatic_sidecar_round_trips_with_raw_label_audit(automatically_cleaned_sidecar: pathlib.Path) -> None:
    manifest, records, quality = oracle.load_and_validate_sidecar(automatically_cleaned_sidecar)
    assert manifest["included_episode_count"] == manifest["excluded_episode_count"] == 1
    assert manifest["included_anchor_count"] == len(records) == 132
    assert {record["source_episode_number"] for record in records} == {0}
    assert records[43]["raw_phase_before"] == 2
    assert records[43]["clean_phase"] == 2
    assert records[43]["subtask_type"] == "1_to_2"
    assert records[20]["raw_phase_before"] == 2
    assert records[20]["clean_phase"] == 1
    audit = quality["episodes"][0]
    assert audit["boundaries"] == [43, 73, 103]
    assert audit["raw_runs"] != audit["clean_runs"]
    assert audit["suppressed_segments"] == [
        {"stable_phase": 1, "candidate_phase": 2, "start": 20, "end": 23, "length": 3}
    ]
    assert audit["longest_unconfirmed_deviation"] == 3


def test_validator_rejects_resealed_wrong_boundary(automatically_cleaned_sidecar: pathlib.Path) -> None:
    quality_path = automatically_cleaned_sidecar / oracle.QUALITY_REPORT_FILENAME
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    audit = quality["episodes"][0]
    audit["boundaries"][0] = 44
    audit["clean_runs"][0].update(end=44, length=44)
    audit["clean_runs"][1].update(start=44, length=29)
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    _reseal_manifest(automatically_cleaned_sidecar)
    with pytest.raises(ValueError, match="Clean phase sequence mismatch"):
        oracle.load_and_validate_sidecar(automatically_cleaned_sidecar)


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
    for episode_index, episode_number in enumerate((0, 13, 3)):
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
    ("source_action_metadata", "expected_included"),
    [
        ({"action_semantics": "next_observed_joint_position_observed_gripper"}, True),
        ({"action_semantics": "executed_joint_target_commanded_gripper"}, True),
        ({"action_semantics": "custom_action_definition"}, True),
        ({"action_semantics": ""}, True),
        ({"action_semantics": "  "}, True),
        ({"action_semantics": None}, True),
        ({"action_semantics": 123}, True),
        ({"action_semantics": {"custom": ["any", "format"]}}, True),
        ({}, True),
        ({"action_semantics": "executed_joint_target_commanded_gripper_effective_v2"}, True),
        ({"action_semantics": "executed_joint_target_commanded_gripper_effective_v2"}, False),
        ({"action_semantics": "executed_joint_target_commanded_gripper"}, False),
        ({}, False),
    ],
    ids=[
        "observed",
        "executed",
        "custom",
        "empty",
        "whitespace",
        "null",
        "integer",
        "object",
        "missing",
        "effective_included",
        "effective_excluded",
        "raw_excluded",
        "missing_excluded",
    ],
)
def test_generator_preserves_source_action_semantics_through_sealing_and_loading(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source_action_metadata: dict, *, expected_included: bool
) -> None:
    source_dir = tmp_path / "export"
    task = "bimanual_pick_plate"
    episode_number = 16
    valid_raw = _phases((1, 8), (2, 3), (1, 8), (2, 20), (3, 20), (4, 20))
    raw = valid_raw if expected_included else _phases((1, 360))
    source_episodes = [(episode_number, raw)]
    if not expected_included:
        # Keep one valid peer: the existing sidecar contract rejects an empty training view.
        source_episodes.append((17, valid_raw))
    prompt = "Pick up the plate."
    state_parts, action_parts, episode_indices, frame_indices = [], [], [], []
    for episode_index, (source_number, phases) in enumerate(source_episodes):
        episode_dir = source_dir / "train" / task / f"episode{source_number}"
        episode_dir.mkdir(parents=True)
        observations = np.arange(len(phases) * 16, dtype=np.float32).reshape(-1, 16)
        states, actions = observations[:-1], observations[1:]
        for name, values in (
            ("state", states),
            ("actions", actions),
            ("phase_before_action", phases[:-1]),
            ("phase_after_action", phases[1:]),
        ):
            np.save(episode_dir / f"{name}.npy", values)
        metadata = {
            "task": prompt,
            "source_task_name": task,
            "source_episode_number": source_number,
            "phase_semantics": oracle.PHASE_SEMANTICS,
            "num_observations": len(phases),
            "num_transitions": len(actions),
            **source_action_metadata,
        }
        if metadata.get("action_semantics") == "executed_joint_target_commanded_gripper_effective_v2":
            metadata["effective_index_file"] = "effective_index.npz"
            np.savez(episode_dir / "effective_index.npz", raw_observation_row=np.arange(len(phases)) * 2)
        (episode_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        state_parts.append(states)
        action_parts.append(actions)
        episode_indices.extend([episode_index] * len(actions))
        frame_indices.extend(range(len(actions)))
    states, actions = np.concatenate(state_parts), np.concatenate(action_parts)
    num_actions = len(actions)

    class FakeHfDataset(dict):
        _fingerprint = "synthetic-fingerprint"

        def __len__(self):
            return num_actions

    hf_dataset = FakeHfDataset(
        state=states,
        actions=actions,
        episode_index=np.asarray(episode_indices),
        frame_index=np.asarray(frame_indices),
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
    assert manifest["num_dataset_rows"] == num_actions
    assert len(records) == len(valid_raw) - 1
    assert manifest["included_episode_count"] == 1
    assert manifest["excluded_episode_count"] == int(not expected_included)
    assert [audit["action_semantics"] for audit in quality["episodes"]] == [
        source_action_metadata.get("action_semantics")
    ] * len(source_episodes)
    audit = quality["episodes"][0]
    if expected_included:
        assert audit["boundaries"] == [19, 39, 59]
        assert audit["longest_unconfirmed_deviation"] == 3
        assert [record["frame_index"] for record in records] == list(range(num_actions))
        assert {record["subtask_type"] for record in records} == set(oracle.SUBTASK_TYPES)
        assert records[8]["raw_phase_before"] == 2
        assert records[8]["clean_phase"] == 1
        assert records[19]["subtask_type"] == "1_to_2"
        assert records[29]["subtask_type"] == "2"
    else:
        assert audit["status"] == "excluded"
        assert audit["reasons"] == ["does_not_end_in_phase_4", "missing_phase"]
        assert {record["source_episode_number"] for record in records} == {17}
        assert records[0]["global_index"] == len(raw) - 1
        assert quality["episodes"][1]["status"] == "included"


def test_validator_accepts_omitted_action_semantics(
    automatically_cleaned_sidecar: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _, records, quality = oracle.load_and_validate_sidecar(automatically_cleaned_sidecar)
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
