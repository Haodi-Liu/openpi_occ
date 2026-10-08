"""Generate the sealed RLBench oracle-phase text sidecar without model inference."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import re
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import tyro

from openpi.shared import rlbench_timing
from openpi.training import oracle_phase_sidecar as oracle

_EPISODE_RE = re.compile(r"episode(\d+)$")
_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    source_export_dir: str
    output_dir: str


def _sorted_source_episode_dirs(split_dir: pathlib.Path) -> list[pathlib.Path]:
    """Mirror build_rlbench_lerobot.py's task/episode lexical ordering."""
    return sorted(
        [item for item in split_dir.glob("*/*") if item.is_dir()],
        key=lambda item: (item.parent.name, item.name),
    )


def _validate_source_layout(split_dir: pathlib.Path) -> list[pathlib.Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Source train split not found: {split_dir}")
    task_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    task_names = {path.name for path in task_dirs}
    unsupported_tasks = sorted(task_names - set(oracle.TASKS))
    if not task_names or unsupported_tasks:
        raise ValueError(
            f"Source export must contain a non-empty subset of the supported oracle tasks {oracle.TASKS}; "
            f"unsupported={unsupported_tasks}."
        )
    for task_dir in task_dirs:
        episode_dirs = [path for path in task_dir.iterdir() if path.is_dir()]
        if not episode_dirs:
            raise ValueError(f"Source task has no episode directories: {task_dir}.")
        invalid_episode_names = sorted(path.name for path in episode_dirs if _EPISODE_RE.fullmatch(path.name) is None)
        if invalid_episode_names:
            raise ValueError(f"Invalid episode directory names for {task_dir.name}: {invalid_episode_names}.")
    return _sorted_source_episode_dirs(split_dir)


def _load_source_episode(
    episode_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    required_paths = {
        name: episode_dir / name
        for name in (
            "state.npy",
            "actions.npy",
            "phase_before_action.npy",
            "phase_after_action.npy",
            "meta.json",
        )
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts in {episode_dir}: {missing}.")

    states = np.asarray(np.load(required_paths["state.npy"], allow_pickle=False), dtype=np.float32)
    actions = np.asarray(np.load(required_paths["actions.npy"], allow_pickle=False), dtype=np.float32)
    phase_before = np.load(required_paths["phase_before_action.npy"], allow_pickle=False)
    phase_after = np.load(required_paths["phase_after_action.npy"], allow_pickle=False)
    try:
        metadata = json.loads(required_paths["meta.json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid source metadata JSON: {required_paths['meta.json']}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Source metadata must be an object: {required_paths['meta.json']}")

    if states.ndim != 2 or states.shape[1:] != (16,) or actions.shape != states.shape:
        raise ValueError(f"Invalid state/action shapes in {episode_dir}: {states.shape}, {actions.shape}.")
    if len(states) == 0 or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Empty or non-finite state/actions in {episode_dir}.")
    raw_phases = oracle.reconstruct_observation_phases(phase_before, phase_after)
    if len(raw_phases) != len(states) + 1:
        raise ValueError(f"Phase/action length mismatch in {episode_dir}.")
    return states, actions, phase_before, phase_after, metadata


def _strict_metadata_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Source metadata {name} must be an integer, got {value!r}.")
    return value


def _load_raw_observation_rows(episode_dir: pathlib.Path, metadata: dict[str, Any]) -> np.ndarray:
    index_file = metadata.get("effective_index_file")
    if index_file:
        with np.load(episode_dir / index_file, allow_pickle=False) as index:
            rows = index["raw_observation_row"]
    else:
        if metadata.get("action_semantics") == "executed_joint_target_commanded_gripper_effective_v2":
            raise ValueError(f"Missing effective_index_file in {episode_dir}.")
        rows = None
    return oracle.validate_raw_observation_rows(rows, metadata["num_observations"])


def _validate_source_metadata(
    episode_dir: pathlib.Path,
    metadata: dict[str, Any],
    num_actions: int,
) -> tuple[str, int, str]:
    required = {
        "task",
        "source_task_name",
        "source_episode_number",
        "phase_semantics",
        "num_observations",
        "num_transitions",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Source metadata is missing fields in {episode_dir}: {missing}.")
    task_name = str(metadata["source_task_name"])
    if task_name != episode_dir.parent.name or task_name not in oracle.TASKS:
        raise ValueError(f"Source task identity mismatch in {episode_dir}.")
    match = _EPISODE_RE.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"Invalid source episode directory name: {episode_dir.name!r}.")
    episode_number = _strict_metadata_int(metadata["source_episode_number"], "source_episode_number")
    if episode_number != int(match.group(1)):
        raise ValueError(f"Source episode number mismatch in {episode_dir}.")
    if metadata["phase_semantics"] != oracle.PHASE_SEMANTICS:
        raise ValueError(f"Source phase semantics mismatch in {episode_dir}.")
    if _strict_metadata_int(metadata["num_transitions"], "num_transitions") != num_actions:
        raise ValueError(f"Source transition count mismatch in {episode_dir}.")
    if _strict_metadata_int(metadata["num_observations"], "num_observations") != num_actions + 1:
        raise ValueError(f"Source observation count mismatch in {episode_dir}.")
    task_prompt = metadata["task"]
    if not isinstance(task_prompt, str) or not task_prompt:
        raise ValueError(f"Source task prompt must be a non-empty string in {episode_dir}.")
    return task_name, episode_number, task_prompt


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _stack_vector_column(hf_dataset: Any, key: str) -> np.ndarray:
    column = hf_dataset[key]
    if len(column) == 0:
        raise ValueError(f"LeRobot column {key!r} is empty.")
    values = np.stack([_to_numpy(value) for value in column]).astype(np.float32, copy=False)
    if values.ndim != 2 or values.shape[1:] != (16,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid LeRobot {key!r} column shape or values: {values.shape}.")
    return values


def _integer_column(hf_dataset: Any, key: str) -> list[int]:
    values = []
    for row, value in enumerate(hf_dataset[key]):
        array = _to_numpy(value)
        if array.size != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
            raise ValueError(f"LeRobot {key!r} row {row} is not one integer scalar.")
        values.append(int(array.reshape(-1)[0]))
    return values


def verify_lerobot_episode_identity(
    *,
    global_start_index: int,
    expected_episode_index: int,
    source_states: np.ndarray,
    source_actions: np.ndarray,
    task_prompt: str,
    dataset_states: np.ndarray,
    dataset_actions: np.ndarray,
    episode_indices: list[int],
    frame_indices: list[int],
    task_indices: list[int],
    task_prompts: dict[int, str],
) -> None:
    """Require exact row identity before attaching source-only phase labels."""
    end = global_start_index + len(source_states)
    if end > len(dataset_states) or end > len(dataset_actions):
        raise ValueError(f"LeRobot dataset ends inside source episode {expected_episode_index}.")
    if episode_indices[global_start_index:end] != [expected_episode_index] * len(source_states):
        raise ValueError(f"LeRobot episode_index mismatch for source episode {expected_episode_index}.")
    if frame_indices[global_start_index:end] != list(range(len(source_states))):
        raise ValueError(f"LeRobot frame_index mismatch for source episode {expected_episode_index}.")
    for global_index in range(global_start_index, end):
        if task_prompts.get(task_indices[global_index]) != task_prompt:
            raise ValueError(f"LeRobot task prompt mismatch at global_index={global_index}.")
    if not np.array_equal(dataset_states[global_start_index:end], source_states):
        raise ValueError(f"LeRobot state values mismatch for source episode {expected_episode_index}.")
    if not np.array_equal(dataset_actions[global_start_index:end], source_actions):
        raise ValueError(f"LeRobot action values mismatch for source episode {expected_episode_index}.")


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_sealed_sidecar(
    output_dir: pathlib.Path,
    records: list[dict[str, Any]],
    quality: dict[str, Any],
    unsealed_manifest: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Oracle sidecar output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    annotations_path = output_dir / oracle.ANNOTATIONS_FILENAME
    quality_path = output_dir / oracle.QUALITY_REPORT_FILENAME
    manifest_path = output_dir / oracle.MANIFEST_FILENAME
    annotations_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    )
    quality_text = json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(annotations_path, annotations_text)
    _atomic_write_text(quality_path, quality_text)
    sealed = oracle.seal_manifest(unsealed_manifest, annotations_path, quality_path)
    _atomic_write_text(manifest_path, json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(args: Args) -> None:
    timing = rlbench_timing.load_runtime_contract()
    source_split_dir = pathlib.Path(args.source_export_dir).expanduser().resolve() / "train"
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Oracle sidecar output already exists: {output_dir}")
    episode_dirs = _validate_source_layout(source_split_dir)

    dataset_metadata = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(args.repo_id)
    if dataset_metadata.root.resolve() != dataset.root.resolve():
        raise ValueError("LeRobot metadata and rows resolved to different dataset roots.")
    hf_dataset = dataset.hf_dataset
    dataset_fingerprint = getattr(hf_dataset, "_fingerprint", None)
    if dataset_fingerprint is None:
        raise ValueError("HuggingFace dataset fingerprint is unavailable.")
    task_prompts = {int(key): str(value) for key, value in dataset_metadata.tasks.items()}

    episode_indices = _integer_column(hf_dataset, "episode_index")
    frame_indices = _integer_column(hf_dataset, "frame_index")
    task_indices = _integer_column(hf_dataset, "task_index")
    dataset_states = _stack_vector_column(hf_dataset, "state")
    dataset_actions = _stack_vector_column(hf_dataset, "actions")
    dataset_rows = len(hf_dataset)
    if not (
        len(episode_indices)
        == len(frame_indices)
        == len(task_indices)
        == len(dataset_states)
        == len(dataset_actions)
        == dataset_rows
    ):
        raise ValueError("LeRobot row columns have inconsistent lengths.")

    records: list[dict[str, Any]] = []
    episode_audits: list[dict[str, Any]] = []
    global_cursor = 0
    for expected_episode_index, episode_dir in enumerate(episode_dirs):
        states, actions, phase_before, phase_after, metadata = _load_source_episode(episode_dir)
        task_name, episode_number, task_prompt = _validate_source_metadata(episode_dir, metadata, len(actions))
        verify_lerobot_episode_identity(
            global_start_index=global_cursor,
            expected_episode_index=expected_episode_index,
            source_states=states,
            source_actions=actions,
            task_prompt=task_prompt,
            dataset_states=dataset_states,
            dataset_actions=dataset_actions,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            task_indices=task_indices,
            task_prompts=task_prompts,
        )
        raw_phases = oracle.reconstruct_observation_phases(phase_before, phase_after)
        raw_observation_row = _load_raw_observation_rows(episode_dir, metadata)
        decision = oracle.assess_source_episode(
            raw_phases,
            len(actions),
            timing.replan_steps,
            source_task_name=task_name,
            source_episode_number=episode_number,
            raw_observation_row=raw_observation_row,
        )
        episode_audits.append(
            oracle.make_episode_audit(
                source_task_name=task_name,
                source_episode_number=episode_number,
                lerobot_episode_index=expected_episode_index,
                global_start_index=global_cursor,
                overall_instruction=task_prompt,
                action_semantics=metadata.get("action_semantics"),
                raw_phases=raw_phases,
                decision=decision,
                replan_steps=timing.replan_steps,
                raw_observation_row=raw_observation_row,
            )
        )
        if decision.phase_source != "automatic" or not decision.included:
            _LOG.info(
                "%s/episode%d: %s via %s; effective_boundaries=%s; raw_observation_boundaries=%s; reasons=%s",
                task_name,
                episode_number,
                "included" if decision.included else "excluded",
                decision.phase_source,
                decision.boundaries,
                tuple(int(raw_observation_row[index]) for index in decision.boundaries),
                decision.reasons,
            )
        records.extend(
            oracle.make_annotation_records(
                source_task_name=task_name,
                source_episode_number=episode_number,
                lerobot_episode_index=expected_episode_index,
                global_start_index=global_cursor,
                overall_instruction=task_prompt,
                raw_phases=raw_phases,
                decision=decision,
            )
        )
        global_cursor += len(actions)

    if global_cursor != dataset_rows:
        raise ValueError(f"Source export has {global_cursor} rows but LeRobot has {dataset_rows}.")
    quality = oracle.make_quality_report(episode_audits)
    unsealed_manifest = oracle.make_unsealed_manifest(
        repo_id=args.repo_id,
        hf_dataset_fingerprint=str(dataset_fingerprint),
        num_dataset_rows=dataset_rows,
        action_horizon=timing.action_horizon,
        replan_steps=timing.replan_steps,
        action_horizon_source=timing.action_horizon_source,
        replan_steps_source=timing.replan_steps_source,
        episodes=episode_audits,
        records=records,
    )
    _write_sealed_sidecar(output_dir, records, quality, unsealed_manifest)

    sealed_manifest, sealed_records, sealed_quality = oracle.load_and_validate_sidecar(output_dir)
    oracle.validate_dataset_binding(
        sealed_manifest,
        sealed_records,
        sealed_quality,
        repo_id=args.repo_id,
        fingerprint=str(dataset_fingerprint),
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        task_indices=task_indices,
        task_prompts=task_prompts,
    )
    rlbench_timing.validate_manifest_timing(sealed_manifest, timing)
    _LOG.info(
        "Wrote %s: %d/%d episodes and %d/%d anchors included.",
        output_dir,
        sealed_manifest["included_episode_count"],
        sealed_manifest["candidate_episode_count"],
        sealed_manifest["included_anchor_count"],
        sealed_manifest["candidate_anchor_count"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
