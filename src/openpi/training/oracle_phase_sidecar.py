"""Deterministic oracle-phase cleaning, annotation, and sidecar validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import itertools
import json
import pathlib
import re
from typing import Any, SupportsIndex

import numpy as np

from openpi.shared import rlbench_timing

PROTOCOL = "oracle_phase_dense_text"
SCHEMA_VERSION = 1
LANGUAGE_CONDITION = "oracle_phase_text"
ACTION_LOSS_PROTOCOL = "pi05_full_action_chunk"
SAMPLING_PROTOCOL = "uniform_included_frame_shuffle"
ACTION_TRAINABLE_PROTOCOL = "pi05_action_allowlist"
FULL_MODEL_TRAINABLE_PROTOCOL = "pi05_full_model_finetune"
TRAIN_CONFIG_NAME = "pi05_rlbench_oracle_phase"
TASKS = (
    "bimanual_edge_phone",
    "bimanual_pick_fork",
    "bimanual_pick_plate",
    "bimanual_pivot_phone",
)
# Retained only to reproduce and validate the already-sealed all-four-task artifacts.
# New subsets and episode layouts are inferred from the source export instead.
LEGACY_EXPECTED_EPISODES_PER_TASK = 150
PHASE_VALUES = (1, 2, 3, 4)
SUBTASK_TYPES = ("1", "1_to_2", "2", "2_to_3", "3", "3_to_4", "4")
SUBTASK_TEXTS = {
    "bimanual_edge_phone": {
        "1": (
            "Use the pushing arm to push the phone farther over the box edge, creating an overhanging end for "
            "the grasping arm."
        ),
        "1_to_2": ("The phone has a stable overhanging end; now begin grasping it with the grasping arm."),
        "2": ("Keep the phone stably overhanging with the pushing arm while the grasping arm secures the exposed end."),
        "2_to_3": "The phone is securely grasped; now stop pushing and move the pushing arm away.",
        "3": ("Hold the phone securely with the grasping arm while moving the pushing arm clear of the lifting path."),
        "3_to_4": ("The pushing arm is clear of the lifting path; now begin lifting the phone with the grasping arm."),
        "4": "Lift the phone with the grasping arm while maintaining a secure grasp.",
    },
    "bimanual_pick_fork": {
        "1": "Use the pressing arm to press down the fork head, raising the handle for the grasping arm.",
        "1_to_2": "The fork handle is raised; now begin grasping it with the grasping arm.",
        "2": ("Keep pressing the fork head so the handle stays raised while the grasping arm secures the handle."),
        "2_to_3": ("The fork handle is securely grasped; now release pressure and move the pressing arm away."),
        "3": ("Hold the fork securely with the grasping arm while moving the pressing arm clear of the lifting path."),
        "3_to_4": ("The pressing arm is clear of the lifting path; now begin lifting the fork with the grasping arm."),
        "4": "Lift the fork with the grasping arm while maintaining a secure grasp.",
    },
    "bimanual_pick_plate": {
        "1": (
            "Use the pressing arm to press down one edge of the plate, tilting the opposite edge upward for the "
            "grasping arm."
        ),
        "1_to_2": "The opposite edge of the plate is raised; now begin grasping it with the grasping arm.",
        "2": (
            "Keep pressing one edge with the pressing arm so the opposite edge stays raised while the grasping arm "
            "secures it."
        ),
        "2_to_3": ("The raised edge is securely grasped; now release pressure and move the pressing arm away."),
        "3": ("Hold the plate securely with the grasping arm while moving the pressing arm clear of the lifting path."),
        "3_to_4": ("The pressing arm is clear of the lifting path; now begin lifting the plate with the grasping arm."),
        "4": "Lift the plate with the grasping arm while maintaining a secure grasp.",
    },
    "bimanual_pivot_phone": {
        "1": (
            "Use the pushing arm to push the phone against the wall and pivot it upward, exposing a raised edge for "
            "the grasping arm."
        ),
        "1_to_2": (
            "The phone is stably pivoted upward against the wall; now begin grasping the raised edge with the "
            "grasping arm."
        ),
        "2": (
            "Keep the phone tilted and stable against the wall with the pushing arm while the grasping arm secures "
            "the raised edge."
        ),
        "2_to_3": "The phone is securely grasped; now stop pushing and move the pushing arm away.",
        "3": ("Hold the phone securely with the grasping arm while moving the pushing arm clear of the lifting path."),
        "3_to_4": ("The pushing arm is clear of the lifting path; now begin lifting the phone with the grasping arm."),
        "4": "Lift the phone with the grasping arm while maintaining a secure grasp.",
    },
}

ANNOTATIONS_FILENAME = "annotations.jsonl"
QUALITY_REPORT_FILENAME = "quality_report.json"
MANIFEST_FILENAME = "manifest.json"
PHASE_SEMANTICS = "observation_phase_type_before_after_action"
PERSISTENCE_FORMULA = "ceil(0.6 * replan_steps)"
PHASE_ALIGNMENT = "phase_before_action[t]=observation[t];phase_after_action[t]=observation[t+1]"
BOUNDARY_RULE = (
    "confirm_after_p_consecutive_observations_then_backdate_to_candidate_start"
    "_then_apply_manual_review_to_excluded_episodes"
)
ANCHOR_RULE = "every_action_frame_uses_one_transition_from_previous_open_closed_k_window_else_current_phase"
ROLE_WORD_POLICY = "task_specific_functional_arm_roles"

# Applied only after automatic exclusion. Keys use source episode numbers, not LeRobot indices.
# Each boundary is the zero-based first observation/action frame of the new phase; None means bad demo.
MANUAL_PHASE_REVIEW: dict[tuple[str, int], tuple[int, int, int] | None] = {
    ("bimanual_edge_phone", 144): None,
    ("bimanual_pick_fork", 0): (77, 190, 228),
    ("bimanual_pick_fork", 1): (82, 185, 223),
    ("bimanual_pick_fork", 4): None,
    ("bimanual_pick_fork", 14): (84, 191, 237),
    ("bimanual_pick_fork", 15): (76, 185, 222),
    ("bimanual_pick_fork", 18): (72, 189, 224),
    ("bimanual_pick_fork", 21): (83, 194, 237),
    ("bimanual_pick_fork", 23): (67, 209, 244),
    ("bimanual_pick_fork", 35): (83, 187, 227),
    ("bimanual_pick_fork", 38): (77, 189, 226),
    ("bimanual_pick_fork", 59): (82, 239, 278),
    ("bimanual_pick_fork", 82): (99, 215, 256),
    ("bimanual_pick_fork", 85): (85, 184, 222),
    ("bimanual_pick_fork", 87): (87, 192, 237),
    ("bimanual_pick_fork", 95): (92, 198, 243),
    ("bimanual_pick_fork", 109): (71, 206, 242),
    ("bimanual_pick_fork", 113): (91, 190, 242),
    ("bimanual_pick_plate", 15): None,
    ("bimanual_pick_plate", 16): (90, 242, 266),
    ("bimanual_pick_plate", 26): (86, 235, 280),
    ("bimanual_pick_plate", 43): (86, 223, 260),
    ("bimanual_pick_plate", 51): (77, 225, 270),
    ("bimanual_pick_plate", 52): (109, 228, 279),
    ("bimanual_pick_plate", 59): (100, 252, 285),
    ("bimanual_pick_plate", 81): (75, 227, 265),
    ("bimanual_pick_plate", 86): (80, 175, 281),
    ("bimanual_pick_plate", 92): None,
    ("bimanual_pick_plate", 95): (88, 205, 245),
    ("bimanual_pick_plate", 104): (104, 220, 261),
    ("bimanual_pick_plate", 107): (82, 211, 249),
    ("bimanual_pick_plate", 112): (86, 220, 262),
    ("bimanual_pick_plate", 129): (77, 239, 278),
    ("bimanual_pick_plate", 148): None,
    ("bimanual_pivot_phone", 66): (102, 200, 263),
}

EXCLUSION_REASONS = (
    "does_not_start_in_phase_1",
    "does_not_end_in_phase_4",
    "non_monotonic_phase_path",
    "non_adjacent_transition",
    "missing_phase",
    "confirmed_run_shorter_than_persistence",
    "long_unconfirmed_oscillation",
    "multiple_transitions_in_lookback_window",
)

_ANNOTATION_FIELDS = {
    "global_index",
    "episode_index",
    "frame_index",
    "source_task_name",
    "source_episode_number",
    "overall_instruction",
    "raw_phase_before",
    "clean_phase",
    "subtask_type",
    "subtask",
}
_QUALITY_FIELDS = {"protocol", "schema_version", "expected_episodes_per_task", "episodes"}
_EPISODE_AUDIT_FIELDS = {
    "source_task_name",
    "source_episode_number",
    "source_episode_relpath",
    "lerobot_episode_index",
    "global_start_index",
    "overall_instruction",
    "num_actions",
    "phase_semantics",
    "persistence",
    "candidate_anchor_count",
    "included_anchor_count",
    "status",
    "reasons",
    "raw_runs",
    "clean_runs",
    "boundaries",
    "suppressed_segments",
    "longest_unconfirmed_deviation",
}
_RUN_FIELDS = {"phase", "start", "end", "length"}
_SUPPRESSED_FIELDS = {"stable_phase", "candidate_phase", "start", "end", "length"}
_BASE_MANIFEST_FIELDS = {
    "protocol",
    "schema_version",
    "status",
    "tasks",
    "expected_episodes_per_task",
    "repo_id",
    "hf_dataset_fingerprint",
    "num_dataset_rows",
    "action_horizon",
    "replan_steps",
    "action_horizon_source",
    "replan_steps_source",
    "persistence",
    "persistence_formula",
    "phase_alignment",
    "boundary_rule",
    "anchor_rule",
    "subtask_types",
    "subtask_texts",
    "subtask_texts_sha256",
    "role_word_policy",
    "source_content_sha256",
    "candidate_episode_count",
    "included_episode_count",
    "excluded_episode_count",
    "candidate_anchor_count",
    "included_anchor_count",
    "excluded_anchor_count",
}
_MANIFEST_FIELDS = _BASE_MANIFEST_FIELDS | {
    "annotations_sha256",
    "quality_report_sha256",
    "manifest_digest",
}
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True)
class PhaseRun:
    phase: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        return {"phase": self.phase, "start": self.start, "end": self.end, "length": self.length}


@dataclasses.dataclass(frozen=True)
class SuppressedSegment:
    stable_phase: int
    candidate_phase: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int]:
        return {
            "stable_phase": self.stable_phase,
            "candidate_phase": self.candidate_phase,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


@dataclasses.dataclass(frozen=True)
class CleanResult:
    clean: np.ndarray
    runs: tuple[PhaseRun, ...]
    suppressed: tuple[SuppressedSegment, ...]
    longest_unconfirmed_deviation: int


@dataclasses.dataclass(frozen=True)
class Anchor:
    frame_index: int
    subtask_type: str
    clean_phase: int


@dataclasses.dataclass(frozen=True)
class EpisodeDecision:
    included: bool
    reasons: tuple[str, ...]
    cleaned: CleanResult
    boundaries: tuple[int, ...]
    anchors: tuple[Anchor, ...]
    candidate_anchor_count: int


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_task_subset(tasks: Any, name: str) -> tuple[str, ...]:
    if isinstance(tasks, str | bytes) or not isinstance(tasks, Sequence) or not tasks:
        raise ValueError(f"{name} must be a non-empty task sequence.")
    if not all(isinstance(task, str) for task in tasks):
        raise ValueError(f"{name} must contain only task names.")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"{name} contains duplicate tasks.")
    unsupported = sorted(set(tasks) - set(TASKS))
    if unsupported:
        raise ValueError(f"{name} contains unsupported oracle tasks: {unsupported}.")
    canonical = tuple(task for task in TASKS if task in tasks)
    if tuple(tasks) != canonical:
        raise ValueError(f"{name} must follow the canonical supported-task order.")
    return canonical


def _subtask_texts_for_tasks(tasks: Sequence[str]) -> dict[str, dict[str, str]]:
    canonical = _canonical_task_subset(tasks, "oracle tasks")
    return {task: dict(SUBTASK_TEXTS[task]) for task in canonical}


def subtask_texts_sha256(tasks: Sequence[str] = TASKS) -> str:
    return canonical_sha256(_subtask_texts_for_tasks(tasks))


def persistence_for_replan_steps(replan_steps: int) -> int:
    replan_steps = _strict_positive_int(replan_steps, "replan_steps")
    return (3 * replan_steps + 4) // 5


def validate_phase_array(array: Any, name: str) -> np.ndarray:
    """Require a non-empty, one-dimensional integer array containing only 1..4."""
    value = np.asarray(array)
    if value.ndim != 1 or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array, got shape {value.shape}.")
    if value.dtype.kind not in "iu" or value.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers in 1..4, got dtype {value.dtype}.")
    if not np.isin(value, PHASE_VALUES).all():
        raise ValueError(f"{name} contains a phase outside 1..4.")
    return value.astype(np.int8, copy=False)


def reconstruct_observation_phases(before: Any, after: Any) -> np.ndarray:
    """Reconstruct all observation phases from action-aligned before/after arrays."""
    before_array = validate_phase_array(before, "phase_before_action")
    after_array = validate_phase_array(after, "phase_after_action")
    if before_array.shape != after_array.shape:
        raise ValueError(f"phase-before/after length mismatch: {before_array.shape} vs {after_array.shape}.")
    if not np.array_equal(before_array[1:], after_array[:-1]):
        raise ValueError("phase-before/after continuity mismatch.")
    return np.concatenate([before_array[:1], after_array]).astype(np.int8, copy=False)


def run_length_encode(phases: Any) -> tuple[PhaseRun, ...]:
    values = validate_phase_array(phases, "phases")
    starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(values)]))
    return tuple(
        PhaseRun(phase=int(values[start]), start=int(start), end=int(end))
        for start, end in zip(starts, ends, strict=True)
    )


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=np.bool_):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def clean_phases(raw: Any, persistence: int) -> CleanResult:
    """Confirm a new phase after P frames and backdate its boundary to the candidate start."""
    raw_array = validate_phase_array(raw, "raw_phases")
    persistence = _strict_positive_int(persistence, "persistence")
    clean = np.full(raw_array.shape, int(raw_array[0]), dtype=np.int8)
    stable = int(raw_array[0])
    candidate: int | None = None
    candidate_start = 0
    candidate_count = 0
    suppressed: list[SuppressedSegment] = []

    def suppress(end: int) -> None:
        nonlocal candidate, candidate_count
        if candidate is not None:
            suppressed.append(
                SuppressedSegment(
                    stable_phase=stable,
                    candidate_phase=candidate,
                    start=candidate_start,
                    end=end,
                )
            )
        candidate = None
        candidate_count = 0

    for index in range(1, len(raw_array)):
        phase = int(raw_array[index])
        if phase == stable:
            suppress(index)
            clean[index] = stable
            continue

        if phase != candidate:
            suppress(index)
            candidate = phase
            candidate_start = index
            candidate_count = 0
        candidate_count += 1
        clean[index] = stable

        if candidate_count == persistence:
            clean[candidate_start : index + 1] = candidate
            stable = candidate
            candidate = None
            candidate_count = 0

    suppress(len(raw_array))
    runs = run_length_encode(clean)
    return CleanResult(
        clean=clean,
        runs=runs,
        suppressed=tuple(suppressed),
        longest_unconfirmed_deviation=_longest_true_run(raw_array != clean),
    )


def _validate_phase_sequence_length(phases: Any, num_actions: int, name: str) -> np.ndarray:
    values = validate_phase_array(phases, name)
    num_actions = _strict_positive_int(num_actions, "num_actions")
    if len(values) != num_actions + 1:
        raise ValueError(f"Expected {name} length {num_actions + 1}, got {len(values)}.")
    return values


def _boundaries(clean: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.flatnonzero(clean[1:] != clean[:-1]) + 1)


def _boundaries_in_period(boundaries: Sequence[int], frame_index: int, replan_steps: int) -> tuple[int, ...]:
    return tuple(boundary for boundary in boundaries if frame_index - replan_steps < boundary <= frame_index)


def classify_anchors(clean: Any, num_actions: int, replan_steps: int) -> tuple[Anchor, ...]:
    """Classify every action frame from the previous open-closed K-step phase window."""
    clean_array = _validate_phase_sequence_length(clean, num_actions, "clean_phases")
    replan_steps = _strict_positive_int(replan_steps, "replan_steps")
    boundaries = _boundaries(clean_array)
    anchors: list[Anchor] = []
    for frame_index in range(num_actions):
        if frame_index == 0:
            phase = int(clean_array[0])
            anchors.append(Anchor(frame_index=0, subtask_type=str(phase), clean_phase=phase))
            continue
        recent = _boundaries_in_period(boundaries, frame_index, replan_steps)
        if len(recent) > 1:
            raise ValueError(f"Multiple phase boundaries in the K-step window ending at frame {frame_index}: {recent}.")
        phase = int(clean_array[frame_index])
        if not recent:
            anchors.append(Anchor(frame_index=frame_index, subtask_type=str(phase), clean_phase=phase))
            continue
        boundary = recent[0]
        previous = int(clean_array[boundary - 1])
        current = int(clean_array[boundary])
        anchors.append(
            Anchor(
                frame_index=frame_index,
                subtask_type=f"{previous}_to_{current}",
                clean_phase=phase,
            )
        )
    return tuple(anchors)


def assess_episode(raw: Any, num_actions: int, replan_steps: int) -> EpisodeDecision:
    """Return all uniquely typed anchors or deterministic whole-episode exclusion reasons."""
    raw_array = _validate_phase_sequence_length(raw, num_actions, "raw_phases")
    replan_steps = _strict_positive_int(replan_steps, "replan_steps")
    persistence = persistence_for_replan_steps(replan_steps)
    cleaned = clean_phases(raw_array, persistence)
    boundaries = _boundaries(cleaned.clean)
    run_values = tuple(run.phase for run in cleaned.runs)
    reasons: list[str] = []

    if run_values[0] != 1:
        reasons.append("does_not_start_in_phase_1")
    if run_values[-1] != 4:
        reasons.append("does_not_end_in_phase_4")
    if any(current < previous for previous, current in itertools.pairwise(run_values)):
        reasons.append("non_monotonic_phase_path")
    if any(abs(current - previous) != 1 for previous, current in itertools.pairwise(run_values)):
        reasons.append("non_adjacent_transition")
    if set(run_values) != set(PHASE_VALUES):
        reasons.append("missing_phase")
    if any(run.length < persistence for run in cleaned.runs):
        reasons.append("confirmed_run_shorter_than_persistence")
    if cleaned.longest_unconfirmed_deviation >= replan_steps:
        reasons.append("long_unconfirmed_oscillation")

    has_multiple = any(
        len(_boundaries_in_period(boundaries, frame_index, replan_steps)) > 1 for frame_index in range(num_actions)
    )
    if has_multiple:
        reasons.append("multiple_transitions_in_lookback_window")

    reasons = [reason for reason in EXCLUSION_REASONS if reason in reasons]
    candidate_anchor_count = num_actions
    included = not reasons and run_values == PHASE_VALUES
    anchors = classify_anchors(cleaned.clean, num_actions, replan_steps) if included else ()
    if included and len(anchors) != candidate_anchor_count:
        raise AssertionError("Internal anchor count mismatch.")
    return EpisodeDecision(
        included=included,
        reasons=tuple(reasons),
        cleaned=cleaned,
        boundaries=boundaries,
        anchors=anchors,
        candidate_anchor_count=candidate_anchor_count,
    )


def assess_source_episode(
    raw: Any,
    num_actions: int,
    replan_steps: int,
    *,
    source_task_name: str,
    source_episode_number: int,
) -> EpisodeDecision:
    """Run unchanged automatic cleaning, then apply the fixed review to excluded source episodes."""
    automatic = assess_episode(raw, num_actions, replan_steps)
    if automatic.included:
        return automatic
    identity = (source_task_name, source_episode_number)
    boundaries = MANUAL_PHASE_REVIEW.get(identity)
    if boundaries is None:
        return automatic
    if not 0 < boundaries[0] < boundaries[1] < boundaries[2] < num_actions:
        raise ValueError(f"Manual phase boundaries {boundaries} are outside source episode {identity}.")

    manual = np.ones(num_actions + 1, dtype=np.int8)
    for phase, boundary in enumerate(boundaries, start=2):
        manual[boundary:] = phase
    reviewed = assess_episode(manual, num_actions, replan_steps)
    if not reviewed.included:
        raise ValueError(f"Manual phase boundaries for {identity} fail quality checks: {reviewed.reasons}.")
    # Keep the raw-label cleaning diagnostics while using the reviewed phases for all training anchors.
    return dataclasses.replace(
        reviewed,
        cleaned=dataclasses.replace(
            reviewed.cleaned,
            suppressed=automatic.cleaned.suppressed,
            longest_unconfirmed_deviation=automatic.cleaned.longest_unconfirmed_deviation,
        ),
    )


def make_episode_audit(
    *,
    source_task_name: str,
    source_episode_number: int,
    lerobot_episode_index: int,
    global_start_index: int,
    overall_instruction: str,
    raw_phases: Any,
    decision: EpisodeDecision,
    replan_steps: int,
    action_semantics: Any = None,
) -> dict[str, Any]:
    """Create a portable episode audit preserving the source export's action semantics."""
    if source_task_name not in TASKS:
        raise ValueError(f"Unsupported oracle task: {source_task_name!r}.")
    source_episode_number = _strict_nonnegative_int(source_episode_number, "source_episode_number")
    lerobot_episode_index = _strict_nonnegative_int(lerobot_episode_index, "lerobot_episode_index")
    global_start_index = _strict_nonnegative_int(global_start_index, "global_start_index")
    if not isinstance(overall_instruction, str) or not overall_instruction:
        raise ValueError("overall_instruction must be a non-empty string.")
    raw_array = validate_phase_array(raw_phases, "raw_phases")
    num_actions = len(raw_array) - 1
    recomputed = assess_source_episode(
        raw_array,
        num_actions,
        replan_steps,
        source_task_name=source_task_name,
        source_episode_number=source_episode_number,
    )
    if _decision_signature(decision) != _decision_signature(recomputed):
        raise ValueError("Episode decision does not match the supplied raw phases.")
    persistence = persistence_for_replan_steps(replan_steps)
    return {
        "source_task_name": source_task_name,
        "source_episode_number": source_episode_number,
        "source_episode_relpath": f"train/{source_task_name}/episode{source_episode_number}",
        "lerobot_episode_index": lerobot_episode_index,
        "global_start_index": global_start_index,
        "overall_instruction": overall_instruction,
        "num_actions": num_actions,
        "action_semantics": action_semantics,
        "phase_semantics": PHASE_SEMANTICS,
        "persistence": persistence,
        "candidate_anchor_count": decision.candidate_anchor_count,
        "included_anchor_count": len(decision.anchors),
        "status": "included" if decision.included else "excluded",
        "reasons": list(decision.reasons),
        "raw_runs": [run.to_dict() for run in run_length_encode(raw_array)],
        "clean_runs": [run.to_dict() for run in decision.cleaned.runs],
        "boundaries": list(decision.boundaries),
        "suppressed_segments": [segment.to_dict() for segment in decision.cleaned.suppressed],
        "longest_unconfirmed_deviation": decision.cleaned.longest_unconfirmed_deviation,
    }


def make_annotation_records(
    *,
    source_task_name: str,
    source_episode_number: int,
    lerobot_episode_index: int,
    global_start_index: int,
    overall_instruction: str,
    raw_phases: Any,
    decision: EpisodeDecision,
) -> list[dict[str, Any]]:
    """Materialize the fixed text record for every included anchor."""
    if source_task_name not in TASKS:
        raise ValueError(f"Unsupported oracle task: {source_task_name!r}.")
    if not decision.included:
        return []
    raw_array = validate_phase_array(raw_phases, "raw_phases")
    texts = SUBTASK_TEXTS[source_task_name]
    records = []
    for anchor in decision.anchors:
        if anchor.frame_index >= len(raw_array) - 1:
            raise ValueError(f"Anchor {anchor.frame_index} is not an action row.")
        if anchor.subtask_type not in texts:
            raise ValueError(f"Unsupported subtask type {anchor.subtask_type!r} for {source_task_name}.")
        records.append(
            {
                "global_index": global_start_index + anchor.frame_index,
                "episode_index": lerobot_episode_index,
                "frame_index": anchor.frame_index,
                "source_task_name": source_task_name,
                "source_episode_number": source_episode_number,
                "overall_instruction": overall_instruction,
                "raw_phase_before": int(raw_array[anchor.frame_index]),
                "clean_phase": anchor.clean_phase,
                "subtask_type": anchor.subtask_type,
                "subtask": texts[anchor.subtask_type],
            }
        )
    return records


def _infer_episode_layout(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    if not episodes:
        raise ValueError("Oracle sidecar requires at least one source episode.")
    identities = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"Episode audit {index} must be a mapping.")
        task = episode.get("source_task_name")
        if not isinstance(task, str) or task not in TASKS:
            raise ValueError(f"Episode audit {index} has an unsupported source task: {task!r}.")
        number = _strict_nonnegative_int(episode.get("source_episode_number"), "source_episode_number")
        identities.append((task, number))

    if len(set(identities)) != len(identities):
        raise ValueError("Oracle source episodes contain duplicate task/episode identities.")
    active_tasks = tuple(task for task in TASKS if any(identity[0] == task for identity in identities))
    task_order = {task: index for index, task in enumerate(TASKS)}
    expected_order = sorted(identities, key=lambda identity: (task_order[identity[0]], f"episode{identity[1]}"))
    if identities != expected_order:
        raise ValueError("Oracle source episodes do not follow canonical task/episode lexical ordering.")
    return active_tasks, tuple(identities)


def _legacy_expected_episode_count(
    tasks: Sequence[str],
    identities: Sequence[tuple[str, int]],
) -> int | None:
    """Preserve the exact legacy marker only for the original all-four x 150 layout."""
    if tuple(tasks) != TASKS:
        return None
    expected_numbers = set(range(LEGACY_EXPECTED_EPISODES_PER_TASK))
    if all(
        {number for identity_task, number in identities if identity_task == task} == expected_numbers for task in tasks
    ):
        return LEGACY_EXPECTED_EPISODES_PER_TASK
    return None


def make_quality_report(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks, identities = _infer_episode_layout(episodes)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "expected_episodes_per_task": _legacy_expected_episode_count(tasks, identities),
        "episodes": [dict(episode) for episode in episodes],
    }


def compute_source_content_sha256(episodes: Sequence[Mapping[str, Any]]) -> str:
    """Bind source phases and portable episode identity without hashing host-specific paths."""
    source_entries = []
    for episode in episodes:
        raw = _phases_from_run_dicts(episode["raw_runs"], int(episode["num_actions"]) + 1, "raw_runs")
        source_entries.append(
            {
                "source_task_name": episode["source_task_name"],
                "source_episode_number": episode["source_episode_number"],
                "source_episode_relpath": episode["source_episode_relpath"],
                "overall_instruction": episode["overall_instruction"],
                "action_semantics": episode.get("action_semantics"),
                "phase_semantics": episode["phase_semantics"],
                "phase_before_action": raw[:-1].tolist(),
                "phase_after_action": raw[1:].tolist(),
            }
        )
    return canonical_sha256(source_entries)


def make_unsealed_manifest(
    *,
    repo_id: str,
    hf_dataset_fingerprint: str,
    num_dataset_rows: int,
    action_horizon: int,
    replan_steps: int,
    action_horizon_source: str,
    replan_steps_source: str,
    episodes: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create manifest semantics; byte-level content hashes are added only after writing."""
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("repo_id must be a non-empty string.")
    if not isinstance(hf_dataset_fingerprint, str) or not hf_dataset_fingerprint:
        raise ValueError("hf_dataset_fingerprint must be a non-empty string.")
    num_dataset_rows = _strict_positive_int(num_dataset_rows, "num_dataset_rows")
    action_horizon = _strict_positive_int(action_horizon, "action_horizon")
    replan_steps = _strict_positive_int(replan_steps, "replan_steps")
    if replan_steps > action_horizon:
        raise ValueError(f"replan_steps={replan_steps} exceeds action_horizon={action_horizon}.")
    tasks, identities = _infer_episode_layout(episodes)
    subtask_texts = _subtask_texts_for_tasks(tasks)
    included_episode_count = sum(episode["status"] == "included" for episode in episodes)
    candidate_anchor_count = sum(int(episode["candidate_anchor_count"]) for episode in episodes)
    included_anchor_count = len(records)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "building",
        "tasks": list(tasks),
        "expected_episodes_per_task": _legacy_expected_episode_count(tasks, identities),
        "repo_id": repo_id,
        "hf_dataset_fingerprint": hf_dataset_fingerprint,
        "num_dataset_rows": num_dataset_rows,
        "action_horizon": action_horizon,
        "replan_steps": replan_steps,
        "action_horizon_source": action_horizon_source,
        "replan_steps_source": replan_steps_source,
        "persistence": persistence_for_replan_steps(replan_steps),
        "persistence_formula": PERSISTENCE_FORMULA,
        "phase_alignment": PHASE_ALIGNMENT,
        "boundary_rule": BOUNDARY_RULE,
        "anchor_rule": ANCHOR_RULE,
        "subtask_types": list(SUBTASK_TYPES),
        "subtask_texts": subtask_texts,
        "subtask_texts_sha256": canonical_sha256(subtask_texts),
        "role_word_policy": ROLE_WORD_POLICY,
        "source_content_sha256": compute_source_content_sha256(episodes),
        "candidate_episode_count": len(episodes),
        "included_episode_count": included_episode_count,
        "excluded_episode_count": len(episodes) - included_episode_count,
        "candidate_anchor_count": candidate_anchor_count,
        "included_anchor_count": included_anchor_count,
        "excluded_anchor_count": candidate_anchor_count - included_anchor_count,
    }


def compute_manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    return canonical_sha256(payload)


def seal_manifest(
    manifest: Mapping[str, Any],
    annotations_path: pathlib.Path | str,
    quality_report_path: pathlib.Path | str,
) -> dict[str, Any]:
    """Bind a complete manifest to the exact annotation and quality-report bytes."""
    _require_exact_keys(manifest, _BASE_MANIFEST_FIELDS, "unsealed manifest")
    sealed = dict(manifest)
    sealed["status"] = "complete"
    sealed["annotations_sha256"] = sha256_file(annotations_path)
    sealed["quality_report_sha256"] = sha256_file(quality_report_path)
    sealed["manifest_digest"] = compute_manifest_digest(sealed)
    return sealed


def read_annotation_records(path: pathlib.Path | str) -> list[dict[str, Any]]:
    records = []
    with pathlib.Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank annotation line at {line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid annotation JSON at line {line_number}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Annotation line {line_number} is not an object.")
            records.append(record)
    return records


def load_and_validate_sidecar(
    directory: pathlib.Path | str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Require exactly three complete files and recompute every semantic and byte digest."""
    sidecar_dir = pathlib.Path(directory).expanduser().resolve()
    if not sidecar_dir.is_dir():
        raise FileNotFoundError(f"Oracle sidecar directory not found: {sidecar_dir}")
    expected_names = {ANNOTATIONS_FILENAME, QUALITY_REPORT_FILENAME, MANIFEST_FILENAME}
    actual_names = {path.name for path in sidecar_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"Oracle sidecar files mismatch: expected {sorted(expected_names)}, got {sorted(actual_names)}."
        )

    annotations_path = sidecar_dir / ANNOTATIONS_FILENAME
    quality_path = sidecar_dir / QUALITY_REPORT_FILENAME
    manifest_path = sidecar_dir / MANIFEST_FILENAME
    if not all(path.is_file() for path in (annotations_path, quality_path, manifest_path)):
        raise ValueError("Every oracle sidecar entry must be a regular file.")

    manifest = _read_json_object(manifest_path)
    _validate_manifest_constants(manifest)
    if manifest["annotations_sha256"] != sha256_file(annotations_path):
        raise ValueError("Oracle annotations digest mismatch.")
    if manifest["quality_report_sha256"] != sha256_file(quality_path):
        raise ValueError("Oracle quality-report digest mismatch.")
    if manifest["manifest_digest"] != compute_manifest_digest(manifest):
        raise ValueError("Oracle manifest digest mismatch.")

    records = read_annotation_records(annotations_path)
    quality = _read_json_object(quality_path)
    _validate_quality_and_records(manifest, records, quality)
    return manifest, records, quality


def validate_dataset_binding(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    *,
    repo_id: str,
    fingerprint: str,
    episode_indices: Sequence[Any],
    frame_indices: Sequence[Any],
    task_indices: Sequence[Any],
    task_prompts: Mapping[int, str],
) -> None:
    """Bind included records and excluded candidate anchors to one live LeRobot dataset."""
    _validate_manifest_constants(manifest)
    _validate_quality_and_records(manifest, records, quality)
    if manifest["repo_id"] != repo_id:
        raise ValueError(f"Oracle repo_id mismatch: artifact={manifest['repo_id']!r}, live={repo_id!r}.")
    # HuggingFace fingerprints include local path/cache identity and can change when an
    # unchanged local Parquet dataset is moved. Keep the sealed value for audit, but do
    # not use the ephemeral live value as a content-integrity gate.

    episodes = [_as_scalar_int(value, "episode_index") for value in episode_indices]
    frames = [_as_scalar_int(value, "frame_index") for value in frame_indices]
    tasks = [_as_scalar_int(value, "task_index") for value in task_indices]
    expected_rows = int(manifest["num_dataset_rows"])
    if not (len(episodes) == len(frames) == len(tasks) == expected_rows):
        raise ValueError("LeRobot index columns do not match num_dataset_rows.")
    normalized_prompts = {int(key): str(value) for key, value in task_prompts.items()}

    for audit in quality["episodes"]:
        start = int(audit["global_start_index"])
        end = start + int(audit["num_actions"])
        episode_index = int(audit["lerobot_episode_index"])
        if episodes[start:end] != [episode_index] * (end - start):
            raise ValueError(f"LeRobot episode_index mismatch for episode {episode_index}.")
        if frames[start:end] != list(range(end - start)):
            raise ValueError(f"LeRobot frame_index mismatch for episode {episode_index}.")
        for global_index in range(start, end):
            prompt = normalized_prompts.get(tasks[global_index])
            if prompt != audit["overall_instruction"]:
                raise ValueError(f"LeRobot task prompt mismatch at global_index={global_index}.")

    for record in records:
        global_index = int(record["global_index"])
        if episodes[global_index] != int(record["episode_index"]):
            raise ValueError(f"Record episode_index mismatch at global_index={global_index}.")
        if frames[global_index] != int(record["frame_index"]):
            raise ValueError(f"Record frame_index mismatch at global_index={global_index}.")


class OraclePhaseAnchorDataset:
    """Read-only dense included-frame view that replaces only the language prompt."""

    def __init__(
        self,
        *,
        dataset: Any,
        hf_dataset: Any,
        annotations_dir: pathlib.Path | str,
        action_horizon: int,
        replan_steps: int,
        repo_id: str,
        task_prompts: Mapping[int, str],
    ) -> None:
        manifest, records, quality = load_and_validate_sidecar(annotations_dir)
        timing = rlbench_timing.TimingContract(
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            action_horizon_source=rlbench_timing.ACTION_HORIZON_SOURCE,
            replan_steps_source=rlbench_timing.REPLAN_STEPS_SOURCE,
        )
        rlbench_timing.validate_manifest_timing(manifest, timing)

        fingerprint = getattr(hf_dataset, "_fingerprint", None)
        if fingerprint is None or not str(fingerprint):
            raise ValueError("HuggingFace dataset fingerprint is unavailable.")
        validate_dataset_binding(
            manifest,
            records,
            quality,
            repo_id=repo_id,
            fingerprint=str(fingerprint),
            episode_indices=hf_dataset["episode_index"],
            frame_indices=hf_dataset["frame_index"],
            task_indices=hf_dataset["task_index"],
            task_prompts=task_prompts,
        )

        self._dataset = dataset
        self._global_indices = tuple(int(record["global_index"]) for record in records)
        self._subtasks = tuple(str(record["subtask"]) for record in records)

    def __len__(self) -> int:
        return len(self._global_indices)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        position = index.__index__()
        item = dict(self._dataset[self._global_indices[position]])
        item["prompt"] = self._subtasks[position]
        return item


def _decision_signature(decision: EpisodeDecision) -> tuple[Any, ...]:
    return (
        decision.included,
        decision.reasons,
        tuple(decision.cleaned.clean.tolist()),
        tuple((run.phase, run.start, run.end) for run in decision.cleaned.runs),
        tuple(
            (segment.stable_phase, segment.candidate_phase, segment.start, segment.end)
            for segment in decision.cleaned.suppressed
        ),
        decision.cleaned.longest_unconfirmed_deviation,
        decision.boundaries,
        decision.anchors,
        decision.candidate_anchor_count,
    )


def _validate_manifest_constants(manifest: Mapping[str, Any]) -> None:
    _require_exact_keys(manifest, _MANIFEST_FIELDS, "manifest")
    _strict_positive_int(manifest["schema_version"], "schema_version")
    if manifest["protocol"] != PROTOCOL or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported oracle protocol or schema version.")
    if manifest["status"] != "complete":
        raise ValueError(f"Oracle sidecar is not complete: {manifest['status']!r}.")
    tasks = _canonical_task_subset(manifest["tasks"], "manifest tasks")
    expected_episodes_per_task = manifest["expected_episodes_per_task"]
    if expected_episodes_per_task is not None:
        _strict_positive_int(expected_episodes_per_task, "expected_episodes_per_task")
    expected_subtask_texts = _subtask_texts_for_tasks(tasks)
    if manifest["subtask_types"] != list(SUBTASK_TYPES) or manifest["subtask_texts"] != expected_subtask_texts:
        raise ValueError("Oracle fixed subtask text table mismatch.")
    if manifest["subtask_texts_sha256"] != canonical_sha256(expected_subtask_texts):
        raise ValueError("Oracle fixed subtask text digest mismatch.")
    constants = {
        "persistence_formula": PERSISTENCE_FORMULA,
        "phase_alignment": PHASE_ALIGNMENT,
        "boundary_rule": BOUNDARY_RULE,
        "anchor_rule": ANCHOR_RULE,
        "role_word_policy": ROLE_WORD_POLICY,
    }
    for key, expected in constants.items():
        if manifest[key] != expected:
            raise ValueError(f"Oracle protocol field {key} mismatch.")
    action_horizon = _strict_positive_int(manifest["action_horizon"], "action_horizon")
    replan_steps = _strict_positive_int(manifest["replan_steps"], "replan_steps")
    if replan_steps > action_horizon:
        raise ValueError("Oracle replan_steps exceeds action_horizon.")
    if manifest["persistence"] != persistence_for_replan_steps(replan_steps):
        raise ValueError("Oracle persistence value does not match its formula.")
    _strict_positive_int(manifest["persistence"], "persistence")
    for key in (
        "repo_id",
        "hf_dataset_fingerprint",
        "action_horizon_source",
        "replan_steps_source",
    ):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"Oracle manifest {key} must be a non-empty string.")
    for key in (
        "num_dataset_rows",
        "candidate_episode_count",
        "candidate_anchor_count",
        "included_anchor_count",
    ):
        _strict_positive_int(manifest[key], key)
    for key in ("included_episode_count", "excluded_episode_count", "excluded_anchor_count"):
        _strict_nonnegative_int(manifest[key], key)
    if expected_episodes_per_task is not None:
        expected_episode_count = len(tasks) * expected_episodes_per_task
        if manifest["candidate_episode_count"] != expected_episode_count:
            raise ValueError("Oracle candidate episode count mismatch.")
    if manifest["included_episode_count"] + manifest["excluded_episode_count"] != manifest["candidate_episode_count"]:
        raise ValueError("Oracle included/excluded episode partition mismatch.")
    if manifest["included_anchor_count"] + manifest["excluded_anchor_count"] != manifest["candidate_anchor_count"]:
        raise ValueError("Oracle included/excluded anchor partition mismatch.")
    for key in (
        "source_content_sha256",
        "annotations_sha256",
        "quality_report_sha256",
        "manifest_digest",
    ):
        if not isinstance(manifest[key], str) or _SHA256_RE.fullmatch(manifest[key]) is None:
            raise ValueError(f"Oracle manifest {key} is not a SHA-256 digest.")


def _validate_quality_and_records(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> None:
    _require_exact_keys(quality, _QUALITY_FIELDS, "quality report")
    _strict_positive_int(quality["schema_version"], "quality schema_version")
    if quality["protocol"] != PROTOCOL or quality["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Quality report protocol or schema mismatch.")
    if quality["expected_episodes_per_task"] != manifest["expected_episodes_per_task"]:
        raise ValueError("Quality report expected episode count mismatch.")
    audits = quality["episodes"]
    if not isinstance(audits, list):
        raise ValueError("Quality report episodes must be a list.")
    tasks = _canonical_task_subset(manifest["tasks"], "manifest tasks")
    expected_episodes_per_task = manifest["expected_episodes_per_task"]
    if expected_episodes_per_task is None:
        audit_tasks, expected_identities_tuple = _infer_episode_layout(audits)
        if audit_tasks != tasks:
            raise ValueError("Quality report task set differs from the manifest.")
        expected_identities = list(expected_identities_tuple)
    else:
        expected_episodes_per_task = _strict_positive_int(expected_episodes_per_task, "expected_episodes_per_task")
        expected_identities = [
            (task, number)
            for task in tasks
            for number in sorted(range(expected_episodes_per_task), key=lambda value: f"episode{value}")
        ]
    if len(audits) != len(expected_identities):
        raise ValueError(f"Expected {len(expected_identities)} quality-report episodes, got {len(audits)}.")
    if manifest["candidate_episode_count"] != len(expected_identities):
        raise ValueError("Oracle candidate episode count mismatch.")

    expected_records: list[dict[str, Any]] = []
    global_cursor = 0
    included_episodes = 0
    candidate_anchors = 0
    for expected_episode_index, (audit, identity) in enumerate(zip(audits, expected_identities, strict=True)):
        _require_exact_keys(
            audit,
            _EPISODE_AUDIT_FIELDS,
            f"episode audit {expected_episode_index}",
            optional=frozenset({"action_semantics"}),
        )
        for key in (
            "source_episode_number",
            "lerobot_episode_index",
            "global_start_index",
            "included_anchor_count",
            "longest_unconfirmed_deviation",
        ):
            _strict_nonnegative_int(audit[key], f"episode audit {key}")
        for key in ("num_actions", "persistence", "candidate_anchor_count"):
            _strict_positive_int(audit[key], f"episode audit {key}")
        task, source_number = identity
        if (audit["source_task_name"], audit["source_episode_number"]) != identity:
            raise ValueError(f"Source episode ordering mismatch at audit {expected_episode_index}.")
        if audit["source_episode_relpath"] != f"train/{task}/episode{source_number}":
            raise ValueError(f"Source relative path mismatch at audit {expected_episode_index}.")
        if audit["lerobot_episode_index"] != expected_episode_index:
            raise ValueError(f"LeRobot episode ordering mismatch at audit {expected_episode_index}.")
        if audit["global_start_index"] != global_cursor:
            raise ValueError(f"Global row continuity mismatch at audit {expected_episode_index}.")
        if not isinstance(audit["overall_instruction"], str) or not audit["overall_instruction"]:
            raise ValueError(f"Empty overall instruction at audit {expected_episode_index}.")
        if audit["phase_semantics"] != PHASE_SEMANTICS:
            raise ValueError(f"Source phase semantics mismatch at audit {expected_episode_index}.")
        if (
            not isinstance(audit["status"], str)
            or not isinstance(audit["reasons"], list)
            or not all(isinstance(reason, str) for reason in audit["reasons"])
        ):
            raise ValueError(f"Invalid status or reasons at audit {expected_episode_index}.")
        if not isinstance(audit["boundaries"], list):
            raise ValueError(f"Boundaries must be a list at audit {expected_episode_index}.")
        for boundary in audit["boundaries"]:
            _strict_positive_int(boundary, "phase boundary")
        num_actions = int(audit["num_actions"])
        if audit["persistence"] != manifest["persistence"]:
            raise ValueError(f"Persistence mismatch at audit {expected_episode_index}.")
        raw = _phases_from_run_dicts(audit["raw_runs"], num_actions + 1, "raw_runs")
        clean = _phases_from_run_dicts(audit["clean_runs"], num_actions + 1, "clean_runs")
        decision = assess_source_episode(
            raw,
            num_actions,
            int(manifest["replan_steps"]),
            source_task_name=task,
            source_episode_number=source_number,
        )
        if not np.array_equal(clean, decision.cleaned.clean):
            raise ValueError(f"Clean phase sequence mismatch at audit {expected_episode_index}.")
        expected_suppressed = [segment.to_dict() for segment in decision.cleaned.suppressed]
        _validate_suppressed_dicts(audit["suppressed_segments"], num_actions + 1)
        if audit["suppressed_segments"] != expected_suppressed:
            raise ValueError(f"Suppressed phase segments mismatch at audit {expected_episode_index}.")
        if audit["boundaries"] != list(decision.boundaries):
            raise ValueError(f"Phase boundaries mismatch at audit {expected_episode_index}.")
        if audit["longest_unconfirmed_deviation"] != decision.cleaned.longest_unconfirmed_deviation:
            raise ValueError(f"Unconfirmed deviation length mismatch at audit {expected_episode_index}.")
        expected_status = "included" if decision.included else "excluded"
        if audit["status"] != expected_status or audit["reasons"] != list(decision.reasons):
            raise ValueError(f"Episode decision mismatch at audit {expected_episode_index}.")
        if audit["candidate_anchor_count"] != decision.candidate_anchor_count:
            raise ValueError(f"Candidate anchor count mismatch at audit {expected_episode_index}.")
        if audit["included_anchor_count"] != len(decision.anchors):
            raise ValueError(f"Included anchor count mismatch at audit {expected_episode_index}.")
        if decision.included:
            included_episodes += 1
            expected_records.extend(
                make_annotation_records(
                    source_task_name=task,
                    source_episode_number=source_number,
                    lerobot_episode_index=expected_episode_index,
                    global_start_index=global_cursor,
                    overall_instruction=audit["overall_instruction"],
                    raw_phases=raw,
                    decision=decision,
                )
            )
        candidate_anchors += decision.candidate_anchor_count
        global_cursor += num_actions

    if global_cursor != manifest["num_dataset_rows"]:
        raise ValueError("Quality-report rows do not match num_dataset_rows.")
    if included_episodes != manifest["included_episode_count"]:
        raise ValueError("Quality-report included episode count mismatch.")
    if len(audits) - included_episodes != manifest["excluded_episode_count"]:
        raise ValueError("Quality-report excluded episode count mismatch.")
    if candidate_anchors != manifest["candidate_anchor_count"]:
        raise ValueError("Quality-report candidate anchor count mismatch.")
    if len(expected_records) != manifest["included_anchor_count"]:
        raise ValueError("Quality-report included anchor count mismatch.")
    if candidate_anchors - len(expected_records) != manifest["excluded_anchor_count"]:
        raise ValueError("Quality-report excluded anchor count mismatch.")
    if compute_source_content_sha256(audits) != manifest["source_content_sha256"]:
        raise ValueError("Oracle source-content digest mismatch.")

    if len(records) != len(expected_records):
        raise ValueError(f"Expected {len(expected_records)} annotations, got {len(records)}.")
    for line_number, (actual, expected) in enumerate(zip(records, expected_records, strict=True), start=1):
        _require_exact_keys(actual, _ANNOTATION_FIELDS, f"annotation line {line_number}")
        for key in (
            "global_index",
            "episode_index",
            "frame_index",
            "source_episode_number",
            "raw_phase_before",
            "clean_phase",
        ):
            _strict_nonnegative_int(actual[key], f"annotation {key}")
        if actual != expected:
            raise ValueError(f"Annotation content mismatch at line {line_number}.")


def _phases_from_run_dicts(runs: Any, expected_length: int, name: str) -> np.ndarray:
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{name} must be a non-empty list.")
    phases = np.empty(expected_length, dtype=np.int8)
    cursor = 0
    previous_phase: int | None = None
    for index, run in enumerate(runs):
        _require_exact_keys(run, _RUN_FIELDS, f"{name}[{index}]")
        phase = _strict_positive_int(run["phase"], f"{name}[{index}].phase")
        start = _strict_nonnegative_int(run["start"], f"{name}[{index}].start")
        end = _strict_positive_int(run["end"], f"{name}[{index}].end")
        length = _strict_positive_int(run["length"], f"{name}[{index}].length")
        if phase not in PHASE_VALUES or start != cursor or end - start != length or end > expected_length:
            raise ValueError(f"Invalid {name} entry at index {index}.")
        if phase == previous_phase:
            raise ValueError(f"Adjacent {name} entries repeat phase {phase}.")
        phases[start:end] = phase
        cursor = end
        previous_phase = phase
    if cursor != expected_length:
        raise ValueError(f"{name} covers {cursor} values, expected {expected_length}.")
    return phases


def _validate_suppressed_dicts(segments: Any, observation_count: int) -> None:
    if not isinstance(segments, list):
        raise ValueError("suppressed_segments must be a list.")
    previous_start = -1
    for index, segment in enumerate(segments):
        _require_exact_keys(segment, _SUPPRESSED_FIELDS, f"suppressed_segments[{index}]")
        stable = _strict_positive_int(segment["stable_phase"], "stable_phase")
        candidate = _strict_positive_int(segment["candidate_phase"], "candidate_phase")
        start = _strict_nonnegative_int(segment["start"], "suppressed start")
        end = _strict_positive_int(segment["end"], "suppressed end")
        length = _strict_positive_int(segment["length"], "suppressed length")
        if stable not in PHASE_VALUES or candidate not in PHASE_VALUES or stable == candidate:
            raise ValueError(f"Invalid suppressed phase values at index {index}.")
        if start < previous_start or end - start != length or end > observation_count:
            raise ValueError(f"Invalid suppressed segment bounds at index {index}.")
        previous_start = start


def _read_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _require_exact_keys(value: Any, expected: set[str], name: str, *, optional: frozenset[str] = frozenset()) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected - optional)
    if missing or extra:
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}.")


def _strict_positive_int(value: Any, name: str) -> int:
    parsed = _strict_nonnegative_int(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {parsed}.")
    return parsed


def _as_scalar_int(value: Any, name: str) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be one integer scalar, got shape={array.shape}, dtype={array.dtype}.")
    return int(array.reshape(-1)[0])
