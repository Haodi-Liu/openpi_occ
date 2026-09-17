"""Single-source runtime timing contract for the π0.5 RLBench hierarchy."""

from collections.abc import Callable
import dataclasses
import os
import pathlib
from typing import Any

import yaml

from openpi.training import config as _config

RLBENCH_CONFIG_NAME = "pi05_rlbench"
OCC_TIMING_FILE_ENV = "OCC_TIMING_FILE"
ACTION_HORIZON_SOURCE = "src/openpi/training/config.py#pi05_rlbench.model.action_horizon"
REPLAN_STEPS_SOURCE = "occ_grasp_models/conf/method/OPENPI_POLICY.yaml#replan_steps"


@dataclasses.dataclass(frozen=True)
class TimingContract:
    """Immutable H/K snapshot shared by generation, training, and serving."""

    action_horizon: int
    replan_steps: int
    action_horizon_source: str
    replan_steps_source: str

    def __post_init__(self) -> None:
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}.")
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                f"Expected 1 <= replan_steps <= action_horizon, got K={self.replan_steps}, H={self.action_horizon}."
            )

    def to_metadata(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


def load_runtime_contract(
    *,
    config_getter: Callable[[str], Any] = _config.get_config,
    occ_policy_config: pathlib.Path | str | None = None,
) -> TimingContract:
    """Read H and K from their sole production sources and freeze the values."""
    train_config = config_getter(RLBENCH_CONFIG_NAME)
    action_horizon = _strict_positive_int(train_config.model.action_horizon, "action_horizon")

    if occ_policy_config is None:
        occ_policy_config = os.environ.get(OCC_TIMING_FILE_ENV)
    if occ_policy_config is None or not str(occ_policy_config).strip():
        raise RuntimeError(
            f"{OCC_TIMING_FILE_ENV} must point to the active occ_grasp_models/conf/method/OPENPI_POLICY.yaml."
        )

    occ_path = pathlib.Path(occ_policy_config).expanduser().resolve()
    if not occ_path.is_file():
        raise FileNotFoundError(f"OCC policy config not found: {occ_path}")
    document = yaml.safe_load(occ_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "replan_steps" not in document:
        raise ValueError(f"OCC policy config is missing replan_steps: {occ_path}")
    replan_steps = _strict_positive_int(document["replan_steps"], "replan_steps")

    return TimingContract(
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        action_horizon_source=ACTION_HORIZON_SOURCE,
        replan_steps_source=REPLAN_STEPS_SOURCE,
    )


def validate_manifest_timing(manifest: dict[str, Any], timing: TimingContract) -> None:
    """Reject artifacts produced under a different timing contract."""
    expected = timing.to_metadata()
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Timing metadata mismatch for {key}: artifact={manifest.get(key)!r}, runtime={value!r}.")


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed
