import types

import pytest

from openpi.shared import rlbench_timing


def _config_getter(action_horizon: int):
    def get_config(name: str):
        assert name == rlbench_timing.RLBENCH_CONFIG_NAME
        return types.SimpleNamespace(model=types.SimpleNamespace(action_horizon=action_horizon))

    return get_config


def _write_occ_config(path, *, replan_steps: int) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(f"replan_steps: {replan_steps}\n", encoding="utf-8")


def test_runtime_contract_uses_environment_path_and_stable_source(tmp_path, monkeypatch):
    first_config = tmp_path / "machine_a" / "OPENPI_POLICY.yaml"
    second_config = tmp_path / "machine_b" / "OPENPI_POLICY.yaml"
    _write_occ_config(first_config, replan_steps=10)
    _write_occ_config(second_config, replan_steps=10)

    monkeypatch.setenv(rlbench_timing.OCC_TIMING_FILE_ENV, str(first_config))
    first = rlbench_timing.load_runtime_contract(config_getter=_config_getter(20))
    monkeypatch.setenv(rlbench_timing.OCC_TIMING_FILE_ENV, str(second_config))
    second = rlbench_timing.load_runtime_contract(config_getter=_config_getter(20))

    assert first == second
    assert first.replan_steps_source == rlbench_timing.REPLAN_STEPS_SOURCE


def test_explicit_occ_path_overrides_environment(tmp_path, monkeypatch):
    environment_config = tmp_path / "environment" / "OPENPI_POLICY.yaml"
    explicit_config = tmp_path / "explicit" / "OPENPI_POLICY.yaml"
    _write_occ_config(environment_config, replan_steps=5)
    _write_occ_config(explicit_config, replan_steps=7)
    monkeypatch.setenv(rlbench_timing.OCC_TIMING_FILE_ENV, str(environment_config))

    timing = rlbench_timing.load_runtime_contract(
        config_getter=_config_getter(20),
        occ_policy_config=explicit_config,
    )

    assert timing.replan_steps == 7
    assert timing.replan_steps_source == rlbench_timing.REPLAN_STEPS_SOURCE


def test_runtime_contract_requires_environment_path(monkeypatch):
    monkeypatch.delenv(rlbench_timing.OCC_TIMING_FILE_ENV, raising=False)

    with pytest.raises(RuntimeError, match=rlbench_timing.OCC_TIMING_FILE_ENV):
        rlbench_timing.load_runtime_contract(config_getter=_config_getter(20))
