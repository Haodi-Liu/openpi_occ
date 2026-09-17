from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.shared import checkpoint_fingerprint
from openpi.training import checkpoints
from openpi.training import config as _config
from openpi.training import oracle_phase_sidecar as oracle
from openpi.training import utils as training_utils


def _toy_model(vlm_value: float, action_value: float) -> nnx.Dict:
    return nnx.Dict(
        PaliGemma=nnx.Dict(img=nnx.Dict(kernel=nnx.Param(jnp.array([vlm_value])))),
        action_in_proj=nnx.Dict(kernel=nnx.Param(jnp.array([action_value]))),
    )


@pytest.mark.parametrize("train_scope", ["action_only", "full_model"])
@pytest.mark.parametrize("ema_decay", [None, 0.999])
def test_checkpoint_fingerprints_match_exported_params_and_restore_training_state(tmp_path, train_scope, ema_decay):
    source_params = nnx.state(_toy_model(1.0, 1.0))
    full_model = train_scope == "full_model"
    model_def, params = nnx.split(_toy_model(3.0 if full_model else 1.0, 3.0))
    ema_params = nnx.state(_toy_model(2.0 if full_model else 1.0, 2.0)) if ema_decay is not None else None
    tx = optax.adam(0.1)
    _, opt_state = tx.update(jax.tree.map(jnp.ones_like, params), tx.init(params), params)
    state = training_utils.TrainState(
        step=jnp.array(5),
        params=params,
        model_def=model_def,
        opt_state=opt_state,
        tx=tx,
        ema_decay=ema_decay,
        ema_params=ema_params,
    )
    data_config = _config.DataConfig(
        checkpoint_provenance={
            "schema_version": 2 if full_model else 1,
            "action_trainable_protocol": (
                oracle.FULL_MODEL_TRAINABLE_PROTOCOL if full_model else oracle.ACTION_TRAINABLE_PROTOCOL
            ),
            "source_vlm_fingerprint": checkpoint_fingerprint.fingerprint_frozen_vlm(source_params),
            "source_action_fingerprint": checkpoint_fingerprint.fingerprint_trainable_action(source_params),
        }
    )
    data_loader = SimpleNamespace(data_config=lambda: data_config)
    # Inspect the exact items handed to Orbax without exercising its I/O machinery.
    manager = mock.Mock()
    checkpoints.save_state(manager, state, data_loader, step=5)
    manager.save.assert_called_once()
    saved_step, items = manager.save.call_args.args
    assert saved_step == 5
    step_dir = tmp_path / str(saved_step)
    items["assets"](step_dir / "assets")
    exported = jax.tree.map(
        lambda array: np.asarray(array, dtype=jnp.bfloat16), items["params"]["params"].to_pure_dict()
    )
    provenance = checkpoint_fingerprint.load_checkpoint_provenance(step_dir)
    expected_params = ema_params if ema_decay is not None else params
    expected_action = checkpoint_fingerprint.fingerprint_trainable_action(expected_params)
    assert provenance["checkpoint_action_fingerprint"] == expected_action
    assert checkpoint_fingerprint.fingerprint_trainable_action(exported) == expected_action
    expected_vlm = checkpoint_fingerprint.fingerprint_frozen_vlm(expected_params)
    assert checkpoint_fingerprint.fingerprint_frozen_vlm(exported) == expected_vlm
    if full_model:
        assert provenance["checkpoint_vlm_fingerprint"] == expected_vlm
    else:
        assert expected_vlm == provenance["source_vlm_fingerprint"]
        assert "checkpoint_vlm_fingerprint" not in provenance

    # Resume must recover both live and EMA weights, plus optimizer state.
    manager.restore.return_value = {"train_state": items["train_state"], "params": items["params"]}
    template = jax.tree.map(jnp.zeros_like, state)
    restored = checkpoints.restore_state(manager, template, data_loader, step=5)
    assert jax.tree.structure(restored) == jax.tree.structure(state)
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(actual, expected)
