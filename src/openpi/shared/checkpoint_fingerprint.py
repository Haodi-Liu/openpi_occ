"""Stable provenance fingerprints for oracle-text π0.5 action checkpoints."""

import hashlib
import json
import pathlib
from typing import Any

import flax.traverse_util
import jax
import ml_dtypes
import numpy as np

from openpi.models import model as _model

OFFICIAL_PI05_BASE_ID = "gs://openpi-assets/checkpoints/pi05_base"
FINGERPRINT_PREFIX = "sha256:"
CHECKPOINT_PROVENANCE_PATH = pathlib.Path("pi05_subtask") / "provenance.json"
_CHUNK_ELEMENTS = 8 * 1024 * 1024


def validate_official_source(source_checkpoint_dir: pathlib.Path | str, source_checkpoint_id: str) -> pathlib.Path:
    """Validate the fixed source identity and return its resolved local directory."""
    if source_checkpoint_id != OFFICIAL_PI05_BASE_ID:
        raise ValueError(
            f"Oracle action training is fixed to the official pi05_base source; got {source_checkpoint_id!r}."
        )
    source_dir = pathlib.Path(source_checkpoint_dir).resolve()
    if source_dir.name != "pi05_base":
        raise ValueError(f"Official source directory must resolve to a pi05_base directory, got {source_dir}.")
    if any(part.startswith("pi05_rlbench") for part in source_dir.parts):
        raise ValueError(f"A project-trained RLBench checkpoint cannot be used as the oracle source: {source_dir}")
    if not (source_dir / "params").is_dir():
        raise FileNotFoundError(f"Official source params directory not found: {source_dir / 'params'}")
    return source_dir


def fingerprint_checkpoint(checkpoint_dir: pathlib.Path | str) -> str:
    """Restore checkpoint parameters on host memory and fingerprint the frozen VLM."""
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    params_dir = checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_dir}")
    params = _model.restore_params(params_dir, restore_type=np.ndarray)
    return fingerprint_frozen_vlm(params)


def fingerprint_frozen_vlm(params_or_state: Any) -> str:
    """Hash vision, tied embedding, and main LLM leaves after bfloat16 canonicalization."""
    pure = _to_pure_mapping(params_or_state)
    flat = flax.traverse_util.flatten_dict(pure)
    selected = [(path, value) for path, value in flat.items() if _is_frozen_vlm_path(path)]
    if not selected:
        raise ValueError("No frozen π0.5 VLM parameter leaves were found for fingerprinting.")

    digest = hashlib.sha256()
    digest.update(b"openpi-pi05-frozen-vlm-bfloat16-v1\0")
    for path, value in sorted(selected, key=lambda item: tuple(map(str, item[0]))):
        array = _unwrap_array(value)
        shape = tuple(int(dim) for dim in array.shape)
        path_text = "/".join(map(str, path))
        header = json.dumps(
            {"path": path_text, "shape": shape, "dtype": "bfloat16"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        for chunk in _iter_bfloat16_little_endian_bytes(array):
            digest.update(chunk)
    digest.update(len(selected).to_bytes(8, "little"))
    return FINGERPRINT_PREFIX + digest.hexdigest()


def fingerprint_trainable_action(params_or_state: Any) -> str:
    """Hash the action allowlist at the bfloat16 precision used for rollout."""
    pure = _to_pure_mapping(params_or_state)
    flat = flax.traverse_util.flatten_dict(pure)
    selected = [(path, value) for path, value in flat.items() if _is_trainable_action_path(path)]
    if not selected:
        raise ValueError("No allowlisted action parameter leaves were found for fingerprinting.")

    digest = hashlib.sha256()
    digest.update(b"openpi-pi05-trainable-action-bfloat16-v1\0")
    for path, value in sorted(selected, key=lambda item: tuple(map(str, item[0]))):
        array = _unwrap_array(value)
        shape = tuple(int(dim) for dim in array.shape)
        path_text = "/".join(map(str, path))
        header = json.dumps(
            {"path": path_text, "shape": shape, "dtype": "bfloat16"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        for chunk in _iter_bfloat16_little_endian_bytes(array):
            digest.update(chunk)
    digest.update(len(selected).to_bytes(8, "little"))
    return FINGERPRINT_PREFIX + digest.hexdigest()


def sha256_file(path: pathlib.Path | str) -> str:
    path = pathlib.Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return FINGERPRINT_PREFIX + digest.hexdigest()


def seal_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Add a digest so checkpoint provenance cannot be silently edited."""
    sealed = dict(provenance)
    sealed.pop("provenance_digest", None)
    payload = json.dumps(sealed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sealed["provenance_digest"] = FINGERPRINT_PREFIX + hashlib.sha256(payload).hexdigest()
    return sealed


def write_checkpoint_provenance(assets_dir: pathlib.Path | str, provenance: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(assets_dir) / CHECKPOINT_PROVENANCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal_provenance(provenance)
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_checkpoint_provenance(checkpoint_dir: pathlib.Path | str) -> dict[str, Any]:
    path = pathlib.Path(checkpoint_dir) / "assets" / CHECKPOINT_PROVENANCE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Oracle action checkpoint provenance not found: {path}")
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint provenance JSON: {path}") from exc
    if not isinstance(provenance, dict) or "provenance_digest" not in provenance:
        raise ValueError(f"Checkpoint provenance is missing its digest: {path}")
    if provenance["provenance_digest"] != seal_provenance(provenance)["provenance_digest"]:
        raise ValueError(f"Checkpoint provenance digest mismatch: {path}")
    return provenance


def _to_pure_mapping(params_or_state: Any) -> dict:
    if hasattr(params_or_state, "to_pure_dict"):
        params_or_state = params_or_state.to_pure_dict()
    if not isinstance(params_or_state, dict):
        raise TypeError(f"Expected a parameter mapping or NNX state, got {type(params_or_state).__name__}.")
    if set(params_or_state) == {"params"} and isinstance(params_or_state["params"], dict):
        return params_or_state["params"]
    return params_or_state


def _is_frozen_vlm_path(path: tuple[Any, ...]) -> bool:
    parts = tuple(map(str, path))
    if len(parts) < 2 or parts[0] != "PaliGemma":
        return False
    if parts[1] == "img":
        return True
    if parts[1] != "llm":
        return False
    # The tied embedder is shared. Every expert-specific action leaf in the
    # current flow model has a component whose name ends in `_1`.
    return not any(part.endswith("_1") for part in parts[2:])


def _is_trainable_action_path(path: tuple[Any, ...]) -> bool:
    parts = tuple(map(str, path))
    if not parts:
        return False
    if parts[0] in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
        return True
    return len(parts) >= 3 and parts[:2] == ("PaliGemma", "llm") and any(part.endswith("_1") for part in parts[2:])


def _unwrap_array(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"Fingerprint leaf is not array-like: {type(value).__name__}.")
    return value


def _iter_bfloat16_little_endian_bytes(array: Any):
    size = int(np.prod(array.shape, dtype=np.int64))
    flat = array.reshape((size,))
    for start in range(0, size, _CHUNK_ELEMENTS):
        chunk = flat[start : start + _CHUNK_ELEMENTS]
        if isinstance(chunk, jax.Array):
            chunk = jax.device_get(chunk)
        canonical = np.asarray(chunk, dtype=ml_dtypes.bfloat16)
        yield canonical.view(np.uint16).astype("<u2", copy=False).tobytes(order="C")
