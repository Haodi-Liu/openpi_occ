import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

RLBENCH_STATE_DIM = 16
RLBENCH_ACTION_DIM = 16
RLBENCH_GRIPPER_INDICES = (7, 15)
# Layout id reported via policy metadata so the closed-loop client can verify
# it is decoding actions with the same convention used at training time.
RLBENCH_ACTION_LAYOUT = "rlbench_bimanual_left_first_joint16"
PICK_PLATE_OVERALL_INSTRUCTION = (
    "Pick up the plate, creating sufficient space for a grasp if direct access is obstructed."
)


def make_rlbench_example() -> dict:
    """Creates a random input example for the RLBench bimanual joint policy."""
    return {
        "observation/state": np.random.rand(RLBENCH_STATE_DIM).astype(np.float32),
        "observation/front_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_left_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_right_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": PICK_PLATE_OVERALL_INSTRUCTION,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}")
    return image


def _parse_prompt(prompt) -> str:
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    if isinstance(prompt, np.bytes_):
        return prompt.decode("utf-8")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.item()
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
    return str(prompt)


@dataclasses.dataclass(frozen=True)
class RLBenchInputs(transforms.DataTransformFn):
    """Converts RLBench joint observations into the OpenPI policy input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        _ = self.model_type  # Kept for API symmetry with other policy adapters.

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/front_rgb"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_left_rgb"]),
                "right_wrist_0_rgb": _parse_image(data["observation/wrist_right_rgb"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # `actions_is_pad` is added by LeRobot when `delta_timestamps` clamps the
        # action chunk against an episode boundary. Pass it through unchanged so
        # the model can mask out those padded timesteps in the loss.
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"], dtype=np.bool_)

        if "prompt" in data:
            inputs["prompt"] = _parse_prompt(data["prompt"])

        return inputs


@dataclasses.dataclass(frozen=True)
class RLBenchOutputs(transforms.DataTransformFn):
    """Slices model outputs back to RLBench's 16D left-first joint action space."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :RLBENCH_ACTION_DIM], dtype=np.float32)
        # Gripper dimensions are trained to ~0/1 absolute values; flow-matching
        # noise leaves them in roughly [-0.1, 1.1]. Clipping back into [0, 1]
        # gives the downstream agent a clean continuous value to threshold,
        # without throwing away the model's confidence near the boundaries.
        actions = actions.copy()
        for gripper_idx in RLBENCH_GRIPPER_INDICES:
            actions[:, gripper_idx] = np.clip(actions[:, gripper_idx], 0.0, 1.0)
        return {"actions": actions}
