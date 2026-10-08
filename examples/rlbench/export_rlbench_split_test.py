import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from examples.rlbench import export_rlbench_split as exporter


def _obs(observed, commanded, phase, *, physics_step=None, left_gripper=0.0, right_gripper=1.0):
    obs = SimpleNamespace(
        left=SimpleNamespace(joint_positions=np.full(7, observed), gripper_open=1.0),
        right=SimpleNamespace(joint_positions=np.full(7, observed + 1), gripper_open=0.0),
        misc={
            "left_executed_demo_joint_position_action": np.full(7, commanded),
            "left_commanded_gripper_state": left_gripper,
            "right_executed_demo_joint_position_action": np.full(7, commanded + 1),
            "right_commanded_gripper_state": right_gripper,
            "phase_type": phase,
        },
    )
    if physics_step is not None:
        obs.misc["physics_step"] = physics_step
    return obs


def _effective_demo():
    return [
        _obs(1.0, 10.0, 1, physics_step=0, left_gripper=1.0),
        _obs(1.0, 10.0, 1, physics_step=0, left_gripper=1.0),
        _obs(2.0, 20.0, 1, physics_step=1, left_gripper=1.0),
        _obs(2.0, 20.0, 2, physics_step=1, left_gripper=1.0),
        _obs(2.5, 20.0, 2, physics_step=2, left_gripper=1.0),
        _obs(3.0, 30.0, 2, physics_step=23, right_gripper=0.0),
        _obs(3.0, 30.0, 3, physics_step=23, right_gripper=0.0),
        _obs(3.5, 30.0, 4, physics_step=25, right_gripper=0.0),
    ]


def _write_episode(data_dir, number, demo):
    episode_dir = data_dir / "fixture.train" / "all_variations" / "episodes" / f"episode{number}"
    episode_dir.mkdir(parents=True)
    for filename, value in (
        ("low_dim_obs.pkl", demo),
        ("variation_descriptions.pkl", ["Pick up the object."]),
        ("variation_number.pkl", 0),
    ):
        with (episode_dir / filename).open("wb") as stream:
            pickle.dump(value, stream)
    for camera in exporter.CAMERA_DIRS:
        (episode_dir / camera).mkdir()
        for raw in range(len(demo)):
            # The exporter references existing image paths without decoding pixels.
            (episode_dir / camera / f"rgb_{raw:04d}.png").touch()
    return episode_dir


class TestExportRLBenchSplit(unittest.TestCase):
    def test_action_and_phase_alignment(self):
        states, actions, phase_before, phase_after = exporter.extract_transitions(
            [_obs(1.0, 10.0, 1), _obs(2.0, 20.0, 2)]
        )

        np.testing.assert_array_equal(states[0, :7], np.full(7, 1.0))
        np.testing.assert_array_equal(actions[0, :7], np.full(7, 20.0))
        self.assertEqual(actions[0, 7], 0.0)
        np.testing.assert_array_equal(actions[0, 8:15], np.full(7, 21.0))
        self.assertEqual(actions[0, 15], 1.0)
        np.testing.assert_array_equal(phase_before, [1])
        np.testing.assert_array_equal(phase_after, [2])

    def test_missing_command_fails(self):
        obs = _obs(1.0, 10.0, 1)
        del obs.misc["left_commanded_gripper_state"]
        with self.assertRaisesRegex(ValueError, "Missing left action command"):
            exporter.extract_action(obs.misc)

    def test_nonfinite_observed_state_fails(self):
        obs = _obs(1.0, 10.0, 1)
        obs.left.joint_positions[3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite joint_positions"):
            exporter.extract_state(obs)

    def test_missing_phase_fails(self):
        obs = _obs(1.0, 10.0, 1)
        del obs.misc["phase_type"]
        with self.assertRaisesRegex(ValueError, "Missing phase_type"):
            exporter.extract_phase(obs.misc)

    def test_nonintegral_or_out_of_range_phase_fails(self):
        for phase in (True, 1.5, 0, 5, np.nan):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                exporter.extract_phase(_obs(1.0, 10.0, phase).misc)

    def test_recorded_boundaries_preserve_holds_simultaneous_changes_and_phase_updates(self):
        demo = _effective_demo()
        index = exporter.build_effective_index(demo)

        np.testing.assert_array_equal(index["raw_observation_row"], [1, 3, 4, 6, 7])
        np.testing.assert_array_equal(index["raw_command_row"], [0, 2, 4, 5, 7])
        np.testing.assert_array_equal(index["physics_step"], [0, 1, 2, 23, 25])
        np.testing.assert_array_equal(index["command_kind"], ["initial", "step", "step", "step", "step"])
        states, actions, before, after = exporter.extract_transitions(demo, index)
        np.testing.assert_array_equal(states[:, 0], [1.0, 2.0, 2.5, 3.0])
        np.testing.assert_array_equal(actions[:, 0], [20.0, 20.0, 30.0, 30.0])
        np.testing.assert_array_equal(actions[:, 7], [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(actions[:, 15], [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(before, [1, 2, 2, 3])
        np.testing.assert_array_equal(after, [2, 2, 3, 4])

    def test_invalid_physics_steps_are_ambiguous(self):
        for steps in ((None, 1), (0, True), (0, 1.0), (1, 2), (0, -1)):
            with self.subTest(steps=steps), self.assertRaises(exporter.TimingAmbiguityError):
                exporter.build_effective_index([_obs(1.0, 10.0, 1, physics_step=step) for step in steps])

    def test_positive_physics_gaps_keep_only_recorded_boundaries(self):
        initial = _obs(1.0, 10.0, 1, physics_step=0)
        cases = {
            "repeated close": _obs(2.0, 10.0, 1, physics_step=2),
            "one gripper": _obs(2.0, 10.0, 1, physics_step=21, left_gripper=1.0),
            "both grippers": _obs(2.0, 10.0, 1, physics_step=21, left_gripper=1.0, right_gripper=0.0),
            "arm and gripper change": _obs(2.0, 20.0, 1, physics_step=21, left_gripper=1.0),
        }
        for name, completed in cases.items():
            with self.subTest(case=name):
                demo = [initial, completed]
                index = exporter.build_effective_index(demo)
                np.testing.assert_array_equal(index["raw_observation_row"], [0, 1])
                np.testing.assert_array_equal(index["raw_command_row"], [0, 1])
                np.testing.assert_array_equal(index["physics_step"], [0, completed.misc["physics_step"]])
                states, actions, before, after = exporter.extract_transitions(demo, index)
                self.assertEqual(states.shape, (1, 16))
                np.testing.assert_array_equal(actions[0], exporter.extract_action(completed.misc))
                np.testing.assert_array_equal(before, [1])
                np.testing.assert_array_equal(after, [1])

    def test_zero_step_command_conflict_and_no_execution_are_rejected(self):
        initial = _obs(1.0, 10.0, 1, physics_step=0)
        for target in (10.0, 20.0):
            with self.subTest(target=target), self.assertRaises(exporter.TimingAmbiguityError):
                exporter.build_effective_index([initial, _obs(1.0, target, 1, physics_step=0)])

    def test_cli_exports_mapped_images_and_reports_pending_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_episode(root / "raw", 0, _effective_demo())
            pending_source = _write_episode(
                root / "raw", 1, [_obs(1.0, 10.0, 1, physics_step=0), _obs(1.0, 20.0, 1, physics_step=0)]
            )
            output = root / "export"
            argv = [
                "export_rlbench_split.py",
                "--data_dir",
                str(root / "raw"),
                "--split",
                "train",
                "--output_dir",
                str(output),
                "--tasks",
                "fixture",
                "--effective_commands",
            ]
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                exporter.main()

            episode = output / "train" / "fixture" / "episode0"
            metadata = json.loads((episode / "meta.json").read_text())
            self.assertEqual(metadata["action_semantics"], exporter.EFFECTIVE_ACTION_SEMANTICS)
            self.assertEqual(metadata["num_observations"], 5)
            self.assertEqual(metadata["raw_num_observations"], 8)
            self.assertEqual(metadata["num_transitions"], 4)
            for camera in exporter.CAMERA_DIRS:
                self.assertEqual(
                    metadata[camera], [str(source / camera / f"rgb_{raw:04d}.png") for raw in (1, 3, 4, 6)]
                )
            with np.load(episode / metadata["effective_index_file"], allow_pickle=False) as index:
                np.testing.assert_array_equal(index["raw_observation_row"], [1, 3, 4, 6, 7])
                np.testing.assert_array_equal(index["raw_command_row"], [0, 2, 4, 5, 7])
                np.testing.assert_array_equal(index["physics_step"], [0, 1, 2, 23, 25])
                np.testing.assert_array_equal(index["command_kind"], ["initial", "step", "step", "step", "step"])
            np.testing.assert_array_equal(np.load(episode / "phase_after_action.npy"), [2, 2, 3, 4])
            self.assertEqual(np.load(episode / "actions.npy").shape, (4, 16))

            report = json.loads((output / "export_report.json").read_text())
            self.assertEqual(report["total_episodes"], 1)
            self.assertEqual(report["total_transitions"], 4)
            self.assertEqual(report["total_pending_episodes"], 1)
            pending = report["tasks"][0]["pending"]
            self.assertEqual(pending[0]["source_episode_dir"], str(pending_source))
            self.assertIn("Different commands at one physics step", pending[0]["reason"])
            self.assertFalse((episode.parent / "episode1").exists())

    def test_legacy_export_does_not_require_physics_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_episode(root / "raw", 0, [_obs(1.0, 10.0, 1), _obs(2.0, 20.0, 2)])
            output = root / "export"
            summary = exporter.export_episode(source, output, "fixture")
            metadata = json.loads((output / "meta.json").read_text())
            self.assertEqual(summary["num_observations"], 2)
            self.assertEqual(summary["num_transitions"], 1)
            self.assertEqual(metadata["action_semantics"], exporter.ACTION_SEMANTICS)
            self.assertIsNone(metadata["effective_index_file"])
            self.assertFalse((output / "effective_index.npz").exists())
            np.testing.assert_array_equal(np.load(output / "actions.npy")[:, 0], [20.0])

    def test_invalid_data_still_raises_before_creating_episode_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = _effective_demo()
            del demo[3].misc["phase_type"]
            source = _write_episode(root / "raw", 0, demo)
            output = root / "export"
            with self.assertRaisesRegex(ValueError, "Missing phase_type"):
                exporter.export_episode(source, output, "fixture", effective_commands=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
# Keep unittest assertions so these tests run in ppi without a pytest dependency.
# ruff: noqa: PT009, PT027
