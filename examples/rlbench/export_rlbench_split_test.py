from types import SimpleNamespace
import unittest

import numpy as np

from examples.rlbench import export_rlbench_split as exporter


def _obs(observed, commanded, phase):
    return SimpleNamespace(
        left=SimpleNamespace(joint_positions=np.full(7, observed), gripper_open=1.0),
        right=SimpleNamespace(joint_positions=np.full(7, observed + 1), gripper_open=0.0),
        misc={
            "left_executed_demo_joint_position_action": np.full(7, commanded),
            "left_commanded_gripper_state": 0.0,
            "right_executed_demo_joint_position_action": np.full(7, commanded + 1),
            "right_commanded_gripper_state": 1.0,
            "phase_type": phase,
        },
    )


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


if __name__ == "__main__":
    unittest.main()
